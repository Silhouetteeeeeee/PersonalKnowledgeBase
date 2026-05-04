import logging
import os

from pydantic import BaseModel, Field

from agent.utils.llm import LLM
from storage.models import query_knowledge_reasoning_path, search_error_records_semantic

logger = logging.getLogger(__name__)


class ReflectionOutput(BaseModel):
    source: str = Field(
        description="'stored_knowledge_wrong' if the stored knowledge is outdated/incorrect, "
                    "'answer_wrong' if the generated answer is a hallucination/error, "
                    "'unresolved' if it cannot determine the source"
    )
    reasoning: str = Field(description="Detailed reasoning for the conclusion, referencing the original reasoning trace if applicable")
    needs_verification_search: bool = Field(
        description="True if a web search should verify the correction"
    )
    suggested_correction: str = Field(
        description="If stored_knowledge_wrong: corrected version of the knowledge. "
                    "If answer_wrong: corrected answer. Empty if unresolved."
    )


def _read_reasoning_trace(knowledge_ids: list[int]) -> str:
    """Read the original reasoning trace MD files for the given knowledge points."""
    traces = []
    for kid in knowledge_ids:
        log_path = query_knowledge_reasoning_path(kid)
        if log_path and os.path.exists(log_path):
            try:
                with open(log_path, "r", encoding="utf-8") as f:
                    content = f.read(3000)  # Limit to avoid huge prompts
                    traces.append(f"--- 知识点 {kid} 的原始推理链路 ---\n{content}")
                logger.info("Read reasoning trace for knowledge %d from %s", kid, log_path)
            except Exception as e:
                logger.warning("Failed to read reasoning trace for knowledge %d: %s", kid, e)
        else:
            logger.info("No reasoning trace file for knowledge %d (path=%s)", kid, log_path)
    return "\n\n".join(traces)


def reflect(state: dict) -> dict:
    contradiction_details = state.get("contradiction_details", "")
    answer = state.get("answer", "")
    knowledge_ids = state.get("contradiction_knowledge_ids", [])
    knowledge_texts = state.get("contradiction_knowledge_texts", [])
    severity = state.get("contradiction_severity", "low")
    confidence = state.get("confidence", 0.0)
    correction_attempts = state.get("correction_attempts", 0)
    user_message = state.get("user_message", "")

    logger.info("Reflecting on contradiction (severity=%s, attempts=%d)", severity, correction_attempts)

    # Read original reasoning traces for the conflicting knowledge points
    original_trace = _read_reasoning_trace(knowledge_ids)

    # Search for similar historical error records
    error_lessons = []
    try:
        error_lessons = search_error_records_semantic(user_message, limit=3)
    except Exception as e:
        logger.warning("Error record search failed: %s", e)

    # Build prompt
    prompt_parts = [
        f"你正在分析一个知识矛盾。需要判断是已有知识错误、还是新生成的回答错误。\n\n",
        f"当前回答：\n{answer}\n\n",
        f"矛盾详情：\n{contradiction_details}\n\n",
        f"已有的知识文本：\n" + "\n".join(f"- {t}" for t in knowledge_texts) if knowledge_texts else "无",
        f"\n\n矛盾严重程度：{severity}",
        f"\n原始回答置信度：{confidence}",
        f"\n当前修正尝试：{correction_attempts + 1} / 2",
    ]

    if original_trace:
        prompt_parts.append(f"\n\n知识点入库时的原始推理链路：\n{original_trace}")
        prompt_parts.append("\n（请分析原始推理是否有缺陷，或知识是否已过时）")

    if error_lessons:
        lessons_text = "\n".join(
            f"- 问题：{e['user_message'][:50]} | 错误：{e['wrong_answer'][:50]} | 修正：{e['correct_answer'][:50]}"
            for e in error_lessons
        )
        prompt_parts.append(f"\n\n历史类似错误记录（供参考）：\n{lessons_text}")

    prompt_parts.append(
        "\n\n请判断矛盾的根源："
        "\n1. stored_knowledge_wrong — 知识库中的数据已过时或不正确"
        "\n2. answer_wrong — 新生成的回答是幻觉或错误"
        "\n3. unresolved — 无法确定"
        "\n\n注意：如果 severity 为 low，可能两者在不同上下文中都正确。"
        "\n如果原始推理链路显示知识入库时本身就存在推理缺陷，更可能归因于 stored_knowledge_wrong。"
    )

    prompt = "\n".join(prompt_parts)

    result = LLM.generate_structured(prompt, ReflectionOutput, use_language=False)

    logger.info("Reflection result: source=%s, needs_search=%s",
                result.source, result.needs_verification_search)
    if result.suggested_correction:
        logger.info("Suggested correction: %s", result.suggested_correction[:80])

    return {
        "reflection_result": result.source,
        "reflection_reasoning": result.reasoning,
        "reflection_correction": result.suggested_correction,
        "force_web_search": result.needs_verification_search,
        "logic_chain": [{
            "node": "reflect",
            "action": f"矛盾分析: {result.source}",
            "reasoning": result.reasoning,
            "source": result.source,
            "suggested_correction": result.suggested_correction[:200] if result.suggested_correction else "",
        }],
    }
