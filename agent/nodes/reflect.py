import logging

from pydantic import BaseModel, Field

from agent.utils.llm import LLM
from storage.models import search_error_records_semantic
from agent.models.nodes import ReflectResult
from agent.models.value_objects import LogicChainStep

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


def reflect(state: dict) -> dict:
    """
    矛盾反思节点：当 fact_check 检测到矛盾时，判断是已有知识错误还是 AI 回答错误。

    三种结论：
    - stored_knowledge_wrong: 知识库中的信息已过时 → 记录错误后搜索纠正
    - answer_wrong: 本次生成的回答是幻觉/错误 → 记录错误后搜索纠正
    - unresolved: 无法判断（可能两者在不同上下文中都正确） → 直接返回
    """
    contradiction_details = state.get("contradiction_details", "")
    answer = state.get("answer", "")
    knowledge_ids = state.get("contradiction_knowledge_ids", [])
    knowledge_texts = state.get("contradiction_knowledge_texts", [])
    severity = state.get("contradiction_severity", "low")
    confidence = state.get("confidence", 0.0)
    correction_attempts = state.get("correction_attempts", 0)
    user_message = state.get("user_message", "")

    logger.info("矛盾反思: severity=%s, attempts=%d", severity, correction_attempts)

    # 搜索历史类似错误记录，帮助 LLM 理解上下文
    error_lessons = []
    try:
        error_lessons = search_error_records_semantic(user_message, limit=3)
    except Exception as e:
        logger.warning("历史错误记录搜索失败: %s", e)

    # 构造 LLM 提示词：分析矛盾的根源
    prompt_parts = [
        f"你正在分析一个知识矛盾。需要判断是已有知识错误、还是新生成的回答错误。\n\n",
        f"当前回答：\n{answer}\n\n",
        f"矛盾详情：\n{contradiction_details}\n\n",
        f"已有的知识文本：\n" + "\n".join(f"- {t}" for t in knowledge_texts) if knowledge_texts else "无",
        f"\n\n矛盾严重程度：{severity}",
        f"\n原始回答置信度：{confidence}",
        f"\n当前修正尝试：{correction_attempts + 1} / 2（最多 2 次）",
    ]

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
    )

    prompt = "\n".join(prompt_parts)

    result = LLM.generate_structured(prompt, ReflectionOutput, use_language=False)

    logger.info("反思结论: source=%s, needs_search=%s",
                result.source, result.needs_verification_search)
    if result.suggested_correction:
        logger.info("建议修正: %s", result.suggested_correction[:80])

    return ReflectResult(
        reflection_result=result.source,
        reflection_reasoning=result.reasoning,
        reflection_correction=result.suggested_correction,
        force_web_search=result.needs_verification_search,
        logic_chain=[LogicChainStep(
            node="reflect",
            action=f"矛盾分析: {result.source}",
            reasoning=result.reasoning,
        )],
    ).model_dump()
