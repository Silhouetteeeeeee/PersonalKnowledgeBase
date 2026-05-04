import logging

from pydantic import BaseModel, Field

from agent.utils.llm import LLM
from storage.models import search_knowledge_points_semantic, get_knowledge_status

logger = logging.getLogger(__name__)


class FactCheckOutput(BaseModel):
    reasoning_trace: str = Field(
        description="Step-by-step reasoning checking for contradictions"
    )
    has_contradiction: bool = Field(
        description="Whether the answer contradicts any stored knowledge"
    )
    explanation: str = Field(
        description="Explanation of the contradiction, or empty if none"
    )
    severity: str = Field(
        description="'high' for clear contradiction, 'medium' for partial inconsistency, 'low' for none"
    )


def _search_related(state: dict) -> tuple[list[dict], str]:
    """Search for active (non-deprecated) knowledge related to the answer."""
    answer = state.get("answer", "")
    if not answer:
        return [], ""

    try:
        related = search_knowledge_points_semantic(answer, limit=5)
    except Exception as e:
        logger.warning("Semantic search for fact check failed: %s", e)
        return [], ""

    if not related:
        return [], ""

    # Filter out deprecated knowledge
    active_related = []
    for k in related:
        status = get_knowledge_status(k["id"])
        if status not in ("deprecated",):
            active_related.append(k)

    if not active_related:
        return [], ""

    knowledge_text = "\n".join(f"- {k['knowledge_text']}" for k in active_related)
    return active_related, knowledge_text


def fact_check(state: dict) -> dict:
    active_related, knowledge_text = _search_related(state)
    if not active_related:
        logger.info("Fact check: no active related knowledge found, skipping")
        return {"contradiction_found": False, "contradiction_details": ""}

    answer = state["answer"]

    # 构造检查 prompt
    prompt = (
        f"请判断以下回答是否与已有的知识相矛盾。\n\n"
        f"已有的知识：\n{knowledge_text}\n\n"
        f"本次的回答：\n{answer}\n\n"
        f"如果回答与任何已有知识矛盾，请指出矛盾点和严重程度。"
    )

    result = LLM.generate_structured(prompt, FactCheckOutput, use_language=False)

    if result.has_contradiction:
        logger.info(
            "Contradiction detected (severity=%s): %s",
            result.severity, result.explanation[:100],
        )

    return {
        "contradiction_found": result.has_contradiction,
        "contradiction_details": result.explanation if result.has_contradiction else "",
        "contradiction_severity": result.severity,
        "contradiction_knowledge_ids": [k["id"] for k in active_related] if result.has_contradiction else [],
        "contradiction_knowledge_texts": [k["knowledge_text"] for k in active_related] if result.has_contradiction else [],
        "logic_chain": [{
            "node": "fact_check",
            "action": "矛盾检测" if result.has_contradiction else "矛盾检测通过",
            "reasoning": result.reasoning_trace,
            "severity": result.severity,
            "contradiction": result.explanation if result.has_contradiction else "",
        }],
    }
