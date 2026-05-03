import logging

from pydantic import BaseModel, Field

from agent.utils.llm import LLM
from storage.models import search_knowledge_points_semantic

logger = logging.getLogger(__name__)


class FactCheckOutput(BaseModel):
    has_contradiction: bool = Field(
        description="Whether the answer contradicts any stored knowledge"
    )
    explanation: str = Field(
        description="Explanation of the contradiction, or empty if none"
    )
    severity: str = Field(
        description="'high' for clear contradiction, 'medium' for partial inconsistency, 'low' for none"
    )


def fact_check(state: dict) -> dict:
    answer = state.get("answer", "")
    if not answer:
        logger.info("Skipping fact check: no answer")
        return {}

    # 用 answer 做语义搜索，查找相关的已有知识
    try:
        related = search_knowledge_points_semantic(answer, limit=5)
    except Exception as e:
        logger.warning("Semantic search for fact check failed: %s", e)
        return {}

    if not related:
        logger.info("Fact check: no related knowledge found, skipping")
        return {"contradiction_found": False, "contradiction_details": ""}

    # 构造检查 prompt
    knowledge_text = "\n".join(f"- {k['knowledge_text']}" for k in related)
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
    }
