import logging

from pydantic import BaseModel, Field

from agent.utils.llm import LLM
from storage.models import (
    save_knowledge_points_bulk_with_embeddings,
    ensure_category,
    find_similar_knowledge,
)

logger = logging.getLogger(__name__)


class DistilledPoint(BaseModel):
    knowledge_text: str = Field(
        description="A concise, standalone knowledge point distilled from the Q&A"
    )
    tags: list[str] = Field(description="Relevant tags for this knowledge point")


class DistillOutput(BaseModel):
    category: str = Field(description="Category for the knowledge, e.g. 'databases/redis'")
    knowledge_points: list[DistilledPoint] = Field(
        description="Knowledge points distilled from the Q&A"
    )


def store(state: dict) -> dict:
    if not state.get("needs_store", True):
        logger.info("Skipping store: needs_store=False")
        return {}

    if not state.get("answer"):
        logger.info("Skipping store: no answer")
        return {}

    if state.get("contradiction_found"):
        logger.info("Skipping store: contradiction detected")
        return {}

    logger.info("Distilling knowledge from Q&A...")
    result = LLM.generate_structured(
        f"Distill the following Q&A into concise, standalone knowledge points.\n\n"
        f"Question: {state['user_message']}\n"
        f"Answer: {state['answer']}",
        DistillOutput,
        use_language=False,
    )

    ensure_category(result.category)

    # Dedup: skip knowledge points that are semantically similar to existing ones
    new_points = []
    for kp in result.knowledge_points:
        try:
            similar = find_similar_knowledge(kp.knowledge_text, threshold=0.25)
        except Exception as e:
            logger.warning("Dedup embedding failed for '%s': %s, saving without dedup",
                           kp.knowledge_text[:30], e)
            similar = []
        if similar:
            logger.info("Skipping duplicate knowledge: '%s' (distance=%.3f)",
                         kp.knowledge_text[:50], similar[0].get("distance", 0))
            continue
        new_points.append(kp)

    if not new_points:
        logger.info("All knowledge points already exist, nothing to store")
        return {}

    knowledge_points = [
        {
            "knowledge_text": kp.knowledge_text,
            "source_question": state["user_message"],
            "category": result.category,
            "tags": kp.tags,
        }
        for kp in new_points
    ]
    save_knowledge_points_bulk_with_embeddings(knowledge_points)

    return {"category": result.category}
