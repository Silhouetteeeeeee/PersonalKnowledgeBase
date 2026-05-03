import logging

from pydantic import BaseModel, Field
from langchain_deepseek import ChatDeepSeek
from server.config import LLM_MODEL, LLM_TEMPERATURE
from agent.utils import with_retry
from storage.models import (
    save_knowledge_points_bulk,
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


model = ChatDeepSeek(model=LLM_MODEL, temperature=LLM_TEMPERATURE)
structured_model = model.with_structured_output(DistillOutput)


def store(state: dict) -> dict:
    # Skip storage if LLM decided this doesn't need storing
    if not state.get("needs_store", True):
        logger.info("Skipping store: needs_store=False")
        return {}

    if not state.get("answer"):
        logger.info("Skipping store: no answer")
        return {}

    logger.info("Distilling knowledge from Q&A...")
    result = with_retry(lambda: structured_model.invoke(
        f"Distill the following Q&A into concise, standalone knowledge points.\n\n"
        f"Question: {state['user_message']}\n"
        f"Answer: {state['answer']}"
    ))

    ensure_category(result.category)

    # Dedup: check each knowledge point against existing ones
    new_points = []
    for kp in result.knowledge_points:
        similar = find_similar_knowledge(kp.knowledge_text)
        if similar:
            logger.info("Skipping duplicate knowledge: '%s' (found %d similar)", kp.knowledge_text[:50], len(similar))
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
    save_knowledge_points_bulk(knowledge_points)

    return {"category": result.category}
