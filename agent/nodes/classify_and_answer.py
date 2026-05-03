import logging

from pydantic import BaseModel, Field

from agent.utils.llm import LLM

logger = logging.getLogger(__name__)


class ClassifyOutput(BaseModel):
    category: str = Field(
        description="Category hierarchy, e.g. 'programming/python' or 'life/health'"
    )
    answer: str = Field(description="Answer to the question")
    confidence: float = Field(
        description="Confidence from 0.0 to 1.0"
    )
    needs_search: bool = Field(
        description="Whether web search is needed for accuracy"
    )
    needs_store: bool = Field(
        description="Whether this Q&A should be stored as knowledge. "
                    "True for factual/educational Q&A, False for casual chat/greetings"
    )


def classify_and_answer(state: dict) -> dict:
    context = ""
    if state.get("stored_knowledge"):
        context = "Relevant past knowledge:\n"
        for k in state["stored_knowledge"]:
            context += f"- {k['knowledge_text']}\n"

    prompt = f"{context} Question: {state['user_message']}"
    logger.info("Classifying question (stored_knowledge=%d)", len(state.get("stored_knowledge", [])))

    result = LLM.generate_structured(prompt, ClassifyOutput)

    logger.info("Classified as category='%s' confidence=%.2f needs_search=%s needs_store=%s",
                result.category, result.confidence, result.needs_search, result.needs_store)
    logger.info("Answer: %s", result.answer[:80])

    return {
        "category": result.category,
        "answer": result.answer,
        "confidence": result.confidence,
        "needs_search": result.needs_search,
        "needs_store": result.needs_store,
    }
