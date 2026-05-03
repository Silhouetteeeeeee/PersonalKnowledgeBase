import logging

from agent.utils.llm import LLM

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the question using the web search results "
    "provided. Be concise and accurate. If the search results don't contain enough "
    "information, say so and provide your best answer. But if you still don't know, "
    "please just say you don't know politely."
)

chain = LLM.build_chain(
    SYSTEM_PROMPT,
    "Web search results:\n{search_results}\n\nQuestion: {question}",
)


def regenerate(state: dict) -> dict:
    search_text = "\n\n".join(state.get("search_results", []))
    if not search_text:
        logger.info("No search results, keeping original answer")
        return {"answer": state.get("answer", "")}

    logger.info("Regenerating answer with %d search results", len(state.get("search_results", [])))
    response = chain.invoke({
        "search_results": search_text,
        "question": state["user_message"],
    })
    logger.info("Regenerated answer: %s", response[:80])
    return {"answer": response}
