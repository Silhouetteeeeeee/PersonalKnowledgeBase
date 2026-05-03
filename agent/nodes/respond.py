import logging

logger = logging.getLogger(__name__)


def respond(state: dict) -> dict:
    answer = state.get("answer", "")
    logger.info("Responding with answer (len=%d): '%s'", len(answer), answer[:80])
    return {"final_response": answer}
