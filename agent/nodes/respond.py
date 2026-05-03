import logging

logger = logging.getLogger(__name__)


def respond(state: dict) -> dict:
    answer = state.get("answer", "")

    if state.get("contradiction_found"):
        details = state.get("contradiction_details", "")
        warning = f"\n\n[矛盾警告] {details}"
        answer += warning
        logger.info("Appended contradiction warning to response")

    logger.info("Responding with answer (len=%d): '%s'", len(answer), answer[:80])
    return {"final_response": answer}
