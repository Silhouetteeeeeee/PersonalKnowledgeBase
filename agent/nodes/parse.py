import logging

logger = logging.getLogger(__name__)


def parse(state: dict) -> dict:
    user_message = state["user_message"].strip()
    logger.info("Parsed message from %s: '%s'", state.get("user_id", "unknown"), user_message[:60])
    return {
        "user_message": user_message,
        "user_id": state.get("user_id", "unknown"),
        "timestamp": state.get("timestamp", ""),
    }
