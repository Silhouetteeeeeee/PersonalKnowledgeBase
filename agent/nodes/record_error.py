import logging

from storage.models import save_error_record_with_embedding

logger = logging.getLogger(__name__)


def record_error(state: dict) -> dict:
    correction_attempts = state.get("correction_attempts", 0)
    user_message = state.get("user_message", "")
    wrong_answer = state.get("answer", "")
    correct_answer = state.get("reflection_correction", "")
    contradiction_details = state.get("contradiction_details", "")

    logger.info("Recording error for: '%s'", user_message[:50])

    record = {
        "user_message": user_message,
        "wrong_answer": wrong_answer,
        "correct_answer": correct_answer,
        "contradiction_details": contradiction_details,
        "error_type": "hallucination_or_error",
    }

    try:
        eid = save_error_record_with_embedding(record)
        logger.info("Saved error record (id=%d)", eid)
    except Exception as e:
        logger.error("Failed to save error record: %s", e)

    return {
        "correction_attempts": correction_attempts + 1,
        "error_recorded": True,
        "logic_chain": [{
            "node": "record_error",
            "action": "记录回答错误",
            "reasoning": f"记录本次回答错误，问题: {user_message[:50]}"
                         + (f"，修正建议: {correct_answer[:50]}" if correct_answer else ""),
        }],
    }
