import time
import logging

logger = logging.getLogger(__name__)

def get_language_instruction() -> str:
    """Return a prompt instruction for the configured output language.

    Returns empty string when English (default), so no extra tokens are used.
    Only applied to user-facing agents (analysts, portfolio manager).
    Internal debate agents stay in English for reasoning quality.
    """
    from server.config import OUTPUT_LANGUAGE as language
    if language.strip().lower() == "english":
        return ""
    return f" Write your entire response in {language}."



def with_retry(fn, max_retries=2, delay=1):
    """Call fn with retry. Retries once on failure (2 attempts total)."""
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            logger.warning("LLM call failed (attempt %d/%d): %s", attempt + 1, max_retries, e)
            time.sleep(delay)
    return None
