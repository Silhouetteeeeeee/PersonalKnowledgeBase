import time
import logging

logger = logging.getLogger(__name__)


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
