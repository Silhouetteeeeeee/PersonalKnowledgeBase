import logging
import re

from server.url_processor import fetch_urls_concurrent

logger = logging.getLogger(__name__)

_URL_PATTERN = re.compile(r'https?://[^\s]+')


def parse(state: dict) -> dict:
    user_message = state["user_message"].strip()
    logger.info("Parsed message from %s: '%s'", state.get("user_id", "unknown"), user_message[:60])

    result = {
        "user_message": user_message,
        "user_id": state.get("user_id", "unknown"),
        "timestamp": state.get("timestamp", ""),
    }

    # URL 提取 + 并发抓取
    urls = _URL_PATTERN.findall(user_message)
    if urls:
        url_contents = fetch_urls_concurrent(urls)
        result["url_contents"] = url_contents
        logger.info("Fetched %d URLs from message", len(url_contents))
    else:
        result["url_contents"] = []

    return result
