import logging

from agent.tools.web_search import *

logger = logging.getLogger(__name__)


def search_web_node(state: dict) -> dict:
    query = state["user_message"]
    logger.info("Searching web for: '%s'", query[:60])
    try:
        results = search_web_from_baidu(query)
    except Exception as e:
        logger.info(f"Web search from Baidu error:{e}")
        results = search_web(query)
    logger.info("Web search returned %d results", len(results))
    return {"search_results": results}
