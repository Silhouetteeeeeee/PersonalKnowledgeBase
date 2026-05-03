import logging

from storage.models import search_knowledge_points

logger = logging.getLogger(__name__)


def retrieve(state: dict) -> dict:
    query = state["user_message"]
    logger.info("Retrieving knowledge for: '%s'", query[:40])
    results = search_knowledge_points(query, limit=5)
    return {"stored_knowledge": results}
