import logging

from storage.models import search_knowledge_points_semantic, search_knowledge_points

logger = logging.getLogger(__name__)


def retrieve(state: dict) -> dict:
    query = state["user_message"]
    logger.info("Retrieving knowledge for: '%s'", query[:40])

    # Try semantic search first, fall back to keyword search on failure or empty results
    try:
        results = search_knowledge_points_semantic(query, limit=5)
    except Exception as e:
        logger.warning("Semantic search failed: %s, falling back to keyword", e)
        results = []

    if not results:
        logger.info("Semantic search returned no results, trying keyword fallback")
        results = search_knowledge_points(query, limit=5)

    return {"stored_knowledge": results}
