import logging

from storage.models import search_knowledge_points_semantic, search_knowledge_points, rerank_knowledge

logger = logging.getLogger(__name__)


def retrieve(state: dict) -> dict:
    query = state.get("search_query") or state["user_message"]
    logger.info("Retrieving knowledge for: '%s'", query[:40])

    # Step 1: Vector search (recall) — expand limit for reranker candidates
    try:
        candidates = search_knowledge_points_semantic(query, threshold=0.6, limit=20)
    except Exception as e:
        logger.warning("Semantic search failed: %s, falling back to keyword", e)
        candidates = []

    # Step 2: Cross-encoder rerank (precision)
    if len(candidates) > 3:
        try:
            results = rerank_knowledge(query, candidates, top_k=5)
            logger.info("Reranked %d candidates → %d results", len(candidates), len(results))
        except Exception as e:
            logger.warning("Reranker failed: %s, using vector ordering", e)
            results = candidates[:5]
    elif candidates:
        logger.info("Only %d candidates, skipping rerank", len(candidates))
        results = candidates[:5]
    else:
        logger.info("Semantic search returned no results, trying keyword fallback")
        results = search_knowledge_points(query, limit=5)

    return {"stored_knowledge": results}
