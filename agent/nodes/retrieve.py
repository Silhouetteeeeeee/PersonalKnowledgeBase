"""Wiki page retrieval: vector search -> read full pages -> expand relations.

Returns page content instead of knowledge_point fragments.
Falls back to old knowledge_points search if no wiki pages exist.
"""

import logging

from storage.models import find_similar_pages, get_related_pages, get_page_by_title
from storage.wiki_storage import read_page

logger = logging.getLogger(__name__)


def retrieve(state: dict) -> dict:
    query = state.get("search_query") or state["user_message"]
    logger.info("Wiki retrieval for: '%s'", query[:40])

    # Step 1: Try wiki page vector search
    try:
        pages = find_similar_pages(query, threshold=0.6, limit=5)
    except Exception as e:
        logger.warning("Page semantic search failed: %s", e)
        pages = []

    if pages:
        return _retrieve_wiki_pages(pages, query)

    # Step 2: No wiki pages found
    logger.info("No wiki pages found, returning empty")
    return {"stored_knowledge": []}


def _retrieve_wiki_pages(pages: list[dict], query: str) -> dict:
    """Read full wiki page content and expand with related pages."""
    # Read pages from disk
    results = []
    for p in pages:
        file_page = read_page(p["file_path"])
        if not file_page:
            continue
        results.append({
            "type": "wiki_page",
            "page_id": p["id"],
            "title": p["title"],
            "content": file_page["body"],
            "tags": file_page["tags"],
            "distance": p.get("distance", 0),
        })

    # Expand with related pages (second pass)
    related_titles = set()
    for r in results:
        related = get_related_pages(r["page_id"])
        for rp in related:
            if rp["title"] not in {x["title"] for x in results}:
                related_titles.add(rp["title"])

    for title in related_titles:
        page = get_page_by_title(title)
        if page:
            file_page = read_page(page["file_path"])
            if file_page:
                results.append({
                    "type": "wiki_page",
                    "page_id": page["id"],
                    "title": title,
                    "content": file_page["body"],
                    "tags": file_page["tags"],
                    "distance": 0,
                })

    logger.info("Retrieved %d wiki pages (including %d relation-expanded)",
                len(results), len(results) - len(pages))
    return {"stored_knowledge": results}


