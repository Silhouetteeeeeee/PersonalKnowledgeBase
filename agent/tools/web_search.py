from duckduckgo_search import DDGS


def search_web(query: str, max_results: int = 5) -> list[str]:
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
            return [r.get("body", "") for r in results if r.get("body")]
    except Exception as e:
        return [f"[Search error: {e}]"]
