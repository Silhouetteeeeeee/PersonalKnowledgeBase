from agent.tools.web_search import search_web


def test_search_returns_strings():
    results = search_web("Python programming", max_results=3)
    assert isinstance(results, list)
    if results and not results[0].startswith("[Search error"):
        assert all(isinstance(r, str) for r in results)
        assert len(results) <= 3
