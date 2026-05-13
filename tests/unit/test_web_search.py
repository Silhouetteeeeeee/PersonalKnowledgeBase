from agent.tools.web_search import search_web


def test_search_error_handling():
    results = search_web("", max_results=0)
    assert isinstance(results, list)
    assert len(results) > 0  # either results or error message
