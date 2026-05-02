from agent.tools.web_search import search_web


def search_web_node(state: dict) -> dict:
    results = search_web(state["user_message"])
    return {"search_results": results}
