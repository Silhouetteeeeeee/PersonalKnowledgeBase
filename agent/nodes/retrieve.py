from storage.models import search_knowledge_points


def retrieve(state: dict) -> dict:
    results = search_knowledge_points(state["user_message"], limit=5)
    return {"stored_knowledge": results}
