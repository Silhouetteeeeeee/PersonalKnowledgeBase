from typing_extensions import TypedDict


class AgentState(TypedDict):
    user_message: str
    user_id: str
    timestamp: str
    category: str
    confidence: float
    needs_search: bool
    needs_store: bool
    search_results: list[str]
    stored_knowledge: list[dict]
    answer: str
    final_response: str
