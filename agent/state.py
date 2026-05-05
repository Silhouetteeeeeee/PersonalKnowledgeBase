from typing import Annotated
from typing_extensions import TypedDict
import operator


class AgentState(TypedDict):
    user_message: str
    user_id: str
    timestamp: str
    category: str
    confidence: float
    needs_store: bool
    search_results: list[str]
    stored_knowledge: list[dict]
    stored_knowledge_ids: list[int]
    answer: str
    final_response: str
    reasoning_log_path: str
    contradiction_found: bool
    contradiction_details: str
    search_time: int

    # Reflection fields
    contradiction_severity: str
    contradiction_knowledge_ids: list[int]
    contradiction_knowledge_texts: list[str]
    reflection_result: str
    reflection_reasoning: str
    reflection_correction: str

    # Control flags
    force_web_search: bool
    correction_attempts: int
    knowledge_corrected: bool
    error_recorded: bool

    # Reasoning trace (accumulates across nodes via operator.add)
    logic_chain: Annotated[list[dict], operator.add]
