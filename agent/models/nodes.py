from pydantic import BaseModel
from agent.models.value_objects import LogicChainStep, UrlContent, StoredKnowledge


class NodeResult(BaseModel):
    logic_chain: list[LogicChainStep] = []


class ClassifyResult(NodeResult):
    answer: str = ""
    confidence: float = 0.0
    needs_store: bool = False


class FactCheckResult(NodeResult):
    contradiction_found: bool = False
    contradiction_details: str = ""
    contradiction_severity: str = ""
    contradiction_knowledge_ids: list[int] = []
    contradiction_knowledge_texts: list[str] = []


class ReflectResult(NodeResult):
    reflection_result: str = ""
    reflection_reasoning: str = ""
    reflection_correction: str = ""
    force_web_search: bool = False


class ParseResult(NodeResult):
    user_message: str = ""
    user_id: str = ""
    timestamp: str = ""
    url_contents: list[UrlContent] = []


class RetrieveResult(NodeResult):
    stored_knowledge: list[StoredKnowledge] = []


class SearchWebResult(NodeResult):
    search_results: list[str] = []


class RegenerateResult(NodeResult):
    answer: str = ""


class RecordErrorResult(NodeResult):
    correction_attempts: int = 0
    error_recorded: bool = False


class UpdateProfileResult(NodeResult):
    user_profile: dict = {}


class RewriteResult(NodeResult):
    search_query: str = ""


class RespondResult(BaseModel):
    final_response: str = ""
