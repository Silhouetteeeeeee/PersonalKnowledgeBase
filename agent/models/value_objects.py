from pydantic import BaseModel


class UrlContent(BaseModel):
    url: str
    title: str | None = None
    content: str = ""


class LogicChainStep(BaseModel):
    node: str
    action: str
    reasoning: str = ""
    confidence: float | None = None
    needs_store: bool | None = None
    search_performed: bool | None = None
    fallback: bool | None = None
    severity: str | None = None


class StoredKnowledge(BaseModel):
    type: str = "wiki_page"
    page_id: int = 0
    title: str = ""
    content: str = ""
    tags: list[str] = []
    distance: float = 0.0


class ContradictionInfo(BaseModel):
    found: bool = False
    details: str = ""
    severity: str = ""
    knowledge_ids: list[int] = []
    knowledge_texts: list[str] = []


class ReflectionInfo(BaseModel):
    result: str = ""
    reasoning: str = ""
    correction: str = ""
