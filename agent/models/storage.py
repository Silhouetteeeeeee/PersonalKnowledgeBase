from pydantic import BaseModel


class WikiPage(BaseModel):
    id: int = 0
    title: str = ""
    file_path: str = ""
    tags: str = ""
    sources: str = ""
    checksum: str = ""
    status: str = "active"
    created_at: str = ""
    updated_at: str = ""
    distance: float = 0.0


class ErrorRecord(BaseModel):
    id: int = 0
    user_message: str = ""
    wrong_answer: str = ""
    correct_answer: str = ""
    contradiction_details: str = ""
    error_type: str = ""
    created_at: str = ""


class FileRecord(BaseModel):
    id: int = 0
    filename: str = ""
    extension: str = ""
    file_hash: str = ""
    text_content: str = ""
    page_ids: str = ""
    user_id: str = ""
    created_at: str = ""


class PageVersion(BaseModel):
    id: int = 0
    page_id: int = 0
    title: str = ""
    content: str = ""
    checksum: str = ""
    source_id: str = ""
    source_question: str = ""
    created_at: str = ""


class Relation(BaseModel):
    id: int = 0
    page_id: int = 0
    related_page_id: int = 0
    relation_type: str = ""
    created_at: str = ""
