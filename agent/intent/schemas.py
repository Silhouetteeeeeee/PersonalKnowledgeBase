"""Intent data models — shared between classifier, registry, and handlers."""

from typing import Any
from pydantic import BaseModel, Field


class ParamSchema(BaseModel):
    """Parameter that an intent expects to extract from user input."""
    name: str
    type: str = "string"
    description: str
    required: bool = True


class IntentNode(BaseModel):
    """A single intent the classifier can recognize. Ragent-style."""
    id: str
    name: str
    description: str
    examples: list[str] = []
    params: list[ParamSchema] = []


# ── LLM output models ──

class IntentResult(BaseModel):
    """A single intent scored by the classifier."""
    id: str = Field(description="Intent ID")
    score: float = Field(ge=0.0, le=1.0, description="Confidence score 0-1")
    reason: str = Field(description="Why this intent was chosen")
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Extracted parameters relevant to this intent",
    )


class IntentClassification(BaseModel):
    """Full LLM output: scored results for ALL intents in one call."""
    results: list[IntentResult] = Field(
        description="All intents with scores, sorted by score descending",
    )


class Suggestions(BaseModel):
    """Clarifying question suggestions when intent is uncertain."""
    suggestions: list[str]
