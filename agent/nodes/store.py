from pydantic import BaseModel, Field
from langchain_deepseek import ChatDeepSeek
from storage.models import save_knowledge_point, ensure_category


class DistilledPoint(BaseModel):
    knowledge_text: str = Field(
        description="A concise, standalone knowledge point distilled from the Q&A"
    )
    tags: list[str] = Field(description="Relevant tags for this knowledge point")


class DistillOutput(BaseModel):
    category: str = Field(description="Category for the knowledge, e.g. 'databases/redis'")
    knowledge_points: list[DistilledPoint] = Field(
        description="Knowledge points distilled from the Q&A"
    )


model = ChatDeepSeek(model="deepseek-chat", temperature=0)
structured_model = model.with_structured_output(DistillOutput)


def store(state: dict) -> dict:
    if not state.get("answer"):
        return {}

    result = structured_model.invoke(
        f"Distill the following Q&A into concise, standalone knowledge points.\n\n"
        f"Question: {state['user_message']}\n"
        f"Answer: {state['answer']}"
    )

    ensure_category(result.category)

    for kp in result.knowledge_points:
        save_knowledge_point(
            knowledge_text=kp.knowledge_text,
            source_question=state["user_message"],
            category=result.category,
            tags=kp.tags,
        )

    return {"category": result.category}
