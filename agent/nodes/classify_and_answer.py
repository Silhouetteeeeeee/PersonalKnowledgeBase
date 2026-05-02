from pydantic import BaseModel, Field
from langchain_deepseek import ChatDeepSeek


class ClassifyOutput(BaseModel):
    category: str = Field(
        description="Category hierarchy, e.g. 'programming/python' or 'life/health'"
    )
    answer: str = Field(description="Answer to the question")
    confidence: float = Field(
        description="Confidence from 0.0 to 1.0"
    )
    needs_search: bool = Field(
        description="Whether web search is needed for accuracy"
    )


model = ChatDeepSeek(model="deepseek-chat", temperature=0)
structured_model = model.with_structured_output(ClassifyOutput)


def classify_and_answer(state: dict) -> dict:
    context = ""
    if state.get("stored_knowledge"):
        context = "Relevant past knowledge:\n"
        for k in state["stored_knowledge"]:
            context += f"- {k['knowledge_text']}\n"

    prompt = f"{context}Question: {state['user_message']}"
    result = structured_model.invoke(prompt)

    return {
        "category": result.category,
        "answer": result.answer,
        "confidence": result.confidence,
        "needs_search": result.needs_search,
    }
