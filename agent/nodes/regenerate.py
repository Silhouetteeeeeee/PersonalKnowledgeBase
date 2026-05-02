from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_deepseek import ChatDeepSeek
from server.config import LLM_MODEL, LLM_TEMPERATURE
from agent.utils import with_retry

model = ChatDeepSeek(model=LLM_MODEL, temperature=LLM_TEMPERATURE)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful assistant. Answer the question using the web search results "
        "provided. Be concise and accurate. If the search results don't contain enough "
        "information, say so and provide your best answer.",
    ),
    ("human", "Web search results:\n{search_results}\n\nQuestion: {question}"),
])

chain = prompt | model | StrOutputParser()


def regenerate(state: dict) -> dict:
    search_text = "\n\n".join(state.get("search_results", []))
    if not search_text:
        return {"answer": state.get("answer", "")}

    response = with_retry(lambda: chain.invoke({
        "search_results": search_text,
        "question": state["user_message"],
    }))
    return {"answer": response}
