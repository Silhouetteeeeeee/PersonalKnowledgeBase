import logging

from pydantic import BaseModel, Field

from agent.utils.llm import LLM

logger = logging.getLogger(__name__)


class RegenerateOutput(BaseModel):
    reasoning_trace: str = Field(
        description="Step-by-step reasoning: how the web search results inform the answer, any corrections from the original answer"
    )
    answer: str = Field(description="The regenerated answer based on web search results")


def regenerate(state: dict) -> dict:
    search_text = "\n\n".join(state.get("search_results", []))
    if not search_text:
        logger.info("No search results, keeping original answer")
        answer = state.get("answer", "")
        return {
            "answer": answer,
            "logic_chain": [{
                "node": "regenerate",
                "action": "无搜索结果，保留原答案",
                "reasoning": "Web search returned no results, keeping original answer unchanged",
            }],
        }

    logger.info("Regenerating answer with %d search results", len(state.get("search_results", [])))
    prompt = (
        f"Web search results:\n{search_text}\n\n"
        f"Question: {state['user_message']}\n\n"
        f"Original answer: {state.get('answer', '')}\n\n"
        f"Please provide an accurate answer based on the web search results."
    )
    result = LLM.generate_structured(prompt, RegenerateOutput, use_language=False)

    logger.info("Regenerated answer: %s", result.answer[:80])
    return {
        "answer": result.answer,
        "logic_chain": [{
            "node": "regenerate",
            "action": "基于搜索结果重新生成答案",
            "reasoning": result.reasoning_trace,
        }],
    }
