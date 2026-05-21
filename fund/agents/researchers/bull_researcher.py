"""Bull Researcher — argues for holding/adding to the fund position."""

import logging

from agent.utils.llm import LLM

logger = logging.getLogger(__name__)


def run_bull_researcher(state: dict) -> dict:
    """Bull argument based on all analyst reports."""
    debate = state["debate_state"]
    prompt = (
        "You are a bullish analyst arguing FOR holding or adding to this fund position. "
        "Use the analyst reports to build your case.\n\n"
        f"Portfolio analysis: {state['portfolio_report']}\n\n"
        f"Holdings analysis: {state['holdings_report']}\n\n"
        f"Performance analysis: {state['performance_report']}\n\n"
        f"Risk analysis: {state['risk_report']}\n\n"
    )
    if debate.get("bear_history"):
        prompt += f"Bear argument to refute: {debate['bear_history']}\n\n"
    prompt += (
        "Be data-driven and specific. Challenge any overly negative assessments "
        "from the bear side. Mark your response with 'Bull Analyst:' prefix."
    )

    result = LLM.generate(prompt, use_language=False)
    logger.info("Bull Researcher: argument %d chars", len(result))

    old_count = debate.get("count", 0)
    old_history = debate.get("history", "")
    return {
        "debate_state": {
            "bull_history": (debate.get("bull_history", "") + "\n\n" + result).strip(),
            "bear_history": debate.get("bear_history", ""),
            "history": (old_history + "\n\n[Bull]: " + result).strip(),
            "current_response": result,
            "count": old_count + 1,
        }
    }
