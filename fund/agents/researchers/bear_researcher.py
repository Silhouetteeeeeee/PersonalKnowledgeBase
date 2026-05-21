"""Bear Researcher — argues for reducing/selling the fund position."""

import logging

from agent.utils.llm import LLM

logger = logging.getLogger(__name__)


def run_bear_researcher(state: dict) -> dict:
    """Bear argument based on all analyst reports."""
    debate = state["debate_state"]
    prompt = (
        "You are a bearish analyst arguing AGAINST holding this fund position. "
        "Use the analyst reports to build your case for reducing or selling.\n\n"
        f"Portfolio analysis: {state['portfolio_report']}\n\n"
        f"Holdings analysis: {state['holdings_report']}\n\n"
        f"Performance analysis: {state['performance_report']}\n\n"
        f"Risk analysis: {state['risk_report']}\n\n"
    )
    if debate.get("bull_history"):
        prompt += f"Bull argument to refute: {debate['bull_history']}\n\n"
    prompt += (
        "Be data-driven and specific. Challenge any overly positive assessments "
        "from the bull side. Mark your response with 'Bear Analyst:' prefix."
    )

    result = LLM.generate(prompt, use_language=False)
    logger.info("Bear Researcher: argument %d chars", len(result))

    old_count = debate.get("count", 0)
    old_history = debate.get("history", "")
    return {
        "debate_state": {
            "bull_history": debate.get("bull_history", ""),
            "bear_history": (debate.get("bear_history", "") + "\n\n" + result).strip(),
            "history": (old_history + "\n\n[Bear]: " + result).strip(),
            "current_response": result,
            "count": old_count + 1,
        }
    }
