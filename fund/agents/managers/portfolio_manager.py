"""Portfolio Manager -- final decision maker with structured output."""

import logging

from agent.utils.llm import LLM
from fund.schemas import FundDecision, render_fund_decision

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a portfolio manager making the final decision on a fund position.
Synthesize all analyses, debate arguments, and past experience into a rating and recommendation.

Consider:
1. All four analyst reports
2. Both bull and bear arguments
3. Past decision history for this fund and similar funds
4. The user's personal holding context (cost basis, portfolio weight)

Output a FundDecision with:
- rating: Strong Buy / Buy / Hold / Reduce / Sell
- summary: one-line conclusion
- analysis: detailed synthesis
- action_advice: concrete steps
- risk_note: key risks"""


def run_portfolio_manager(state: dict) -> dict:
    """Synthesize all inputs into a final FundDecision."""
    debate = state.get("debate_state", {})

    prompt_parts = [
        SYSTEM_PROMPT,
        f"\nFund: {state['fund_code']} {state['fund_name']}",
        f"\n## Portfolio Analysis\n{state['portfolio_report']}",
        f"\n## Holdings Analysis\n{state['holdings_report']}",
        f"\n## Performance Analysis\n{state['performance_report']}",
        f"\n## Risk Analysis\n{state['risk_report']}",
    ]

    if debate.get("bull_history"):
        prompt_parts.append(f"\n## Bull Case\n{debate['bull_history']}")
    if debate.get("bear_history"):
        prompt_parts.append(f"\n## Bear Case\n{debate['bear_history']}")

    if state.get("past_context"):
        prompt_parts.append(f"\n## Past Decision History\n{state['past_context']}")

    prompt = "\n".join(prompt_parts)

    result = LLM.generate_structured(prompt, FundDecision, use_language=False)
    if result is None:
        logger.error("Portfolio Manager: structured output returned None")
        return {"final_decision": "分析暂时不可用，请稍后再试。"}

    rendered = render_fund_decision(result)
    logger.info("Portfolio Manager: decision rating=%s", result.rating.value)
    return {"final_decision": rendered}
