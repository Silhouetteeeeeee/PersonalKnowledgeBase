"""Portfolio Analyst — analyzes user's holding context for this fund."""

import logging

from agent.utils.llm import LLM
from fund.utils.portfolio_tools import get_holding, get_portfolio

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a portfolio analyst evaluating a user's fund holding.
Your job is to analyze the user's position in this fund and its role in their overall portfolio.

Cover:
1. Cost basis and current P&L (use the holding data provided)
2. Position sizing (% of total portfolio)
3. Whether this fund fits the user's apparent diversification strategy
4. Alternative funds the user already holds that overlap with this one

Be concise. Use specific numbers."""


def run_portfolio_analyst(state: dict) -> dict:
    """Analyze user's holding context for the fund."""
    user_id = state["user_id"]
    fund_code = state["fund_code"]

    holding = get_holding(user_id, fund_code)
    portfolio = get_portfolio(user_id)

    data_lines = [
        f"Fund code: {fund_code}",
        f"User holding: {holding}",
        f"Full portfolio ({len(portfolio)} funds):",
    ]
    for h in portfolio:
        data_lines.append(f"  - {h['fund_code']} {h['fund_name']}: {h['shares']} shares @ {h['cost_price']}")
    data_block = "\n".join(data_lines)

    prompt = f"{SYSTEM_PROMPT}\n\n{data_block}"
    result = LLM.generate(prompt, use_language=False)
    logger.info("Portfolio Analyst: report %d chars", len(result))
    return {"portfolio_report": result}
