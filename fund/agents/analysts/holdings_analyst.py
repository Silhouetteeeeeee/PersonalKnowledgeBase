"""Holdings Analyst — analyzes fund's underlying portfolio holdings."""

import logging

from agent.utils.llm import LLM
from fund.utils.fund_data_tools import get_fund_holdings, get_manager_info

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a holdings analyst evaluating a fund's underlying portfolio.
Analyze the fund's stock/bond holdings, sector allocation, and management.

Cover:
1. Top holdings and their weights (concentrated or diversified?)
2. Sector/category distribution
3. Style drift — has the holding profile changed vs prior periods?
4. Manager stability and changes

Be concise. Use specific data."""


def run_holdings_analyst(state: dict) -> dict:
    """Analyze fund holdings structure."""
    fund_code = state["fund_code"]

    holdings = get_fund_holdings(fund_code)
    manager_info = get_manager_info(fund_code)

    data_lines = [
        f"Fund code: {fund_code}",
        f"Top holdings ({len(holdings)} positions):",
    ]
    for h in holdings[:10]:
        data_lines.append(f"  - {h}")
    data_lines.append(f"\nManager history ({len(manager_info)} records):")
    for m in manager_info[:5]:
        data_lines.append(f"  - {m}")
    data_block = "\n".join(data_lines)

    prompt = f"{SYSTEM_PROMPT}\n\n{data_block}"
    result = LLM.generate(prompt, use_language=False)
    logger.info("Holdings Analyst: report %d chars", len(result))
    return {"holdings_report": result}
