"""Performance Analyst — analyzes fund returns and rankings."""

import logging

from agent.utils.llm import LLM
from fund.utils.fund_data_tools import get_fund_nav, get_fund_rankings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a performance analyst evaluating a fund's returns.
Analyze the fund's performance across recent periods and versus peers.

Cover:
1. Recent returns (1m/3m/6m/1y) — calculate from NAV data provided
2. Peer group ranking (what percentile?)
3. Maximum drawdown in the past year
4. Risk-adjusted return (note: sharpe not available, use return/drawdown ratio)

Be concise. Use specific numbers."""


def run_performance_analyst(state: dict) -> dict:
    """Analyze fund performance."""
    fund_code = state["fund_code"]

    nav_records = get_fund_nav(fund_code, days=365)
    rankings = get_fund_rankings(fund_code)

    data_lines = [
        f"Fund code: {fund_code}",
        f"NAV records: {len(nav_records)} days available",
        f"Latest NAV: {nav_records[0] if nav_records else 'N/A'}",
        f"Earliest NAV: {nav_records[-1] if nav_records else 'N/A'}",
        f"\nRankings ({len(rankings)} records):",
    ]
    for r in rankings[:3]:
        data_lines.append(f"  - {r}")
    data_block = "\n".join(data_lines)

    prompt = f"{SYSTEM_PROMPT}\n\n{data_block}"
    result = LLM.generate(prompt, use_language=False)
    logger.info("Performance Analyst: report %d chars", len(result))
    return {"performance_report": result}
