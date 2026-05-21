"""Risk Analyst — evaluates fund risk factors."""

import logging

from agent.utils.llm import LLM
from fund.utils.fund_data_tools import get_fund_nav, get_manager_info

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a risk analyst evaluating a fund's risk profile.
Analyze the fund's risk factors for a retail investor holding it.

Cover:
1. Volatility — how much does this fund swing? (price range / average)
2. Drawdown risk — largest peak-to-trough in past 6 months
3. Concentration risk — top 3 holdings weight, sector concentration
4. Manager risk — how long has current manager been in place? any frequent changes?
5. Liquidity / size risk — fund size too small to be viable?

Be concise. Highlight real concerns, don't over-flag normal risks."""


def run_risk_analyst(state: dict) -> dict:
    """Analyze fund risk."""
    fund_code = state["fund_code"]

    nav_records = get_fund_nav(fund_code, days=180)
    manager_info = get_manager_info(fund_code)

    data_lines = [
        f"Fund code: {fund_code}",
        f"NAV records (6mo): {len(nav_records)} days",
    ]
    if nav_records:
        navs = [r["nav"] for r in nav_records if r["nav"]]
        if navs:
            data_lines.append(f"NAV range: {min(navs):.4f} ~ {max(navs):.4f}")
            data_lines.append(f"NAV current: {navs[0]:.4f}")
    data_lines.append(f"\nManager history ({len(manager_info)} records):")
    for m in manager_info[:5]:
        data_lines.append(f"  - {m}")
    data_block = "\n".join(data_lines)

    prompt = f"{SYSTEM_PROMPT}\n\n{data_block}"
    result = LLM.generate(prompt, use_language=False)
    logger.info("Risk Analyst: report %d chars", len(result))
    return {"risk_report": result}
