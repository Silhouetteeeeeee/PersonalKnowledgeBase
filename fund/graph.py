"""Fund Bot LangGraph pipeline."""

import logging

from langgraph.graph import StateGraph

from fund.state import FundAgentState, FundDebateState
from fund.agents.analysts.portfolio_analyst import run_portfolio_analyst
from fund.agents.analysts.holdings_analyst import run_holdings_analyst
from fund.agents.analysts.performance_analyst import run_performance_analyst
from fund.agents.analysts.risk_analyst import run_risk_analyst
from fund.agents.researchers.bull_researcher import run_bull_researcher
from fund.agents.researchers.bear_researcher import run_bear_researcher
from fund.agents.managers.portfolio_manager import run_portfolio_manager

logger = logging.getLogger(__name__)

MAX_DEBATE_ROUNDS = 1


def debate_router(state: dict) -> str:
    """Route between bull/bear debate and portfolio manager."""
    debate = state.get("debate_state", {})
    count = debate.get("count", 0)
    if count >= 2 * MAX_DEBATE_ROUNDS:
        return "portfolio_manager"
    if count % 2 == 0:
        return "bull_researcher"
    else:
        return "bear_researcher"


def build_fund_graph() -> StateGraph:
    """Build (but do not compile) the fund analysis StateGraph."""
    builder = StateGraph(FundAgentState)

    builder.add_node("portfolio_analyst", run_portfolio_analyst)
    builder.add_node("holdings_analyst", run_holdings_analyst)
    builder.add_node("performance_analyst", run_performance_analyst)
    builder.add_node("risk_analyst", run_risk_analyst)
    builder.add_node("bull_researcher", run_bull_researcher)
    builder.add_node("bear_researcher", run_bear_researcher)
    builder.add_node("portfolio_manager", run_portfolio_manager)

    builder.set_entry_point("portfolio_analyst")

    builder.add_edge("portfolio_analyst", "holdings_analyst")
    builder.add_edge("holdings_analyst", "performance_analyst")
    builder.add_edge("performance_analyst", "risk_analyst")
    builder.add_edge("risk_analyst", "bull_researcher")

    builder.add_conditional_edges(
        "bull_researcher",
        debate_router,
        {
            "bull_researcher": "bull_researcher",
            "bear_researcher": "bear_researcher",
            "portfolio_manager": "portfolio_manager",
        },
    )
    builder.add_conditional_edges(
        "bear_researcher",
        debate_router,
        {
            "bull_researcher": "bull_researcher",
            "bear_researcher": "bear_researcher",
            "portfolio_manager": "portfolio_manager",
        },
    )

    builder.add_edge("portfolio_manager", "__end__")

    logger.info("Fund graph built (uncompiled)")
    return builder
