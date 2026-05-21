"""Fund Agent state definitions."""

from typing_extensions import TypedDict


class FundDebateState(TypedDict):
    bull_history: str
    bear_history: str
    history: str
    current_response: str
    count: int


class FundAgentState(TypedDict):
    # Input
    user_message: str
    user_id: str
    fund_code: str
    fund_name: str
    intent: str              # analyze | status | portfolio_overview | add_holding

    # User holding context
    user_holding: dict       # {"shares": 1000, "cost": 1.5, ...}

    # Analyst reports
    portfolio_report: str
    holdings_report: str
    performance_report: str
    risk_report: str

    # Debate state
    debate_state: FundDebateState

    # Output
    final_decision: str

    # Memory injection
    past_context: str
