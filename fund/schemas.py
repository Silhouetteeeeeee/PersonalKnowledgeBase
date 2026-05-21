"""Pydantic schemas for Fund Bot structured output."""

from enum import Enum
from pydantic import BaseModel, Field


class FundRating(str, Enum):
    STRONG_BUY = "Strong Buy"
    BUY = "Buy"
    HOLD = "Hold"
    REDUCE = "Reduce"
    SELL = "Sell"


class FundDecision(BaseModel):
    rating: FundRating = Field(
        description="基金评级，基于分析师的报告和辩论结果"
    )
    summary: str = Field(
        description="一句话结论，用户能快速理解"
    )
    analysis: str = Field(
        description="详细分析：持仓背景、业绩表现、风险评估"
    )
    action_advice: str = Field(
        description="具体操作建议：加仓/定投/持有/减仓/清仓及理由"
    )
    risk_note: str = Field(
        description="关键风险提示"
    )


def render_fund_decision(d: FundDecision) -> str:
    """Render FundDecision as markdown."""
    lines = [
        f"**评级**: {d.rating.value}",
        "",
        f"**结论**: {d.summary}",
        "",
        f"**分析**: {d.analysis}",
        "",
        f"**操作建议**: {d.action_advice}",
        "",
        f"**风险提示**: {d.risk_note}",
    ]
    return "\n".join(lines)
