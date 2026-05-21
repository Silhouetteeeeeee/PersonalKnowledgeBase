"""Tests for fund schemas and rendering."""

from fund.schemas import FundRating, FundDecision, render_fund_decision


def test_fund_rating_values():
    assert FundRating.STRONG_BUY.value == "Strong Buy"
    assert FundRating.BUY.value == "Buy"
    assert FundRating.HOLD.value == "Hold"
    assert FundRating.REDUCE.value == "Reduce"
    assert FundRating.SELL.value == "Sell"


def test_fund_decision_model():
    d = FundDecision(
        rating=FundRating.BUY,
        summary="值得买入",
        analysis="业绩优异",
        action_advice="建议定投",
        risk_note="注意回撤",
    )
    assert d.rating == FundRating.BUY
    assert d.summary == "值得买入"


def test_render_fund_decision():
    d = FundDecision(
        rating=FundRating.HOLD,
        summary="继续持有",
        analysis="表现平稳",
        action_advice="持有观察",
        risk_note="市场风险",
    )
    rendered = render_fund_decision(d)
    assert "**评级**: Hold" in rendered
    assert "**结论**: 继续持有" in rendered
    assert "**分析**: 表现平稳" in rendered
    assert "**操作建议**: 持有观察" in rendered
    assert "**风险提示**: 市场风险" in rendered
