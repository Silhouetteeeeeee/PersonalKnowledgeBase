"""Tests for FundBot intent parsing."""

from fund.bot import parse_intent


def test_parse_add_holding():
    r = parse_intent("添加基金 110011 1000份 1.5")
    assert r["intent"] == "add_holding"
    assert r["fund_code"] == "110011"
    assert r["shares"] == 1000.0
    assert r["cost"] == 1.5


def test_parse_add_holding_no_cost():
    r = parse_intent("添加基金 110011 1000")
    assert r["intent"] == "add_holding"
    assert r["fund_code"] == "110011"
    assert r["shares"] == 1000.0


def test_parse_remove_holding():
    r = parse_intent("删除基金 110011")
    assert r["intent"] == "remove_holding"
    assert r["fund_code"] == "110011"

    r = parse_intent("移除 110011")
    assert r["intent"] == "remove_holding"


def test_parse_portfolio_overview():
    for cmd in ("我的持仓", "持仓", "组合", "我的组合"):
        r = parse_intent(cmd)
        assert r["intent"] == "portfolio_overview", f"failed for {cmd}"


def test_parse_fund_analyze():
    r = parse_intent("分析 110011")
    assert r["intent"] == "fund_analyze"
    assert r["query"] == "110011"

    r = parse_intent("看看 110011")
    assert r["intent"] == "fund_analyze"


def test_parse_fund_status():
    r = parse_intent("查一下 110011")
    assert r["intent"] == "fund_status"
    assert r["fund_code"] == "110011"

    # Pure 6-digit code
    r = parse_intent("110011")
    assert r["intent"] == "fund_status"
    assert r["fund_code"] == "110011"


def test_parse_fund_search():
    r = parse_intent("易方达蓝筹")
    assert r["intent"] == "fund_search"
    assert r["query"] == "易方达蓝筹"
