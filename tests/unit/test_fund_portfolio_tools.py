"""Tests for portfolio CRUD tools."""

import pytest
from storage.database import init_db


@pytest.fixture(autouse=True)
def temp_db(monkeypatch, tmp_path):
    db_dir = tmp_path / "data"
    db_dir.mkdir()
    monkeypatch.setattr("storage.database.DB_DIR", str(db_dir))
    monkeypatch.setattr("storage.database.DB_PATH", str(db_dir / "knowledge.db"))
    init_db()


def test_add_and_get_holding():
    from fund.utils.portfolio_tools import add_holding, get_holding, get_portfolio

    result = add_holding("user1", "110011", "易方达蓝筹", 1000.0, 1.5)
    assert result["success"] is True

    h = get_holding("user1", "110011")
    assert h is not None
    assert h["fund_code"] == "110011"
    assert h["shares"] == 1000.0
    assert h["cost_price"] == 1.5

    portfolio = get_portfolio("user1")
    assert len(portfolio) == 1


def test_remove_holding():
    from fund.utils.portfolio_tools import add_holding, remove_holding, get_portfolio

    add_holding("user1", "110011", "易方达蓝筹", 1000.0, 1.5)
    remove_holding("user1", "110011")
    assert len(get_portfolio("user1")) == 0


def test_add_holding_updates_existing():
    from fund.utils.portfolio_tools import add_holding, get_holding

    add_holding("user1", "110011", "易方达蓝筹", 1000.0, 1.5)
    add_holding("user1", "110011", "易方达蓝筹", 2000.0, 2.0)

    h = get_holding("user1", "110011")
    assert h["shares"] == 2000.0
    assert h["cost_price"] == 2.0
