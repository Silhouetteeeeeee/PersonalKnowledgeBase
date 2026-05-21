"""Tests for fund decision memory."""

import pytest
from storage.database import init_db


@pytest.fixture(autouse=True)
def temp_db(monkeypatch, tmp_path):
    db_dir = tmp_path / "data"
    db_dir.mkdir()
    monkeypatch.setattr("storage.database.DB_DIR", str(db_dir))
    monkeypatch.setattr("storage.database.DB_PATH", str(db_dir / "knowledge.db"))
    init_db()


def test_store_decision():
    from fund.utils.memory import FundMemory

    mem = FundMemory()
    mem.store_decision("user1", "110011", "Buy", "test analysis", 1.5)

    from storage.database import get_connection
    conn = get_connection()
    try:
        row = conn.execute("SELECT * FROM fund_decisions").fetchone()
        assert row is not None
        assert row["user_id"] == "user1"
        assert row["fund_code"] == "110011"
        assert row["status"] == "pending"
    finally:
        conn.close()


def test_get_past_context_empty():
    from fund.utils.memory import FundMemory

    mem = FundMemory()
    ctx = mem.get_past_context("user1", "110011")
    assert ctx == ""


def test_reflect_returns_string():
    from fund.utils.memory import FundMemory

    mem = FundMemory()
    reflection = mem._reflect("Buy decision", 0.05)
    assert isinstance(reflection, str)
    assert len(reflection) > 0
