"""User portfolio CRUD and analysis tools."""

import logging
from typing import Optional

from storage.database import get_connection

logger = logging.getLogger(__name__)


def add_holding(user_id: str, fund_code: str, fund_name: str,
                shares: float, cost_price: float, notes: str = "") -> dict:
    """Add a fund holding to user portfolio. Returns {"success": bool, "message": str}."""
    conn = get_connection()
    try:
        conn.execute(
            """INSERT INTO user_portfolio (user_id, fund_code, fund_name, shares, cost_price, notes)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(user_id, fund_code) DO UPDATE SET
                   shares = excluded.shares,
                   cost_price = excluded.cost_price,
                   notes = excluded.notes,
                   updated_at = datetime('now', 'localtime')""",
            (user_id, fund_code, fund_name, shares, cost_price, notes),
        )
        conn.commit()
        return {"success": True, "message": f"已添加/更新 {fund_code} {fund_name}"}
    except Exception as e:
        conn.rollback()
        logger.error("add_holding failed: %s", e)
        return {"success": False, "message": f"添加失败: {e}"}
    finally:
        conn.close()


def remove_holding(user_id: str, fund_code: str) -> dict:
    """Remove a fund holding."""
    conn = get_connection()
    try:
        conn.execute(
            "DELETE FROM user_portfolio WHERE user_id = ? AND fund_code = ?",
            (user_id, fund_code),
        )
        conn.commit()
        return {"success": True, "message": f"已移除 {fund_code}"}
    except Exception as e:
        conn.rollback()
        return {"success": False, "message": f"移除失败: {e}"}
    finally:
        conn.close()


def get_portfolio(user_id: str) -> list[dict]:
    """Get all holdings for a user."""
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM user_portfolio WHERE user_id = ? ORDER BY added_at DESC",
            (user_id,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_holding(user_id: str, fund_code: str) -> Optional[dict]:
    """Get a single holding by fund code."""
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM user_portfolio WHERE user_id = ? AND fund_code = ?",
            (user_id, fund_code),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()
