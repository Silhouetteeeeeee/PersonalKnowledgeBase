"""akshare-based fund data tools with SQLite cache layer."""

import json
import logging
from datetime import datetime, timedelta
from typing import Optional

import akshare as ak

from storage.database import get_connection

logger = logging.getLogger(__name__)

CACHE_TTL = {
    "fund_info": timedelta(hours=24),
    "fund_nav": timedelta(hours=1),
    "fund_holdings": timedelta(days=7),
}


def _cache_valid(updated_at: Optional[str], ttl: timedelta) -> bool:
    if not updated_at:
        return False
    try:
        updated = datetime.fromisoformat(updated_at)
        return datetime.now() - updated < ttl
    except (ValueError, TypeError):
        return False


def search_fund(keyword: str) -> list[dict]:
    """Search fund by code or name keyword."""
    try:
        df = ak.fund_name_em()
        df = df[df["基金代码"].str.contains(keyword, case=False) |
                df["基金简称"].str.contains(keyword, case=False)]
        return df.head(10).to_dict("records")
    except Exception as e:
        logger.error("search_fund(%s) failed: %s", keyword, e)
        return []


def get_fund_info(code: str) -> Optional[dict]:
    """Get fund basic info with 24h cache."""
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM fund_info WHERE code = ?", (code,)
        ).fetchone()
        if row and _cache_valid(row["updated_at"], CACHE_TTL["fund_info"]):
            return dict(row)
    finally:
        conn.close()

    try:
        df = ak.fund_open_fund_info_em(symbol=code, indicator="基金概况")
        if df.empty:
            return None
        row = df.iloc[0]
        info = {
            "code": code,
            "name": row.get("基金简称", ""),
            "fund_type": row.get("基金类型", ""),
            "company": row.get("管理人", ""),
            "established_date": row.get("成立日", ""),
            "fund_size": row.get("最新规模", 0),
            "manager": row.get("基金经理", ""),
        }
        conn = get_connection()
        try:
            conn.execute(
                """INSERT OR REPLACE INTO fund_info
                   (code, name, fund_type, company, established_date, fund_size, manager, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now', 'localtime'))""",
                (info["code"], info["name"], info["fund_type"], info["company"],
                 info["established_date"], info["fund_size"], info["manager"]),
            )
            conn.commit()
        finally:
            conn.close()
        return info
    except Exception as e:
        logger.error("get_fund_info(%s) failed: %s", code, e)
        return None


def get_fund_nav(code: str, days: int = 120) -> list[dict]:
    """Get fund NAV history with 1h cache."""
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT updated_at FROM fund_nav_cache WHERE fund_code = ? LIMIT 1", (code,)
        ).fetchone()
        use_cache = row and _cache_valid(row["updated_at"], CACHE_TTL["fund_nav"])
    finally:
        conn.close()

    if use_cache:
        conn = get_connection()
        try:
            rows = conn.execute(
                "SELECT * FROM fund_nav_cache WHERE fund_code = ? ORDER BY date DESC LIMIT ?",
                (code, days),
            ).fetchall()
            if rows:
                return [dict(r) for r in rows]
        finally:
            conn.close()

    try:
        df = ak.fund_open_fund_info_em(symbol=code, indicator="单位净值走势")
        if df.empty:
            return []
        records = []
        conn = get_connection()
        try:
            for _, row in df.iterrows():
                date = str(row.iloc[0])[:10]
                nav = float(row.iloc[1]) if row.iloc[1] else 0.0
                total_nav = float(row.iloc[2]) if row.iloc[2] else 0.0
                daily_return = float(row.iloc[3]) if len(row) > 3 and row.iloc[3] else 0.0
                conn.execute(
                    """INSERT OR REPLACE INTO fund_nav_cache
                       (fund_code, date, nav, total_nav, daily_return, updated_at)
                       VALUES (?, ?, ?, ?, ?, datetime('now', 'localtime'))""",
                    (code, date, nav, total_nav, daily_return),
                )
                records.append({
                    "fund_code": code, "date": date,
                    "nav": nav, "total_nav": total_nav,
                    "daily_return": daily_return,
                })
            conn.commit()
        finally:
            conn.close()
        return sorted(records, key=lambda x: x["date"], reverse=True)[:days]
    except Exception as e:
        logger.error("get_fund_nav(%s) failed: %s", code, e)
        return []


def get_fund_holdings(code: str, year: Optional[int] = None) -> list[dict]:
    """Get fund top holdings with 7d cache."""
    if year is None:
        year = datetime.now().year

    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM fund_holdings_cache WHERE fund_code = ? AND report_date LIKE ?",
            (code, f"{year}%"),
        ).fetchall()
        for r in rows:
            r_dict = dict(r)
            if _cache_valid(r_dict.get("updated_at"), CACHE_TTL["fund_holdings"]):
                return json.loads(r["holdings_json"])
    finally:
        conn.close()

    try:
        df = ak.fund_portfolio_hold_em(symbol=code, date=str(year))
        if df.empty:
            return []
        records = df.to_dict("records")
        conn = get_connection()
        try:
            conn.execute(
                """INSERT OR REPLACE INTO fund_holdings_cache
                   (fund_code, report_date, holdings_json, sectors_json, updated_at)
                   VALUES (?, ?, ?, ?, datetime('now', 'localtime'))""",
                (code, str(year), json.dumps(records, ensure_ascii=False), "{}"),
            )
            conn.commit()
        finally:
            conn.close()
        return records
    except Exception as e:
        logger.error("get_fund_holdings(%s, %s) failed: %s", code, year, e)
        return []


def get_fund_rankings(code: str) -> list[dict]:
    """Get fund rankings in its category."""
    try:
        df = ak.fund_open_fund_rank_em()
        if df.empty:
            return []
        return df[df["基金代码"] == code].to_dict("records")
    except Exception as e:
        logger.error("get_fund_rankings(%s) failed: %s", code, e)
        return []


def get_manager_info(code: str) -> list[dict]:
    """Get fund manager change history."""
    try:
        df = ak.fund_manager_em(symbol=code)
        if df.empty:
            return []
        return df.to_dict("records")
    except Exception as e:
        logger.error("get_manager_info(%s) failed: %s", code, e)
        return []


def get_index_data(symbol: str = "sh000300", days: int = 60) -> list[dict]:
    """Get market index data (default: CSI 300)."""
    try:
        df = ak.stock_zh_index_daily(symbol=symbol)
        if df.empty:
            return []
        df = df.tail(days)
        return df.to_dict("records")
    except Exception as e:
        logger.error("get_index_data(%s) failed: %s", symbol, e)
        return []
