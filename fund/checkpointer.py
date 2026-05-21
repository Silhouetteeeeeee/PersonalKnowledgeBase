"""SqliteSaver checkpoint per-user, adapted from TradingAgents pattern."""

import hashlib
import sqlite3
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

from langgraph.checkpoint.sqlite import SqliteSaver
from storage.database import DB_DIR

logger = logging.getLogger(__name__)

CHECKPOINT_DIR = Path(DB_DIR) / "fund_checkpoints"


def _ensure_dir():
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


def _db_path(user_id: str) -> Path:
    _ensure_dir()
    return CHECKPOINT_DIR / f"user_{user_id}.db"


def thread_id(user_id: str, fund_code: str, date: str) -> str:
    raw = f"{user_id}:{fund_code}:{date}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


@contextmanager
def get_checkpointer(user_id: str):
    db = _db_path(user_id)
    conn = sqlite3.connect(str(db), check_same_thread=False)
    try:
        saver = SqliteSaver(conn)
        saver.setup()
        yield saver
    finally:
        conn.close()


def has_checkpoint(user_id: str, fund_code: str, date: str) -> bool:
    db = _db_path(user_id)
    if not db.exists():
        return False
    tid = thread_id(user_id, fund_code, date)
    try:
        with get_checkpointer(user_id) as saver:
            return saver.get_tuple({"configurable": {"thread_id": tid}}) is not None
    except Exception:
        return False


def clear_checkpoint(user_id: str, fund_code: str, date: str):
    """Clear checkpoint for a completed analysis."""
    db = _db_path(user_id)
    if not db.exists():
        return
    tid = thread_id(user_id, fund_code, date)
    conn = sqlite3.connect(str(db))
    try:
        for table in ("writes", "checkpoints"):
            conn.execute(f"DELETE FROM {table} WHERE thread_id = ?", (tid,))
        conn.commit()
    except sqlite3.OperationalError:
        pass
    finally:
        conn.close()
