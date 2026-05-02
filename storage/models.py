import json
from .database import get_connection


def save_knowledge_point(
    knowledge_text: str,
    source_question: str,
    category: str,
    tags: list[str],
) -> int:
    conn = get_connection()
    cur = conn.execute(
        """INSERT INTO knowledge_points (knowledge_text, source_question, category, tags)
           VALUES (?, ?, ?, ?)""",
        (knowledge_text, source_question, category, json.dumps(tags)),
    )
    conn.commit()
    conn.close()
    return cur.lastrowid


def search_knowledge_points(query: str, limit: int = 5) -> list[dict]:
    conn = get_connection()
    like = f"%{query}%"
    cur = conn.execute(
        """SELECT * FROM knowledge_points
           WHERE knowledge_text LIKE ? OR source_question LIKE ?
           ORDER BY created_at DESC LIMIT ?""",
        (like, like, limit),
    )
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def get_all_categories() -> list[str]:
    conn = get_connection()
    cur = conn.execute("SELECT DISTINCT category FROM knowledge_points ORDER BY category")
    rows = [r["category"] for r in cur.fetchall()]
    conn.close()
    return rows


def ensure_category(name: str, description: str = "") -> int:
    conn = get_connection()
    cur = conn.execute(
        "INSERT OR IGNORE INTO categories (name, description) VALUES (?, ?)",
        (name, description),
    )
    conn.commit()
    conn.close()
    return cur.lastrowid
