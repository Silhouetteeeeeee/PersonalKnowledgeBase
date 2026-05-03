import json
import logging

import jieba

from .database import get_connection

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> list[str]:
    """Tokenize text: jieba for Chinese, whitespace split for others."""
    words = jieba.lcut(text)
    # Filter out single-character noise, keep meaningful words
    return [w.strip() for w in words if len(w.strip()) >= 2]


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


def save_knowledge_points_bulk(knowledge_points: list[dict]) -> list[int]:
    """Save multiple knowledge points in a single transaction."""
    conn = get_connection()
    ids = []
    try:
        for kp in knowledge_points:
            cur = conn.execute(
                """INSERT INTO knowledge_points (knowledge_text, source_question, category, tags)
                   VALUES (?, ?, ?, ?)""",
                (kp["knowledge_text"], kp["source_question"], kp["category"], json.dumps(kp["tags"])),
            )
            ids.append(cur.lastrowid)
        conn.commit()
        logger.info("Saved %d knowledge points in category '%s'", len(knowledge_points), knowledge_points[0]["category"])
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return ids


def search_knowledge_points(query: str, limit: int = 5) -> list[dict]:
    conn = get_connection()

    # Extract keywords from both methods
    words = query.strip().split()
    chinese_words = _tokenize(query)

    # Combine both: for short ASCII words use split, for Chinese use jieba
    all_keywords = set()
    for w in words:
        if any(ord(c) > 127 for c in w):
            # Chinese/unicode word from split — already in chinese_words
            if len(w) >= 2:
                all_keywords.add(w)
        else:
            # English/ASCII word
            all_keywords.add(w)
    all_keywords.update(chinese_words)

    if not all_keywords:
        conn.close()
        return []

    conditions = []
    params = []
    for word in sorted(all_keywords, key=len, reverse=True):
        like = f"%{word}%"
        conditions.append("(knowledge_text LIKE ? OR source_question LIKE ?)")
        params.extend([like, like])

    sql = (
        """SELECT * FROM knowledge_points
           WHERE {}
           ORDER BY created_at DESC LIMIT ?"""
    ).format(" OR ".join(conditions))
    params.append(limit)

    cur = conn.execute(sql, params)
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()

    logger.info("Retrieved %d knowledge points for query '%s' (keywords: %s)", len(rows), query[:30], list(all_keywords)[:8])
    return rows


def find_similar_knowledge(knowledge_text: str, threshold: float = 0.7) -> list[dict]:
    """Find similar knowledge points by keyword overlap."""
    words = _tokenize(knowledge_text)
    if not words:
        return []

    conn = get_connection()
    conditions = []
    params = []
    for w in words:
        conditions.append("knowledge_text LIKE ?")
        params.append(f"%{w}%")

    sql = "SELECT * FROM knowledge_points WHERE {} ORDER BY created_at DESC LIMIT 3".format(
        " OR ".join(conditions)
    )
    cur = conn.execute(sql, params)
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
