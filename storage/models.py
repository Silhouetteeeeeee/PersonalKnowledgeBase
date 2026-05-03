import json
import logging
import threading
import time
from typing import Optional

import jieba

from sqlite_vec import serialize_float32
from .database import get_connection

logger = logging.getLogger(__name__)

# ── Embedding model (lazy-loaded singleton, thread-safe) ──

_embedder: Optional[any] = None
_embedder_lock = threading.Lock()


def _get_embedder():
    global _embedder
    if _embedder is None:
        with _embedder_lock:
            if _embedder is None:
                from fastembed import TextEmbedding
                logger.info("Loading embedding model (BAAI/bge-small-zh-v1.5)...")
                t0 = time.time()
                _embedder = TextEmbedding("BAAI/bge-small-zh-v1.5")
                logger.info("Embedding model loaded in %.2fs (dim=512)", time.time() - t0)
    return _embedder


def generate_embedding(text: str) -> list[float]:
    """Generate a 512-dim embedding vector for the given text."""
    model = _get_embedder()
    vec = next(model.embed(text))
    return vec.tolist()


# ── Tokenization (used by keyword search fallback) ──

def _tokenize(text: str) -> list[str]:
    words = jieba.lcut(text)
    return [w.strip() for w in words if len(w.strip()) >= 2]


# ── CRUD ──

def save_knowledge_point(
    knowledge_text: str,
    source_question: str,
    category: str,
    tags: list[str],
) -> int:
    conn = get_connection()
    try:
        cur = conn.execute(
            """INSERT INTO knowledge_points (knowledge_text, source_question, category, tags)
               VALUES (?, ?, ?, ?)""",
            (knowledge_text, source_question, category, json.dumps(tags)),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def save_knowledge_points_bulk(knowledge_points: list[dict]) -> list[int]:
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
        logger.info("Saved %d knowledge points in category '%s'",
                     len(knowledge_points), knowledge_points[0]["category"])
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return ids


def save_knowledge_points_bulk_with_embeddings(knowledge_points: list[dict]) -> list[int]:
    """Save knowledge points and their embeddings in a single transaction."""
    conn = get_connection()
    ids = []
    try:
        for kp in knowledge_points:
            cur = conn.execute(
                """INSERT INTO knowledge_points (knowledge_text, source_question, category, tags)
                   VALUES (?, ?, ?, ?)""",
                (kp["knowledge_text"], kp["source_question"], kp["category"], json.dumps(kp["tags"])),
            )
            kid = cur.lastrowid
            ids.append(kid)

            # Generate and store embedding
            embedding = generate_embedding(kp["knowledge_text"])
            conn.execute(
                "INSERT INTO knowledge_vectors(rowid, embedding) VALUES (?, ?)",
                (kid, serialize_float32(embedding)),
            )
        conn.commit()
        logger.info("Saved %d knowledge points with embeddings in category '%s'",
                     len(knowledge_points), knowledge_points[0].get("category", ""))
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return ids


# ── Semantic search ──

def find_similar_knowledge(knowledge_text: str, threshold: float = 0.25) -> list[dict]:
    """Find semantically similar knowledge points using vector search.

    Args:
        knowledge_text: Text to compare against stored knowledge.
        threshold: Cosine distance threshold. 0=identical, ~0.2=very similar,
                   >0.5=unrelated. Default 0.25 (~0.75 cosine similarity).

    Returns:
        List of matching knowledge point dicts (up to 3), ordered by similarity.
    """
    embedding = generate_embedding(knowledge_text)
    conn = get_connection()
    try:
        rows = conn.execute(
            """SELECT kp.*, v.distance
               FROM (
                   SELECT rowid, distance
                   FROM knowledge_vectors
                   WHERE embedding MATCH ?
                     AND k = 20
               ) v
               JOIN knowledge_points kp ON kp.id = v.rowid
               WHERE v.distance <= ?
               ORDER BY v.distance
               LIMIT 3""",
            (serialize_float32(embedding), threshold),
        ).fetchall()
        results = [dict(r) for r in rows]
        logger.info("Semantic search found %d similar points (threshold=%.2f)",
                     len(results), threshold)
        return results
    finally:
        conn.close()


def search_knowledge_points_semantic(query: str, limit: int = 5) -> list[dict]:
    """Semantic search for the retrieve node."""
    embedding = generate_embedding(query)
    conn = get_connection()
    try:
        rows = conn.execute(
            """SELECT kp.*, v.distance
               FROM (
                   SELECT rowid, distance
                   FROM knowledge_vectors
                   WHERE embedding MATCH ?
                     AND k = ?
               ) v
               JOIN knowledge_points kp ON kp.id = v.rowid
               ORDER BY v.distance""",
            (serialize_float32(embedding), limit * 4),
        ).fetchall()
        results = [dict(r) for r in rows][:limit]
        logger.info("Semantic retrieval found %d results for query '%s'",
                     len(results), query[:30])
        return results
    finally:
        conn.close()


# ── Keyword search (fallback) ──

def search_knowledge_points(query: str, limit: int = 5) -> list[dict]:
    conn = get_connection()
    try:
        words = query.strip().split()
        chinese_words = _tokenize(query)

        all_keywords = set()
        for w in words:
            if any(ord(c) > 127 for c in w):
                if len(w) >= 2:
                    all_keywords.add(w)
            else:
                all_keywords.add(w)
        all_keywords.update(chinese_words)

        if not all_keywords:
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
        return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()


# ── Category helpers ──

def get_all_categories() -> list[str]:
    conn = get_connection()
    try:
        cur = conn.execute("SELECT DISTINCT category FROM knowledge_points ORDER BY category")
        return [r["category"] for r in cur.fetchall()]
    finally:
        conn.close()


def ensure_category(name: str, description: str = "") -> int:
    conn = get_connection()
    try:
        cur = conn.execute(
            "INSERT OR IGNORE INTO categories (name, description) VALUES (?, ?)",
            (name, description),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


# ── File record helpers ──

def save_file_record(
    file_name: str,
    file_type: str,
    file_hash: str,
    extracted_text: str,
    knowledge_ids: list[int],
    source_user_id: str,
) -> int:
    conn = get_connection()
    try:
        cur = conn.execute(
            """INSERT INTO file_records (file_name, file_type, file_hash, extracted_text, knowledge_ids, source_user_id)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (file_name, file_type, file_hash, extracted_text, json.dumps(knowledge_ids), source_user_id),
        )
        conn.commit()
        logger.info("Saved file record: %s (type=%s, %d knowledge points)", file_name, file_type, len(knowledge_ids))
        return cur.lastrowid
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def get_file_records(limit: int = 10) -> list[dict]:
    conn = get_connection()
    try:
        cur = conn.execute(
            "SELECT * FROM file_records ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )
        return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()


def get_file_record_by_hash(file_hash: str) -> dict | None:
    """Check if a file with the given hash has already been processed."""
    conn = get_connection()
    try:
        cur = conn.execute(
            "SELECT * FROM file_records WHERE file_hash = ?",
            (file_hash,),
        )
        row = cur.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()
