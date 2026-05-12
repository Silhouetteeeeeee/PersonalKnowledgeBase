import json
import logging
import re
import sqlite3
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


_reranker: Optional[any] = None
_reranker_lock = threading.Lock()


def _get_reranker_model_path() -> str:
    """Resolve reranker model path via ModelScope (works in China)."""
    from modelscope.hub.snapshot_download import snapshot_download
    try:
        path = snapshot_download("BAAI/bge-reranker-v2-m3")
        logger.info("Reranker model path (ModelScope): %s", path)
        return path
    except Exception as e:
        logger.warning("ModelScope download failed: %s, trying HuggingFace ID", e)
        return "BAAI/bge-reranker-v2-m3"


def _get_reranker():
    global _reranker
    if _reranker is None:
        with _reranker_lock:
            if _reranker is None:
                from sentence_transformers import CrossEncoder
                model_path = _get_reranker_model_path()
                logger.info("Loading reranker model from %s ...", model_path)
                t0 = time.time()
                _reranker = CrossEncoder(model_path)
                logger.info("Reranker model loaded in %.2fs", time.time() - t0)
    return _reranker


def rerank_knowledge(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    """Rerank candidate knowledge points using a cross-encoder model.

    Takes (query, knowledge_text) pairs through the reranker to get relevance
    scores, then returns top_k candidates sorted by score descending.
    """
    pairs = [(query, doc["knowledge_text"]) for doc in candidates]
    model = _get_reranker()
    scores = model.predict(pairs)

    scored = list(zip(candidates, scores))
    scored.sort(key=lambda x: x[1], reverse=True)

    results = []
    for doc, score in scored[:top_k]:
        doc["relevance_score"] = float(score)
        results.append(doc)
    return results


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
                """INSERT INTO knowledge_points (knowledge_text, source_question, category, tags, status, corrected_text, reasoning_log_path)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (kp["knowledge_text"], kp["source_question"], kp["category"],
                 json.dumps(kp.get("tags", [])),
                 kp.get("status", "active"),
                 kp.get("corrected_text", ""),
                 kp.get("reasoning_log_path", "")),
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
                """INSERT INTO knowledge_points (knowledge_text, source_question, category, tags, status, corrected_text, reasoning_log_path)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (kp["knowledge_text"], kp["source_question"], kp["category"],
                 json.dumps(kp.get("tags", [])),
                 kp.get("status", "active"),
                 kp.get("corrected_text", ""),
                 kp.get("reasoning_log_path", "")),
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


def search_knowledge_points_semantic(query: str, threshold: float = 0.25, limit: int = 5) -> list[dict]:
    """Semantic search for the retrieve node."""
    embedding = generate_embedding(query)
    conn = get_connection()
    try:
        rows = conn.execute(
            """
                SELECT kp.*, v.distance
                FROM (
                    SELECT rowid, distance
                    FROM knowledge_vectors
                    WHERE 
                        embedding MATCH ?
                        AND k = ?
                ) v
                JOIN knowledge_points kp ON kp.id = v.rowid
                where 
                    distance <= ?
                ORDER BY v.distance""",
            (serialize_float32(embedding), limit, threshold),
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


def normalize_category_str(category: str, max_depth: int = 4) -> str:
    """Lightweight format normalization for category strings.

    - lowercase
    - unify separators (\\ | ｜ · > → /)
    - strip whitespace per level
    - remove hyphens/underscores (re-ranker → reranker)
    - truncate to max_depth levels
    - idempotent: normalize(x) == normalize(normalize(x))
    """
    s = category.lower().strip()
    s = re.sub(r'[\\｜|·>→／]', '/', s)
    parts = [p.strip() for p in s.split('/') if p.strip()]
    parts = [p.replace('-', '').replace('_', '') for p in parts]
    return '/'.join(parts[:max_depth])


def get_normalized_categories(max_count: int = 20) -> str:
    """Return a formatted string of existing categories for prompt injection.

    Returns e.g. "目前已存在的分类：databases/redis, ai/rag（共 2 个分类）"
    Truncates to max_count entries with "...等 N 个分类" suffix.
    """
    cats = get_all_categories()
    if not cats:
        return ""
    display = [normalize_category_str(c) for c in cats[:max_count]]
    suffix = f"（共 {len(cats)} 个分类）" if len(cats) <= max_count else f"...等 {len(cats)} 个分类"
    return f"目前已存在的分类：{', '.join(display)}{suffix}"


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


# ── Wiki page helpers ──

def _page_row_to_dict(row: sqlite3.Row) -> dict:
    """Convert a pages table row to a dict with parsed JSON fields."""
    d = dict(row)
    if isinstance(d.get("tags"), str):
        d["tags"] = json.loads(d["tags"])
    if isinstance(d.get("sources"), str):
        d["sources"] = json.loads(d["sources"])
    return d


def upsert_page(title: str, file_path: str, tags: list[str],
                sources: list[str], checksum: str,
                content: str) -> int:
    """Insert or update a wiki page record. Returns page id."""
    conn = get_connection()
    try:
        existing = conn.execute(
            "SELECT id FROM pages WHERE title = ?", (title,)
        ).fetchone()

        if existing:
            pid = existing["id"]
            conn.execute(
                """UPDATE pages SET file_path=?, tags=?, sources=?,
                   checksum=?, updated_at=datetime('now')
                   WHERE id=?""",
                (file_path, json.dumps(tags), json.dumps(sources),
                 checksum, pid),
            )
            logger.info("Updated page index: '%s' (id=%d)", title, pid)
        else:
            cur = conn.execute(
                """INSERT INTO pages (title, file_path, tags, sources, checksum)
                   VALUES (?, ?, ?, ?, ?)""",
                (title, file_path, json.dumps(tags), json.dumps(sources), checksum),
            )
            pid = cur.lastrowid
            logger.info("Created page index: '%s' (id=%d)", title, pid)

        # Update embedding: delete existing then insert (INSERT OR REPLACE
        # is not supported on vec0 virtual tables)
        conn.execute(
            "DELETE FROM page_vectors WHERE rowid = ?", (pid,)
        )
        embedding = generate_embedding(content)
        conn.execute(
            "INSERT INTO page_vectors(rowid, embedding) VALUES (?, ?)",
            (pid, serialize_float32(embedding)),
        )

        conn.commit()
        return pid
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def update_page_relations(page_id: int, linked_titles: list[str]) -> None:
    """Replace page_relations for a given page with fresh [[wikilink]] data."""
    conn = get_connection()
    try:
        conn.execute(
            "DELETE FROM page_relations WHERE source_id = ?", (page_id,)
        )
        for link_title in linked_titles:
            conn.execute(
                """INSERT INTO page_relations (source_id, target_title)
                   VALUES (?, ?)""",
                (page_id, link_title),
            )
        conn.commit()
        logger.info("Updated %d relations for page id=%d", len(linked_titles), page_id)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def find_similar_pages(query: str, threshold: float = 0.6, limit: int = 5) -> list[dict]:
    """Search wiki pages by semantic similarity. Returns list of page dicts with distance."""
    embedding = generate_embedding(query)
    conn = get_connection()
    try:
        rows = conn.execute(
            """SELECT p.*, v.distance
               FROM (
                   SELECT rowid, distance
                   FROM page_vectors
                   WHERE embedding MATCH ?
                     AND k = ?
               ) v
               JOIN pages p ON p.id = v.rowid
               WHERE p.status = 'active'
                 AND v.distance <= ?
               ORDER BY v.distance""",
            (serialize_float32(embedding), limit * 4, threshold),
        ).fetchall()
        results = [_page_row_to_dict(r) for r in rows][:limit]
        logger.info("Page semantic search: %d results for '%s'", len(results), query[:30])
        return results
    finally:
        conn.close()


def get_related_pages(page_id: int) -> list[dict]:
    """Get pages linked via page_relations to the given page (bidirectional)."""
    conn = get_connection()
    try:
        rows = conn.execute(
            """SELECT p.* FROM pages p
               JOIN page_relations r ON r.target_title = p.title
               WHERE r.source_id = ? AND p.status = 'active'
               UNION
               SELECT p.* FROM pages p
               JOIN page_relations r ON r.source_id = p.id
               WHERE r.target_title = (SELECT title FROM pages WHERE id = ?)
                 AND p.status = 'active'
               """,
            (page_id, page_id),
        ).fetchall()
        return [_page_row_to_dict(r) for r in rows]
    finally:
        conn.close()


def get_all_pages_index() -> list[dict]:
    """Get all active pages for building index.md."""
    conn = get_connection()
    try:
        rows = conn.execute(
            """SELECT id, title, tags, sources, updated_at
               FROM pages WHERE status = 'active'
               ORDER BY updated_at DESC LIMIT 50"""
        ).fetchall()
        return [_page_row_to_dict(r) for r in rows]
    finally:
        conn.close()


def get_page_by_title(title: str) -> Optional[dict]:
    """Look up a page by exact title match."""
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM pages WHERE title = ? AND status = 'active'",
            (title,),
        ).fetchone()
        return _page_row_to_dict(row) if row else None
    finally:
        conn.close()


# ── Reflection helpers ──

def save_error_record(
    user_message: str,
    wrong_answer: str,
    correct_answer: str,
    category: str,
    contradiction_details: str,
    error_type: str,
) -> int:
    conn = get_connection()
    try:
        cur = conn.execute(
            """INSERT INTO error_records (user_message, wrong_answer, correct_answer, category, contradiction_details, error_type)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (user_message, wrong_answer, correct_answer, category, contradiction_details, error_type),
        )
        conn.commit()
        logger.info("Saved error record: type=%s, category=%s", error_type, category)
        return cur.lastrowid
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def save_error_record_with_embedding(record: dict) -> int:
    """Save error record and store its embedding for semantic search."""
    conn = get_connection()
    try:
        cur = conn.execute(
            """INSERT INTO error_records (user_message, wrong_answer, correct_answer, category, contradiction_details, error_type)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (record["user_message"], record["wrong_answer"], record["correct_answer"],
             record.get("category", ""), record.get("contradiction_details", ""),
             record.get("error_type", "unknown")),
        )
        eid = cur.lastrowid
        # Store error record in knowledge_vectors with an offset rowid to avoid collision
        embedding = generate_embedding(record["user_message"] + " " + record["wrong_answer"])
        conn.execute(
            "INSERT INTO error_vectors(rowid, embedding) VALUES (?, ?)",
            (eid, serialize_float32(embedding)),
        )
        conn.commit()
        logger.info("Saved error record with embedding (id=%d)", eid)
        return eid
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def update_knowledge_status(knowledge_id: int, status: str, corrected_text: str = "") -> None:
    conn = get_connection()
    try:
        conn.execute(
            "UPDATE knowledge_points SET status = ?, corrected_text = ?, updated_at = datetime('now') WHERE id = ?",
            (status, corrected_text, knowledge_id),
        )
        conn.commit()
        logger.info("Updated knowledge point %d status to '%s'", knowledge_id, status)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def get_knowledge_status(knowledge_id: int) -> str:
    conn = get_connection()
    try:
        cur = conn.execute(
            "SELECT status FROM knowledge_points WHERE id = ?",
            (knowledge_id,),
        )
        row = cur.fetchone()
        return row["status"] if row else "active"
    finally:
        conn.close()


def query_knowledge_reasoning_path(knowledge_id: int) -> str:
    """Get the reasoning log file path for a knowledge point."""
    conn = get_connection()
    try:
        cur = conn.execute(
            "SELECT reasoning_log_path FROM knowledge_points WHERE id = ?",
            (knowledge_id,),
        )
        row = cur.fetchone()
        return row["reasoning_log_path"] if row else ""
    finally:
        conn.close()


def update_knowledge_reasoning_path(knowledge_ids: list[int], log_path: str) -> None:
    """Update reasoning_log_path for multiple knowledge points after saving the MD file."""
    conn = get_connection()
    try:
        for kid in knowledge_ids:
            conn.execute(
                "UPDATE knowledge_points SET reasoning_log_path = ? WHERE id = ?",
                (log_path, kid),
            )
        conn.commit()
        logger.info("Updated reasoning_log_path for %d knowledge points to '%s'",
                     len(knowledge_ids), log_path)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def search_error_records_semantic(query: str, limit: int = 3) -> list[dict]:
    """Search error records by semantic similarity to avoid repeating past mistakes."""
    embedding = generate_embedding(query)
    conn = get_connection()
    try:
        rows = conn.execute(
            """SELECT er.*, v.distance
               FROM (
                   SELECT rowid, distance
                   FROM error_vectors
                   WHERE embedding MATCH ?
                     AND k = ?
               ) v
               JOIN error_records er ON er.id = v.rowid
               ORDER BY v.distance""",
            (serialize_float32(embedding), limit * 4),
        ).fetchall()
        results = [dict(r) for r in rows][:limit]
        if results:
            logger.info("Found %d similar error records for query '%s'",
                         len(results), query[:30])
        return results
    finally:
        conn.close()
