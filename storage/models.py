import json
import logging
import sqlite3
import threading
import time
from datetime import datetime, timedelta
from typing import Optional

import jieba

from agent.models.storage import WikiPage

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
    """生成 512 维文本嵌入向量，用于向量相似度搜索。"""
    model = _get_embedder()
    vec = next(model.embed(text))
    return vec.tolist()


# ── Tokenization (used by keyword search fallback) ──

def _tokenize(text: str) -> list[str]:
    words = jieba.lcut(text)
    return [w.strip() for w in words if len(w.strip()) >= 2]


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
                (page_id, link_title.lower().strip()),
            )
        conn.commit()
        logger.info("Updated %d relations for page id=%d", len(linked_titles), page_id)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ── Version helpers ──

def save_page_version(
    page_id: int,
    title: str,
    content: str,
    checksum: str,
    source_id: str = "",
    source_question: str = "",
) -> int:
    """Save a new version of a wiki page. Auto-increments version number per page_id.

    Returns the version number that was saved.
    """
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT COALESCE(MAX(version), 0) + 1 FROM page_versions WHERE page_id = ?",
            (page_id,),
        ).fetchone()
        next_ver = row[0]

        conn.execute(
            """INSERT INTO page_versions (page_id, version, title, content, checksum, source_id, source_question)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (page_id, next_ver, title, content, checksum, source_id, source_question),
        )
        conn.commit()
        logger.info("Saved version %d for page '%s' (id=%d)", next_ver, title, page_id)
        return next_ver
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def get_page_versions(page_id: int, limit: int = 20) -> list[dict]:
    """List versions for a page, most recent first."""
    conn = get_connection()
    try:
        rows = conn.execute(
            """SELECT id, version, title, source_id, source_question, change_summary, created_at
               FROM page_versions WHERE page_id = ?
               ORDER BY version DESC LIMIT ?""",
            (page_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_page_version(page_id: int, version: int) -> dict | None:
    """Get a specific version's full content."""
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM page_versions WHERE page_id = ? AND version = ?",
            (page_id, version),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def cleanup_old_versions(days: int = 30) -> int:
    """Delete versions older than `days`, keeping at least 1 per page.

    Returns number of deleted rows.
    """
    from datetime import datetime, timedelta
    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")

    conn = get_connection()
    try:
        deleted = conn.execute(
            """DELETE FROM page_versions WHERE created_at < ? AND id NOT IN (
                   SELECT MAX(id) FROM page_versions GROUP BY page_id
               )""",
            (cutoff,),
        ).rowcount
        conn.commit()
        if deleted:
            logger.info("Cleaned up %d old page versions (cutoff=%s)", deleted, cutoff)
        return deleted
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def find_similar_pages(query: str, threshold: float = 0.6, limit: int = 5) -> list[WikiPage]:
    """Search wiki pages by semantic similarity. Returns list of WikiPage with distance."""
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
        results = [WikiPage(**dict(r)) for r in rows][:limit]
        logger.info("Page semantic search: %d results for '%s'", len(results), query[:30])
        return results
    finally:
        conn.close()


def get_related_pages(page_id: int) -> list[WikiPage]:
    """Get pages linked via page_relations to the given page (bidirectional)."""
    conn = get_connection()
    try:
        rows = conn.execute(
            """SELECT p.* FROM pages p
               JOIN page_relations r ON r.target_title = p.title COLLATE NOCASE
               WHERE r.source_id = ? AND p.status = 'active'
               UNION
               SELECT p.* FROM pages p
               JOIN page_relations r ON r.source_id = p.id
               WHERE r.target_title = (SELECT title FROM pages WHERE id = ?) COLLATE NOCASE
                 AND p.status = 'active'
               """,
            (page_id, page_id),
        ).fetchall()
        return [WikiPage(**dict(r)) for r in rows]
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
    """Look up a page by title (case-insensitive)."""
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM pages WHERE title = ? COLLATE NOCASE AND status = 'active'",
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
        # Store error record in error_vectors with an offset rowid to avoid collision
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


# ── Spaced repetition helpers ──

def init_review_schedule(page_id: int, next_review_at: str | None = None) -> int:
    """Insert a new review schedule for a page. Returns schedule id."""
    conn = get_connection()
    try:
        if next_review_at is None:
            next_review_at = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")
        conn.execute(
            """INSERT OR IGNORE INTO review_schedule (page_id, next_review_at)
               VALUES (?, ?)""",
            (page_id, next_review_at),
        )
        conn.commit()
        row = conn.execute(
            "SELECT id FROM review_schedule WHERE page_id = ?", (page_id,)
        ).fetchone()
        sid = row["id"] if row else 0
        if sid:
            logger.info("Initialized review schedule for page_id=%d (schedule_id=%d)", page_id, sid)
        return sid
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def get_due_reviews(limit: int = 10) -> list[dict]:
    """Query all schedules where next_review_at <= now, up to limit."""
    conn = get_connection()
    try:
        rows = conn.execute(
            """SELECT rs.*, p.title, p.file_path
               FROM review_schedule rs
               JOIN pages p ON p.id = rs.page_id
               WHERE rs.next_review_at <= datetime('now', 'localtime')
                 AND p.status = 'active'
               ORDER BY rs.next_review_at ASC
               LIMIT ?""",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def update_review_schedule(
    schedule_id: int,
    easiness_factor: float,
    interval_days: int,
    repetitions: int,
    next_review_at: str,
    quality: int,
) -> None:
    """Update SM-2 parameters after a review."""
    conn = get_connection()
    try:
        conn.execute(
            """UPDATE review_schedule
               SET easiness_factor = ?, interval_days = ?, repetitions = ?,
                   next_review_at = ?, last_reviewed_at = datetime('now', 'localtime'),
                   last_quality = ?
               WHERE id = ?""",
            (easiness_factor, interval_days, repetitions, next_review_at, quality, schedule_id),
        )
        conn.commit()
        logger.info("Updated review schedule id=%d: EF=%.2f interval=%d reps=%d quality=%d",
                    schedule_id, easiness_factor, interval_days, repetitions, quality)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def has_pending_review(page_id: int) -> bool:
    """Check if a page has a pending (unanswered) review."""
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT 1 FROM sent_reviews WHERE page_id = ? AND status = 'pending' LIMIT 1",
            (page_id,),
        ).fetchone()
        return row is not None
    finally:
        conn.close()


def get_sent_review_by_marker(marker_id: str) -> dict | None:
    """Look up a sent_review by its marker_id."""
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM sent_reviews WHERE marker_id = ?", (marker_id,),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def mark_review_answered(sent_id: int) -> None:
    """Mark a sent_review as answered."""
    conn = get_connection()
    try:
        conn.execute(
            "UPDATE sent_reviews SET status = 'reviewed' WHERE id = ?", (sent_id,),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def get_reviewed_pages_since(days: int = 7) -> list[dict]:
    """Get pages reviewed in the last N days (for weekly integration)."""
    conn = get_connection()
    try:
        rows = conn.execute(
            """SELECT DISTINCT rs.page_id, p.title, p.file_path
               FROM review_schedule rs
               JOIN pages p ON p.id = rs.page_id
               WHERE rs.last_reviewed_at >= datetime('now', 'localtime', ?)
                 AND rs.last_quality >= 3
                 AND p.status = 'active'
               ORDER BY rs.last_reviewed_at DESC""",
            (f'-{days} days',),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def record_sent_review(schedule_id: int, page_id: int, marker_id: str) -> int:
    """Record that a review was sent to the user."""
    conn = get_connection()
    try:
        cur = conn.execute(
            "INSERT INTO sent_reviews (schedule_id, page_id, marker_id) VALUES (?, ?, ?)",
            (schedule_id, page_id, marker_id),
        )
        conn.commit()
        logger.info("Recorded sent review: marker=%s page_id=%d", marker_id, page_id)
        return cur.lastrowid
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def process_review_feedback(
    schedule_id: int,
    sent_id: int,
    easiness_factor: float,
    interval_days: int,
    repetitions: int,
    next_review_at: str,
    quality: int,
) -> None:
    """Atomically update SM-2 schedule and mark sent_review as answered.

    Performed in a single transaction for consistency.
    """
    conn = get_connection()
    try:
        conn.execute(
            """UPDATE review_schedule
               SET easiness_factor = ?, interval_days = ?, repetitions = ?,
                   next_review_at = ?, last_reviewed_at = datetime('now', 'localtime'),
                   last_quality = ?
               WHERE id = ?""",
            (easiness_factor, interval_days, repetitions, next_review_at, quality, schedule_id),
        )
        conn.execute(
            "UPDATE sent_reviews SET status = 'reviewed' WHERE id = ?", (sent_id,),
        )
        conn.commit()
        logger.info("Processed review feedback: schedule=%d sent=%d quality=%d next=%s",
                    schedule_id, sent_id, quality, next_review_at)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
