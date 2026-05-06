import logging

from storage.database import get_connection

logger = logging.getLogger(__name__)


class MessageHistory:
    """Store and retrieve conversation messages per session."""

    @staticmethod
    def get_recent(session_id: int, limit: int = 12) -> list[dict]:
        """Get most recent messages for a session, ordered chronologically."""
        conn = get_connection()
        try:
            rows = conn.execute(
                """
                SELECT id, session_id, user_id, role, content, category, created_at
                FROM messages
                WHERE session_id=?
                ORDER BY created_at DESC LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()
            # Reverse to chronological order
            return [dict(r) for r in reversed(rows)]
        finally:
            conn.close()

    @staticmethod
    def add_message(session_id: int, user_id: str, role: str, content: str, category: str = ""):
        """Insert a message record."""
        conn = get_connection()
        try:
            conn.execute(
                "INSERT INTO messages (session_id, user_id, role, content, category) VALUES (?, ?, ?, ?, ?)",
                (session_id, user_id, role, content, category),
            )
            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def get_session_messages(session_id: int) -> list[dict]:
        """Get ALL messages for a session (used by episodic summarization)."""
        conn = get_connection()
        try:
            rows = conn.execute(
                """
                SELECT id, session_id, user_id, role, content, category, created_at
                FROM messages
                WHERE session_id=?
                ORDER BY created_at ASC
                """,
                (session_id,),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()
