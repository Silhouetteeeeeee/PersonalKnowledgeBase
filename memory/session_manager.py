import logging
from datetime import datetime

from storage.database import get_connection

logger = logging.getLogger(__name__)

SESSION_TIMEOUT_MINUTES = 30


class SessionManager:
    """Manage conversation sessions with 30-minute inactivity timeout."""

    def lookup(self, user_id: str) -> dict:
        """Find active session for user, or create new one.

        If an active session exists and was active within 30 minutes, return it.
        Otherwise, archive stale sessions and create a fresh one.
        """
        conn = get_connection()
        try:
            row = conn.execute(
                """
                SELECT id, user_id, status, created_at, last_active_at
                FROM sessions
                WHERE user_id=? AND status='active'
                  AND datetime(last_active_at) > datetime('now', ?)
                ORDER BY last_active_at DESC LIMIT 1
                """,
                (user_id, f'-{SESSION_TIMEOUT_MINUTES} minutes'),
            ).fetchone()

            if row is not None:
                self.refresh(row["id"])
                return dict(row)

            # Archive any remaining active sessions for this user
            conn.execute(
                "UPDATE sessions SET status='archived' WHERE user_id=? AND status='active'",
                (user_id,),
            )

            cursor = conn.execute(
                "INSERT INTO sessions (user_id) VALUES (?)",
                (user_id,),
            )
            conn.commit()
            session_id = cursor.lastrowid
            logger.info("Created new session %s for user %s", session_id, user_id)
            return {
                "id": session_id,
                "user_id": user_id,
                "status": "active",
                "created_at": datetime.now().isoformat(),
                "last_active_at": datetime.now().isoformat(),
            }
        finally:
            conn.close()

    @staticmethod
    def refresh(session_id: int):
        """Update last_active_at timestamp for session."""
        conn = get_connection()
        try:
            conn.execute(
                "UPDATE sessions SET last_active_at=datetime('now','localtime') WHERE id=?",
                (session_id,),
            )
            conn.commit()
        finally:
            conn.close()
