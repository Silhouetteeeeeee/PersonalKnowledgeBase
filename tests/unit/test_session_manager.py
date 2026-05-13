from memory.session_manager import SessionManager


class TestSessionManager:
    def test_lookup_new_user(self):
        """First-time user gets a new session."""
        manager = SessionManager()
        session = manager.lookup("new_user_1")
        assert session["user_id"] == "new_user_1"
        assert session["status"] == "active"
        assert session["id"] > 0

    def test_lookup_reuses_active_session(self):
        """Same user within 30min gets same session."""
        manager = SessionManager()
        s1 = manager.lookup("user_reuse")
        s2 = manager.lookup("user_reuse")
        assert s1["id"] == s2["id"]

    def test_lookup_different_users_get_different_sessions(self):
        """Different users get different sessions."""
        manager = SessionManager()
        s1 = manager.lookup("user_a")
        s2 = manager.lookup("user_b")
        assert s1["id"] != s2["id"]

    def test_refresh_updates_last_active(self):
        """refresh() updates last_active_at."""
        from datetime import datetime
        from storage.database import get_connection

        manager = SessionManager()
        session = manager.lookup("user_refresh")
        old_active = session["last_active_at"]
        manager.refresh(session["id"])
        conn = get_connection()
        try:
            row = conn.execute(
                "SELECT last_active_at FROM sessions WHERE id=?", (session["id"],)
            ).fetchone()
            old_dt = datetime.fromisoformat(old_active)
            db_dt = datetime.strptime(row["last_active_at"], "%Y-%m-%d %H:%M:%S")
            assert db_dt >= old_dt
        finally:
            conn.close()

    def test_lookup_archives_stale(self):
        """Session older than 30min gets archived, new one created."""
        from storage.database import get_connection

        # Clean up any previous test data for isolation
        conn = get_connection()
        try:
            conn.execute("DELETE FROM sessions WHERE user_id='user_stale'")
            conn.execute(
                "INSERT INTO sessions (user_id, status, last_active_at) VALUES (?, 'active', datetime('now','-31 minutes'))",
                ("user_stale",),
            )
            conn.commit()
        finally:
            conn.close()

        manager = SessionManager()
        session = manager.lookup("user_stale")

        conn = get_connection()
        try:
            stale_count = conn.execute(
                "SELECT COUNT(*) as c FROM sessions WHERE user_id='user_stale' AND status='active'"
            ).fetchone()["c"]
            assert stale_count == 1
        finally:
            conn.close()
        assert session["status"] == "active"
