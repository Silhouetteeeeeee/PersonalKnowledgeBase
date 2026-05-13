from memory.models import init_memory_tables
from storage.database import get_connection


class TestMemoryModels:
    def test_init_memory_tables_creates_tables(self):
        """Calling init_memory_tables creates sessions and messages tables."""
        init_memory_tables()
        conn = get_connection()
        try:
            tables = {
                r["name"]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            assert "sessions" in tables
            assert "messages" in tables
        finally:
            conn.close()

    def test_init_memory_tables_idempotent(self):
        """Calling init_memory_tables multiple times does not error."""
        init_memory_tables()
        init_memory_tables()
        init_memory_tables()
        # If we reach here without exceptions, the test passes
        assert True
