import pytest
from storage.database import init_db, get_connection, DB_DIR, DB_PATH


@pytest.fixture(autouse=True)
def temp_db(monkeypatch, tmp_path):
    db_dir = tmp_path / "data"
    db_dir.mkdir()
    monkeypatch.setattr("storage.database.DB_DIR", str(db_dir))
    monkeypatch.setattr("storage.database.DB_PATH", str(db_dir / "knowledge.db"))
    init_db()


def test_save_and_search():
    from storage.models import save_knowledge_point, search_knowledge_points

    save_knowledge_point(
        "Redis RDB creates point-in-time snapshots",
        "What is Redis persistence?",
        "databases/redis",
        ["redis", "persistence"],
    )
    save_knowledge_point(
        "Python list comprehensions provide concise list creation",
        "How do list comprehensions work?",
        "programming/python",
        ["python", "lists"],
    )

    results = search_knowledge_points("Redis")
    assert len(results) == 1
    assert "Redis RDB" in results[0]["knowledge_text"]

    python_results = search_knowledge_points("python")
    assert isinstance(python_results, list)


def test_save_returns_id():
    from storage.models import save_knowledge_point

    id1 = save_knowledge_point("K1", "Q1", "cat/a", ["a"])
    id2 = save_knowledge_point("K2", "Q2", "cat/b", ["b"])
    assert id2 > id1


def test_ensure_category():
    from storage.models import ensure_category
    from storage.database import get_connection

    ensure_category("databases/redis", "Redis related knowledge")
    conn = get_connection()
    cur = conn.execute("SELECT * FROM categories WHERE name = ?", ("databases/redis",))
    row = dict(cur.fetchone())
    conn.close()
    assert row["name"] == "databases/redis"
    assert row["description"] == "Redis related knowledge"
