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


def test_bulk_save_with_embeddings():
    from storage.models import save_knowledge_points_bulk_with_embeddings, find_similar_knowledge

    points = [
        {
            "knowledge_text": "Redis RDB creates point-in-time snapshots of data",
            "source_question": "What is Redis persistence?",
            "category": "databases/redis",
            "tags": ["redis", "persistence"],
        },
    ]
    ids = save_knowledge_points_bulk_with_embeddings(points)
    assert len(ids) == 1

    # Verify embedding was stored by searching semantically
    similar = find_similar_knowledge("Redis RDB snapshots", threshold=0.5)
    assert len(similar) == 1
    assert similar[0]["id"] == ids[0]


def test_find_similar_knowledge_semantic():
    from storage.models import save_knowledge_points_bulk_with_embeddings, find_similar_knowledge

    points = [
        {
            "knowledge_text": "Python is a high-level programming language",
            "source_question": "What is Python?",
            "category": "programming/python",
            "tags": ["python"],
        },
        {
            "knowledge_text": "Redis is an in-memory data store",
            "source_question": "What is Redis?",
            "category": "databases/redis",
            "tags": ["redis"],
        },
    ]
    save_knowledge_points_bulk_with_embeddings(points)

    # Similar text should find a match within threshold
    similar = find_similar_knowledge("Python is a programming language", threshold=0.3)
    assert len(similar) == 1
    assert "Python" in similar[0]["knowledge_text"]

    # Unrelated text should not find matches with strict threshold
    unrelated = find_similar_knowledge("The weather is nice today", threshold=0.3)
    assert len(unrelated) == 0


def test_normalize_category_str():
    from storage.models import normalize_category_str

    # lowercase
    assert normalize_category_str("RAG/ReRank") == "rag/rerank"

    # unify separators
    assert normalize_category_str("databases \\ redis") == "databases/redis"
    assert normalize_category_str("life | health") == "life/health"
    assert normalize_category_str("programming·python") == "programming/python"
    assert normalize_category_str("frameworks→pytorch") == "frameworks/pytorch"
    assert normalize_category_str("AI／RAG") == "ai/rag"

    # strip whitespace
    assert normalize_category_str("  ai  /  rag  ") == "ai/rag"

    # remove hyphens and underscores
    assert normalize_category_str("re-ranker") == "reranker"
    assert normalize_category_str("re_ranker") == "reranker"

    # depth truncation
    long = "a/b/c/d/e/f"
    assert normalize_category_str(long, max_depth=4) == "a/b/c/d"
    assert normalize_category_str(long, max_depth=2) == "a/b"

    # idempotent
    raw = "  RAG \\ ReRank-Re_ranking "
    once = normalize_category_str(raw)
    twice = normalize_category_str(once)
    assert once == twice


def test_get_normalized_categories_empty():
    from storage.models import get_normalized_categories

    result = get_normalized_categories()
    assert result == ""


def test_get_normalized_categories_with_data():
    from storage.models import (
        save_knowledge_point,
        get_normalized_categories,
    )

    save_knowledge_point("text1", "q1", "databases/redis", ["redis"])
    save_knowledge_point("text2", "q2", "ai/rag", ["rag"])

    result = get_normalized_categories()
    assert "databases/redis" in result
    assert "ai/rag" in result
    assert "目前已存在的分类" in result
    assert "2 个分类" in result
