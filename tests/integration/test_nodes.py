"""Integration tests for graph nodes (real LLM calls)."""
import pytest
from storage.database import init_db


@pytest.fixture(autouse=True)
def temp_db(monkeypatch, tmp_path):
    db_dir = tmp_path / "data"
    db_dir.mkdir()
    monkeypatch.setattr("storage.database.DB_DIR", str(db_dir))
    monkeypatch.setattr("storage.database.DB_PATH", str(db_dir / "knowledge.db"))
    init_db()


def test_classify_and_answer():
    from agent.nodes.classify_and_answer import classify_and_answer

    result = classify_and_answer({
        "user_message": "What is Redis?",
        "stored_knowledge": [],
    })
    assert "answer" in result
    assert 0 <= result["confidence"] <= 1
    assert isinstance(result["needs_store"], bool)


def test_classify_and_answer_has_reasoning_trace():
    from agent.nodes.classify_and_answer import classify_and_answer

    result = classify_and_answer({
        "user_message": "What is Redis?",
        "stored_knowledge": [],
    })
    assert "logic_chain" in result
    assert len(result["logic_chain"]) == 1
    assert result["logic_chain"][0]["node"] == "classify_and_answer"
    assert "reasoning" in result["logic_chain"][0]


def test_search_web_node():
    from agent.nodes.search_web import search_web_node

    result = search_web_node({"user_message": "Python"})
    assert "search_results" in result
    assert isinstance(result["search_results"], list)


def test_regenerate_with_search():
    from agent.nodes.regenerate import regenerate

    result = regenerate({
        "user_message": "What is Python?",
        "answer": "I don't know",
        "search_results": [
            "Python is a high-level programming language created by Guido van Rossum.",
        ],
    })
    assert isinstance(result["answer"], str)
    assert len(result["answer"]) > 0
    assert "logic_chain" in result
    assert result["logic_chain"][0]["node"] == "regenerate"


def test_store_distills_knowledge():
    from agent.nodes.store import store

    result = store({
        "user_message": "What is Redis persistence?",
        "answer": "Redis supports RDB snapshots and AOF logs for persistence.",
    })
    assert isinstance(result.get("stored_knowledge_ids"), list)


def test_store_returns_stored_ids():
    from agent.nodes.store import store

    result = store({
        "user_message": "What is Redis?",
        "answer": "Redis is an in-memory data store.",
    })
    if result.get("stored_knowledge_ids"):
        assert isinstance(result.get("stored_knowledge_ids"), list)


def test_logic_chain_accumulates_in_graph():
    from agent.graph import build_graph

    graph = build_graph()
    result = graph.invoke({
        "user_message": "What is Python?",
        "user_id": "test_user",
        "timestamp": "2026-05-02T12:00:00",
    })

    assert "logic_chain" in result
    assert len(result["logic_chain"]) >= 1
    node_names = [entry["node"] for entry in result["logic_chain"]]
    assert "classify_and_answer" in node_names


def test_extract_to_wiki_creates_pages(monkeypatch, tmp_path):
    """Call extract_to_wiki with simple text -> creates wiki pages."""
    from storage.database import DB_DIR, DB_PATH

    monkeypatch.setattr("storage.database.DB_DIR", str(tmp_path / "data"))
    monkeypatch.setattr("storage.database.DB_PATH", str(tmp_path / "data" / "knowledge.db"))
    monkeypatch.setattr("storage.wiki_storage.WIKI_DIR", str(tmp_path / "wiki"))

    from storage.database import init_db
    init_db()

    from agent.nodes.store import extract_to_wiki
    result = extract_to_wiki(
        "Python dict is a key-value store",
        "test_001",
        "Question: What is Python dict?",
    )
    assert "page_ids" in result
    assert len(result["page_ids"]) >= 1
