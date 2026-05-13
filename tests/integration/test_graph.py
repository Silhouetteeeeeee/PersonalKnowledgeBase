"""Integration tests: full graph with real LLM calls."""
import pytest
from storage.database import init_db


@pytest.fixture(autouse=True)
def temp_db(monkeypatch, tmp_path):
    db_dir = tmp_path / "data"
    db_dir.mkdir()
    monkeypatch.setattr("storage.database.DB_DIR", str(db_dir))
    monkeypatch.setattr("storage.database.DB_PATH", str(db_dir / "knowledge.db"))
    init_db()


def test_graph_short_circuit():
    """Test graph runs end-to-end with a simple question the LLM knows."""
    from agent.graph import build_graph

    graph = build_graph()
    result = graph.invoke({
        "user_message": "What is Python?",
        "user_id": "test_user",
        "timestamp": "2026-05-02T12:00:00",
        "user_profile": {},
    })
    assert result["final_response"]
    assert len(result["final_response"]) > 0


def test_graph_search_query_flows_to_retrieve(monkeypatch, tmp_path):
    """search_query from rewrite_query should be used by retrieve."""
    from storage.models import upsert_page
    from storage.wiki_storage import WIKI_DIR, write_page

    monkeypatch.setattr("storage.wiki_storage.WIKI_DIR", str(tmp_path / "wiki"))

    file_path = "pages/python-dict.md"

    upsert_page(
        title="Python dict",
        file_path=file_path,
        tags=["python", "dict"],
        sources=["test"],
        checksum="abc",
        content="Python dict is a key-value store",
    )
    write_page(file_path, "---\ntitle: Python dict\ntags: [python, dict]\n---\n\nPython dict is a key-value store")

    monkeypatch.setattr(
        "agent.nodes.rewrite_query.MessageHistory.get_recent",
        lambda session_id, limit=12: [
            {"role": "user", "content": "Python dict 是什么？"},
            {"role": "assistant", "content": "Python dict 是键值对集合"},
        ],
    )

    class FakeModel:
        def invoke(self, prompt):
            return type("AIMessage", (), {"content": "Python dict differences"})

    monkeypatch.setattr(
        "agent.nodes.rewrite_query.LLM.get_model_for",
        lambda task, temperature=None: FakeModel(),
    )

    from agent.graph import build_graph
    g = build_graph()

    result = g.invoke({
        "user_message": "Java Map and its differences",
        "user_id": "test_user",
        "session_id": "1",
        "search_query": "",
        "timestamp": "2026-05-02T12:00:00",
    })

    stored = result.get("stored_knowledge", [])
    assert len(stored) >= 1
    assert stored[0]["type"] == "wiki_page"
    assert stored[0]["title"] == "Python dict"
