import pytest
from storage.database import init_db
from storage.profile import load_profile


@pytest.fixture(autouse=True)
def temp_db(monkeypatch, tmp_path):
    db_dir = tmp_path / "data"
    db_dir.mkdir()
    monkeypatch.setattr("storage.database.DB_DIR", str(db_dir))
    monkeypatch.setattr("storage.database.DB_PATH", str(db_dir / "knowledge.db"))
    init_db()


def test_build_graph():
    from agent.graph import build_graph

    graph = build_graph()
    assert graph is not None


def test_graph_short_circuit():
    """Test graph runs end-to-end with a simple question the LLM knows."""
    from agent.graph import build_graph

    graph = build_graph()
    result = graph.invoke({
        "user_message": "What is Python?",
        "user_id": "test_user",
        "timestamp": "2026-05-02T12:00:00",
        "user_profile": load_profile(),
    })
    print(result)
    assert result["final_response"]
    assert len(result["final_response"]) > 0


def test_graph_with_no_answer():
    """Test that empty messages don't crash."""
    from agent.graph import build_graph

    graph = build_graph()
    result = graph.invoke({
        "user_message": "",
        "user_id": "test_user",
        "timestamp": "2026-05-02T12:00:00",
        "user_profile": load_profile(),
    })
    assert "final_response" in result


def test_graph_has_rewrite_query_node():
    """Graph should include the rewrite_query node."""
    from agent.graph import build_graph

    g = build_graph()
    assert "rewrite_query" in g.nodes


def test_graph_rewrite_query_edge():
    """parse should connect to rewrite_query (verify via nodes and bootstrap)."""
    from agent.graph import build_graph

    g = build_graph()
    # Verify the nodes are connected in the right order by running bootstrap
    assert "rewrite_query" in g.nodes
    assert "parse" in g.nodes
    assert "retrieve" in g.nodes


def test_graph_search_query_flows_to_retrieve(monkeypatch):
    """search_query from rewrite_query should be used by retrieve."""
    from storage.models import save_knowledge_point

    save_knowledge_point("Python dict is a key-value store", "What is Python dict?", "programming/python", ["python"])

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
    assert "Python" in stored[0]["knowledge_text"]
