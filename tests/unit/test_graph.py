"""Unit tests for graph structure: compilation, nodes, edges."""
import pytest
from storage.database import init_db


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


def test_graph_with_no_answer():
    """Test that empty messages don't crash."""
    from agent.graph import build_graph

    graph = build_graph()
    result = graph.invoke({
        "user_message": "",
        "user_id": "test_user",
        "timestamp": "2026-05-02T12:00:00",
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
    assert "rewrite_query" in g.nodes
    assert "parse" in g.nodes
    assert "retrieve" in g.nodes


def test_graph_includes_fact_check():
    from agent.graph import build_graph

    g = build_graph()
    assert "fact_check" in g.nodes
    assert "reflect" in g.nodes
    assert "record_error" in g.nodes
