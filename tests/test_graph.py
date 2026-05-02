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


def test_graph_short_circuit():
    """Test graph runs end-to-end with a simple question the LLM knows."""
    from agent.graph import build_graph

    graph = build_graph()
    result = graph.invoke({
        "user_message": "What is Python?",
        "user_id": "test_user",
        "timestamp": "2026-05-02T12:00:00",
    })
    assert result["final_response"]
    assert len(result["final_response"]) > 0
    assert "category" in result


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
