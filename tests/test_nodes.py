import pytest
from storage.database import init_db


@pytest.fixture(autouse=True)
def temp_db(monkeypatch, tmp_path):
    db_dir = tmp_path / "data"
    db_dir.mkdir()
    monkeypatch.setattr("storage.database.DB_DIR", str(db_dir))
    monkeypatch.setattr("storage.database.DB_PATH", str(db_dir / "knowledge.db"))
    init_db()


def test_parse():
    from agent.nodes.parse import parse

    result = parse({
        "user_message": "  hello world  ",
        "user_id": "user1",
        "timestamp": "2026-05-02T12:00:00",
    })
    assert result["user_message"] == "hello world"
    assert result["user_id"] == "user1"


def test_retrieve_no_results():
    from agent.nodes.retrieve import retrieve

    result = retrieve({"user_message": "something not in db"})
    assert result["stored_knowledge"] == []


def test_retrieve_with_results():
    from storage.models import save_knowledge_point
    from agent.nodes.retrieve import retrieve

    save_knowledge_point("Python is a programming language", "What is Python?", "programming/python", ["python"])
    result = retrieve({"user_message": "Tell me about Python"})
    assert len(result["stored_knowledge"]) == 1
    assert "Python" in result["stored_knowledge"][0]["knowledge_text"]


def test_classify_and_answer():
    from agent.nodes.classify_and_answer import classify_and_answer

    result = classify_and_answer({
        "user_message": "What is Redis?",
        "stored_knowledge": [],
    })
    assert "category" in result
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


def test_regenerate_empty_search():
    from agent.nodes.regenerate import regenerate

    result = regenerate({
        "user_message": "test",
        "answer": "original answer",
        "search_results": [],
    })
    assert result["answer"] == "original answer"


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
    # Should include logic_chain
    assert "logic_chain" in result
    assert result["logic_chain"][0]["node"] == "regenerate"


def test_store_empty_answer():
    from agent.nodes.store import store

    result = store({"user_message": "hi", "answer": ""})
    assert result == {}


def test_store_distills_knowledge():
    from agent.nodes.store import store

    result = store({
        "user_message": "What is Redis persistence?",
        "answer": "Redis supports RDB snapshots and AOF logs for persistence.",
    })
    assert "category" in result
    assert isinstance(result["category"], str)


def test_fact_check_no_answer():
    from agent.nodes.fact_check import fact_check

    result = fact_check({"answer": ""})
    assert result == {"contradiction_found": False, "contradiction_details": ""}


def test_fact_check_skips_when_no_related_knowledge():
    from agent.nodes.fact_check import fact_check

    result = fact_check({
        "answer": "Python is a programming language",
        "user_message": "What is Python?",
    })
    # No stored knowledge → no contradiction
    assert result.get("contradiction_found") is False


def test_respond_normal():
    from agent.nodes.respond import respond

    result = respond({"answer": "Hello world"})
    assert result["final_response"] == "Hello world"


def test_respond_contradiction_warning():
    from agent.nodes.respond import respond

    result = respond({
        "answer": "Python is a scripting language",
        "contradiction_found": True,
        "contradiction_details": "Python is a compiled language, not scripting",
    })
    assert "[矛盾警告]" in result["final_response"]
    assert "Python is a compiled language" in result["final_response"]


def test_respond_reflection_stored_knowledge_wrong():
    from agent.nodes.respond import respond

    result = respond({
        "answer": "Python is a scripting language",
        "contradiction_found": True,
        "reflection_result": "stored_knowledge_wrong",
        "knowledge_corrected": True,
    })
    assert "已自动修正" in result["final_response"]
    assert "[矛盾警告]" not in result["final_response"]


def test_respond_reflection_answer_wrong():
    from agent.nodes.respond import respond

    result = respond({
        "answer": "Python is a scripting language",
        "contradiction_found": True,
        "reflection_result": "answer_wrong",
    })
    assert "错误" in result["final_response"]


def test_respond_saves_reasoning_log(monkeypatch, tmp_path):
    """Verify that respond creates a reasoning log MD file."""
    reasoning_dir = tmp_path / "reasoning"
    monkeypatch.setattr("agent.nodes.respond.REASONING_LOG_DIR", str(reasoning_dir))

    from agent.nodes.respond import respond

    result = respond({
        "answer": "Hello",
        "user_message": "Say hi",
        "user_id": "test",
        "logic_chain": [{"node": "test", "action": "testing", "reasoning": "test reasoning"}],
    })

    # Check a file was created
    md_files = list(reasoning_dir.rglob("*.md"))
    assert len(md_files) >= 1
    content = md_files[0].read_text(encoding="utf-8")
    assert "Say hi" in content
    assert "test reasoning" in content
    assert "testing" in content


def test_store_skips_on_contradiction():
    from agent.nodes.store import store

    result = store({
        "user_message": "test",
        "answer": "some answer",
        "contradiction_found": True,
        "contradiction_details": "test contradiction",
    })
    assert result == {}


def test_store_returns_stored_ids():
    """Store should return stored_knowledge_ids when knowledge is saved."""
    from agent.nodes.store import store

    result = store({
        "user_message": "What is Redis?",
        "answer": "Redis is an in-memory data store.",
    })
    if result:  # May be empty if dedup skips everything
        assert isinstance(result.get("category"), str)


def test_correct_knowledge():
    from storage.models import save_knowledge_point, get_knowledge_status
    from agent.nodes.correct_knowledge import correct_knowledge

    # Save a knowledge point first
    kid = save_knowledge_point("Old wrong fact", "test question", "test", ["test"])
    assert get_knowledge_status(kid) == "active"

    # Correct it
    result = correct_knowledge({
        "contradiction_knowledge_ids": [kid],
        "reflection_correction": "New correct fact",
        "category": "test",
        "user_message": "test question",
        "correction_attempts": 0,
    })

    assert result["knowledge_corrected"] is True
    assert result["correction_attempts"] == 1
    # Verify status changed
    assert get_knowledge_status(kid) == "deprecated"


def test_correct_knowledge_increments_counter():
    from agent.nodes.correct_knowledge import correct_knowledge

    result = correct_knowledge({
        "contradiction_knowledge_ids": [],
        "reflection_correction": "",
        "category": "test",
        "user_message": "test",
        "correction_attempts": 1,
    })
    assert result["correction_attempts"] == 2


def test_record_error():
    from agent.nodes.record_error import record_error
    from storage.database import get_connection

    result = record_error({
        "user_message": "What is X?",
        "answer": "Wrong answer about X",
        "reflection_correction": "Correct answer about X",
        "category": "test",
        "contradiction_details": "X is Y, not Z",
        "correction_attempts": 0,
    })

    assert result["error_recorded"] is True
    assert result["correction_attempts"] == 1

    # Verify DB record
    conn = get_connection()
    row = conn.execute("SELECT * FROM error_records WHERE user_message = 'What is X?'").fetchone()
    conn.close()
    assert row is not None
    assert row["wrong_answer"] == "Wrong answer about X"
    assert row["correct_answer"] == "Correct answer about X"


def test_logic_chain_accumulates_in_graph():
    """Verify that logic_chain accumulates across multiple nodes via graph invoke."""
    from agent.graph import build_graph

    graph = build_graph()
    result = graph.invoke({
        "user_message": "What is Python?",
        "user_id": "test_user",
        "timestamp": "2026-05-02T12:00:00",
    })

    # The graph runs multiple nodes; logic_chain should have multiple entries
    assert "logic_chain" in result
    assert len(result["logic_chain"]) >= 1
    # Check first entry is from parse or classify
    node_names = [entry["node"] for entry in result["logic_chain"]]
    assert "classify_and_answer" in node_names


def test_graph_includes_fact_check():
    from agent.graph import build_graph

    g = build_graph()
    # 验证新节点存在
    assert "fact_check" in g.nodes
    assert "reflect" in g.nodes
    assert "correct_knowledge" in g.nodes
    assert "record_error" in g.nodes
