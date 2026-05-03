
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
    assert isinstance(result["needs_search"], bool)
    assert isinstance(result["needs_store"], bool)


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


def test_store_skips_on_contradiction():
    from agent.nodes.store import store

    result = store({
        "user_message": "test",
        "answer": "some answer",
        "contradiction_found": True,
        "contradiction_details": "test contradiction",
    })
    assert result == {}


def test_graph_includes_fact_check():
    from agent.graph import build_graph

    g = build_graph()
    # 验证新节点存在
    assert "fact_check" in g.nodes
