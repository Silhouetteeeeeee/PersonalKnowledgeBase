"""Unit tests for individual graph nodes (no real LLM calls)."""
import pytest
from storage.database import init_db
from agent.models.value_objects import UrlContent


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


def test_store_empty_answer():
    from agent.nodes.store import _sync_background_store

    # Should not raise on empty answer
    _sync_background_store({"user_message": "hi", "answer": ""})


def test_fact_check_no_answer():
    from agent.nodes.fact_check import fact_check

    result = fact_check({"answer": ""})
    assert result["contradiction_found"] is False
    assert result["contradiction_details"] == ""


def test_fact_check_skips_when_no_related_knowledge():
    from agent.nodes.fact_check import fact_check

    result = fact_check({
        "answer": "Python is a programming language",
        "user_message": "What is Python?",
    })
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
    })
    assert "已标记待审核" in result["final_response"]
    assert "[矛盾警告]" not in result["final_response"]


def test_respond_reflection_answer_wrong():
    from agent.nodes.respond import respond

    result = respond({
        "answer": "Python is a scripting language",
        "contradiction_found": True,
        "reflection_result": "answer_wrong",
    })
    assert "错误" in result["final_response"]


def test_reasoning_log_is_saved(monkeypatch):
    """Verify that _save_reasoning_log writes log content correctly."""
    from unittest.mock import mock_open

    from agent.nodes.store import _save_reasoning_log

    written = {}

    def fake_open(path, mode="r", encoding=None):
        f = mock_open()()
        f.write = lambda s: written.update({"content": s})
        return f

    monkeypatch.setattr("builtins.open", fake_open)
    monkeypatch.setattr("agent.nodes.store.os.makedirs", lambda p, exist_ok: None)

    _save_reasoning_log({
        "user_message": "Say hi",
        "user_id": "test",
        "logic_chain": [{"node": "test", "action": "testing", "reasoning": "test reasoning"}],
    })

    content = written.get("content", "")
    assert "Say hi" in content
    assert "test reasoning" in content


def test_store_skips_on_contradiction():
    from agent.nodes.store import _sync_background_store

    # Should not raise when contradiction is set
    _sync_background_store({
        "user_message": "test",
        "answer": "some answer",
        "contradiction_found": True,
        "contradiction_details": "test contradiction",
    })


def test_regenerate_empty_search():
    from agent.nodes.regenerate import regenerate

    result = regenerate({
        "user_message": "test",
        "answer": "original answer",
        "search_results": [],
    })
    assert result["answer"] == "original answer"


def test_record_error():
    from agent.nodes.record_error import record_error
    from storage.database import get_connection

    result = record_error({
        "user_message": "What is X?",
        "answer": "Wrong answer about X",
        "reflection_correction": "Correct answer about X",
        "contradiction_details": "X is Y, not Z",
        "correction_attempts": 0,
    })

    assert result["error_recorded"] is True
    assert result["correction_attempts"] == 1


def test_parse_with_urls(mocker):
    """含 URL 的消息 → parse 输出 url_contents。"""
    mocker.patch('agent.nodes.parse.fetch_urls_concurrent', return_value=[
        UrlContent(url="https://example.com", title="测试页面", content="正文"),
    ])
    from agent.nodes.parse import parse
    result = parse({
        "user_message": "总结 https://example.com 的内容",
        "user_id": "user1",
        "timestamp": "",
    })
    assert len(result["url_contents"]) == 1
    assert result["url_contents"][0].title == "测试页面"


def test_parse_without_urls():
    """无 URL 的消息 → url_contents 为空列表。"""
    from agent.nodes.parse import parse
    result = parse({
        "user_message": "你好，今天天气怎么样",
        "user_id": "user1",
        "timestamp": "",
    })
    assert result["url_contents"] == []


def test_parse_with_multiple_urls(mocker):
    """多个 URL 全部提取。"""
    mocker.patch('agent.nodes.parse.fetch_urls_concurrent', return_value=[
        UrlContent(url="https://a.com", title="A", content="内容A"),
        UrlContent(url="https://b.com", title="B", content="内容B"),
    ])
    from agent.nodes.parse import parse
    result = parse({
        "user_message": "比较 https://a.com 和 https://b.com",
        "user_id": "user1",
        "timestamp": "",
    })
    assert len(result["url_contents"]) == 2


def test_parse_with_only_url(mocker):
    """消息纯 URL，没有附加文字。"""
    mocker.patch('agent.nodes.parse.fetch_urls_concurrent', return_value=[
        UrlContent(url="https://example.com", title=None, content="正文内容"),
    ])
    from agent.nodes.parse import parse
    result = parse({
        "user_message": "https://example.com",
        "user_id": "user1",
        "timestamp": "",
    })
    assert len(result["url_contents"]) == 1
    assert result["url_contents"][0].title is None
