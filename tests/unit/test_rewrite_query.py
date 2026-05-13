"""Tests for the rewrite_query node."""
import pytest


def test_skip_rewrite_when_no_history(monkeypatch):
    """Conversation with no history → returns original unchanged."""
    monkeypatch.setattr(
        "agent.nodes.rewrite_query.MessageHistory.get_recent",
        lambda session_id, limit=12: [],
    )
    from agent.nodes.rewrite_query import rewrite_query

    result = rewrite_query({
        "user_message": "Java Map 和它的区别？",
        "session_id": "1",
    })
    assert result["search_query"] == "Java Map 和它的区别？"


def test_skip_rewrite_when_only_one_message(monkeypatch):
    """Only 1 prior message → returns original unchanged."""
    monkeypatch.setattr(
        "agent.nodes.rewrite_query.MessageHistory.get_recent",
        lambda session_id, limit=12: [
            {"role": "user", "content": "Python dict 是什么？"},
        ],
    )
    from agent.nodes.rewrite_query import rewrite_query

    result = rewrite_query({
        "user_message": "Java Map 和它的区别？",
        "session_id": "1",
    })
    assert result["search_query"] == "Java Map 和它的区别？"


def test_skip_rewrite_when_only_assistant_message(monkeypatch):
    """Only 1 assistant message → returns original unchanged."""
    monkeypatch.setattr(
        "agent.nodes.rewrite_query.MessageHistory.get_recent",
        lambda session_id, limit=12: [
            {"role": "assistant", "content": "Python dict 是键值对集合"},
        ],
    )
    from agent.nodes.rewrite_query import rewrite_query

    result = rewrite_query({
        "user_message": "Java Map 和它的区别？",
        "session_id": "1",
    })
    assert result["search_query"] == "Java Map 和它的区别？"


def test_rewrite_with_full_history(monkeypatch):
    """Has 2+ history messages → LLM called, rewritten query returned."""
    monkeypatch.setattr(
        "agent.nodes.rewrite_query.MessageHistory.get_recent",
        lambda session_id, limit=12: [
            {"role": "user", "content": "Python dict 是什么？"},
            {"role": "assistant", "content": "Python dict 是键值对集合"},
        ],
    )

    class FakeModel:
        def invoke(self, prompt):
            assert "Python dict" in prompt
            assert "Java Map" in prompt
            return type("AIMessage", (), {"content": "Java Map 和 Python dict 的区别"})()

    monkeypatch.setattr(
        "agent.nodes.rewrite_query.LLM.get_model_for",
        lambda task, temperature=None: FakeModel(),
    )

    from agent.nodes.rewrite_query import rewrite_query

    result = rewrite_query({
        "user_message": "Java Map 和它的区别？",
        "session_id": "1",
    })
    assert result["search_query"] == "Java Map 和 Python dict 的区别"


def test_fallback_when_llm_returns_empty(monkeypatch):
    """LLM returns empty → falls back to original."""
    monkeypatch.setattr(
        "agent.nodes.rewrite_query.MessageHistory.get_recent",
        lambda session_id, limit=12: [
            {"role": "user", "content": "Python dict"},
            {"role": "assistant", "content": "Key-value store"},
        ],
    )

    class FakeModel:
        def invoke(self, prompt):
            return type("AIMessage", (), {"content": ""})()

    monkeypatch.setattr(
        "agent.nodes.rewrite_query.LLM.get_model_for",
        lambda task, temperature=None: FakeModel(),
    )

    from agent.nodes.rewrite_query import rewrite_query

    result = rewrite_query({
        "user_message": "Java Map 和它的区别？",
        "session_id": "1",
    })
    assert result["search_query"] == "Java Map 和它的区别？"


def test_fallback_when_llm_raises(monkeypatch):
    """LLM exception → falls back to original."""
    monkeypatch.setattr(
        "agent.nodes.rewrite_query.MessageHistory.get_recent",
        lambda session_id, limit=12: [
            {"role": "user", "content": "Python dict"},
            {"role": "assistant", "content": "Key-value store"},
        ],
    )

    class FakeModel:
        def invoke(self, prompt):
            raise RuntimeError("API timeout")

    monkeypatch.setattr(
        "agent.nodes.rewrite_query.LLM.get_model_for",
        lambda task, temperature=None: FakeModel(),
    )

    from agent.nodes.rewrite_query import rewrite_query

    result = rewrite_query({
        "user_message": "Java Map 和它的区别？",
        "session_id": "1",
    })
    assert result["search_query"] == "Java Map 和它的区别？"


def test_rewrite_converts_session_id(monkeypatch):
    """session_id conversion from string to int works."""
    captured = {}

    def fake_get_recent(session_id, limit=12):
        captured["sid"] = session_id
        return []

    monkeypatch.setattr(
        "agent.nodes.rewrite_query.MessageHistory.get_recent",
        fake_get_recent,
    )

    from agent.nodes.rewrite_query import rewrite_query

    rewrite_query({
        "user_message": "test",
        "session_id": "42",
    })
    assert captured["sid"] == 42
    assert isinstance(captured["sid"], int)
