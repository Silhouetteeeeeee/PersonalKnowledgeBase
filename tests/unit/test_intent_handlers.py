"""Tests for intent handlers in isolation (no real LLM calls)."""

import pytest
from agent.intent.handlers import (
    handle_chitchat,
    handle_link,
    handle_personal_info,
    handle_stub,
    handle_low_confidence,
    dispatch_intent_handler,
    HANDLER_MAP,
)
from agent.intent.registry import INTENTS, validate_registry


class TestChitchatHandler:
    def test_generates_friendly_reply(self, monkeypatch):
        """Chitchat handler returns a friendly LLM-generated reply."""
        monkeypatch.setattr(
            "agent.intent.handlers.LLM.generate",
            lambda prompt, use_language=True: "你好！今天有什么可以帮你的吗？",
        )
        result = handle_chitchat({"user_message": "hello", "intent": "chitchat"})
        assert "你好" in result["answer"]
        assert result.get("needs_store") is False

    def test_llm_failure_fallback(self, monkeypatch):
        """Chitchat handler has a hardcoded fallback when LLM fails."""
        def raise_error(*a, **kw):
            raise RuntimeError("LLM error")
        monkeypatch.setattr("agent.intent.handlers.LLM.generate", raise_error)
        result = handle_chitchat({"user_message": "hello"})
        assert len(result["answer"]) > 0
        assert "?" in result["answer"] or "？" in result["answer"]


class TestLinkHandler:
    def test_summarizes_url_contents(self, monkeypatch):
        """Link handler summarizes URL contents from parse node."""
        from agent.models.value_objects import UrlContent
        monkeypatch.setattr(
            "agent.intent.handlers.LLM.generate",
            lambda prompt, use_language=True: "This is a summary of the article.",
        )
        result = handle_link({
            "user_message": "总结一下",
            "url_contents": [
                UrlContent(url="https://example.com", title="Test Article", content="Article body"),
            ],
        })
        assert "summary" in result["answer"].lower() or "is" in result["answer"]

    def test_no_urls_fallback(self):
        """No URL contents → clear fallback message."""
        result = handle_link({"user_message": "总结", "url_contents": []})
        assert "没有检测到" in result["answer"] or "没有" in result["answer"]

    def test_llm_failure_uses_titles(self, monkeypatch):
        """When LLM fails, at least show URL titles."""
        def raise_error(*a, **kw):
            raise RuntimeError("LLM error")
        monkeypatch.setattr("agent.intent.handlers.LLM.generate", raise_error)
        from agent.models.value_objects import UrlContent
        result = handle_link({
            "user_message": "总结",
            "url_contents": [
                UrlContent(url="https://example.com", title="Test Title", content="body"),
            ],
        })
        assert "Test Title" in result["answer"]


class TestPersonalInfoHandler:
    def test_updates_profile_and_confirms(self, mocker):
        """Personal info handler calls update_profile and returns confirmation."""
        # mock at the source module so the lazy import inside handle_personal_info picks it up
        mock_update = mocker.patch("agent.nodes.update_profile.update_profile")
        mock_update.return_value = {
            "logic_chain": [{"node": "update_profile", "action": "更新 1 个画像字段", "reasoning": ""}],
            "user_profile": {"identity": {"name": "Alice"}},
        }
        result = handle_personal_info({
            "user_message": "我叫Alice",
            "user_id": "u1",
            "intent": "personal_info",
            "intent_params": {"info_type": "姓名"},
            "timestamp": "",
            "confidence": 0.0,
            "answer": "",
        })
        assert "已记录" in result["answer"] or "已了解" in result["answer"]
        assert result["user_profile"]["identity"]["name"] == "Alice"


class TestDispatchHandler:
    def test_dispatch_known_intent(self, mocker):
        """dispatch routes to correct handler."""
        mock_fn = mocker.MagicMock(return_value={"answer": "Hi!", "needs_store": False})
        mocker.patch.dict("agent.intent.handlers.HANDLER_MAP", {"chitchat": mock_fn})
        result = dispatch_intent_handler({"intent": "chitchat", "user_message": "hi"})
        mock_fn.assert_called_once()

    def test_dispatch_unknown_intent(self):
        """Unknown intent → stub handler."""
        result = dispatch_intent_handler({"intent": "nonexistent"})
        assert "answer" in result

    def test_dispatch_handler_raises(self, mocker):
        """Handler exception → graceful error message."""
        def raise_error(state):
            raise ValueError("test error")
        mocker.patch.dict("agent.intent.handlers.HANDLER_MAP", {"test_error": raise_error})
        result = dispatch_intent_handler({"intent": "test_error"})
        assert "answer" in result

    def test_handler_map_covers_all_intents(self):
        """HANDLER_MAP covers all INTENTS (except knowledge_qa which uses standard path)."""
        warnings = validate_registry(set(HANDLER_MAP.keys()))
        assert not warnings, f"Registry validation: {warnings}"
        assert "knowledge_qa" not in HANDLER_MAP


class TestLowConfidenceHandler:
    def test_generates_clarifying_question(self):
        """Low confidence handler returns suggestions in a friendly question."""
        result = handle_low_confidence({
            "low_confidence_suggestions": ["查知识", "聊聊天"],
        })
        assert "不太确定" in result["answer"] or "想" in result["answer"]

    def test_fallback_suggestions(self):
        """When no suggestions provided, uses defaults."""
        result = handle_low_confidence({"intent": "low_confidence"})
        assert result["answer"]


class TestStubHandler:
    def test_stub_returns_under_construction(self):
        """Stub handler returns appropriate 'under construction' message."""
        result = handle_stub({"intent": "learning_plan"})
        assert "开发中" in result["answer"] or "敬请期待" in result["answer"]

    def test_stub_with_unknown_intent(self):
        """Stub handler works for unknown intents too."""
        result = handle_stub({"intent": "unknown"})
        assert len(result["answer"]) > 0
