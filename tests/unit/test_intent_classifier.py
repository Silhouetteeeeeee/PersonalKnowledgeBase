"""Tests for intent classifier in isolation (no real LLM calls)."""

import pytest
from agent.intent.classifier import classify_intent, _build_intents_section, MIN_CONFIDENCE
from agent.intent.schemas import IntentClassification, IntentResult
from agent.intent.registry import INTENTS, validate_registry
from agent.intent.handlers import HANDLER_MAP


# ── Intent registry tests ──

class TestIntentRegistry:
    def test_all_intents_have_handlers(self):
        """Every intent in INTENTS has a corresponding handler in HANDLER_MAP."""
        warnings = validate_registry(set(HANDLER_MAP.keys()))
        assert not warnings, f"Registry validation warnings: {warnings}"

    def test_knowledge_qa_not_in_handler_map(self):
        """knowledge_qa is handled by the standard graph path, not HANDLER_MAP."""
        assert "knowledge_qa" not in HANDLER_MAP
        # All non-knowledge_qa intents should have a handler
        with_handlers = {n.id for n in INTENTS} - {"knowledge_qa"}
        registered = set(HANDLER_MAP.keys()) - {"low_confidence"}
        assert with_handlers == registered

    def test_intents_have_examples(self):
        """Every intent should have at least 2 examples for the LLM prompt."""
        for n in INTENTS:
            assert len(n.examples) >= 2, f"{n.id} has < 2 examples"

    def test_intents_have_descriptions(self):
        """Every intent must have a non-empty description."""
        for n in INTENTS:
            assert n.description, f"{n.id} has empty description"


# ── Prompt building tests ──

class TestPromptBuilding:
    def test_build_intents_section_contains_all(self):
        """The prompt section includes every intent ID."""
        section = _build_intents_section()
        for n in INTENTS:
            assert n.id in section, f"Missing {n.id} in prompt"
            assert n.name in section, f"Missing {n.name} in prompt"

    def test_build_intents_section_contains_examples(self):
        """At least one example per intent appears in prompt."""
        section = _build_intents_section()
        for n in INTENTS:
            found = any(ex in section for ex in n.examples[:2])
            assert found, f"No examples from {n.id} in prompt"


# ── Classifier logic tests ──

class TestClassifierLogic:
    def test_empty_message_defaults_to_kg(self):
        """Empty user_message → knowledge_qa with confidence 0."""
        result = classify_intent({"user_message": ""})
        assert result["intent"] == "knowledge_qa"
        assert result["intent_confidence"] == 0.0

    def test_llm_returns_top_intent(self, mocker):
        """Valid classification → top intent selected."""
        mocker.patch(
            "agent.intent.classifier.LLM.generate_structured",
            return_value=IntentClassification(
                results=[IntentResult(id="chitchat", score=0.95, reason="Greeting", params={})]
            ),
        )
        result = classify_intent({"user_message": "你好"})
        assert result["intent"] == "chitchat"
        assert result["intent_confidence"] == 0.95

    def test_llm_returns_with_params(self, mocker):
        """Params are extracted alongside intent."""
        mocker.patch(
            "agent.intent.classifier.LLM.generate_structured",
            return_value=IntentClassification(
                results=[IntentResult(
                    id="knowledge_qa", score=0.92, reason="Query about Python",
                    params={"query": "Python decorator"},
                )]
            ),
        )
        result = classify_intent({"user_message": "Python装饰器是什么"})
        assert result["intent"] == "knowledge_qa"
        assert result["intent_params"].get("query") == "Python decorator"

    def test_all_below_threshold_returns_low_confidence(self, mocker):
        """All scores < threshold → low_confidence with fallback suggestions."""
        # 第一次调用返回低于阈值的结果 → 触发 low_confidence
        # _low_confidence_result 中的 LLM 调用会异常 → 使用 fallback 建议
        call_count = [0]
        def mock_llm(*a, **kw):
            call_count[0] += 1
            if call_count[0] == 1:
                return IntentClassification(
                    results=[IntentResult(id="knowledge_qa", score=0.2, reason="vague", params={})]
                )
            raise RuntimeError("模拟第二次 LLM 调用异常")
        mocker.patch("agent.intent.classifier.LLM.generate_structured", side_effect=mock_llm)
        result = classify_intent({"user_message": "嗯"})
        assert result["intent"] == "low_confidence"
        assert len(result.get("low_confidence_suggestions", [])) > 0

    def test_llm_returns_empty_list(self, mocker):
        """Empty results list → low_confidence."""
        call_count = [0]
        def mock_llm(*a, **kw):
            call_count[0] += 1
            if call_count[0] == 1:
                return IntentClassification(results=[])
            raise RuntimeError("模拟第二次 LLM 调用异常")
        mocker.patch("agent.intent.classifier.LLM.generate_structured", side_effect=mock_llm)
        result = classify_intent({"user_message": "test"})
        assert result["intent"] == "low_confidence"

    def test_llm_returns_none(self, mocker):
        """LLM returns None → default to knowledge_qa."""
        mocker.patch(
            "agent.intent.classifier.LLM.generate_structured",
            return_value=None,
        )
        result = classify_intent({"user_message": "test"})
        assert result["intent"] == "knowledge_qa"

    def test_llm_raises_exception(self, mocker):
        """LLM raises exception → default to knowledge_qa."""
        mocker.patch(
            "agent.intent.classifier.LLM.generate_structured",
            side_effect=RuntimeError("API timeout"),
        )
        result = classify_intent({"user_message": "test"})
        assert result["intent"] == "knowledge_qa"

    def test_low_confidence_generates_suggestions(self, mocker):
        """Low confidence result includes human-readable suggestions."""
        # 第一次调用返回空结果，第二次在 _low_confidence_result 中异常 → fallback
        call_count = [0]
        def mock_llm(*a, **kw):
            call_count[0] += 1
            if call_count[0] == 1:
                return IntentClassification(results=[])
            raise RuntimeError("触发 fallback 建议")
        mocker.patch("agent.intent.classifier.LLM.generate_structured", side_effect=mock_llm)
        result = classify_intent({"user_message": "嗯"})
        assert result["intent"] == "low_confidence"
        assert isinstance(result.get("low_confidence_suggestions", []), list)
