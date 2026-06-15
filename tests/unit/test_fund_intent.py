"""Tests for LLM-based intent classification."""

from fund.intent.classifier import classify, validate_params, MIN_SCORE
from fund.intent.schemas import INTENTS, IntentNode, ParamSchema


def test_intents_defined():
    """All intents have required fields."""
    for n in INTENTS:
        assert n.id, f"missing id in {n}"
        assert n.name, f"missing name in {n}"
        assert n.description, f"missing description in {n}"
        assert n.examples, f"missing examples in {n}"
        assert n.kind in ("fund", "portfolio", "system"), f"invalid kind in {n}"
        for p in n.params:
            assert p.name, f"missing param name in {n.id}"
            assert p.type in ("fund_code", "fund_name", "shares", "cost", "query")


def test_intent_ids_unique():
    """No duplicate intent IDs."""
    ids = [n.id for n in INTENTS]
    assert len(ids) == len(set(ids))


def test_validate_params_ok():
    intent = IntentNode(id="test", name="test", description="", examples=[],
                        kind="system", params=[])
    ok, msg = validate_params(intent, {})
    assert ok is True


def test_validate_params_missing():
    intent = IntentNode(id="test", name="test", description="", examples=[],
                        kind="system",
                        params=[ParamSchema(name="code", type="fund_code", description="基金代码")])
    ok, msg = validate_params(intent, {})
    assert ok is False
    assert "code" in msg


def test_validate_params_partial():
    intent = IntentNode(id="test", name="test", description="", examples=[],
                        kind="system",
                        params=[
                            ParamSchema(name="code", type="fund_code", description="基金代码"),
                            ParamSchema(name="shares", type="shares", description="份额"),
                        ])
    ok, msg = validate_params(intent, {"code": "110011"})
    assert ok is False
    assert "shares" in msg
    assert "code" not in msg


def test_min_score_constant():
    assert MIN_SCORE == 0.35
