"""Tests for agent utility functions."""

from agent.utils.agent_utils import build_context_block


def test_build_context_block_empty():
    result = build_context_block({})
    assert result == ""


def test_build_context_block_with_profile():
    result = build_context_block({
        "user_profile": {"basic": {"name": "Alice"}},
    })
    assert "Alice" in result
    assert "用户画像" in result


def test_build_context_block_with_knowledge():
    result = build_context_block({
        "stored_knowledge": [
            {"knowledge_text": "Python is a programming language"},
        ],
    })
    assert "Python" in result
    assert "相关知识" in result


def test_build_context_block_with_history():
    result = build_context_block({
        "message_history": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ],
    })
    assert "用户: Hello" in result
    assert "助手: Hi there" in result


def test_build_context_block_full():
    result = build_context_block({
        "user_profile": {"basic": {"name": "Bob"}},
        "stored_knowledge": [{"knowledge_text": "Redis is fast"}],
        "message_history": [{"role": "user", "content": "What is Redis?"}],
        "episodic_memories": ["User asked about databases before"],
    })
    assert "Bob" in result
    assert "Redis is fast" in result
    assert "What is Redis?" in result
    assert "databases" in result
