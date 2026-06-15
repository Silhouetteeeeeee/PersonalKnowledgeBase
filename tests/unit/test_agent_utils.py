"""Tests for agent utility functions."""

import pytest
from agent.utils.agent_utils import build_context_block
from agent.models.value_objects import UrlContent


@pytest.mark.skip(reason="Integration test requiring live akshare API access")
def test_fund_utils():
    import fund.utils.fund_data_tools as utils

    utils.get_fund_info("008163")
    utils.get_fund_nav("008163")
    utils.get_fund_holdings("008163")
    utils.get_fund_rankings("008163")
    utils.get_manager_info("008163")
    utils.get_index_data()



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


def test_build_context_block_with_url_contents():
    """url_contents 非空时包含网页内容块。"""
    result = build_context_block({
        "url_contents": [
            UrlContent(url="https://example.com", title="测试标题", content="正文内容")
        ],
    })
    assert "网页内容" in result
    assert "测试标题" in result
    assert "正文内容" in result


def test_build_context_block_with_multiple_urls():
    """多个 URL 时全部包含。"""
    result = build_context_block({
        "url_contents": [
            UrlContent(url="https://a.com", title="文章A", content="内容A"),
            UrlContent(url="https://b.com", title=None, content="内容B的内容"),
        ],
    })
    assert "文章A" in result
    assert "内容B" in result


def test_build_context_block_empty_url_contents():
    """url_contents 为空列表不影响输出。"""
    result = build_context_block({
        "url_contents": [],
        "user_profile": {"basic": {"name": "test"}},
    })
    assert "网页内容" not in result
    assert "test" in result


def test_build_context_block_with_url_only_message():
    """纯 URL 消息含 '请直接总结' 指令标记。"""
    result = build_context_block({
        "url_contents": [
            UrlContent(url="https://example.com", title="文章", content="正文")
        ],
        "user_message": "https://example.com",
    })
    assert "请直接总结" in result
    assert "没有附加问题" in result
