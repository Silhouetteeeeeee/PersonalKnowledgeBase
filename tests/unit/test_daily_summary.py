"""Unit tests for daily knowledge summary."""
import json
import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from storage.database import init_db


@pytest.fixture(autouse=True)
def temp_db(monkeypatch, tmp_path):
    db_dir = tmp_path / "data"
    db_dir.mkdir()
    monkeypatch.setattr("storage.database.DB_DIR", str(db_dir))
    monkeypatch.setattr("storage.database.DB_PATH", str(db_dir / "knowledge.db"))
    init_db()


@pytest.fixture
def mock_pages():
    return [
        {
            "id": 1,
            "title": "TCP三次握手",
            "file_path": "pages/tcp-handshake.md",
            "tags": '["TCP","网络"]',
            "sources": '["conv_001"]',
            "created_at": (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S"),
            "updated_at": (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S"),
            "content": "TCP三次握手是建立可靠连接的过程...",
            "source_questions": ["TCP三次握手的过程是什么"],
        },
    ]


def test_get_yesterday_pages_empty(monkeypatch, tmp_path):
    """No pages yesterday → empty list."""
    # Patch wiki dir
    wiki_dir = tmp_path / "wiki"
    monkeypatch.setattr("storage.wiki_storage.WIKI_DIR", str(wiki_dir))

    from server.daily_summary import _get_yesterday_pages

    pages = _get_yesterday_pages()
    assert pages == []


def test_get_yesterday_pages_with_data(monkeypatch, tmp_path):
    """Pages created yesterday → returned with content."""
    # Patch wiki dir
    wiki_dir = tmp_path / "wiki"
    pages_dir = wiki_dir / "pages"
    pages_dir.mkdir(parents=True)
    monkeypatch.setattr("storage.wiki_storage.WIKI_DIR", str(wiki_dir))

    # Insert a page record into DB
    from storage.database import get_connection

    conn = get_connection()
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")
    conn.execute(
        "INSERT INTO pages (id, title, file_path, tags, sources, created_at, updated_at) "
        "VALUES (1, 'Test Page', 'pages/test.md', '[]', '[]', ?, ?)",
        (yesterday, yesterday),
    )
    conn.commit()
    conn.close()

    # Create the wiki file on disk
    test_page = pages_dir / "test.md"
    test_page.write_text(
        "---\ntitle: Test Page\ntags: []\nsources: []\ncreated: 2026-05-13\nupdated: 2026-05-13\n---\n\nTest content body",
        encoding="utf-8",
    )

    from server.daily_summary import _get_yesterday_pages

    pages = _get_yesterday_pages()
    assert len(pages) == 1
    assert pages[0]["title"] == "Test Page"
    assert "Test content body" in pages[0]["content"]


def test_generate_summary_text(mock_pages):
    """Verify LLM summary generation produces markdown."""
    from server.daily_summary import _generate_summary_text

    mock_summary = "📅 昨日知识汇总\n\n📌 新增页面（1篇）\n\n**1. TCP三次握手**\n用户问：TCP三次握手的过程是什么？\n→ 内容摘要"

    with patch("server.daily_summary.LLM.generate_structured") as mock_llm:
        mock_llm.return_value = MagicMock(summary=mock_summary)
        result = _generate_summary_text(mock_pages)
        assert "📅" in result
        assert "TCP三次握手" in result


def test_send_daily_summary_skips_when_empty():
    """No pages → no send_message call."""
    from server.daily_summary import send_daily_summary
    import asyncio

    mock_client = MagicMock()

    with patch("server.daily_summary._get_yesterday_pages", return_value=[]):
        result = asyncio.run(send_daily_summary(mock_client, "user1"))
        assert result is False
        mock_client.send_message.assert_not_called()


def test_send_daily_summary_sends(mock_pages):
    """Has pages → calls send_message with markdown."""
    from server.daily_summary import send_daily_summary
    import asyncio

    mock_client = MagicMock()

    with patch("server.daily_summary._get_yesterday_pages", return_value=mock_pages):
        with patch("server.daily_summary._generate_summary_text", return_value="📅 昨日知识汇总\n\n内容..."):
            result = asyncio.run(send_daily_summary(mock_client, "user1"))
            assert result is True
            mock_client.send_message.assert_called_once()


def test_split_new_vs_updated():
    """Verify page splitting logic: new vs updated."""
    from server.daily_summary import _split_pages

    now = datetime.now()
    yesterday = now.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    yesterday_str = yesterday.strftime("%Y-%m-%d %H:%M:%S")

    pages = [
        {"id": 1, "title": "New Page", "created_at": yesterday_str, "updated_at": yesterday_str},
        {"id": 2, "title": "Updated Page", "created_at": "2026-01-01 10:00:00", "updated_at": yesterday_str},
    ]

    new, updated = _split_pages(pages, yesterday, today)
    assert len(new) == 1
    assert new[0]["title"] == "New Page"
    assert len(updated) == 1
    assert updated[0]["title"] == "Updated Page"
