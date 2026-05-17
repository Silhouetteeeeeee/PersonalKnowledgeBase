"""Tests for wiki knowledge base."""
import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from storage.wiki_storage import (
    title_to_filename,
    parse_frontmatter,
    build_frontmatter,
    extract_wikilinks,
    compute_checksum,
)
from storage.wiki_index import get_index_for_prompt


@pytest.fixture(autouse=True)
def temp_db(monkeypatch, tmp_path):
    """Use a temporary database for all wiki tests."""
    db_dir = tmp_path / "data"
    db_dir.mkdir()
    monkeypatch.setattr("storage.database.DB_DIR", str(db_dir))
    monkeypatch.setattr("storage.database.DB_PATH", str(db_dir / "knowledge.db"))
    from storage.database import init_db
    init_db()


class TestWikiStorage:
    def test_title_to_filename(self):
        assert title_to_filename("Django ORM") == "django-orm.md"
        assert title_to_filename("Python 基础").endswith(".md")
        assert title_to_filename("HTTP/2 协议").endswith(".md")
        result = title_to_filename("C++ 模板元编程")
        assert result.endswith(".md")
        assert " " not in result

    def test_parse_frontmatter(self):
        content = """---
title: Test
tags: [a, b, c]
sources: [conv_001]
---

Body text here
"""
        meta, body = parse_frontmatter(content)
        assert meta["title"] == "Test"
        assert meta["tags"] == ["a", "b", "c"]
        assert meta["sources"] == ["conv_001"]
        assert body == "Body text here"

    def test_parse_frontmatter_no_frontmatter(self):
        content = "Just some text"
        meta, body = parse_frontmatter(content)
        assert meta == {}
        assert body == "Just some text"

    def test_build_frontmatter(self):
        result = build_frontmatter("Test", ["a"], ["conv_001"], "2025-01-01", "2025-01-02")
        assert "title: Test" in result
        assert "tags: [a]" in result
        assert "created: 2025-01-01" in result

    def test_extract_wikilinks(self):
        content = "See [[Django]] and [[Python]] and [[Django#ORM]]"
        links = extract_wikilinks(content)
        assert "Django" in links
        assert "Python" in links
        assert len(links) == 2  # Django should appear once

    def test_extract_wikilinks_empty(self):
        assert extract_wikilinks("No links here") == []

    def test_compute_checksum(self):
        c1 = compute_checksum("hello")
        c2 = compute_checksum("hello")
        c3 = compute_checksum("world")
        assert c1 == c2
        assert c1 != c3
        assert len(c1) == 64  # SHA256 hex


class TestWikiIndex:
    def test_get_index_for_prompt(self):
        result = get_index_for_prompt()
        assert isinstance(result, str)


class TestWikiDB:
    def test_upsert_and_search_page(self):
        from storage.models import upsert_page, find_similar_pages, get_page_by_title

        pid = upsert_page(
            title="Test Wiki Page",
            file_path="wiki/pages/test-wiki-page.md",
            tags=["test", "wiki"],
            sources=["conv_test_001"],
            checksum="abc123",
            content="This is a test wiki page about testing wiki pages.",
        )
        assert pid > 0

        found = get_page_by_title("Test Wiki Page")
        assert found is not None
        assert found["title"] == "Test Wiki Page"

        results = find_similar_pages("test wiki", threshold=1.0, limit=5)
        titles = [r.title for r in results]
        assert "Test Wiki Page" in titles

    def test_page_relations(self):
        from storage.models import upsert_page, update_page_relations, get_related_pages

        pid1 = upsert_page(
            "Page A", "wiki/pages/page-a.md", ["a"], ["conv_001"], "abc",
            "Content about [[Page B]]"
        )
        upsert_page(
            "Page B", "wiki/pages/page-b.md", ["b"], ["conv_001"], "def", "Content"
        )

        update_page_relations(pid1, ["Page B"])

        related = get_related_pages(pid1)
        titles = [r.title for r in related]
        assert "Page B" in titles
