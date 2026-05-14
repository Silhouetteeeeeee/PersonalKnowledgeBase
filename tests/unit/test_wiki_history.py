"""Unit tests for wiki revision history CLI."""
import pytest
from unittest.mock import patch


def test_format_log():
    """Verify log formatting is correct."""
    from wiki_history import format_log
    versions = [
        {"version": 3, "created_at": "2026-05-14 15:30:00",
         "source_question": "TCP四次挥手", "change_summary": ""},
        {"version": 2, "created_at": "2026-05-13 20:00:00",
         "source_question": "TCP状态转换", "change_summary": ""},
        {"version": 1, "created_at": "2026-05-13 10:00:00",
         "source_question": "TCP三次握手的过程", "change_summary": ""},
    ]
    output = format_log("TCP三次握手", versions)
    assert "v3" in output
    assert "v2" in output
    assert "v1" in output
    assert "TCP三次握手" in output


def test_format_show():
    """Verify show returns full content for a specific version."""
    from wiki_history import format_show
    version = {
        "version": 2, "title": "TCP三次握手",
        "content": "---\ntitle: TCP三次握手\n---\n\nContent body",
        "created_at": "2026-05-14 15:30:00",
        "source_question": "TCP四次挥手",
    }
    output = format_show(version)
    assert "v2" in output
    assert "TCP三次握手" in output
    assert "Content body" in output


def test_format_diff():
    """Verify diff produces unified diff lines."""
    from wiki_history import format_diff
    old_content = "line1\nline2\nline3\n"
    new_content = "line1\nline2_modified\nline3\nline4\n"
    output = format_diff("Test Page", 1, 2, old_content, new_content)
    assert "--- Test Page v1" in output
    assert "+++ Test Page v2" in output
    assert "-line2" in output
    assert "+line2_modified" in output
    assert "+line4" in output


def test_diff_identical():
    """Diff of identical content shows no changes."""
    from wiki_history import format_diff
    content = "line1\nline2\nline3\n"
    output = format_diff("Test", 1, 2, content, content)
    assert "No differences" in output


def test_handle_log_not_found():
    """Non-existent page returns error."""
    from wiki_history import handle_log
    with patch("wiki_history.get_page_by_title", return_value=None):
        result = handle_log("NonExistent")
        assert "not found" in result.lower()


def test_handle_log_no_versions():
    """Existing page with no versions returns appropriate message."""
    from wiki_history import handle_log
    with patch("wiki_history.get_page_by_title", return_value={"id": 1}):
        with patch("wiki_history.get_page_versions", return_value=[]):
            result = handle_log("EmptyPage")
            assert "No version history" in result
