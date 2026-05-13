import subprocess

import pytest

from server.claude_bridge import (
    _split_long_message,
    _check_claude,
    ClaudeCodeBridge,
)


# ── _split_long_message ──────────────────────────────────────

class TestSplitLongMessage:
    def test_under_limit(self):
        assert _split_long_message("hello", 100) == ["hello"]

    def test_at_limit(self):
        text = "a" * 100
        assert _split_long_message(text, 100) == [text]

    def test_splits_by_line(self):
        text = "a" * 60 + "\n" + "b" * 60
        result = _split_long_message(text, 80)
        assert len(result) == 2
        assert all(len(chunk) <= 80 for chunk in result)

    def test_long_line_without_break(self):
        text = "a" * 200
        result = _split_long_message(text, 100)
        assert len(result) >= 1

    def test_empty_text(self):
        assert _split_long_message("", 100) == [""]

    def test_single_newline(self):
        # Single newline passes through since it's under max_len
        assert _split_long_message("\n", 100) == ["\n"]


# ── _check_claude ────────────────────────────────────────────

class TestCheckClaude:
    def test_returns_true_when_found(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda cmd: "/usr/bin/claude")
        assert _check_claude() is True

    def test_returns_false_when_missing(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda cmd: None)
        assert _check_claude() is False


# ── ClaudeCodeBridge ─────────────────────────────────────────

class FakeResult:
    """Simulate subprocess.run result."""
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class TestClaudeCodeBridge:
    def test_handle_empty_returns_usage(self):
        bridge = ClaudeCodeBridge()
        result = bridge.handle("")
        assert "用法" in result

    def test_handle_exit_resets_conversation(self):
        bridge = ClaudeCodeBridge()
        bridge._has_conversation = True
        result = bridge.handle("exit")
        assert "已关闭" in result
        assert bridge._has_conversation is False

    def test_first_call_no_continue(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda cmd: "/usr/bin/claude")
        calls = []

        def fake_run(*args, **kwargs):
            calls.append(list(args[0]))
            return FakeResult(stdout="Hello from Claude")

        monkeypatch.setattr(subprocess, "run", fake_run)

        bridge = ClaudeCodeBridge()
        result = bridge.handle("hello")

        assert "Hello" in result
        assert "--continue" not in calls[0]
        assert bridge._has_conversation is True

    def test_subsequent_call_with_continue(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda cmd: "/usr/bin/claude")
        calls = []

        def fake_run(*args, **kwargs):
            calls.append(list(args[0]))
            return FakeResult(stdout="Again from Claude")

        monkeypatch.setattr(subprocess, "run", fake_run)

        bridge = ClaudeCodeBridge()
        bridge._has_conversation = True
        result = bridge.handle("follow up")

        assert "Again" in result
        assert "--continue" in calls[0]

    def test_timeout_returns_error(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda cmd: "/usr/bin/claude")

        def fake_run(*args, **kwargs):
            raise subprocess.TimeoutExpired(cmd="claude", timeout=120)

        monkeypatch.setattr(subprocess, "run", fake_run)

        bridge = ClaudeCodeBridge()
        result = bridge.handle("hello")
        assert "超时" in result
        assert bridge._has_conversation is False

    def test_claude_not_found_returns_error(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda cmd: None)

        bridge = ClaudeCodeBridge()
        result = bridge.handle("hello")
        assert "未检测到" in result
        assert bridge._has_conversation is False

    def test_file_not_found_error(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda cmd: "/usr/bin/claude")

        def fake_run(*args, **kwargs):
            raise FileNotFoundError("claude not found")

        monkeypatch.setattr(subprocess, "run", fake_run)

        bridge = ClaudeCodeBridge()
        result = bridge.handle("hello")
        assert "未找到" in result
        assert bridge._has_conversation is False

    def test_split_long_response(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda cmd: "/usr/bin/claude")
        # ~6000 chars to exceed MAX_MSG_LENGTH (4000)
        long_output = ("hello world this is a test\n" * 200)

        def fake_run(*args, **kwargs):
            return FakeResult(stdout=long_output)

        monkeypatch.setattr(subprocess, "run", fake_run)

        bridge = ClaudeCodeBridge()
        result = bridge.handle("hello")
        assert "---" in result
