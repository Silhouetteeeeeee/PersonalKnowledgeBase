import subprocess

import pytest

from server.claude_bridge import (
    _strip_ansi,
    _split_long_message,
    _parse_prompt_from_output,
    ClaudeCodeBridge,
)


# ── _strip_ansi ──────────────────────────────────────────────

class TestStripAnsi:
    def test_removes_simple_csi(self):
        assert _strip_ansi("\x1b[31mred\x1b[0m") == "red"

    def test_removes_cursor_movements(self):
        assert _strip_ansi("abc\x1b[3Ddef") == "abcdef"

    def test_removes_erase_display(self):
        assert _strip_ansi("a\x1b[2Jb") == "ab"

    def test_handles_empty(self):
        assert _strip_ansi("") == ""

    def test_handles_no_ansi(self):
        assert _strip_ansi("hello world") == "hello world"

    def test_strips_control_chars(self):
        assert _strip_ansi("line1\x00line2") == "line1line2"

    def test_normalizes_crlf(self):
        assert _strip_ansi("a\r\nb\rc") == "a\nb\nc"


# ── _split_long_message ──────────────────────────────────────

class TestSplitLongMessage:
    def test_under_limit(self):
        assert _split_long_message("hello", 100) == ["hello"]

    def test_at_limit(self):
        text = "a" * 100
        assert _split_long_message(text, 100) == [text]

    def test_splits_by_paragraph(self):
        text = "a" * 60 + "\n" + "b" * 60
        result = _split_long_message(text, 80)
        assert len(result) == 2
        assert all(len(chunk) <= 80 for chunk in result)

    def test_long_line_without_break(self):
        text = "a" * 200
        result = _split_long_message(text, 100)
        assert len(result) >= 1


# ── _parse_prompt_from_output ────────────────────────────────

class TestParsePrompt:
    def test_detects_gt_prompt(self):
        assert _parse_prompt_from_output("hello\n> ") is not None

    def test_detects_arrow_prompt(self):
        assert _parse_prompt_from_output("hello\n❯ ") is not None

    def test_returns_none_when_no_prompt(self):
        assert _parse_prompt_from_output("still running...") is None

    def test_returns_none_on_empty(self):
        assert _parse_prompt_from_output("") is None


# ── ClaudeCodeBridge — unit tests with mocked subprocess ─────

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

    def test_handle_exit(self):
        bridge = ClaudeCodeBridge()
        bridge._session_active = True
        result = bridge.handle("exit")
        assert "已关闭" in result
        assert not bridge._session_active

    def test_ensure_session_creates_when_missing(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda cmd: f"/usr/bin/{cmd}")
        calls = []

        def fake_run(*args, **kwargs):
            cmd = args[0]
            calls.append(cmd)
            if cmd[0] == "tmux" and cmd[1] == "has-session":
                return FakeResult(returncode=1)
            if cmd[0] == "tmux" and cmd[1] == "new-session":
                return FakeResult()
            if cmd[0] == "tmux" and cmd[1] == "capture-pane":
                return FakeResult(stdout="> ")
            return FakeResult()

        monkeypatch.setattr(subprocess, "run", fake_run)

        bridge = ClaudeCodeBridge()
        bridge._ensure_session()

        new_session_calls = [c for c in calls if "new-session" in c]
        assert len(new_session_calls) == 1

    def test_ensure_session_reuses_existing(self, monkeypatch):
        calls = []

        def fake_run(*args, **kwargs):
            cmd = args[0]
            calls.append(cmd)
            if cmd[0] == "tmux" and cmd[1] == "has-session":
                return FakeResult(returncode=0)
            return FakeResult()

        monkeypatch.setattr(subprocess, "run", fake_run)

        bridge = ClaudeCodeBridge()
        bridge._session_active = True
        bridge._ensure_session()

        has_session_calls = [c for c in calls if "has-session" in c]
        assert len(has_session_calls) == 1
        new_session_calls = [c for c in calls if "new-session" in c]
        assert len(new_session_calls) == 0

    def test_ensure_session_fails_without_tmux(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda cmd: None if cmd == "tmux" else "/usr/bin/claude")

        bridge = ClaudeCodeBridge()
        with pytest.raises(FileNotFoundError, match="tmux"):
            bridge._ensure_session()

    def test_ensure_session_fails_without_claude(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda cmd: None if cmd == "claude" else "/usr/bin/tmux")

        bridge = ClaudeCodeBridge()
        with pytest.raises(FileNotFoundError, match="claude"):
            bridge._ensure_session()

    def test_send_keys_called(self, monkeypatch):
        calls = []

        def fake_run(*args, **kwargs):
            cmd = args[0]
            calls.append(cmd)
            return FakeResult()

        monkeypatch.setattr(subprocess, "run", fake_run)

        bridge = ClaudeCodeBridge()
        bridge._send("hello world")

        # Should call set-buffer, paste-buffer, then send-keys Enter
        assert any("set-buffer" in c for c in calls)
        assert any("hello world" in c for c in calls)
        assert any("Enter" in c for c in calls)

    def test_handle_send_and_capture(self, monkeypatch):
        """Verify handle() creates session, sends keys, captures output."""
        monkeypatch.setattr("shutil.which", lambda cmd: f"/usr/bin/{cmd}")
        calls = []

        def fake_run(*args, **kwargs):
            cmd = args[0]
            calls.append(cmd)
            if cmd[0] == "tmux" and cmd[1] == "has-session":
                return FakeResult(returncode=1)
            if cmd[0] == "tmux" and cmd[1] == "new-session":
                return FakeResult()
            if cmd[0] == "tmux" and cmd[1] == "capture-pane":
                return FakeResult(stdout="> ")
            if cmd[0] == "tmux" and cmd[1] == "send-keys":
                return FakeResult()
            return FakeResult()

        monkeypatch.setattr(subprocess, "run", fake_run)
        monkeypatch.setattr("server.claude_bridge.POLL_INTERVAL", 0.01)
        monkeypatch.setattr("server.claude_bridge.STABLE_THRESHOLD", 2)

        bridge = ClaudeCodeBridge()
        result = bridge.handle("hello")

        # Should have output (at minimum the prompt text)
        assert result
        # Should have called new-session
        assert any("new-session" in c for c in calls)
        # Should have called send-keys with input
        assert any("hello" in c for c in calls)
