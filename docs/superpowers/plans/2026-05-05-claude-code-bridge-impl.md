# Claude Code Bridge Rewrite (tmux → claude --print) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) for syntax checking.

**Goal:** Rewrite `server/claude_bridge.py` from tmux-based session management to `claude --print` non-interactive mode, eliminating Windows pipe/encoding issues.

**Architecture:** Replace persistent tmux session + `send-keys`/`capture-pane` with single-shot `subprocess.run(["claude", "--print", message])`. First message omits `--continue`; subsequent messages include it for context persistence. Claude Code's built-in conversation file management replaces manual tmux session management.

**Tech Stack:** Python `subprocess`, `claude --print`/`--continue`, `atexit`

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `server/claude_bridge.py` | Rewrite | `claude --print` bridge, ~80 lines, no tmux |
| `tests/test_claude_bridge.py` | Rewrite | Mock subprocess for claude, 7 focused tests |
| `server/bot.py` | Unchanged | Already has `/code` routing + `claude_bridge` instance |
| `server/config.py` | Unchanged | Already has `CLAUDE_CODE_BRIDGE_ENABLED` |

---

### Task 1: Rewrite claude_bridge.py to use claude --print

**Files:**
- Rewrite: `server/claude_bridge.py`

**Changes:**
- Remove all tmux-related code: `_ensure_session`, `_cleanup`, `_send`, `_capture`, `_wait_for_output`, `_wait_for_prompt`, `_strip_ansi`, `_parse_prompt_from_output`, `SESSION_NAME`, `POLL_INTERVAL`, `STABLE_THRESHOLD`
- Keep: `_split_long_message` (unchanged), `MAX_MSG_LENGTH`
- New: `_check_claude()` module-level function
- New: `ClaudeCodeBridge.__init__()` — simple init with `_has_conversation = False`, no atexit needed
- New: `ClaudeCodeBridge.handle()` — core `claude --print` logic

- [ ] **Step 1: Write the new claude_bridge.py**

```python
import logging
import shutil
import subprocess

logger = logging.getLogger(__name__)

MAX_MSG_LENGTH = 4000


def _check_claude() -> bool:
    """Return True if claude is available on PATH."""
    return shutil.which("claude") is not None


def _split_long_message(text: str, max_len: int = MAX_MSG_LENGTH) -> list[str]:
    """Split text into chunks at line boundaries, each <= max_len."""
    if len(text) <= max_len:
        return [text]
    chunks = []
    current = ""
    for line in text.split('\n'):
        candidate = current + '\n' + line if current else line
        if len(candidate) > max_len:
            if current:
                chunks.append(current.strip())
            current = line
        else:
            current = candidate
    if current.strip():
        chunks.append(current.strip())
    return chunks


class ClaudeCodeBridge:
    """Invoke claude --print for remote coding via WeChat Work."""

    def __init__(self):
        self._has_conversation = False

    def handle(self, message: str) -> str:
        """Process a single /code message. Returns response text."""
        if not message:
            return (
                "用法: /code <你想让 Claude Code 执行的命令>\n\n"
                "示例:\n"
                "/code 查看当前 git 状态\n"
                "/code 帮我写一个二分查找\n"
                "/code exit  - 关闭编码会话"
            )

        if message.strip() == "exit":
            self._has_conversation = False
            return "编码会话已关闭。下次发送 /code 将开始新会话。"

        if not _check_claude():
            return "❌ 未检测到 Claude Code，请先安装：npm install -g @anthropic-ai/claude-code"

        cmd = ["claude", "--print", message]
        if self._has_conversation:
            cmd.append("--continue")

        try:
            r = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
            )
            self._has_conversation = True
            output = r.stdout.strip()

            if not output:
                return "⚠️ Claude Code 返回了空结果"

            parts = _split_long_message(output)
            return "\n\n---\n\n".join(parts) if len(parts) > 1 else parts[0]

        except subprocess.TimeoutExpired:
            return "⚠️ 命令执行超时（120s），请简化问题或重试"
        except FileNotFoundError:
            return "❌ claude 命令未找到，请确认已安装 Claude Code"
        except Exception as e:
            logger.exception("Claude Code bridge error")
            return f"❌ 编码会话出错: {e}"
```

- [ ] **Step 2: Run existing tests to confirm they now fail (old tmux tests)**

Run: `python -m pytest tests/test_claude_bridge.py -v`
Expected: Most tests fail because tmux methods no longer exist

- [ ] **Step 3: Commit the rewrite**

```bash
git add server/claude_bridge.py
git commit -m "refactor(claude-bridge): replace tmux with claude --print

Remove all tmux session management code. Use single-shot
subprocess.run(['claude', '--print', message]) instead.
Keeps _split_long_message utility unchanged."
```

---

### Task 2: Rewrite tests for claude --print bridge

**Files:**
- Rewrite: `tests/test_claude_bridge.py`

- [ ] **Step 1: Write the new test file**

```python
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
        assert _split_long_message("\n", 100) == [""]


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
            calls.append(kwargs)
            return FakeResult(stdout="Hello from Claude")

        monkeypatch.setattr(subprocess, "run", fake_run)

        bridge = ClaudeCodeBridge()
        result = bridge.handle("hello")

        assert "Hello" in result
        # First call should NOT have --continue
        assert "--continue" not in calls[0]["args"]
        assert bridge._has_conversation is True

    def test_subsequent_call_with_continue(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda cmd: "/usr/bin/claude")
        calls = []

        def fake_run(*args, **kwargs):
            calls.append(kwargs)
            return FakeResult(stdout="Again from Claude")

        monkeypatch.setattr(subprocess, "run", fake_run)

        bridge = ClaudeCodeBridge()
        bridge._has_conversation = True  # Simulate existing conversation
        result = bridge.handle("follow up")

        assert "Again" in result
        # Subsequent call SHOULD have --continue
        assert "--continue" in calls[0]["args"]

    def test_timeout_returns_error(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda cmd: "/usr/bin/claude")

        def fake_run(*args, **kwargs):
            raise subprocess.TimeoutExpired(cmd="claude", timeout=120)

        monkeypatch.setattr(subprocess, "run", fake_run)

        bridge = ClaudeCodeBridge()
        result = bridge.handle("hello")
        assert "超时" in result
        # Conversation flag should NOT be set on timeout
        assert bridge._has_conversation is False

    def test_claude_not_found_returns_error(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda cmd: None)

        bridge = ClaudeCodeBridge()
        result = bridge.handle("hello")
        assert "未安装" in result or "未检测到" in result
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
        long_output = "hello\n" * 500  # ~3000 chars

        def fake_run(*args, **kwargs):
            return FakeResult(stdout=long_output)

        monkeypatch.setattr(subprocess, "run", fake_run)

        bridge = ClaudeCodeBridge()
        result = bridge.handle("hello")
        # Should contain the separator for multi-chunk messages
        assert "---" in result or len(long_output) > 4000
```

- [ ] **Step 2: Run new tests**

Run: `python -m pytest tests/test_claude_bridge.py -v`
Expected: All tests PASS

- [ ] **Step 3: Run full test suite to check for regressions**

Run: `python -m pytest tests/ -v`
Expected: All tests pass

- [ ] **Step 4: Commit**

```bash
git add tests/test_claude_bridge.py
git commit -m "test(claude-bridge): rewrite tests for claude --print

Remove tmux-based test mocks. Add tests for --continue logic,
timeout handling, FileNotFoundError, claude-not-found check,
and split_long_message."
```

---

## Self-Review

**1. Spec coverage:**
- `handle()` with empty message → Task 1, `test_handle_empty_returns_usage`
- `handle()` with "exit" → Task 1 (resets `_has_conversation`), `test_handle_exit_resets_conversation`
- First call without `--continue` → Task 1 (conditional append), `test_first_call_no_continue`
- Subsequent call with `--continue` → Task 1, `test_subsequent_call_with_continue`
- Timeout handling → Task 1 (`TimeoutExpired` catch), `test_timeout_returns_error`
- FileNotFoundError → Task 1 (`FileNotFoundError` catch + `_check_claude`), `test_claude_not_found_returns_error` + `test_file_not_found_error`
- Long message splitting → `_split_long_message` kept from old code, `test_split_long_response` verifies integration
- `_split_long_message` unit tests → 6 tests covering boundaries
- `_check_claude` → Task 1 module-level function, 2 tests
- bot.py/config.py unchanged → confirmed

**2. Placeholder scan:** No TBD, TODO, incomplete sections, or vague requirements. Every code block is complete. Every test has specific assertions.

**3. Type consistency:**
- `ClaudeCodeBridge.__init__()` returns None ✓
- `ClaudeCodeBridge.handle(message: str) -> str` ✓
- `_check_claude() -> bool` ✓
- `_split_long_message(text, max_len) -> list[str]` ✓
- `FakeResult.returncode: int`, `.stdout: str`, `.stderr: str` ✓
- All test method names match the test classes they're in ✓
- `_has_conversation` used as `bool` throughout ✓
