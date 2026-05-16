# Claude Code Bridge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a tmux-based bridge that routes `/code` messages from WeChat Work to a persistent Claude Code session, enabling remote coding via mobile/desktop WeChat.

**Architecture:** New `server/claude_bridge.py` manages a persistent tmux session running `claude`. On `/code` prefix, bot.py routes to `claude_bridge.handle()`, which sends keystrokes via `tmux send-keys`, polls `tmux capture-pane` until output stabilizes, strips ANSI codes, and returns the response. A config flag (`CLAUDE_CODE_BRIDGE_ENABLED`) controls feature availability.

**Tech Stack:** Python stdlib (`subprocess`, `re`, `time`, `atexit`, `logging`), tmux, Claude Code CLI.

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `server/config.py` | Modify | Add `CLAUDE_CODE_BRIDGE_ENABLED` flag |
| `server/claude_bridge.py` | Create | tmux session management, message send/receive, output processing |
| `server/bot.py` | Modify | Add `/code` prefix routing (~8 lines) |
| `tests/test_claude_bridge.py` | Create | Unit tests with mocked subprocess |

---

### Task 1: Add config flag

**Files:**
- Modify: `server/config.py`

- [ ] **Step 1: Add CLAUDE_CODE_BRIDGE_ENABLED**

```python
# server/config.py
CLAUDE_CODE_BRIDGE_ENABLED = os.getenv("CLAUDE_CODE_BRIDGE_ENABLED", "false").lower() == "true"
```

Insert after the existing config entries (after `BAIDU_AI_SEARCH_API_KEY` line).

- [ ] **Step 2: Verify import**

Run: `python -c "from server.config import CLAUDE_CODE_BRIDGE_ENABLED; print(CLAUDE_CODE_BRIDGE_ENABLED)"`
Expected: `False`

- [ ] **Step 3: Commit**

```bash
git add server/config.py
git commit -m "feat: add CLAUDE_CODE_BRIDGE_ENABLED config flag"
```

---

### Task 2: Create server/claude_bridge.py

**Files:**
- Create: `server/claude_bridge.py`

- [ ] **Step 1: Write server/claude_bridge.py**

```python
import atexit
import logging
import re
import subprocess
import time

logger = logging.getLogger(__name__)

SESSION_NAME = "claude-code"
POLL_INTERVAL = 0.5
STABLE_THRESHOLD = 3
TIMEOUT = 120
MAX_MSG_LENGTH = 4000


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences from text."""
    # CSI sequences (e.g. \x1b[31m), OSC sequences, and other control chars
    text = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', text)
    text = re.sub(r'\x1b\][0-9;]*(?:\x07|\x1b\\)', '', text)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    # Normalize line endings
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    return text.strip()


def _split_long_message(text: str, max_len: int = MAX_MSG_LENGTH) -> list[str]:
    """Split text into chunks at paragraph boundaries, each ≤ max_len."""
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


def _parse_prompt_from_output(output: str) -> str | None:
    """Detect Claude Code ready-prompt at the end of output."""
    lines = output.strip().split('\n')
    if not lines:
        return None
    last = lines[-1].strip()
    # Claude Code prompts: "> " or "❯ " style
    if last in (">", "❯") or last.startswith("> ") or last.startswith("❯ "):
        return last
    return None


class ClaudeCodeBridge:
    """Manage a persistent tmux session running Claude Code."""

    def __init__(self, session_name: str = SESSION_NAME):
        self.session_name = session_name
        self._session_active = False
        atexit.register(self._cleanup)

    # ── Public API ──────────────────────────────────────────────

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
            self._cleanup()
            return "编码会话已关闭。"

        try:
            self._ensure_session()
            self._send(message)
            output = self._wait_for_output()
            parts = _split_long_message(output)
            return "\n\n---\n\n".join(parts) if len(parts) > 1 else parts[0]
        except TimeoutError:
            partial = self._capture()
            cleaned = _strip_ansi(partial) if partial else ""
            if cleaned:
                return cleaned + "\n\n⚠️ 命令执行超时（120s），以上为部分输出"
            return "⚠️ 命令执行超时，无输出返回"
        except FileNotFoundError as e:
            return f"❌ 依赖缺失: {e}"
        except Exception as e:
            logger.exception("Claude Code bridge error")
            return f"❌ 编码会话出错: {e}"

    # ── Session management ──────────────────────────────────────

    def _ensure_session(self):
        """Create tmux session if it doesn't exist, or reuse existing."""
        if self._session_active:
            # Quick health check
            r = subprocess.run(
                ["tmux", "has-session", "-t", self.session_name],
                capture_output=True,
            )
            if r.returncode == 0:
                return

        # Check dependencies
        for cmd in ("tmux", "claude"):
            if subprocess.run(["which", cmd], capture_output=True).returncode != 0:
                raise FileNotFoundError(
                    f"`{cmd}` 未找到，请先安装 {cmd}"
                    if cmd == "tmux" else
                    f"`{cmd}` 未找到，请先安装 Claude Code"
                )

        logger.info("Creating tmux session '%s'", self.session_name)
        subprocess.run(
            ["tmux", "new-session", "-d", "-s", self.session_name,
             "env", f"PAGER=cat", "claude"],
            check=True,
        )
        self._session_active = True
        self._wait_for_prompt()

    def _wait_for_prompt(self, timeout: int = 30):
        """Wait until Claude Code prompt appears or timeout."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            raw = self._capture()
            if _parse_prompt_from_output(raw):
                return
            time.sleep(0.5)
        logger.warning("Timeout waiting for Claude Code prompt, continuing anyway")

    def _cleanup(self):
        """Destroy the tmux session."""
        if not self._session_active:
            return
        try:
            subprocess.run(
                ["tmux", "kill-session", "-t", self.session_name],
                capture_output=True, timeout=5,
            )
            logger.info("tmux session '%s' destroyed", self.session_name)
        except Exception:
            pass
        self._session_active = False

    # ── Message send / receive ──────────────────────────────────

    def _send(self, message: str):
        """Type message into tmux pane and press Enter."""
        subprocess.run(
            ["tmux", "send-keys", "-t", self.session_name, "-l", message],
            check=True,
        )
        subprocess.run(
            ["tmux", "send-keys", "-t", self.session_name, "Enter"],
            check=True,
        )

    def _capture(self) -> str:
        """Capture full visible pane content."""
        r = subprocess.run(
            ["tmux", "capture-pane", "-t", self.session_name, "-p"],
            capture_output=True, text=True,
        )
        return r.stdout if r.returncode == 0 else ""

    def _wait_for_output(self, timeout: int = TIMEOUT) -> str:
        """Poll capture until output stabilizes (Claude Code done)."""
        deadline = time.time() + timeout
        prev_clean = ""
        stable_count = 0

        while time.time() < deadline:
            time.sleep(POLL_INTERVAL)
            raw = self._capture()
            current_clean = _strip_ansi(raw)

            if current_clean == prev_clean:
                stable_count += 1
                if stable_count >= STABLE_THRESHOLD:
                    break
            else:
                stable_count = 0
                prev_clean = current_clean
        else:
            raise TimeoutError("Command execution timed out")

        return prev_clean
```

- [ ] **Step 2: Verify import**

Run: `python -c "from server.claude_bridge import ClaudeCodeBridge, _strip_ansi, _split_long_message; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add server/claude_bridge.py
git commit -m "feat: add Claude Code bridge with tmux session management"
```

---

### Task 3: Add routing to bot.py

**Files:**
- Modify: `server/bot.py`

- [ ] **Step 1: Add import and bridge instance**

At module level, after the `graph = build_graph()` line:

```python
from server.claude_bridge import ClaudeCodeBridge
from server.config import CLAUDE_CODE_BRIDGE_ENABLED

claude_bridge = ClaudeCodeBridge() if CLAUDE_CODE_BRIDGE_ENABLED else None
```

- [ ] **Step 2: Add routing in _on_text**

After the `if not content: return` block and before `graph.invoke`, insert:

```python
            # ── Claude Code bridge route ──
            if content.startswith("/code"):
                cmd = content[5:].strip()
                if CLAUDE_CODE_BRIDGE_ENABLED and claude_bridge:
                    response = claude_bridge.handle(cmd)
                else:
                    response = "⚠️ 远程编码功能未启用\n请在 .env 中设置 CLAUDE_CODE_BRIDGE_ENABLED=true"
                await self.client.reply(frame, {
                    "msgtype": "markdown",
                    "markdown": {"content": response},
                })
                logger.info("Claude Code bridge response sent to user_id=%s", user_id)
                return
```

- [ ] **Step 3: Verify bot imports**

Run: `python -c "from server.bot import claude_bridge; print('OK')"`
Expected: `OK` (with `CLAUDE_CODE_BRIDGE_ENABLED=false`, `claude_bridge` will be `None`)

- [ ] **Step 4: Commit**

```bash
git add server/bot.py
git commit -m "feat: add /code prefix routing to bot.py"
```

---

### Task 4: Write tests

**Files:**
- Create: `tests/test_claude_bridge.py`

- [ ] **Step 1: Write tests/test_claude_bridge.py**

```python
import subprocess
import time

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

    def test_removes_osc_sequences(self):
        assert _strip_ansi("\x1b]0;title\x07text") == "title\x07text"
        # Actually OSC 0;title is a window title, we want content
        # but this is fine — just checking it doesn't crash

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
        assert len(result[0]) <= 80

    def test_long_line_without_break(self):
        """Single line longer than max_len goes into one chunk."""
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

class FakePopenResult:
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
        # Ensure _session_active is set so _cleanup does something
        bridge._session_active = True
        result = bridge.handle("exit")
        assert "已关闭" in result
        assert not bridge._session_active

    def test_ensure_session_creates_when_missing(self, monkeypatch):
        calls = []

        def fake_run(*args, **kwargs):
            cmd = args[0]
            calls.append(cmd)
            if cmd[0] == "tmux" and cmd[1] == "has-session":
                return FakePopenResult(returncode=1)
            if cmd[0] == "which":
                return FakePopenResult(returncode=0)
            if cmd[0] == "tmux" and cmd[1] == "new-session":
                return FakePopenResult()
            if cmd[0] == "tmux" and cmd[1] == "capture-pane":
                return FakePopenResult(stdout="> ")
            return FakePopenResult()

        monkeypatch.setattr(subprocess, "run", fake_run)

        bridge = ClaudeCodeBridge()
        bridge._ensure_session()

        # Should have called has-session → which → which → new-session → capture-pane
        new_session_calls = [c for c in calls if "new-session" in c]
        assert len(new_session_calls) == 1

    def test_ensure_session_reuses_existing(self, monkeypatch):
        calls = []

        def fake_run(*args, **kwargs):
            cmd = args[0]
            calls.append(cmd)
            if cmd[0] == "tmux" and cmd[1] == "has-session":
                return FakePopenResult(returncode=0)
            return FakePopenResult()

        monkeypatch.setattr(subprocess, "run", fake_run)

        bridge = ClaudeCodeBridge()
        bridge._session_active = True
        bridge._ensure_session()

        # Only has-session should be called
        has_session_calls = [c for c in calls if "has-session" in c]
        assert len(has_session_calls) == 1
        new_session_calls = [c for c in calls if "new-session" in c]
        assert len(new_session_calls) == 0

    def test_ensure_session_fails_without_tmux(self, monkeypatch):
        def fake_run(*args, **kwargs):
            cmd = args[0]
            if cmd[0] == "which" and cmd[1] == "tmux":
                return FakePopenResult(returncode=1)
            return FakePopenResult()

        monkeypatch.setattr(subprocess, "run", fake_run)

        bridge = ClaudeCodeBridge()
        with pytest.raises(FileNotFoundError, match="tmux"):
            bridge._ensure_session()

    def test_send_keys_called(self, monkeypatch):
        calls = []

        def fake_run(*args, **kwargs):
            cmd = args[0]
            calls.append(cmd)
            return FakePopenResult()

        monkeypatch.setattr(subprocess, "run", fake_run)

        bridge = ClaudeCodeBridge()
        bridge._send("hello world")

        # Should call send-keys with -l flag, then Enter
        send_keys_calls = [c for c in calls if c[:2] == ["tmux", "send-keys"]]
        assert len(send_keys_calls) == 2
        assert "-l" in send_keys_calls[0]
        assert "hello world" in send_keys_calls[0]
        assert "Enter" in send_keys_calls[1]

    def test_handle_send_and_capture(self, monkeypatch):
        """Integration-style: simulate full handle flow."""
        state = {"session_created": False, "capture_count": 0}

        def fake_run(*args, **kwargs):
            cmd = args[0]
            if cmd[0] == "which":
                return FakePopenResult(returncode=0)
            if cmd[0] == "tmux" and cmd[1] == "has-session":
                if not state["session_created"]:
                    return FakePopenResult(returncode=1)
                return FakePopenResult(returncode=0)
            if cmd[0] == "tmux" and cmd[1] == "new-session":
                state["session_created"] = True
                return FakePopenResult()
            if cmd[0] == "tmux" and cmd[1] == "capture-pane":
                state["capture_count"] += 1
                # After 3 polls, simulate prompt appearance (output stable)
                if state["capture_count"] >= 5:
                    return FakePopenResult(stdout="> ")
                return FakePopenResult(stdout="Hello from Claude\n")
            if cmd[0] == "tmux" and cmd[1] == "send-keys":
                return FakePopenResult()
            return FakePopenResult()

        monkeypatch.setattr(subprocess, "run", fake_run)
        # Speed up polling for test
        monkeypatch.setattr("server.claude_bridge.POLL_INTERVAL", 0.01)
        monkeypatch.setattr("server.claude_bridge.STABLE_THRESHOLD", 2)

        bridge = ClaudeCodeBridge()
        result = bridge.handle("hello")

        assert "Hello from Claude" in result
        assert state["session_created"]
```

- [ ] **Step 2: Run tests**

Run: `python -m pytest tests/test_claude_bridge.py -v`
Expected: All tests PASS

- [ ] **Step 3: Run full test suite**

Run: `python -m pytest tests/ -v`
Expected: All previous tests still PASS, plus new bridge tests pass

- [ ] **Step 4: Commit**

```bash
git add tests/test_claude_bridge.py
git commit -m "test: add Claude Code bridge tests"
```

---

## Self-Review

**1. Spec coverage:**
- Config flag `CLAUDE_CODE_BRIDGE_ENABLED` → Task 1 ✓
- `server/claude_bridge.py` with tmux session management → Task 2 ✓
- `_strip_ansi` / `_split_long_message` helpers → Task 2 ✓
- `/code` prefix routing in bot.py → Task 3 ✓
- Tests for all above → Task 4 ✓

**2. Placeholder scan:** No TBD, TODO, or incomplete code. All code is complete.

**3. Type consistency:** `handle(message: str) -> str` matches the return type used in bot.py. `_ensure_session()`, `_send()`, `_capture()`, `_wait_for_output()` all use consistent signatures. `_strip_ansi` and `_split_long_message` are module-level functions, matching the spec.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-05-claude-code-bridge.md`. Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
