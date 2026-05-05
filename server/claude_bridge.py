import atexit
import logging
import re
import shutil
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
    text = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', text)
    text = re.sub(r'\x1b\][0-9;]*(?:\x07|\x1b\\)', '', text)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    return text.strip()


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


def _parse_prompt_from_output(output: str) -> str | None:
    """Detect Claude Code ready-prompt at the end of output."""
    lines = output.strip().split('\n')
    if not lines:
        return None
    last = lines[-1].strip()
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
            r = subprocess.run(
                ["tmux", "has-session", "-t", self.session_name],
                capture_output=True,
            )
            if r.returncode == 0:
                return

        for cmd in ("tmux", "claude"):
            if shutil.which(cmd) is None:
                raise FileNotFoundError(
                    f"`{cmd}` 未找到，请先安装 {cmd}"
                    if cmd == "tmux" else
                    f"`{cmd}` 未找到，请先安装 Claude Code"
                )

        logger.info("Creating tmux session '%s'", self.session_name)
        subprocess.run(
            ["tmux", "new-session", "-d", "-s", self.session_name,
             "env", "PAGER=cat", "claude"],
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
        """Send message via tmux send-keys (handles UTF-8 without -l flag)."""
        subprocess.run(
            ["tmux", "send-keys", "-t", self.session_name, message],
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
