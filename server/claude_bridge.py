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

        cmd = [r"C:\Program Files\nodejs\claude.cmd", "--print", message, "--resume", "fund", "--dangerously-skip-permissions"]
        if self._has_conversation:
            cmd.append("--continue")

        try:
            r = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,
                encoding='utf-8',
                errors='replace'
            )
            self._has_conversation = True
            output = r.stdout.strip()

            if not output:
                return "⚠️ Claude Code 返回了空结果"

            parts = _split_long_message(output)
            return "\n\n---\n\n".join(parts) if len(parts) > 1 else parts[0]

        except subprocess.TimeoutExpired:
            return "⚠️ 命令执行超时（1800），请简化问题或重试"
        except FileNotFoundError:
            return "❌ claude 命令未找到，请确认已安装 Claude Code"
        except Exception as e:
            logger.exception("Claude Code bridge error")
            return f"❌ 编码会话出错: {e}"
