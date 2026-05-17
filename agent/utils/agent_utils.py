import time
import logging
import re

from agent.models.value_objects import UrlContent

logger = logging.getLogger(__name__)

def build_url_context(urls: list[UrlContent], chars: int = -1) -> str:
    """
    将urls拼接成md格式
    :param urls:
    :param chars:
    :return:
    """
    parts = [f"爬取内容如下（摘要{chars if chars > 0 else '全部'}字符）"]
    for uc in urls:
        parts.append("")
        parts.append(f"### URL: {uc.url}")
        if uc.title:
            parts.append(f"> 标题：{uc.title}")
        content = uc.content
        if content:
            parts.append(f"全文：")
            parts.append(f"{content[:chars] if chars > 0 else content}")
    return "\n".join(parts)

def build_context_block(state: dict) -> str:
    """Build shared context block: user profile, stored knowledge,
    message history, episodic memories. Returns empty string if none available."""
    parts = []

    profile = state.get("user_profile", {})
    if profile and any(v for v in profile.values() if v):
        profile_summary = _summarize_profile(profile)
        parts.append("")
        parts.append("## 用户画像")
        parts.append(profile_summary)

    if state.get("stored_knowledge"):
        parts.append("")
        parts.append("## 相关知识")
        for k in state["stored_knowledge"]:
            if k.get("type") == "wiki_page":
                parts.append(f"- [{k['title']}]: {k['content'][:200]}")
            else:
                parts.append(f"- {k.get('knowledge_text', k.get('title', ''))}")

    if state.get("message_history"):
        parts.append("")
        parts.append("## 近期对话历史")
        for msg in state["message_history"]:
            if isinstance(msg, str):
                parts.append(msg)
            elif isinstance(msg, dict):
                role = "用户" if msg.get("role") == "user" else "助手"
                content = msg.get("content", "")
                parts.append(f"{role}: {content}")

    if state.get("episodic_memories"):
        if isinstance(state["episodic_memories"], list) and state["episodic_memories"]:
            parts.append("")
            parts.append("## 历史相关记忆")
            parts.extend(state["episodic_memories"] if all(isinstance(m, str) for m in state["episodic_memories"]) else [str(m) for m in state["episodic_memories"]])

    # ── URL 网页内容 ──
    url_contents = state.get("url_contents", [])
    if url_contents:
        parts.append("")
        parts.append("## 用户提供的网页内容")
        parts.append(build_url_context(url_contents))

        # 纯 URL 消息（无附加文字）→ 追加总结指令
        user_message = state.get("user_message", "")
        url_pattern = re.compile(r'https?://[^\s]+')
        if not url_pattern.sub('', user_message).strip():
            parts.append("")
            parts.append("用户只发送了网页链接，没有附加问题。请直接总结这篇文章的核心内容，用中文输出。")

    return "\n".join(parts)


def _summarize_profile(profile: dict) -> str:
    """Flatten profile dict into a readable summary string."""
    lines = []
    for section, data in profile.items():
        if section == "updated_at" or not data:
            continue
        if isinstance(data, dict):
            for k, v in data.items():
                if v:
                    lines.append(f"- {section}.{k}: {v}")
        elif isinstance(data, list):
            if data:
                lines.append(f"- {section}: {', '.join(str(x) for x in data)}")
    return "\n".join(lines) if lines else "（暂无用户画像信息）"


def get_language_instruction() -> str:
    """Return a prompt instruction for the configured output language.

    Returns empty string when English (default), so no extra tokens are used.
    Only applied to user-facing agents (analysts, portfolio manager).
    Internal debate agents stay in English for reasoning quality.
    """
    from server.config import OUTPUT_LANGUAGE as language
    if language.strip().lower() == "english":
        return ""
    return f" Write your entire response in {language}."



def with_retry(fn, max_retries=2, delay=1):
    """Call fn with retry. Retries once on failure (2 attempts total)."""
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            logger.warning("LLM call failed (attempt %d/%d): %s", attempt + 1, max_retries, e)
            time.sleep(delay)
    return None
