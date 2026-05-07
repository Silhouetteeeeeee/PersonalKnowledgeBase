"""Rewrite user query with conversation context for standalone retrieval."""
import logging

from memory.message_history import MessageHistory
from agent.utils.llm import LLM

logger = logging.getLogger(__name__)

REWRITE_PROMPT = (
    "你是一个查询改写助手。根据对话历史，将用户的最新问题改写为一个"
    "不需要上下文就能理解的独立问题。\n\n"
    "要求：\n"
    "- 补全指代（如"它"→"Python dict"、"区别呢"→"A和B的区别"）\n"
    "- 补全省略的部分\n"
    "- 不要添加不存在的信息\n"
    "- 如果问题已经是独立的，保持原文\n"
    "- 只输出改写后的文本，不要任何解释\n\n"
    "对话历史：\n{history}\n\n"
    "用户最新消息：{message}"
)


def rewrite_query(state: dict) -> dict:
    """Rewrite the user message as a standalone query using conversation history.

    Falls back to the original user_message when:
    - Less than 2 prior messages exist (new or short conversation)
    - LLM call fails or returns empty
    """
    user_message = state["user_message"]
    session_id = int(state["session_id"])

    history = MessageHistory.get_recent(session_id)
    if len(history) < 2:
        logger.debug("Skipping rewrite: only %d history messages", len(history))
        return {"search_query": user_message}

    lines = []
    for msg in history:
        role = "用户" if msg["role"] == "user" else "助手"
        content = (msg.get("content") or msg.get("content", ""))[:200]
        lines.append(f"{role}：{content}")
    history_text = "\n".join(lines)

    prompt = REWRITE_PROMPT.format(history=history_text, message=user_message)
    try:
        model = LLM.get_model_for("rewrite")
        rewritten = model.invoke(prompt)
        if hasattr(rewritten, "content"):
            rewritten = rewritten.content
        rewritten = rewritten.strip()
        if not rewritten:
            logger.warning("Rewrite returned empty, falling back to original")
            return {"search_query": user_message}
        logger.info("Rewrote query: '%s' → '%s'", user_message[:40], rewritten[:60])
        return {"search_query": rewritten}
    except Exception as e:
        logger.warning("Rewrite query failed: %s, falling back to original", e)
        return {"search_query": user_message}
