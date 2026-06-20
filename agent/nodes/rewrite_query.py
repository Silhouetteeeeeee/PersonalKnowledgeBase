"""Rewrite user query with conversation context for standalone retrieval."""
import logging
import re

from memory.message_history import MessageHistory
from agent.utils.llm import LLM
from agent.models.nodes import RewriteResult
from agent.models.value_objects import LogicChainStep

logger = logging.getLogger(__name__)

REWRITE_PROMPT = (
    '你是一个查询改写助手。根据对话历史，将用户的最新问题改写为一个'
    '不需要上下文就能理解的独立问题。\n\n'
    '要求：\n'
    '- 补全指代（如"它"→"Python dict"、"区别呢"→"A和B的区别"）\n'
    '- 补全省略的部分\n'
    '- 不要添加不存在的信息\n'
    '- 如果问题已经是独立的，保持原文\n'
    '- 只输出改写后的文本，不要任何解释\n\n'
    '对话历史：\n{history}\n\n'
    '用户最新消息：{message}'
)


def rewrite_query(state: dict) -> dict:
    """Rewrite the user message as a standalone query using conversation history.

    Falls back to the original user_message when:
    - Less than 2 prior messages exist (new or short conversation)
    - LLM call fails or returns empty
    """
    user_message = state["user_message"]
    url_contents = state.get("url_contents", [])

    # ── URL 场景检索策略 ──
    if url_contents:
        url_pattern = re.compile(r'https?://[^\s]+')
        question_only = url_pattern.sub('', user_message).strip()

        candidates = []
        for uc in url_contents:
            if uc.title:
                candidates.append(uc.title)
            else:
                content = uc.content
                first_sentence = re.split(r'[。.!?]', content)[0].strip()
                candidates.append(first_sentence[:100] if first_sentence else content[:100])

        if len(question_only) >= 10:
            return RewriteResult(search_query=question_only, logic_chain=[LogicChainStep(
                node="rewrite_query",
                action="URL 场景：使用用户问题",
                reasoning=f"用户有附加问题（{len(question_only)} 字），直接用作检索查询",
            )]).model_dump()

        query = "；".join(candidates)[:300]
        logger.info("URL query (no user question): '%s'", query[:60])
        return RewriteResult(search_query=query if query else user_message, logic_chain=[LogicChainStep(
            node="rewrite_query",
            action="URL 场景：使用文章标题/首句",
            reasoning=f"用户无附加问题，用 URL 内容的标题/首句作为检索查询",
        )]).model_dump()

    # ── 原有逻辑（无 URL）──
    session_id_raw = state.get("session_id")
    if session_id_raw is None:
        logger.debug("Skipping rewrite: no session_id in state")
        return RewriteResult(search_query=user_message, logic_chain=[LogicChainStep(
            node="rewrite_query",
            action="跳过改写",
            reasoning="无 session_id",
        )]).model_dump()
    # 兼容字符串和数字两种格式的 session_id
    try:
        session_id = int(session_id_raw)
    except (ValueError, TypeError):
        logger.debug("Skipping rewrite: non-integer session_id '%s'", session_id_raw)
        return RewriteResult(search_query=user_message, logic_chain=[LogicChainStep(
            node="rewrite_query",
            action="跳过改写",
            reasoning=f"非数字 session_id（{session_id_raw}），无需改写",
        )]).model_dump()

    history = MessageHistory.get_recent(session_id)
    if len(history) < 2:
        logger.debug("Skipping rewrite: only %d history messages", len(history))
        return RewriteResult(search_query=user_message, logic_chain=[LogicChainStep(
            node="rewrite_query",
            action="跳过改写",
            reasoning=f"历史消息不足（{len(history)} 条），无需改写",
        )]).model_dump()

    lines = []
    for msg in history:
        role = "用户" if msg["role"] == "user" else "助手"
        content = (msg.get("content") or "")[:200]
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
            return RewriteResult(search_query=user_message, logic_chain=[LogicChainStep(
                node="rewrite_query",
                action="改写返回空",
                reasoning="LLM 改写结果为空，回退到原始用户消息",
            )]).model_dump()
        logger.info("Rewrote query: '%s' → '%s'", user_message[:40], rewritten[:60])
        return RewriteResult(search_query=rewritten, logic_chain=[LogicChainStep(
            node="rewrite_query",
            action="查询改写",
            reasoning=f"原查询：「{user_message}」→ 改写为：「{rewritten}」",
        )]).model_dump()
    except Exception as e:
        logger.warning("Rewrite query failed: %s, falling back to original", e)
        return RewriteResult(search_query=user_message, logic_chain=[LogicChainStep(
            node="rewrite_query",
            action="改写失败",
            reasoning=f"LLM 异常：{e}，回退到原始用户消息",
        )]).model_dump()
