"""Intent handler functions. Each is a pure function (state) → dict.

Called by dispatch_intent_handler() which is registered as a single
LangGraph node. Adding a new intent = adding one handler here and one
entry in HANDLER_MAP — zero graph changes.
"""

import logging

from agent.utils.llm import LLM
from agent.utils.agent_utils import build_url_context
from agent.nodes.update_profile import update_profile
from agent.nodes.search_web import search_web_node
from agent.nodes.regenerate import regenerate
from storage.models import get_all_pages_index, find_similar_pages, get_page_by_title
from storage.wiki_storage import read_page
from agent.models.nodes import (
    ChitChatResult,
    LinkHandlingResult,
    PersonalInfoResult,
    ErrorFeedbackResult,
    KnowledgeMgmtResult,
    LowConfidenceResult,
    StubResult,
)
from agent.models.value_objects import LogicChainStep

logger = logging.getLogger(__name__)


# ── Handlers ──────────────────────────────────────────────────────


def handle_chitchat(state: dict) -> dict:
    """Lightweight LLM generation for chitchat/greetings. No retrieval, no tools."""
    user_message = state.get("user_message", "")
    prompt = (
        f"用户说: {user_message}\n\n"
        f"这是一个轻松的闲聊场景。请用友好、自然的语气回复用户。"
        f"保持简短（1-2句话），不要检索知识库。"
    )
    try:
        answer = LLM.generate(prompt, use_language=True)
    except Exception as e:
        logger.warning("Chitchat generation failed: %s", e)
        answer = "你好！有什么我可以帮你的吗？😊"

    return ChitChatResult(
        answer=answer,
        logic_chain=[LogicChainStep(
            node="handle_chitchat",
            action="闲聊回复",
            reasoning="轻量LLM生成，无需检索知识",
        )],
    ).model_dump()


def handle_link(state: dict) -> dict:
    """Summarize/extract from URL contents already fetched by parse node."""
    url_contents = state.get("url_contents", [])
    if not url_contents:
        return LinkHandlingResult(
            answer="没有检测到需要处理的网页链接。",
            logic_chain=[LogicChainStep(
                node="handle_link",
                action="无链接",
                reasoning="state.url_contents 为空",
            )],
        ).model_dump()

    context = build_url_context(url_contents)
    user_message = state.get("user_message", "")
    prompt = (
        f"请根据以下网页内容回答用户的问题。\n\n{context}\n\n"
        f"用户问题：{user_message}\n\n"
        f"要求：\n"
        f"- 提取核心观点和关键信息\n"
        f"- 用中文输出\n"
        f"- 如果用户没有附加问题，直接生成文章摘要"
    )
    try:
        answer = LLM.generate(prompt, use_language=True)
    except Exception as e:
        logger.warning("Link summarization failed: %s", e)
        # 至少返回标题
        titles = [uc.title for uc in url_contents if uc.title]
        if titles:
            answer = f"链接标题：\n" + "\n".join(f"- {t}" for t in titles)
        else:
            answer = "无法生成摘要。请确保链接可访问。"

    return LinkHandlingResult(
        answer=answer,
        logic_chain=[LogicChainStep(
            node="handle_link",
            action=f"处理 {len(url_contents)} 个链接",
            reasoning="基于 parse 节点提取的 URL 内容生成回复",
        )],
    ).model_dump()


def handle_personal_info(state: dict) -> dict:
    """Record personal info and confirm. Delegates to existing update_profile node."""
    profile_result = update_profile(state)
    updated = profile_result.get("user_profile", {})

    # Build confirmation message
    intent_params = state.get("intent_params", {})
    info_type = intent_params.get("info_type", "")
    if info_type:
        answer = f"好的，已记录您的{info_type}信息。📝"
    elif updated:
        answer = "好的，已更新您的个人信息。我会在后续对话中参考这些信息。📝"
    else:
        answer = "好的，已了解。"

    return PersonalInfoResult(
        answer=answer,
        user_profile=updated,
        logic_chain=[LogicChainStep(
            node="handle_personal_info",
            action="记录个人信息",
            reasoning=f"调用 update_profile 提取并持久化用户画像",
        )],
    ).model_dump()


def handle_error_feedback(state: dict) -> dict:
    """Handle error feedback: search the web and regenerate a corrected answer."""
    logger.info("Handling error feedback via web search + regeneration")

    # Run search
    search_result = search_web_node(state)
    state["search_results"] = search_result.get("search_results", [])

    # Regenerate answer with search results
    regen_result = regenerate(state)
    corrected_answer = regen_result.get("answer", "")

    if not corrected_answer:
        corrected_answer = "感谢您的反馈！已记录这个问题，我会在后续回答中注意纠正。"

    return ErrorFeedbackResult(
        answer=corrected_answer,
        search_results=state["search_results"],
        logic_chain=[LogicChainStep(
            node="handle_error_feedback",
            action="错误反馈处理",
            reasoning="收到纠错 → 网络搜索 → 重新生成答案",
        )],
    ).model_dump()


def handle_knowledge_mgmt(state: dict) -> dict:
    """Knowledge management: list, search, delete, or organize wiki pages."""
    logger.info("Handling knowledge management request")

    user_message = state.get("user_message", "")

    # Step 1: classify the management action
    action = _classify_mgmt_action(user_message)
    if action is None:
        return KnowledgeMgmtResult(
            answer="我没太理解您想对知识库做什么操作。您可以：\n\n"
                   "1. 查看知识列表 — 「我有哪些知识」\n"
                   "2. 搜索知识 — 「关于XX的知识」\n"
                   "3. 删除知识 — 「删除关于XX的知识」\n"
                   "4. 知识统计 — 「我的知识库有多大」",
            logic_chain=[LogicChainStep(
                node="handle_knowledge_mgmt",
                action="意图不明确",
                reasoning="无法识别知识管理操作类型",
            )],
        ).model_dump()

    # Step 2: execute the action
    if action["action"] == "list":
        result = _mgmt_list_pages()
    elif action["action"] == "search":
        result = _mgmt_search_pages(action.get("target", user_message))
    elif action["action"] == "delete":
        result = _mgmt_delete_page(action.get("target", ""))
    elif action["action"] == "stats":
        result = _mgmt_stats()
    elif action["action"] == "organize":
        result = _mgmt_organize(action.get("target", ""))
    else:
        result = "暂不支持该知识管理操作。"

    return KnowledgeMgmtResult(
        answer=result,
        logic_chain=[LogicChainStep(
            node="handle_knowledge_mgmt",
            action=f"知识管理: {action['action']}",
            reasoning=action.get("reason", ""),
        )],
    ).model_dump()


def _classify_mgmt_action(user_message: str) -> dict | None:
    """LLM classifies the specific knowledge management action.

    Returns dict with keys: action, target, reason.
    """
    from pydantic import BaseModel, Field

    class MgmtAction(BaseModel):
        action: str = Field(
            description="操作类型: list=查看列表, search=搜索, delete=删除, "
                        "stats=统计, organize=整理归类"
        )
        target: str = Field(description="搜索或删除的目标主题/标题")
        reason: str = Field(description="分类理由")

    prompt = (
        f"用户消息: {user_message}\n\n"
        "判断用户想对知识库做什么操作：\n"
        "- list: 查看知识列表/目录/索引。示例：「我有哪些知识」「帮我看看知识库」\n"
        "- search: 搜索/查询某个主题的知识。示例：「关于Redis的知识」「Python相关知识」\n"
        "- delete: 删除某个知识页面。示例：「删除关于XX的页面」「把XX删掉」\n"
        "- stats: 知识库统计。示例：「我有多少知识」「知识库有多大」\n"
        "- organize: 整理/归类知识。示例：「帮我整理一下知识」「归类相关知识点」\n\n"
        "无法判断时，action 设为空字符串。\n"
        "target 填写用户提到的具体主题（如果有的话）。"
    )
    try:
        result = LLM.generate_structured(prompt, MgmtAction, use_language=False)
        if not result.action:
            return None
        return {"action": result.action, "target": result.target, "reason": result.reason}
    except Exception:
        return None


def _mgmt_list_pages() -> str:
    """List all active wiki pages in a formatted message."""
    pages = get_all_pages_index()
    if not pages:
        return "📚 知识库目前还是空的，没有已保存的知识页面。\n\n您可以向我提问来积累知识，系统会自动从对话中提取知识点。"

    total = len(pages)
    lines = [f"📚 当前共有 **{total}** 篇知识页面：\n"]
    for p in pages:
        title = p.get("title", "未命名")
        tags = p.get("tags", [])
        if isinstance(tags, str):
            try:
                import json
                tags = json.loads(tags)
            except (json.JSONDecodeError, TypeError):
                tags = []
        tag_str = f" [{', '.join(tags[:3])}]" if tags else ""
        lines.append(f"- **{title}**{tag_str}")
    return "\n".join(lines)


def _mgmt_search_pages(query: str) -> str:
    """Search for pages matching the query."""
    if not query or len(query.strip()) < 2:
        return "请告诉我您想搜索什么主题，比如「关于Redis的知识」。"
    try:
        results = find_similar_pages(query, threshold=0.4, limit=5)
    except Exception as e:
        logger.warning("Knowledge search failed: %s", e)
        return "搜索知识库时出了点问题，请稍后再试。"

    if not results:
        return f"未找到与「{query}」相关的知识页面。\n\n您可以继续提问，系统会自动从对话中积累相关知识。"

    from storage.wiki_storage import read_page
    lines = [f"🔍 找到以下与「{query}」相关的知识页面：\n"]
    for r in results:
        page_data = read_page(r.file_path)
        snippet = (page_data["body"][:150] + "...") if page_data and page_data.get("body") else "(内容读取失败)"
        lines.append(f"### {r.title}")
        lines.append(f"> 相关度: {(1 - r.distance) * 100:.0f}%")
        lines.append(f"> {snippet}")
        lines.append("")
    return "\n".join(lines)


def _mgmt_delete_page(target: str) -> str:
    """Soft-delete a wiki page by title."""
    if not target or len(target.strip()) < 2:
        return "请告诉我您想删除哪个知识页面，比如「删除关于Redis的页面」。\n\n⚠️ 删除操作不可撤销，请谨慎操作。"

    from storage.database import get_connection

    # Try exact match first, then fuzzy
    page = get_page_by_title(target.strip())
    if page is None:
        # Fuzzy: find similar pages and show options
        try:
            similar = find_similar_pages(target, threshold=0.5, limit=3)
        except Exception:
            similar = []

        if similar:
            lines = [f"未找到完全匹配「{target}」的页面。您是不是想删除以下某个页面？\n"]
            for s in similar:
                lines.append(f"- {s.title}")
            lines.append("\n请回复更精确的页面标题。")
            return "\n".join(lines)
        return f"未找到标题为「{target}」的知识页面。"

    # Confirm and soft-delete
    title = page["title"]
    page_id = page["id"]
    conn = get_connection()
    try:
        conn.execute(
            "UPDATE pages SET status='inactive' WHERE id=? AND status='active'",
            (page_id,),
        )
        conn.commit()
        logger.info("Soft-deleted page '%s' (id=%d)", title, page_id)
    except Exception as e:
        conn.rollback()
        logger.error("Failed to delete page '%s': %s", title, e)
        return f"删除「{title}」时出了点问题，请稍后再试。"
    finally:
        conn.close()

    return f"🗑️ 已删除知识页面「{title}」。\n\n如果误删了，可以联系管理员恢复。"


def _mgmt_stats() -> str:
    """Return knowledge base statistics."""
    from storage.database import get_connection

    conn = get_connection()
    try:
        total = conn.execute(
            "SELECT COUNT(*) FROM pages WHERE status='active'"
        ).fetchone()[0]
        total_tags = conn.execute(
            "SELECT COUNT(DISTINCT id) FROM page_relations"
        ).fetchone()[0]
        recent = conn.execute(
            "SELECT COUNT(*) FROM pages WHERE status='active' "
            "AND updated_at >= datetime('now', '-7 days')"
        ).fetchone()[0]
    except Exception:
        total = total_tags = recent = 0
    finally:
        conn.close()

    return (
        f"📊 **知识库统计**\n\n"
        f"- 知识页面: **{total}** 篇\n"
        f"- 页面关联: **{total_tags}** 条\n"
        f"- 本周更新: **{recent}** 篇\n\n"
        f"继续保持学习节奏！"
    )


def _mgmt_organize(target: str) -> str:
    """LLM-based organization suggestions."""
    pages = get_all_pages_index()
    if not pages or len(pages) < 2:
        return "知识库中的页面还不多，暂时不需要整理。多提问积累更多知识后再来整理吧！"

    # Build a compact index for LLM
    index_lines = []
    for p in pages:
        tags = p.get("tags", [])
        if isinstance(tags, str):
            try:
                import json
                tags = json.loads(tags)
            except (json.JSONDecodeError, TypeError):
                tags = []
        tag_str = f" [{', '.join(tags)}]" if tags else ""
        index_lines.append(f"- {p['title']}{tag_str}")

    index_text = "\n".join(index_lines)
    prompt = (
        f"以下是知识库中的所有页面标题和标签：\n\n{index_text}\n\n"
        f"用户要求整理归类知识。请分析这些页面之间的关系，"
        f"给出1-3条归类建议（哪些页面应该归为一组，用什么主题标签）。"
        f"保持简洁，用中文输出。"
    )
    try:
        suggestion = LLM.generate(prompt, use_language=True)
    except Exception as e:
        logger.warning("Knowledge organization failed: %s", e)
        return "整理分析时出了点问题，请稍后再试。"

    return f"📂 **知识归类建议**\n\n{suggestion}\n\n---\n这些建议可以帮助您更好地组织知识库。"


def handle_low_confidence(state: dict) -> dict:
    """Clarifying question when intent is uncertain."""
    suggestions = state.get("low_confidence_suggestions", [])
    if not suggestions:
        suggestions = ["我想查询某个知识点", "我想和你聊聊天", "我想记录一些个人信息"]

    text = "我不太确定您的意思。您是想：\n\n"
    for i, s in enumerate(suggestions, 1):
        text += f"{i}. {s}\n"
    text += "\n请选择或重新描述您的问题，我会更好地帮助您！😊"

    return LowConfidenceResult(
        answer=text,
        logic_chain=[LogicChainStep(
            node="handle_low_confidence",
            action="追问用户",
            reasoning=f"生成 {len(suggestions)} 个澄清建议",
        )],
    ).model_dump()


def _intent_display_name(intent_id: str) -> str:
    """Look up the human-readable name for an intent from INTENTS."""
    from agent.intent.registry import INTENTS
    for n in INTENTS:
        if n.id == intent_id:
            return n.name
    return intent_id


def handle_stub(state: dict) -> dict:
    """Generic stub for future intents (learning_plan, todo, etc.)."""
    intent = state.get("intent", "unknown")
    name = _intent_display_name(intent)
    return StubResult(
        answer=f"「{name}」功能正在开发中，敬请期待！🚧",
        logic_chain=[LogicChainStep(
            node=f"handle_stub",
            action=f"Stub: {intent}",
            reasoning="功能尚未实现",
        )],
    ).model_dump()


# ════════════════════════════════════════════════════════════════════
# Dispatcher
# ════════════════════════════════════════════════════════════════════

HANDLER_MAP: dict[str, callable] = {
    "chitchat": handle_chitchat,
    "link_handling": handle_link,
    "personal_info": handle_personal_info,
    "error_feedback": handle_error_feedback,
    "knowledge_mgmt": handle_knowledge_mgmt,
    "low_confidence": handle_low_confidence,
    "learning_plan": handle_stub,
    "todo": handle_stub,
}


def dispatch_intent_handler(state: dict) -> dict:
    """LangGraph node: dispatches to the correct handler based on state['intent'].

    This is the ONLY graph node needed for all non-knowledge_qa intents.
    Adding a new intent = add a handler function + register in HANDLER_MAP.
    """
    intent = state.get("intent", "")
    logger.info("Dispatching intent handler: %s", intent)

    handler = HANDLER_MAP.get(intent)
    if handler:
        try:
            return handler(state)
        except Exception as e:
            logger.exception("Handler '%s' failed: %s", intent, e)
            return StubResult(
                answer=f"抱歉，处理您的请求时出了点问题，请稍后再试。🙏",
                logic_chain=[LogicChainStep(
                    node="dispatch_intent_handler",
                    action=f"Handler 异常: {intent}",
                    reasoning=str(e),
                )],
            ).model_dump()

    logger.warning("No handler for intent '%s', falling back to stub", intent)
    return handle_stub(state)
