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
    """Knowledge management — stub for now, will be enhanced later."""
    return KnowledgeMgmtResult(
        answer=(
            "📚 知识管理功能正在开发中。目前您可以通过日常问答来积累知识，"
            "系统会自动从对话中提取并整理知识页面。"
        ),
        logic_chain=[LogicChainStep(
            node="handle_knowledge_mgmt",
            action="知识管理",
            reasoning="Stub: 功能尚未实现",
        )],
    ).model_dump()


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
