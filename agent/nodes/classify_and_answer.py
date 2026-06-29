import logging
import re

from langgraph.prebuilt import ToolRuntime
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from agent.state import AgentState
from agent.utils.llm import LLM
from agent.utils.agent_utils import build_context_block
from agent.tools.web_search import search_web_from_baidu
from agent.models.nodes import ClassifyResult
from agent.models.value_objects import LogicChainStep

logger = logging.getLogger(__name__)

# Agent recursion safety — prevents infinite tool-calling loops
MAX_AGENT_STEPS = 5


class ClassifyOutput(BaseModel):
    reasoning_trace: str = Field(
        description="Step-by-step reasoning: why this confidence level, what knowledge was considered"
    )
    answer: str = Field(
        description="The COMPLETE answer to the question. "
                    "This is the final response shown to the user. "
                    "Include ALL details, code, explanations — do NOT summarize. "
                    "For code questions, include the full working code."
    )
    confidence: float = Field(
        description="Confidence from 0.0 to 1.0"
    )
    needs_store: bool = Field(
        description="Whether this Q&A should be stored as knowledge. "
                    "True for factual/educational Q&A, False for casual chat/greetings"
    )


def _build_system_prompt(state: dict) -> str:
    """Build the system prompt with classification rules and optional context."""
    parts = [
        "你是一个专业的智能问答助手，以「金牌讲师」的风格回答问题。",
        "",
        "## 回答要求",
        "- 给出准确定义，解释核心原理",
        "- 分点/分条列出关键内容，用标题组织层次",
        "- 提供具体的例子或对比，帮助理解",
        "- 回答要完整、详细，不要过于简略",
        "- 对于技术类问题，从基本原理讲到实际应用",
        "",
        "## 网络搜索",
        "- 仅在完全不知道答案或需要最新信息时搜索",
        f"- 最多搜索 {MAX_AGENT_STEPS} 次，搜索失败则用已有知识回答",
        "",
        "## 知识拓展",
        "- 基于存储的知识进行关联思考，帮助用户建立更完整的知识网络",
        "- 可以补充相关知识点、对比差异、联系实际应用场景",
        "- 如果用户问题涉及多个知识点，可以串联起来给出综合性的回答",
        "- 不要编造不存在的内容，所有拓展需基于已有知识的合理延伸",
        "",
        "## 输出要求",
        "- 事实性、教育性问答需要存储（needs_store=true）",
        "- 问候、闲聊、个人观点不需要存储（needs_store=false）",
    ]

    context = build_context_block(state)
    if context:
        parts.append(context)

    return "\n".join(parts)


def _fallback_answer(state: dict) -> dict:
    """Generate a direct structured answer without tools when the agent loop fails."""
    logger.warning("Agent loop failed, falling back to direct generation")
    prompt = _build_system_prompt(state) + (
        f"## 用户问题\n"
        f"{state['user_message']}\n\n"
        f"请按照上述要求进行分析，并以结构化格式输出结果。"
    )
    try:
        result = LLM.generate_structured(prompt, ClassifyOutput)
    except Exception as e:
        logger.error("Fallback generation also failed: %s", e)
        return ClassifyResult(
            answer="",
            confidence=0.0,
            needs_store=False,
            logic_chain=[LogicChainStep(
                node="classify_and_answer",
                action="生成答案失败",
                reasoning=f"Agent and fallback both failed: {e}",
            )],
        ).model_dump()

    return ClassifyResult(
        answer=result.answer,
        confidence=result.confidence * 0.8,
        needs_store=result.needs_store,
        logic_chain=[LogicChainStep(
            node="classify_and_answer",
            action="网络搜索超时，基于知识直接回答",
            reasoning=result.reasoning_trace,
            confidence=result.confidence * 0.8,
            needs_store=result.needs_store,
            search_performed=False,
            fallback=True,
        )],
    ).model_dump()


def classify_and_answer(state: dict) -> dict:
    logger.info(
        "Classifying question (stored_knowledge=%d)",
        len(state.get("stored_knowledge", [])),
    )

    # ── URL 全部抓取失败 → 跳过 Agent 直接返回 ──
    url_contents = state.get("url_contents", [])
    user_message = state.get("user_message", "")
    if url_contents and all(
        uc.content == "[抓取失败]" for uc in url_contents
    ) and not re.sub(r'https?://[^\s]+', '', user_message).strip():
        failed = [uc.url for uc in url_contents]
        logger.warning("All %d URLs failed to fetch, skipping agent", len(failed))
        return ClassifyResult(
            answer=f"抱歉，以下网页内容抓取失败：\n" + "\n".join(f"- {u}" for u in failed) + "\n\n请检查链接是否可访问或稍后重试。",
            confidence=1.0,
            needs_store=False,
            logic_chain=[LogicChainStep(
                node="classify_and_answer",
                action="URL 全部抓取失败",
                reasoning=f"{len(url_contents)} 个 URL 全部抓取失败，跳过 Agent 直接返回",
            )],
        ).model_dump()

    agent = create_react_agent(
        model=LLM.get_model(temperature=0.3),
        tools=[web_search_tool],
        prompt=_build_system_prompt(state),
        response_format=ClassifyOutput,
    )
    # 避免无限循环
    # 2倍留给agent思考的轮数
    agent.recursion_limit = MAX_AGENT_STEPS * 2

    try:
        result = agent.invoke({
            "messages": [("user", state["user_message"])],
        })
        structured = result.get("structured_response")
    except Exception as e:
        logger.error("Agent recursion limit exceeded or error: %s", e)
        return _fallback_answer(state)

    if structured is None:
        logger.error("Agent did not produce structured response")
        return _fallback_answer(state)

    # Determine if search was performed by checking for tool calls
    search_performed = any(
        hasattr(m, "tool_calls") and m.tool_calls
        for m in result["messages"]
    )

    logger.info(
        "Classified with confidence=%.2f needs_store=%s",
        structured.confidence, structured.needs_store,
    )

    # 获取最终答案：优先用结构化输出的 answer，如果过短则用 Agent 消息体内容 暂不采用 对结果不准确有影响
    answer_text = structured.answer
    # if len(answer_text.strip()) < 200:
    #     messages = result.get("messages", [])
    #     if messages:
    #         last_ai = next(
    #             (m.content for m in reversed(messages)
    #              if hasattr(m, "content") and isinstance(getattr(m, "content", ""), str)
    #              and len(m.content or "") > 200),
    #             None
    #         )
    #         if last_ai:
    #             logger.info("结构化 answer 过短（%d字），使用 Agent 消息体内容（%d字）",
    #                         len(answer_text), len(last_ai))
    #             answer_text = last_ai

    logger.info("Answer: %s", answer_text[:80])

    return ClassifyResult(
        answer=answer_text,
        confidence=structured.confidence,
        needs_store=structured.needs_store,
        logic_chain=[LogicChainStep(
            node="classify_and_answer",
            action="搜索后生成答案" if search_performed else "生成初始答案",
            reasoning=structured.reasoning_trace,
            confidence=structured.confidence,
            needs_store=structured.needs_store,
            search_performed=search_performed,
        )],
    ).model_dump()


@tool
def web_search_tool(query: str, runtime: ToolRuntime[AgentState]) -> str:
    """Search the web when you lack the information needed to answer. Do NOT search for common knowledge. If search fails, answer from existing knowledge — do NOT retry."""
    state = runtime.state
    search_time = state.get("search_time", 0)
    if search_time > MAX_AGENT_STEPS:
        return "__EXCEED_SEARCH_LIMIT__ Please answer based on your existing knowledge. Do NOT search again."
    logger.info(f"Need to search on the web. {search_time}st question is {query}")
    results = search_web_from_baidu(query)
    state["search_time"] = search_time + 1
    if not results:
        return "__SEARCH_UNAVAILABLE__ Please answer based on your existing knowledge. Do NOT search again."
    return "\n".join(results)

