import logging
import re

from langgraph.prebuilt import ToolRuntime
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain.agents import create_agent as create_react_agent

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
    answer: str = Field(description="Answer to the question")
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
        "你是一个专业的智能问答助手。分析问题并生成准确、有用的回答。",
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
        model=LLM.get_model(),
        tools=[web_search_tool],
        system_prompt=_build_system_prompt(state),
        response_format=ClassifyOutput,
    )
    # Prevent infinite tool-calling loops
    agent.recursion_limit = MAX_AGENT_STEPS

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
    logger.info("Answer: %s", structured.answer[:80])

    return ClassifyResult(
        answer=structured.answer,
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

