import logging
from datetime import datetime

from langgraph.prebuilt import ToolRuntime
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain.agents import create_agent as create_react_agent

from agent.state import AgentState
from agent.utils.llm import LLM
from agent.tools.web_search import search_web, search_web_from_baidu

logger = logging.getLogger(__name__)

# Agent recursion safety — prevents infinite tool-calling loops
MAX_AGENT_STEPS = 5


class ClassifyOutput(BaseModel):
    reasoning_trace: str = Field(
        description="Step-by-step reasoning: why this category, why this confidence level, what knowledge was considered"
    )
    category: str = Field(
        description="Category hierarchy, e.g. 'programming/python' or 'life/health'"
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
    context = ""
    if state.get("stored_knowledge"):
        context = "Relevant past knowledge:\n"
        for k in state["stored_knowledge"]:
            context += f"- {k['knowledge_text']}\n"

    return (
        f"你是一个专业的智能问答助手，负责分析问题并生成准确、有用的回答。"
        f"同时你也是用户的生活助手，你应该记录用户的生活习惯，基本信息\n\n"
        f"## 任务要求\n"
        f"1. 仔细分析用户的问题，理解其意图和背景\n"
        f"2. 结合提供的相关知识（如果有）进行推理\n"
        f"3. 对问题进行分类，确定所属领域层级（如 'programming/python' 或 'life/health'）\n"
        f"4. 评估回答的置信度（0.0-1.0），考虑知识完整性和不确定性\n"
        f"5. 判断是否应该存储此问答作为知识（事实性、教育性内容需要存储；闲聊、问候不需要）\n"
        f"6. 生成清晰、准确、有帮助的回答\n"
        f"7. 详细记录你的推理过程\n\n"
        f"8. 如果用户并非提问，无需进行网络搜索，以一个朋友的角度进行回答，或者简单地回答收到即可"
        f"## ⚠️ 网络搜索使用规则（重要）\n"
        f"1. **仅在必要时搜索**：只有当你完全不知道答案，或需要最新信息时才搜索\n"
        f"2. **最多搜索1次**：每次对话只能调用一次网络搜索工具，禁止重复搜索\n"
        f"3. **基于现有知识优先**：优先使用你的训练知识和提供的上下文回答问题\n"
        f"4. **搜索失败处理**：如果搜索返回空结果或失败，立即停止搜索，用已有知识回答\n"
        f"5. **常识问题不搜索**：编程基础、数学公式、历史事实等常识性问题无需搜索\n\n"
        f"## 分类指南\n"
        f"- programming: 编程、软件开发、算法等技术问题\n"
        f"- life: 日常生活、健康、饮食等非技术问题\n"
        f"- education: 学术、学习、考试等教育相关问题\n"
        f"- personal: 用户的信息，比如用户的生活习惯personal/habits、基本信息personal/info\n"
        f"可以使用更细粒度的子分类，如 'programming/python'、'life/health' 等\n\n"
        f"## 置信度评估标准\n"
        f"- 0.9-1.0: 非常确定，有充分的知识支持，答案明确无歧义\n"
        f"- 0.7-0.9: 比较确定，有相关知识但可能存在细节不确定\n"
        f"- 0.5-0.7: 中等确定，部分信息缺失或存在多种解释\n"
        f"- 0.3-0.5: 不太确定，知识有限或问题模糊\n"
        f"- 0.0-0.3: 非常不确定，缺乏相关知识或问题不清晰\n\n"
        f"## 知识存储判断标准\n"
        f"需要存储的情况：\n"
        f"- 事实性信息（定义、概念、原理、方法等）\n"
        f"- 教育性内容（教程、示例、最佳实践等）\n"
        f"- 有价值的问答对，可能对后续问题有帮助\n"
        f"- 用户提供的基本信息，生活习惯，学习计划等等\n\n"
        f"不需要存储的情况：\n"
        f"- 简单的问候、感谢等社交性对话\n"
        f"- 纯主观的个人观点或偏好\n"
        f"- 临时性的、无长期价值的内容\n"
        f"- 重复的、已有的知识内容\n\n"
        f"{context}"
    )


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
        return {
            "answer": "",
            "category": "unknown",
            "confidence": 0.0,
            "needs_store": False,
            "logic_chain": [{
                "node": "classify_and_answer",
                "action": "生成答案失败",
                "reasoning": f"Agent and fallback both failed: {e}",
            }],
        }

    return {
        "category": result.category,
        "answer": result.answer,
        "confidence": result.confidence * 0.8,
        "needs_store": result.needs_store,
        "logic_chain": [{
            "node": "classify_and_answer",
            "action": "网络搜索超时，基于知识直接回答",
            "reasoning": result.reasoning_trace,
            "category": result.category,
            "confidence": result.confidence * 0.8,
            "needs_store": result.needs_store,
            "search_performed": False,
            "fallback": True,
        }],
    }


def classify_and_answer(state: dict) -> dict:
    logger.info(
        "Classifying question (stored_knowledge=%d)",
        len(state.get("stored_knowledge", [])),
    )

    agent = create_react_agent(
        model=LLM.get_model(),
        tools=[web_search_tool],
        system_prompt=_build_system_prompt(state),
        response_format=ClassifyOutput,
    )
    # Prevent infinite tool-calling loops
    agent.recursion_limit = MAX_AGENT_STEPS

    try:
        state["search_time"] = 0
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
        "Classified as category='%s' confidence=%.2f needs_store=%s",
        structured.category, structured.confidence, structured.needs_store,
    )
    logger.info("Answer: %s", structured.answer[:80])

    return {
        "category": structured.category,
        "answer": structured.answer,
        "confidence": structured.confidence,
        "needs_store": structured.needs_store,
        "logic_chain": [{
            "node": "classify_and_answer",
            "action": "搜索后生成答案" if search_performed else "生成初始答案",
            "reasoning": structured.reasoning_trace,
            "category": structured.category,
            "confidence": structured.confidence,
            "needs_store": structured.needs_store,
            "search_performed": search_performed,
        }],
    }


@tool
def web_search_tool(query: str, runtime: ToolRuntime[AgentState]) -> str:
    """
        只有当你确实不知道相关信息的时候需要进行网络搜索来给你更多的资料生成回答，对于常识问题无需网络搜索。
        如果网络搜索失败请不要一直重试，以你已有的知识进行回答问题
        请不要一直
    """
    state = runtime.state
    if state['search_time'] > MAX_AGENT_STEPS:
        return "__EXCEED_SEARCH_LIMIT__ Please answer based on your existing knowledge. Do NOT search again."
    logger.info(f"Need to search on the web. {state.get('search_time', 0)}st question is {query}")
    results = search_web_from_baidu(query)
    state['search_time'] = state.get('search_time', 0) + 1
    if not results:
        return "__SEARCH_UNAVAILABLE__ Please answer based on your existing knowledge. Do NOT search again."
    return "\n".join(results)

@tool
def get_current_time() -> datetime:
    """
        获取当前时间
        :return: 返回当前时间 e.g.  2026-05-04 15:30:45.123456
    """
    return datetime.now()