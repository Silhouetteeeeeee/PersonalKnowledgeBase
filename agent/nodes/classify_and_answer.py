import logging

from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain.agents import create_agent as create_react_agent

from agent.utils.llm import LLM
from agent.tools.web_search import search_web

logger = logging.getLogger(__name__)


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
        f"你是一个专业的智能问答助手，负责分析问题并生成准确、有用的回答。\n\n"
        f"## 任务要求\n"
        f"1. 仔细分析用户的问题，理解其意图和背景\n"
        f"2. 结合提供的相关知识（如果有）进行推理\n"
        f"3. 对问题进行分类，确定所属领域层级（如 'programming/python' 或 'life/health'）\n"
        f"4. 评估回答的置信度（0.0-1.0），考虑知识完整性和不确定性\n"
        f"5. 判断是否应该存储此问答作为知识（事实性、教育性内容需要存储；闲聊、问候不需要）\n"
        f"6. 生成清晰、准确、有帮助的回答\n"
        f"7. 详细记录你的推理过程\n\n"
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
        f"- 有价值的问答对，可能对后续问题有帮助\n\n"
        f"不需要存储的情况：\n"
        f"- 简单的问候、感谢等社交性对话\n"
        f"- 纯主观的个人观点或偏好\n"
        f"- 临时性的、无长期价值的内容\n"
        f"- 重复的、已有的知识内容\n\n"
        f"{context}"
    )


def classify_and_answer(state: dict) -> dict:
    logger.info(
        "Classifying question (stored_knowledge=%d)",
        len(state.get("stored_knowledge", [])),
    )

    @tool
    def web_search_tool(query: str) -> str:
        """Search the web for current information when the question involves recent events, facts, statistics, or topics where accuracy verification is needed. Do NOT use for common knowledge, basic concepts, or greetings."""
        results = search_web(query)
        return "\n".join(results) if results else "未找到相关结果。"

    agent = create_react_agent(
        model=LLM.get_model(),
        tools=[web_search_tool],
        system_prompt=_build_system_prompt(state),
        response_format=ClassifyOutput,
    )

    result = agent.invoke({
        "messages": [("user", state["user_message"])],
    })

    structured = result.get("structured_response")
    if structured is None:
        logger.error("Agent did not produce structured response")
        return {
            "answer": "",
            "category": "unknown",
            "confidence": 0.0,
            "needs_store": False,
            "logic_chain": [{
                "node": "classify_and_answer",
                "action": "生成答案失败",
                "reasoning": "Agent failed to produce structured output",
            }],
        }

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
