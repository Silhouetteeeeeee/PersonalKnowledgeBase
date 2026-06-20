"""
重新生成答案节点：结合网络搜索结果修正原始答案。

用在 contradiction 循环中：
  record_error → search_web → regenerate → fact_check（再次验证）
当搜索结果为空时，保留原答案不变。
"""

import logging

from pydantic import BaseModel, Field

from agent.utils.agent_utils import build_context_block
from agent.utils.llm import LLM
from agent.models.nodes import RegenerateResult
from agent.models.value_objects import LogicChainStep

logger = logging.getLogger(__name__)


class RegenerateOutput(BaseModel):
    reasoning_trace: str = Field(
        description="Step-by-step reasoning: how the web search results inform the answer, any corrections from the original answer"
    )
    answer: str = Field(description="The regenerated answer based on web search results")


def regenerate(state: dict) -> dict:
    """
    重新生成答案：基于网络搜索到的实时信息修正原始回答。

    流程：
    1. 检查是否有搜索结果
    2. 如果有，将搜索结果 + 上下文 + 原答案 一起送入 LLM 重新生成
    3. 如果没有，保留原始答案不变
    """
    search_text = "\n\n".join(state.get("search_results", []))
    if not search_text:
        logger.info("无搜索结果，保留原答案")
        answer = state.get("answer", "")
        return RegenerateResult(answer=answer, logic_chain=[LogicChainStep(
            node="regenerate",
            action="无搜索结果，保留原答案",
            reasoning="Web search returned no results, keeping original answer unchanged",
        )]).model_dump()

    logger.info("基于 %d 条搜索结果重新生成答案", len(state.get("search_results", [])))

    context = build_context_block(state)

    prompt = (
        f"{context}\n\n"
        f"## 网络搜索结果\n{search_text}\n\n"
        f"## 用户问题\n{state['user_message']}\n\n"
        f"## 原答案\n{state.get('answer', '')}\n\n"
        f"请基于搜索结果的真实信息，结合上述背景，生成一个准确且风格一致的答案。"
    )
    result = LLM.generate_structured(prompt, RegenerateOutput, use_language=False)

    logger.info("重新生成答案: %s", result.answer[:80])
    return RegenerateResult(answer=result.answer, logic_chain=[LogicChainStep(
        node="regenerate",
        action="基于搜索结果重新生成答案",
        reasoning=result.reasoning_trace,
    )]).model_dump()
