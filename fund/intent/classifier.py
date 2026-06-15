"""LLM-based intent classifier — single-shot scoring against all intent nodes.

Usage:
    result = classify("帮我看看易方达蓝筹", INTENTS)
    if result and result.score >= MIN_SCORE:
        handler = route[result.id]
        handler(frame, user_id, result.params)
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from agent.utils.llm import LLM
from fund.intent.schemas import IntentNode

logger = logging.getLogger(__name__)

MIN_SCORE = 0.35
MAX_RESULTS = 1


class IntentResult(BaseModel):
    """Single intent classification result."""
    id: str = Field(description="意图ID")
    score: float = Field(..., ge=0.0, le=1.0, description="置信度分数(0-1)")
    reason: str = Field(description="分类理由")
    params: dict[str, Any] = Field(default_factory=dict, description="提取的参数")


class IntentClassification(BaseModel):
    """Full classification output from LLM."""
    results: list[IntentResult] = Field(description="按置信度降序排列的意图列表")


SYSTEM_PROMPT_HEADER = """你是一个基金助手意图分类器。你的任务是根据用户消息判断其意图，并提取必要的参数。

## 可用意图列表

{intents_section}

## 评分标准

- score > 0.8: 强匹配 — 用户表述与意图名称、描述或示例高度一致
- 0.35 ~ 0.8: 中等匹配 — 部分要素匹配，可能是该意图
- < 0.35: 弱匹配 — 不匹配，应排除

## 规则

1. 如果用户消息只涉及一个意图，返回一个结果
2. 如果用户消息涉及多个独立意图，最多返回 2 个结果
3. 如果所有意图分数都 < 0.35，返回空数组
4. 参数必须从用户消息中提取，不要编造
5. 对于 fund_analyze / fund_status / fund_search 意图，query 可以是基金代码或基金名称"""


def _build_intents_section(intents: list[IntentNode]) -> str:
    """Serialize intent nodes into the prompt section (mirrors ragent's intent-classifier.st)."""
    lines = []
    for n in intents:
        lines.append(f"### {n.id}")
        lines.append(f"  名称: {n.name}")
        lines.append(f"  描述: {n.description}")
        lines.append(f"  示例: {' | '.join(n.examples)}")
        if n.params:
            lines.append("  参数:")
            for p in n.params:
                req = "必填" if p.required else "可选"
                lines.append(f"    - {p.name} ({p.type}): {p.description} [{req}]")
        lines.append("")
    return "\n".join(lines)


def classify(
    user_message: str,
    intents: list[IntentNode],
) -> IntentResult | None:
    """Classify user message against intent nodes. Returns top result or None."""
    intents_section = _build_intents_section(intents)
    prompt = SYSTEM_PROMPT_HEADER.format(intents_section=intents_section)
    prompt += f"\n## 用户消息\n{user_message}"

    try:
        classification: IntentClassification | None = LLM.generate_structured(
            prompt, IntentClassification, use_language=False
        )
    except Exception as e:
        logger.error("Intent classification failed: %s", e)
        return None

    if not classification or not classification.results:
        logger.info("No intents matched for: %s", user_message[:60])
        return None

    # Filter and sort
    valid = [r for r in classification.results if r.score >= MIN_SCORE]
    valid.sort(key=lambda r: r.score, reverse=True)

    if not valid:
        logger.info("All intents below threshold for: %s", user_message[:60])
        return None

    top = valid[0]
    logger.info("Intent: %s (score=%.2f, reason=%s)", top.id, top.score, top.reason)
    return top


def validate_params(intent: IntentNode, params: dict) -> tuple[bool, str]:
    """Check that all required params are present."""
    missing = [p.name for p in intent.params if p.required and p.name not in params]
    if missing:
        return False, f"缺少必要参数: {', '.join(missing)}"
    return True, ""
