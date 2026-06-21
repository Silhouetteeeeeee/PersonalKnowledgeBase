"""Intent classification: single LLM call scores ALL intents simultaneously.

Ragent-style pattern: instead of regex or a separate classifier model,
the LLM scores every intent type in one shot and returns structured output.
"""

import logging

from agent.utils.llm import LLM
from agent.intent.schemas import IntentClassification, IntentResult, Suggestions
from agent.intent.registry import INTENTS
from agent.models.nodes import IntentClassifyResult
from agent.models.value_objects import LogicChainStep

logger = logging.getLogger(__name__)

# Intents below this threshold are discarded (same as fund/intent/classifier).
MIN_CONFIDENCE = 0.35


def _build_intents_section() -> str:
    """Serialize INTENTS into a prompt section the LLM can score."""
    lines = []
    for n in INTENTS:
        lines.append(f"### {n.id}")
        lines.append(f"  名称: {n.name}")
        lines.append(f"  描述: {n.description}")
        lines.append(f"  示例: {' | '.join(n.examples[:4])}")
        if n.params:
            lines.append("  参数:")
            for p in n.params:
                req = "必填" if p.required else "可选"
                lines.append(f"    - {p.name} ({p.type}): {p.description} [{req}]")
        lines.append("")
    return "\n".join(lines)


CLASSIFY_SYSTEM_PROMPT = """你是一个知识助手的意图分类器。根据用户消息判断其意图，并提取必要参数。

## 可用意图

{intents_section}

## 评分标准

- score > 0.8: 强匹配 — 用户表述与意图名称、描述或示例高度一致
- 0.35 ~ 0.8: 中等匹配 — 部分要素匹配，可能是该意图
- < 0.35: 弱匹配 — 应排除

## 规则

1. 对每个可用意图独立评分，返回所有意图的评分结果
2. 评分结果按置信度降序排列
3. 参数必须从用户消息中提取，不要编造
4. 注意检查用户消息中是否包含 URL — 如果包含且问题是总结/分析链接，优先匹配 link_handling
5. 如果所有意图分数都 < 0.35，返回空数组"""


def classify_intent(state: dict) -> dict:
    """Intent classification graph node.

    Single LLM call scores all 8 intents. Returns the top intent
    (or low_confidence / default) plus extracted params.
    """
    user_message = state.get("user_message", "")
    if not user_message:
        logger.info("Empty message → default to knowledge_qa")
        return IntentClassifyResult(
            intent="knowledge_qa",
            intent_confidence=0.0,
        ).model_dump()

    intents_section = _build_intents_section()
    prompt = CLASSIFY_SYSTEM_PROMPT.format(intents_section=intents_section)
    prompt += f"\n## 用户消息\n{user_message}"

    try:
        classification: IntentClassification | None = LLM.generate_structured(
            prompt, IntentClassification, use_language=False,
        )
    except Exception as e:
        logger.error("Intent classification failed: %s → default to knowledge_qa", e)
        return IntentClassifyResult(
            intent="knowledge_qa",
            intent_confidence=0.0,
            intent_reasoning=f"Classification error: {e}",
        ).model_dump()

    if classification is None:
        logger.info("LLM returned None → default to knowledge_qa")
        return IntentClassifyResult(
            intent="knowledge_qa",
            intent_confidence=0.0,
            intent_reasoning="Classifier returned None",
        ).model_dump()
    if not classification.results:
        logger.info("No intents matched → low_confidence for: %s", user_message[:60])
        return _low_confidence_result(user_message)

    # Filter by threshold
    valid = [r for r in classification.results if r.score >= MIN_CONFIDENCE]
    if not valid:
        return _low_confidence_result(user_message)

    top = valid[0]
    logger.info("Intent: %s (score=%.2f) params=%s", top.id, top.score, top.params)

    return IntentClassifyResult(
        intent=top.id,
        intent_confidence=top.score,
        intent_params=top.params,
        intent_reasoning=top.reason,
        logic_chain=[LogicChainStep(
            node="classify_intent",
            action=f"意图: {top.id} (score={top.score:.2f})",
            reasoning=top.reason,
            confidence=top.score,
        )],
    ).model_dump()


def _low_confidence_result(user_message: str) -> dict:
    """Generate clarifying suggestions when no intent is confidently matched."""
    logger.info("Low confidence for '%s', generating suggestions", user_message[:60])

    prompt = (
        "用户消息意图不够明确。生成2-3个简短(10-20字)的追问选项，"
        "帮助用户明确意图。可能的意图方向：\n"
        + "\n".join(f"- {n.id}: {n.name} — {n.description}" for n in INTENTS)
        + "\n\n返回JSON格式：{'suggestions': ['选项1', '选项2']}"
    )
    try:
        result = LLM.generate_structured(prompt, Suggestions, use_language=False)
        suggestions = result.suggestions[:3]
    except Exception:
        suggestions = [
            "我想查询某个知识点",
            "我想和你聊聊天",
            "我想记录一些个人信息",
        ]

    return IntentClassifyResult(
        intent="low_confidence",
        intent_confidence=0.0,
        intent_params={},
        intent_reasoning="所有意图评分均低于阈值",
        low_confidence_suggestions=suggestions,
        logic_chain=[LogicChainStep(
            node="classify_intent",
            action="意图不明确",
            reasoning=f"所有意图评分 < {MIN_CONFIDENCE}，生成{len(suggestions)}个追问建议",
        )],
    ).model_dump()
