import logging

from storage.models import save_error_record_with_embedding
from agent.models.nodes import RecordErrorResult
from agent.models.value_objects import LogicChainStep

logger = logging.getLogger(__name__)


def record_error(state: dict) -> dict:
    """
    错误记录节点：将检测到的回答错误存入数据库（含向量索引），
    供后续回答参考，避免重复犯同类错误。

    correction_attempts 递增，驱动 Graph 的 contradiction 循环继续。
    """
    correction_attempts = state.get("correction_attempts", 0)
    user_message = state.get("user_message", "")
    wrong_answer = state.get("answer", "")
    correct_answer = state.get("reflection_correction", "")
    contradiction_details = state.get("contradiction_details", "")

    logger.info("记录错误: user_message='%s'", user_message[:50])

    record = {
        "user_message": user_message,
        "wrong_answer": wrong_answer,
        "correct_answer": correct_answer,
        "contradiction_details": contradiction_details,
        "error_type": "hallucination_or_error",
    }

    try:
        eid = save_error_record_with_embedding(record)
        logger.info("错误记录已保存（id=%d）", eid)
    except Exception as e:
        logger.error("错误记录保存失败: %s", e)

    return RecordErrorResult(
        correction_attempts=correction_attempts + 1,
        error_recorded=True,
        logic_chain=[LogicChainStep(
            node="record_error",
            action="记录回答错误",
            reasoning=f"记录本次回答错误，问题: {user_message[:50]}"
                     + (f"，修正建议: {correct_answer[:50]}" if correct_answer else ""),
        )],
    ).model_dump()
