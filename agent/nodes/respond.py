import logging
import os
from datetime import datetime

from storage.models import update_knowledge_reasoning_path

logger = logging.getLogger(__name__)

REASONING_LOG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data", "reasoning",
)


def _save_reasoning_log(state: dict) -> str:
    """Save the logic_chain to a local MD file and return the file path."""
    user_id = state.get("user_id", "unknown")
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H%M%S")
    log_dir = os.path.join(REASONING_LOG_DIR, date_str)
    os.makedirs(log_dir, exist_ok=True)

    # Sanitize user_id for filename
    safe_user = "".join(c if c.isalnum() else "_" for c in user_id)
    filename = f"{safe_user}_{time_str}.md"
    file_path = os.path.join(log_dir, filename)

    lines = [
        f"# 推理链路 - {now.strftime('%Y-%m-%d %H:%M:%S')}",
        f"",
        f"## 用户消息",
        f"{state.get('user_message', '')}",
        f"",
    ]

    if state.get("contradiction_found"):
        lines.append(f"> ⚠️ 本次检测到矛盾：{state.get('contradiction_details', '')}")
        if state.get("reflection_result"):
            lines.append(f"> 反思结论：{state['reflection_result']}")
            lines.append(f"> 反思推理：{state.get('reflection_reasoning', '')}")
        lines.append("")

    chain = state.get("logic_chain", [])
    for step in chain:
        node = step.get("node", "unknown")
        action = step.get("action", "")
        reasoning = step.get("reasoning", "")

        lines.append(f"## {node} — {action}")
        if reasoning:
            lines.append(f"\n**思考过程：**\n{reasoning}")
        lines.append("")

    if state.get("knowledge_corrected"):
        lines.append("## 知识库修正\n知识库中过时/错误的知识已被标记废弃并替换。")
    if state.get("error_recorded"):
        lines.append("## 错误记录\n本次回答中的错误已被记录，将用于后续改进。")

    lines.append("---")
    lines.append(f"_生成时间: {now.strftime('%Y-%m-%d %H:%M:%S')}_")

    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        logger.info("Reasoning log saved to %s", file_path)
    except Exception as e:
        logger.error("Failed to save reasoning log: %s", e)
        return ""

    return file_path


def _update_reasoning_paths(state: dict, log_path: str) -> None:
    """Update the reasoning_log_path for knowledge points stored in this turn."""
    stored_ids = state.get("stored_knowledge_ids", [])
    if stored_ids and log_path:
        try:
            update_knowledge_reasoning_path(stored_ids, log_path)
        except Exception as e:
            logger.warning("Failed to update reasoning_log_path: %s", e)


def respond(state: dict) -> dict:
    answer = state.get("answer", "")

    # Handle contradiction warnings
    if state.get("contradiction_found"):
        reflection_result = state.get("reflection_result", "")

        if reflection_result == "unresolved":
            details = state.get("contradiction_details", "")
            warning = f"\n\n[矛盾警告] 无法判断矛盾来源，请人工复核。\n详情：{details}"
            answer += warning
            logger.info("Appended unresolved contradiction warning")
        elif reflection_result == "stored_knowledge_wrong":
            warning = "\n\n[检测到知识库中存在过时信息]"
            if state.get("knowledge_corrected"):
                warning += " 已自动修正。"
            else:
                warning += " 已标记待审核。"
            answer += warning
            logger.info("Appended knowledge correction notice")
        elif reflection_result == "answer_wrong":
            warning = "\n\n[已记录本次回答中的错误，将用于后续改进]"
            answer += warning
            logger.info("Appended error recording notice")
        else:
            # Fallback: no reflection result yet (shouldn't happen in normal flow)
            details = state.get("contradiction_details", "")
            warning = f"\n\n[矛盾警告] {details}"
            answer += warning
            logger.info("Appended generic contradiction warning (no reflection)")

    # Save reasoning log to MD file
    log_path = _save_reasoning_log(state)

    # Update knowledge points with the reasoning log path
    _update_reasoning_paths(state, log_path)

    logger.info("Responding with answer (len=%d): '%s'", len(answer), answer[:80])
    return {"final_response": answer}
