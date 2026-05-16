import logging

logger = logging.getLogger(__name__)


def respond(state: dict) -> dict:
    answer = state.get("answer", "")

    # Handle contradiction warnings
    if state.get("contradiction_found"):
        reflection_result = state.get("reflection_result", "")

        if reflection_result == "unresolved":
            details = state.get("contradiction_details", "")
            answer += f"\n\n[矛盾警告] 无法判断矛盾来源，请人工复核。\n详情：{details}"
        elif reflection_result == "stored_knowledge_wrong":
            answer += "\n\n[检测到知识库中存在过时信息] 已标记待审核。"
        elif reflection_result == "answer_wrong":
            answer += "\n\n[已记录本次回答中的错误，将用于后续改进]"
        else:
            details = state.get("contradiction_details", "")
            answer += f"\n\n[矛盾警告] {details}"

    logger.info("Responding with answer (len=%d): '%s'", len(answer), answer[:80])
    return {"final_response": answer}
