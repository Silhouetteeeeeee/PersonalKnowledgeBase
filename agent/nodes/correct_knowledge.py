import logging

from storage.models import update_knowledge_status, save_knowledge_points_bulk_with_embeddings

logger = logging.getLogger(__name__)


def correct_knowledge(state: dict) -> dict:
    knowledge_ids = state.get("contradiction_knowledge_ids", [])
    correction = state.get("reflection_correction", "")
    user_message = state.get("user_message", "")
    correction_attempts = state.get("correction_attempts", 0)

    logger.info("Correcting knowledge: %d ids, correction=%s", len(knowledge_ids), bool(correction))

    if not knowledge_ids:
        logger.warning("correct_knowledge: no knowledge IDs to correct")
        return {"correction_attempts": correction_attempts + 1, "knowledge_corrected": False}

    # Mark deprecated
    for kid in knowledge_ids:
        update_knowledge_status(kid, "deprecated", corrected_text=correction)
        logger.info("Marked knowledge %d as deprecated", kid)

    # Save corrected version
    if correction:
        ids = save_knowledge_points_bulk_with_embeddings([{
            "knowledge_text": correction,
            "source_question": f"[auto-corrected] {user_message}",
            "category": "uncategorized",
            "tags": ["auto-corrected"],
            "status": "active",
        }])
        logger.info("Saved corrected knowledge (id=%s)", ids)

    return {
        "correction_attempts": correction_attempts + 1,
        "knowledge_corrected": True,
        "logic_chain": [{
            "node": "correct_knowledge",
            "action": "修正知识库",
            "reasoning": f"将 {len(knowledge_ids)} 条冲突知识标记为 deprecated"
                         + (f"，保存修正版本" if correction else ""),
            "knowledge_ids": knowledge_ids,
        }],
    }
