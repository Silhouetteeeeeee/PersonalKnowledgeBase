import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from pydantic import BaseModel, Field

from agent.utils.llm import LLM
from storage.models import (
    save_knowledge_points_bulk_with_embeddings,
    ensure_category,
    find_similar_knowledge,
    get_normalized_categories,
    normalize_category_str,
)

logger = logging.getLogger(__name__)

class DistilledPoint(BaseModel):
    knowledge_text: str = Field(
        description="A concise, standalone knowledge point distilled from the Q&A"
    )
    tags: list[str] = Field(description="Relevant tags for this knowledge point")


class DistillOutput(BaseModel):
    category: str = Field(
        description="Category for the knowledge. "
                    "Support up to four-tier hierarchical categories (e.g., databases/nosql/redis/commands)"
    )
    knowledge_points: list[DistilledPoint] = Field(
        description="Knowledge points distilled from the Q&A"
    )

def _check_duplicate(kp) -> tuple:
    """检查知识点是否重复，返回 (kp, is_duplicate)"""
    try:
        similar = find_similar_knowledge(kp.knowledge_text, threshold=0.25)
        if similar:
            logger.info("Skipping duplicate knowledge: '%s' (distance=%.3f)",
                       kp.knowledge_text[:50], similar[0].get("distance", 0))
            return kp, True
    except Exception as e:
        logger.warning("Dedup embedding failed for '%s': %s, saving without dedup",
                      kp.knowledge_text[:30], e)
    return kp, False

def store(state: dict) -> dict:
    if not state.get("needs_store", True):
        logger.info("Skipping store: needs_store=False")
        return {}

    if not state.get("answer"):
        logger.info("Skipping store: no answer")
        return {}

    if state.get("contradiction_found"):
        logger.info("Skipping store: contradiction detected")
        return {}

    logger.info("Distilling knowledge from Q&A...")
    existing_cats = get_normalized_categories()
    prompt = (
        f"Distill the following Q&A into concise, standalone knowledge points.\n\n"
        f"Question: {state['user_message']}\n"
        f"Answer: {state['answer']}"
    )
    if existing_cats:
        prompt += (f"\n\n已有分类：{existing_cats}\n请优先选择最匹配的已有分类，仅当完全不匹配时创建新分类。"
                   f"如果当前知识点属于某个已有分类的子类别，请使用多级分类格式（如：）")
    result = LLM.generate_structured(prompt, DistillOutput, use_language=False)

    result.category = normalize_category_str(result.category)
    ensure_category(result.category)

    # Dedup: skip knowledge points that are semantically similar to existing ones
    new_points = []
    # 并行去重检查
    with ThreadPoolExecutor(max_workers=16, thread_name_prefix="dedup") as executor:
        futures = {executor.submit(_check_duplicate, kp): kp
                   for kp in result.knowledge_points}

        # 收集非重复的知识点
        new_points = []
        for future in as_completed(futures):
            kp, is_duplicate = future.result()
            if not is_duplicate:
                new_points.append(kp)

    if not new_points:
        logger.info("All knowledge points already exist, nothing to store")
        return {}

    reasoning_log_path = state.get("reasoning_log_path", "")
    knowledge_points = [
        {
            "knowledge_text": kp.knowledge_text,
            "source_question": state["user_message"],
            "category": result.category,
            "tags": kp.tags,
            "reasoning_log_path": reasoning_log_path,
        }
        for kp in new_points
    ]
    ids = save_knowledge_points_bulk_with_embeddings(knowledge_points)

    return {
        "category": result.category,
        "stored_knowledge_ids": ids,
        "logic_chain": [{
            "node": "store",
            "action": f"存储 {len(ids)} 条知识点",
            "reasoning": f"分类: {result.category}, 知识点数: {len(ids)}",
        }],
    }
