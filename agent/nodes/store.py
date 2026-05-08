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
        description="A concise, standalone knowledge point distilled from the input content"
    )
    category: str = Field(
        description="Category for the knowledge point. Each knowledge point has different category."
                    "Support up to four-tier hierarchical categories (e.g., databases/nosql/redis/commands)"
    )
    tags: list[str] = Field(description="Relevant tags for this knowledge point")


class DistillOutput(BaseModel):
    category: str = Field(
        description="Category for the knowledge. "
                    "Support up to four-tier hierarchical categories (e.g., databases/nosql/redis/commands)"
    )
    knowledge_points: list[DistilledPoint] = Field(
        description="Knowledge points distilled from the input content"
    )


_BASE_DISTILL_PROMPT = (
    "## Role\n"
    "You are a knowledge refinement expert. Extract key knowledge points from the content, "
    "enrich them with explanatory context, and classify them precisely.\n\n"
    "## 知识提炼要求\n"
    "1. 每个知识点应该自包含、可独立理解，让读者只看知识点就能学到完整内容\n"
    "2. 在核心事实基础上，补充原理、机制、上下文等解释性内容\n"
    "3. 严格保持单一范畴——一个知识点只聚焦一个概念，不要发散到相关但不属于同一主题的内容\n"
    "4. 不要单纯摘抄原话，要用自己的语言组织、提炼、丰富\n"
    "5. 保持简洁精准，每条约 50-150 字，不做过度的展开\n\n"
    "## Fixed Top-Level Categories\n"
    "programming, mathematics, physics, chemistry, biology, history, literature, art, philosophy, "
    "economics, law, medicine, education, career, life, sports, other\n\n"
    "## Rules\n"
    "1. Path format: level1/level2/level3/level4 (lowercase English, no spaces)\n"
    "2. Deeper content = deeper path (e.g. 'programming/python/web/django')\n"
    "3. Pick the closest fixed top-level, then freely extend sub-levels\n"
    "4. Terms must be standard lowercase English, no mixed-language\n"
    "5. Multiple core topics use & separator (e.g. 'programming&mathematics')\n"
    "6. Unclassifiable -> 'other'\n\n"
    "## Examples\n"
    "- What is Python -> programming/python\n"
    "- Django routing -> programming/python/web/django\n"
    "- Hello -> other\n\n"
    "## Per-Knowledge-Point Categories\n"
    "Each distilled knowledge point may have its own category, "
    "which can be more specific than the overall Q&A category."
)


def _check_duplicate(kp: DistilledPoint) -> tuple[DistilledPoint, bool]:
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


def _distill_and_save(prompt: str, source_question: str, reasoning_log_path: str = "") -> dict:
    """Full pipeline: LLM distill → category normalize → dedup → save.

    Returns {"stored_knowledge_ids": list[int], "category": str} or empty dict if nothing saved.
    """
    existing_cats = get_normalized_categories()
    if existing_cats:
        prompt += (
            f"\n\n已有分类：{existing_cats}\n"
            f"请优先选择最匹配的已有分类，仅当完全不匹配时创建新分类。"
        )

    result = LLM.generate_structured(prompt, DistillOutput, use_language=False, model="deepseek-v4-pro")
    if result is None:
        logger.error("LLM.generate_structured returned None")
        return {}

    for d in result.knowledge_points:
        d.category = normalize_category_str(d.category)
        ensure_category(d.category)

    # Parallel dedup
    new_points = []
    with ThreadPoolExecutor(max_workers=16, thread_name_prefix="dedup") as executor:
        futures = {executor.submit(_check_duplicate, kp): kp
                   for kp in result.knowledge_points}

        for future in as_completed(futures):
            kp, is_duplicate = future.result()
            if not is_duplicate:
                new_points.append(kp)

    if not new_points:
        logger.info("All knowledge points already exist, nothing to store")
        return {}

    knowledge_points = [
        {
            "knowledge_text": kp.knowledge_text,
            "source_question": source_question,
            "category": kp.category,
            "tags": kp.tags,
            "reasoning_log_path": reasoning_log_path,
        }
        for kp in new_points
    ]
    ids = save_knowledge_points_bulk_with_embeddings(knowledge_points)

    return {
        "stored_knowledge_ids": ids,
        "category": result.category,
    }


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
    prompt = (
        _BASE_DISTILL_PROMPT + "\n\n"
        f"Question: {state['user_message']}\n"
        f"Answer: {state['answer']}"
    )
    reasoning_log_path = state.get("reasoning_log_path", "")

    saved = _distill_and_save(
        prompt=prompt,
        source_question=state["user_message"],
        reasoning_log_path=reasoning_log_path,
    )

    if not saved:
        return {}

    return {
        "stored_knowledge_ids": saved["stored_knowledge_ids"],
        "logic_chain": [{
            "node": "store",
            "action": f"存储 {len(saved['stored_knowledge_ids'])} 条知识点",
            "reasoning": f"分类: {saved['category']}, 知识点数: {len(saved['stored_knowledge_ids'])}",
        }],
    }
