# Knowledge Distillation Prompt Unification Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Unify the two different knowledge distillation prompts (Q&A path in `store()` vs file path in `_process_and_store_file()`) into a shared `_BASE_DISTILL_PROMPT` + `_distill_and_save()` pipeline.

**Architecture:** Extract the classification prompt prefix and the full distill→dedup→save pipeline into shared functions in `store.py`. Both callers use the same base prompt with their own content suffix, and call the same pipeline function.

**Tech Stack:** LangChain, Pydantic, ThreadPoolExecutor

---

## File Structure

| File | Change | Responsibility |
|------|--------|---------------|
| `agent/nodes/store.py` | Modify | Add `_BASE_DISTILL_PROMPT`, `_distill_and_save()`; refactor `store()` |
| `server/bot.py:212-280` | Modify | Use shared prompt + pipeline, fix per-point category |

---

### Task 1: Refactor store.py — add shared prompt and pipeline function

**Files:**
- Modify: `agent/nodes/store.py`

**Changes:**
1. Extract `_BASE_DISTILL_PROMPT` constant (everything before the Q&A lines in current prompt)
2. Add `_distill_and_save(prompt, source_question, reasoning_log_path="") -> dict`:
   - Calls `LLM.generate_structured(prompt, DistillOutput, use_language=False)`
   - Normalizes each point's category via `normalize_category_str`
   - Calls `ensure_category()` per point
   - Parallel dedup via `_check_duplicate` + `ThreadPoolExecutor`
   - Calls `save_knowledge_points_bulk_with_embeddings()`
   - Returns `{"stored_knowledge_ids": ids, "category": result.category}` (category needed for store()'s logic_chain)
3. Refactor `store()` to use both

- [ ] **Step 1: Write the refactored `store.py`**

```python
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
        description="Knowledge points distilled from the Q&A"
    )


_BASE_DISTILL_PROMPT = (
    "## Role\n"
    "You are a knowledge classification expert. Categorize the following Q&A with precise hierarchical paths.\n\n"
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

    result = LLM.generate_structured(prompt, DistillOutput, use_language=False)

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
```

- [ ] **Step 2: Run store tests to verify**

Run: `python -m pytest tests/test_nodes.py::test_store_empty_answer tests/test_nodes.py::test_store_distills_knowledge tests/test_nodes.py::test_store_skips_on_contradiction tests/test_nodes.py::test_store_returns_stored_ids -v`

Expected: All 4 PASS

- [ ] **Step 3: Commit**

```bash
git add agent/nodes/store.py
git commit -m "refactor: extract shared _BASE_DISTILL_PROMPT and _distill_and_save pipeline"
```

---

### Task 2: Refactor _process_and_store_file to use shared pipeline

**Files:**
- Modify: `server/bot.py` (~line 212-280)

**Changes:**
1. Import `_BASE_DISTILL_PROMPT` and `_distill_and_save` from `agent.nodes.store` (lazy, inside function)
2. Build prompt using `_BASE_DISTILL_PROMPT` + file-specific content
3. Call `_distill_and_save()` instead of manual LLM + save
4. Read category from saved result instead of `result.category`
5. Remove unused `DistillOutput` import

- [ ] **Step 1: Write the refactored `_process_and_store_file`**

Replace function body after text extraction (line 251 onward):

```python
    # LLM 蒸馏为知识点（使用与 store 节点相同的提示词模板）
    from agent.nodes.store import _BASE_DISTILL_PROMPT, _distill_and_save

    prompt = (
        _BASE_DISTILL_PROMPT + "\n\n"
        f"请将以下从文件「{filename}」中提取的文字内容提炼为知识点。"
        f"如果内容涉及多个主题，请分多条知识点存储，并为每条添加合适的标签。\n\n"
        f"内容：\n{text}"
    )
    saved = _distill_and_save(
        prompt=prompt,
        source_question=f"[文件] {filename}",
    )

    if not saved:
        return f"未能从文件「{filename}」中提取到知识点。"

    # 记录文件处理记录
    save_file_record(filename, ext, file_hash, text, saved["stored_knowledge_ids"], user_id)

    reply = (
        f"已从文件「{filename}」中提取并存储了 {len(saved['stored_knowledge_ids'])} 条知识点\n"
        f"分类：{saved['category']}"
    )
    logger.info("File processed: %s → %d points in '%s'", filename, len(saved['stored_knowledge_ids']), saved['category'])
    return reply
```

Also remove the unused `DistillOutput` import from the imports block:

```python
    # Remove this line:
    from agent.nodes.store import DistillOutput
```

- [ ] **Step 2: Run full test suite to verify**

Run: `python -m pytest tests/ -v`
Expected: Previously passing tests still pass (92 pass, 1 pre-existing failure)

- [ ] **Step 3: Commit**

```bash
git add server/bot.py
git commit -m "refactor: _process_and_store_file uses shared distillation pipeline"
```

---

## Self-Review

**1. Spec coverage:** All spec requirements covered: `_BASE_DISTILL_PROMPT` extracted, `_distill_and_save()` pipeline handles LLM→normalize→ensure_category→dedup→save, `store()` refactored to use it, `_process_and_store_file()` refactored to use it with per-point categories fixed.

**2. Placeholder scan:** No TBD, TODO, or "implement later." All code is complete.

**3. Type consistency:** `_distill_and_save()` returns `{"stored_knowledge_ids": ids, "category": str}`. `store()` and `_process_and_store_file()` both consume this consistently. `store()` adds `logic_chain` on top. No type mismatches.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-08-knowledge-distillation-unification.md`. Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
