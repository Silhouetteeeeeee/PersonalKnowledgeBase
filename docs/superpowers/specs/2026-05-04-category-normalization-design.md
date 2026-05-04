# Category Normalization Design

> **Problem:** LLM generates inconsistent categories for the same topic (e.g., `RAG/ReRank`, `ai/rag`, `RAG/re-ranking`), polluting the category hierarchy.

**Goal:** Ensure that similar topics converge to a consistent category path without restricting the LLM's ability to create new categories when genuinely needed.

**Non-goal:** Semantic merging of different categories (e.g., not auto-merging `ai/rag` into `programming/rag` — the LLM chooses the parent, we only normalize format and hint at existing options).

---

## Approach

Two-pronged: (1) inform the LLM of existing categories via prompt, (2) lightweight format normalization as a safety net.

Only the `store` node is modified — `classify_and_answer`'s category is not persisted, so it requires no changes.

---

## Data Flow

```
store(state):
  1. existing_categories = get_all_categories()       # ← NEW
  2. Inject into distill prompt: "已有分类: a/b, c/d..."
  3. LLM generates DistillOutput with category
  4. category = normalize_category_str(category)        # ← NEW
  5. ensure_category(category)
  6. save_knowledge_points_bulk_with_embeddings(...)
```

---

## Changes

### `storage/models.py` — Two new functions

```python
def normalize_category_str(category: str, max_depth: int = 4) -> str:
    """
    Lightweight format normalization — no semantic mapping.
    - lowercase
    - unify separators (\ | ｜ · > → /)
    - strip whitespace per level
    - remove hyphens/underscores (re-ranker → reranker)
    - truncate to max_depth levels
    """
```

```python
def get_normalized_categories(max_count: int = 20) -> str:
    """
    Returns a formatted string of existing categories for prompt injection,
    e.g. "目前已存在的分类：databases/redis, ai/rag, ...（共 N 个分类）"
    If count > max_count, shows first max_count + "...等 N 个分类"
    """
```

### `agent/nodes/store.py` — Prompt injection + normalization

- In the `store()` function, before calling LLM, fetch existing categories via `get_normalized_categories()`
- Append to the distill prompt: `f"\n目前已存在的分类：{category_list}\n请优先选择最匹配的已有分类，仅当完全不匹配时创建新分类。"`
- After LLM returns `DistillOutput`, call `normalize_category_str(result.category)` before `ensure_category()`

---

## Edge Cases

| Case | Handling |
|------|----------|
| Empty DB (first run) | Prompt has no categories to show; LLM creates freely |
| All categories exhausted (≥20) | Show first 20 + "...等 N 个分类" |
| LLM ignores prompt entirely | `normalize_category_str()` ensures at least format consistency |
| Single-level category (e.g. `testing`) | Treated as-is, no depth issues |
| Already-normalized category | `normalize_category_str()` is idempotent — safe to call multiple times |

---

## Files Changed

| File | Change |
|------|--------|
| `storage/models.py` | + `normalize_category_str()`, + `get_normalized_categories()` |
| `agent/nodes/store.py` | Inject categories into distill prompt; normalize output |

No changes to `classify_and_answer.py`, `agent/graph.py`, or `agent/state.py`.

---

## Testing

| Test | What it verifies |
|------|-----------------|
| `test_normalize_category_str` | lowercase, separator unification, depth truncation, idempotency |
| `test_normalize_category_str_hyphens` | `re-ranker` → `reranker`, `re_ranker` → `reranker` |
| `test_store_with_existing_categories` | Prompt contains "已存在的分类" when categories exist in DB |
| `test_store_empty_db_first_category` | No "已存在的分类" in prompt when DB is empty |
