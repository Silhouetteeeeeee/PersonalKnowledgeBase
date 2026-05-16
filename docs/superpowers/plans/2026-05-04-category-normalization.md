# Category Normalization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ensure LLM-generated knowledge categories converge to consistent names by informing the LLM of existing categories and normalizing format.

**Architecture:** Two lightweight functions in `storage/models.py` (format normalization + category list formatter), plus prompt injection in `store.py`. No new dependencies, no embedding calls, no graph changes.

**Tech Stack:** Python stdlib `re`, existing `storage/models.py` helpers.

---

## File Structure

| File | Change | Responsibility |
|------|--------|---------------|
| `storage/models.py` | Add | `normalize_category_str()` + `get_normalized_categories()` |
| `agent/nodes/store.py` | Modify | Inject existing categories into distill prompt; normalize category |
| `tests/test_storage.py` | Modify | Add category normalization tests |

---

### Task 1: Add normalization helpers to storage/models.py

**Files:**
- Modify: `storage/models.py` (after `ensure_category`, around line 304)

- [ ] **Step 1: Add import and new functions**

Add after `ensure_category()`:

```python
import re


def normalize_category_str(category: str, max_depth: int = 4) -> str:
    """Lightweight format normalization for category strings.

    - lowercase
    - unify separators (\\ | ｜ · > → /)
    - strip whitespace per level
    - remove hyphens/underscores (re-ranker → reranker)
    - truncate to max_depth levels
    - idempotent: normalize(x) == normalize(normalize(x))
    """
    s = category.lower().strip()
    s = re.sub(r'[\\｜|·>]', '/', s)
    parts = [p.strip() for p in s.split('/') if p.strip()]
    parts = [p.replace('-', '').replace('_', '') for p in parts]
    return '/'.join(parts[:max_depth])


def get_normalized_categories(max_count: int = 20) -> str:
    """Return a formatted string of existing categories for prompt injection.

    Returns e.g. "目前已存在的分类：databases/redis, ai/rag（共 2 个分类）"
    Truncates to max_count entries with "...等 N 个分类" suffix.
    """
    cats = get_all_categories()
    if not cats:
        return ""
    display = cats[:max_count]
    suffix = f"（共 {len(cats)} 个分类）" if len(cats) <= max_count else f"…等 {len(cats)} 个分类"
    return f"目前已存在的分类：{', '.join(display)}{suffix}"
```

- [ ] **Step 2: Run existing storage tests**

Run: `python -m pytest tests/test_storage.py -v`
Expected: All 5 tests PASS

- [ ] **Step 3: Commit**

```bash
git add storage/models.py
git commit -m "feat: add normalize_category_str and get_normalized_categories"
```

---

### Task 2: Update store node to inject categories and normalize

**Files:**
- Modify: `agent/nodes/store.py`

- [ ] **Step 1: Update store() with prompt injection + normalization**

Two changes to `store()`:

Change 1: inject existing categories into the distill prompt (around line 55-61):

```python
    logger.info("Distilling knowledge from Q&A...")
    existing_cats = get_normalized_categories()
    prompt = (
        f"Distill the following Q&A into concise, standalone knowledge points.\n\n"
        f"Question: {state['user_message']}\n"
        f"Answer: {state['answer']}"
    )
    if existing_cats:
        prompt += f"\n\n{existing_cats}\n请优先选择最匹配的已有分类，仅当完全不匹配时创建新分类。"
    result = LLM.generate_structured(prompt, DistillOutput, use_language=False)
```

Change 2: normalize category before ensure_category (around line 63):

```python
    result.category = normalize_category_str(result.category)
    ensure_category(result.category)
```

And add the import at the top of the file:

```python
from storage.models import (
    save_knowledge_points_bulk_with_embeddings,
    ensure_category,
    find_similar_knowledge,
    get_normalized_categories,   # ← add
    normalize_category_str,      # ← add
)
```

- [ ] **Step 2: Run existing store tests**

Run: `python -m pytest tests/test_nodes.py::test_store_empty_answer tests/test_nodes.py::test_store_distills_knowledge tests/test_nodes.py::test_store_skips_on_contradiction tests/test_nodes.py::test_store_returns_stored_ids -v`
Expected: All 4 PASS

- [ ] **Step 3: Commit**

```bash
git add agent/nodes/store.py
git commit -m "feat: inject existing categories into distill prompt"
```

---

### Task 3: Add tests for category normalization

**Files:**
- Modify: `tests/test_storage.py`

- [ ] **Step 1: Add normalization tests**

Add after `test_find_similar_knowledge_semantic`:

```python
def test_normalize_category_str():
    from storage.models import normalize_category_str

    # lowercase
    assert normalize_category_str("RAG/ReRank") == "rag/rerank"

    # unify separators
    assert normalize_category_str("databases \\ redis") == "databases/redis"
    assert normalize_category_str("life | health") == "life/health"
    assert normalize_category_str("programming·python") == "programming/python"

    # strip whitespace
    assert normalize_category_str("  ai  /  rag  ") == "ai/rag"

    # remove hyphens and underscores
    assert normalize_category_str("re-ranker") == "reranker"
    assert normalize_category_str("re_ranker") == "reranker"

    # depth truncation
    long = "a/b/c/d/e/f"
    assert normalize_category_str(long, max_depth=4) == "a/b/c/d"
    assert normalize_category_str(long, max_depth=2) == "a/b"

    # idempotent
    raw = "  RAG \\ ReRank-Re_ranking "
    once = normalize_category_str(raw)
    twice = normalize_category_str(once)
    assert once == twice


def test_get_normalized_categories_empty():
    from storage.models import get_normalized_categories

    result = get_normalized_categories()
    assert result == ""


def test_get_normalized_categories_with_data():
    from storage.models import (
        save_knowledge_point,
        get_normalized_categories,
    )

    save_knowledge_point("text1", "q1", "databases/redis", ["redis"])
    save_knowledge_point("text2", "q2", "ai/rag", ["rag"])

    result = get_normalized_categories()
    assert "databases/redis" in result
    assert "ai/rag" in result
    assert "目前已存在的分类" in result
    assert "2 个分类" in result
```

- [ ] **Step 2: Run the new tests**

Run: `python -m pytest tests/test_storage.py::test_normalize_category_str tests/test_storage.py::test_get_normalized_categories_empty tests/test_storage.py::test_get_normalized_categories_with_data -v`
Expected: All 3 PASS

- [ ] **Step 3: Run full test suite**

Run: `python -m pytest tests/ -v`
Expected: All tests pass

- [ ] **Step 4: Commit**

```bash
git add tests/test_storage.py
git commit -m "test: add category normalization tests"
```

---

## Self-Review

**1. Spec coverage:**
- `normalize_category_str()` → Task 1 step 1 ✓
- `get_normalized_categories()` → Task 1 step 1 ✓
- Prompt injection in store → Task 2 step 1 ✓
- Normalization before ensure_category → Task 2 step 1 ✓
- Tests for normalize → Task 3 step 1 ✓
- Tests for prompt injection sign → Task 3 step 1 (test confirms "目前已存在的分类" appears) ✓

**2. Placeholder scan:** No TBD, TODO, or incomplete code. All code blocks are complete.

**3. Type consistency:** `normalize_category_str(category: str, max_depth: int = 4) -> str` is consistently used in both definition (models.py) and call site (store.py). `get_normalized_categories(max_count: int = 20) -> str` matches usage in store.py.

---

## Execution Handoff

Plan complete. Two execution options:

**1. Subagent-Driven (recommended)** — dispatch a fresh subagent per task, review between tasks

**2. Inline Execution** — execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
