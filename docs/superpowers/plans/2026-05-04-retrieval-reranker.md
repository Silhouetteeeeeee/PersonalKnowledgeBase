# Reranker Retrieval Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add cross-encoder reranker to the retrieval pipeline to filter out semantically close but irrelevant knowledge.

**Architecture:** Vector search recall expanded from top 5 to top 20, then `BAAI/bge-reranker-v2-m3` scores each (query, doc) pair, re-ranks by score, returns top 5. Reranker is a lazy-loaded singleton in `storage/models.py`, same pattern as the existing embedder.

**Tech Stack:** `FlagEmbedding` + `BAAI/bge-reranker-v2-m3`

---

## File Structure

| File | Change | Responsibility |
|------|--------|---------------|
| `storage/models.py` | Add | `_get_reranker()` singleton + `rerank_knowledge()` function |
| `agent/nodes/retrieve.py` | Modify | Vector search limit 20, call reranker, fallback logic |
| `tests/test_nodes.py` | Modify | Add rerank tests (normal, fallback, skip) |
| `requirements.txt` | Modify | Add `FlagEmbedding` |

---

### Task 1: Install dependency and add reranker to storage/models.py

**Files:**
- Modify: `storage/models.py`
- Modify: `requirements.txt`

- [ ] **Step 1: Add FlagEmbedding to requirements.txt**

```
# In requirements.txt, add:
FlagEmbedding>=1.3.0
```

- [ ] **Step 2: Install the dependency**

Run: `pip install -r requirements.txt`
Expected: FlagEmbedding and its dependencies (torch, transformers) install successfully

- [ ] **Step 3: Add reranker singleton and rerank function to storage/models.py**

Add after the embedder section (after `_get_embedder`):

```python
_reranker: Optional[any] = None
_reranker_lock = threading.Lock()


def _get_reranker():
    global _reranker
    if _reranker is None:
        with _reranker_lock:
            if _reranker is None:
                from FlagEmbedding import FlagReranker
                logger.info("Loading reranker model (BAAI/bge-reranker-v2-m3)...")
                t0 = time.time()
                _reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=False)
                logger.info("Reranker model loaded in %.2fs", time.time() - t0)
    return _reranker


def rerank_knowledge(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    """Rerank candidate knowledge points using a cross-encoder model.

    Takes (query, knowledge_text) pairs through the reranker to get relevance
    scores, then returns top_k candidates sorted by score descending.
    """
    pairs = [(query, doc["knowledge_text"]) for doc in candidates]
    model = _get_reranker()
    scores = model.compute_score(pairs)

    scored = list(zip(candidates, scores))
    scored.sort(key=lambda x: x[1], reverse=True)

    results = []
    for doc, score in scored[:top_k]:
        doc["relevance_score"] = score
        results.append(doc)
    return results
```

- [ ] **Step 4: Run tests to verify no regression**

Run: `python -m pytest tests/test_storage.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add requirements.txt storage/models.py
git commit -m "feat: add cross-encoder reranker to storage/models"
```

---

### Task 2: Update retrieve node to use reranker

**Files:**
- Modify: `agent/nodes/retrieve.py`

- [ ] **Step 1: Rewrite retrieve.py**

```python
import logging

from storage.models import search_knowledge_points_semantic, search_knowledge_points, rerank_knowledge

logger = logging.getLogger(__name__)


def retrieve(state: dict) -> dict:
    query = state["user_message"]
    logger.info("Retrieving knowledge for: '%s'", query[:40])

    # Step 1: Vector search (recall) — expand limit for reranker candidates
    try:
        candidates = search_knowledge_points_semantic(query, limit=20)
    except Exception as e:
        logger.warning("Semantic search failed: %s, falling back to keyword", e)
        candidates = []

    # Step 2: Cross-encoder rerank (precision)
    if len(candidates) > 3:
        try:
            results = rerank_knowledge(query, candidates, top_k=5)
            logger.info("Reranked %d candidates → %d results", len(candidates), len(results))
        except Exception as e:
            logger.warning("Reranker failed: %s, using vector ordering", e)
            results = candidates[:5]
    elif candidates:
        logger.info("Only %d candidates, skipping rerank", len(candidates))
        results = candidates[:5]
    else:
        logger.info("Semantic search returned no results, trying keyword fallback")
        results = search_knowledge_points(query, limit=5)

    return {"stored_knowledge": results}
```

- [ ] **Step 2: Run existing retrieve tests**

Run: `python -m pytest tests/test_nodes.py::test_retrieve_no_results tests/test_nodes.py::test_retrieve_with_results -v`
Expected: Both PASS

- [ ] **Step 3: Commit**

```bash
git add agent/nodes/retrieve.py
git commit -m "feat: integrate reranker into retrieve node"
```

---

### Task 3: Add rerank tests

**Files:**
- Modify: `tests/test_nodes.py`

Add three new tests after `test_retrieve_with_results`.

- [ ] **Step 1: Add test_rerank_reorders_results**

```python
def test_retrieve_reranks_when_enough_candidates(monkeypatch):
    """Verify that retrieve calls reranker when >3 candidates exist."""
    from agent.nodes.retrieve import retrieve

    fake_candidates = [
        {"knowledge_text": f"doc {i}", "distance": i * 0.1}
        for i in range(10)
    ]

    def mock_semantic(*args, **kwargs):
        return fake_candidates

    def mock_rerank(query, candidates, top_k):
        # Reverse order to verify reranking happens
        scored = list(reversed(candidates))
        for doc in scored[:top_k]:
            doc["relevance_score"] = 1.0
        return scored[:top_k]

    monkeypatch.setattr("agent.nodes.retrieve.search_knowledge_points_semantic", mock_semantic)
    monkeypatch.setattr("agent.nodes.retrieve.rerank_knowledge", mock_rerank)

    result = retrieve({"user_message": "test query"})
    assert len(result["stored_knowledge"]) == 5
    # Should be reversed order (doc 9, doc 8, ... doc 5)
    assert result["stored_knowledge"][0]["knowledge_text"] == "doc 9"
    assert "relevance_score" in result["stored_knowledge"][0]
```

- [ ] **Step 2: Add test_retrieve_fallback_when_few_candidates**

```python
def test_retrieve_skips_rerank_when_few_candidates(monkeypatch):
    """Verify that retrieve skips reranker when ≤3 candidates exist."""
    from agent.nodes.retrieve import retrieve

    fake_candidates = [
        {"knowledge_text": f"doc {i}", "distance": i * 0.1}
        for i in range(3)
    ]

    monkeypatch.setattr("agent.nodes.retrieve.search_knowledge_points_semantic", lambda *a, **kw: fake_candidates)
    monkeypatch.setattr("agent.nodes.retrieve.rerank_knowledge", lambda *a, **kw: (_ for _ in ()).throw(Exception("should not be called")))

    result = retrieve({"user_message": "test"})
    assert len(result["stored_knowledge"]) == 3
```

- [ ] **Step 3: Add test_retrieve_fallback_when_reranker_fails**

```python
def test_retrieve_fallback_when_reranker_fails(monkeypatch):
    """Verify that retrieve falls back to vector ordering when reranker fails."""
    from agent.nodes.retrieve import retrieve

    fake_candidates = [
        {"knowledge_text": f"doc {i}", "distance": i * 0.1}
        for i in range(10)
    ]

    monkeypatch.setattr("agent.nodes.retrieve.search_knowledge_points_semantic", lambda *a, **kw: fake_candidates)
    monkeypatch.setattr("agent.nodes.retrieve.rerank_knowledge", lambda *a, **kw: (_ for _ in ()).throw(Exception("reranker crash")))

    result = retrieve({"user_message": "test"})
    assert len(result["stored_knowledge"]) == 5
    # Should preserve original vector order (doc 0, doc 1, ... doc 4)
    assert result["stored_knowledge"][0]["knowledge_text"] == "doc 0"
```

- [ ] **Step 4: Run new tests**

Run: `python -m pytest tests/test_nodes.py::test_retrieve_reranks_when_enough_candidates tests/test_nodes.py::test_retrieve_skips_rerank_when_few_candidates tests/test_nodes.py::test_retrieve_fallback_when_reranker_fails -v`
Expected: All 3 PASS

- [ ] **Step 5: Run full test suite**

Run: `python -m pytest tests/ -v`
Expected: All tests pass (new rerank tests + existing tests)

- [ ] **Step 6: Commit**

```bash
git add tests/test_nodes.py
git commit -m "test: add rerank tests for retrieve node"
```

---

## Self-Review

**1. Spec coverage:**
- `_get_reranker()` singleton → Task 1
- `rerank_knowledge()` function → Task 1
- retrieve.py rerank integration + fallback → Task 2
- Normal rerank test → Task 3 Step 1
- Skip when ≤3 candidates → Task 3 Step 2
- Reranker failure fallback → Task 3 Step 3
- Vector limit 20 → Task 2 Step 1

**2. Placeholder scan:** No TBD, TODO, or incomplete code blocks. All implementations complete.

**3. Type consistency:** `rerank_knowledge(query: str, candidates: list[dict], top_k: int = 5)` is consistently used in both models.py (definition) and retrieve.py (call site). Return type `list[dict]` matches what retrieve returns in `stored_knowledge`.

---

## Execution Handoff

Plan complete. Two execution options:

**1. Subagent-Driven (recommended)** — dispatch a fresh subagent per task, review between tasks

**2. Inline Execution** — execute tasks in this session using executing-plans

**Which approach?**
