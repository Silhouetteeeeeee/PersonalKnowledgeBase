# Knowledge Distillation Prompt Unification

## Problem

`server/bot.py:_process_and_store_file` and `agent/nodes/store.py:store()` both extract knowledge points via LLM structured output, but use different prompts and inconsistent processing pipelines:

| Aspect | `store()` | `_process_and_store_file()` |
|--------|-----------|-----------------------------|
| Prompt | Detailed (role, categories, rules, examples) | Minimal ("提炼为知识点") |
| Per-point category | Yes (`DistilledPoint.category`) | No (uses `result.category` for all) |
| Dedup | Yes (`_check_duplicate`) | No |
| Category normalization | Yes | No |

## Design

### Shared pipeline function

Add to `agent/nodes/store.py`:

- `_BASE_DISTILL_PROMPT` — shared prompt prefix (role, category list, rules, examples)
- `_distill_and_save(prompt, source_question, reasoning_log_path="") -> dict` — full pipeline:
  1. LLM `generate_structured(prompt, DistillOutput)`
  2. Normalize each `DistilledPoint.category` via `normalize_category_str`
  3. `ensure_category()` for each point's category
  4. Parallel dedup via `_check_duplicate` + `ThreadPoolExecutor`
  5. `save_knowledge_points_bulk_with_embeddings()`
  6. Return `{"stored_knowledge_ids": ids}`

### Changes

| File | Change |
|------|--------|
| `agent/nodes/store.py` | Extract `_BASE_DISTILL_PROMPT`; add `_distill_and_save()`; refactor `store()` to use it |
| `server/bot.py` | Import and use `_BASE_DISTILL_PROMPT` and `_distill_and_save()`; fix per-point category |

### Unchanged

- `DistillOutput`, `DistilledPoint` models
- File hash dedup, file saving, file records in `_process_and_store_file`
- `store()` logic_chain format, tests
