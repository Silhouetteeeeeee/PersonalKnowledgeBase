# Knowledge Enrichment Enhancement

## Problem

Current knowledge distillation extracts verbatim content from Q&A or files without enriching it. Knowledge points like "Redis supports RDB snapshots" are stored as-is, lacking explanatory context (mechanism, principles, trade-offs). This limits the learning value when users retrieve these points later.

Additionally, the answer generation (`classify_and_answer`) does not encourage leveraging stored knowledge for associative thinking or cross-topic connections.

## Design

### Storage enrichment — `_BASE_DISTILL_PROMPT`

Change the role from "knowledge classification expert" to "knowledge refinement expert" with explicit enrichment rules:

- Each knowledge point must be self-contained and independently understandable
- Supplement core facts with explanatory context (principles, mechanism, background)
- **Strict single-topic scope** — one concept per point, no cross-topic drifting
- ~50-150 characters per point (Chinese), concise but substantive
- Not verbatim extraction — restructure and enrich in own words

### Query extension — `classify_and_answer` system prompt

Add a "知识拓展" (Knowledge Extension) section that guides the LLM to:

- Think associatively based on stored knowledge
- Connect related knowledge points and compare differences
- Tie to practical application scenarios
- Never fabricate — all extensions must be reasonable extrapolations of existing knowledge

## Files Changed

| File | Change |
|------|--------|
| `agent/nodes/store.py` | `_BASE_DISTILL_PROMPT` — new role description + enrichment rules |
| `agent/nodes/classify_and_answer.py` | `_build_system_prompt()` — add "知识拓展" section |

## Unchanged

- `DistilledPoint` / `DistillOutput` / `ClassifyOutput` schemas
- `_distill_and_save()` pipeline logic
- `store()` / `_process_and_store_file()` flow
- All tests
