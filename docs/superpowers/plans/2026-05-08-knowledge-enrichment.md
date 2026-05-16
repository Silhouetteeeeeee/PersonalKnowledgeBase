# Knowledge Enrichment Enhancement Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enrich stored knowledge points with explanatory context (not just verbatim extraction), and guide answer generation to leverage knowledge associatively.

**Architecture:** Two prompt-only changes — update `_BASE_DISTILL_PROMPT` to instruct the LLM to enrich knowledge points, and add a "知识拓展" section to `classify_and_answer`'s system prompt to encourage associative thinking.

**Tech Stack:** Prompt engineering (no code changes)

---

## File Structure

| File | Change | Responsibility |
|------|--------|---------------|
| `agent/nodes/store.py` | Modify | Update `_BASE_DISTILL_PROMPT` role + enrichment rules |
| `agent/nodes/classify_and_answer.py` | Modify | Add "知识拓展" section to `_build_system_prompt` |

---

### Task 1: Enrich `_BASE_DISTILL_PROMPT` with knowledge enrichment instructions

**Files:**
- Modify: `agent/nodes/store.py:39-59` — `_BASE_DISTILL_PROMPT` constant

- [ ] **Step 1: Update `_BASE_DISTILL_PROMPT`**

Change the role and add enrichment rules:

```python
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
```

Key changes from current version:
- Role changed from "knowledge classification expert" to "knowledge refinement expert"
- First `## Role` sentence changed from "Categorize the following content with precise hierarchical paths" to "Extract key knowledge points from the content, enrich them with explanatory context, and classify them precisely"
- Added new `## 知识提炼要求` section with 5 rules about enrichment and single-topic scope

- [ ] **Step 2: Run store tests to verify**

Run: `python -m pytest tests/test_nodes.py::test_store_empty_answer tests/test_nodes.py::test_store_distills_knowledge tests/test_nodes.py::test_store_skips_on_contradiction tests/test_nodes.py::test_store_returns_stored_ids -v`

Expected: All 4 PASS (prompt change should not affect test logic)

- [ ] **Step 3: Commit**

```bash
git add agent/nodes/store.py
git commit -m "feat: enrich _BASE_DISTILL_PROMPT with knowledge refinement instructions"
```

---

### Task 2: Add knowledge extension guidance to classify_and_answer

**Files:**
- Modify: `agent/nodes/classify_and_answer.py:33-51` — `_build_system_prompt()` function

- [ ] **Step 1: Add "知识拓展" section to system prompt**

Change `_build_system_prompt` from:

```python
def _build_system_prompt(state: dict) -> str:
    """Build the system prompt with classification rules and optional context."""
    parts = [
        "你是一个专业的智能问答助手。分析问题并生成准确、有用的回答。",
        "",
        "## 网络搜索",
        "- 仅在完全不知道答案或需要最新信息时搜索",
        "- 最多搜索 1 次，搜索失败则用已有知识回答",
        "",
        "## 输出要求",
        "- 事实性、教育性问答需要存储（needs_store=true）",
        "- 问候、闲聊、个人观点不需要存储（needs_store=false）",
    ]

    context = build_context_block(state)
    if context:
        parts.append(context)

    return "\n".join(parts)
```

To:

```python
def _build_system_prompt(state: dict) -> str:
    """Build the system prompt with classification rules and optional context."""
    parts = [
        "你是一个专业的智能问答助手。分析问题并生成准确、有用的回答。",
        "",
        "## 网络搜索",
        "- 仅在完全不知道答案或需要最新信息时搜索",
        "- 最多搜索 1 次，搜索失败则用已有知识回答",
        "",
        "## 知识拓展",
        "- 基于存储的知识进行关联思考，帮助用户建立更完整的知识网络",
        "- 可以补充相关知识点、对比差异、联系实际应用场景",
        "- 如果用户问题涉及多个知识点，可以串联起来给出综合性的回答",
        "- 不要编造不存在的内容，所有拓展需基于已有知识的合理延伸",
        "",
        "## 输出要求",
        "- 事实性、教育性问答需要存储（needs_store=true）",
        "- 问候、闲聊、个人观点不需要存储（needs_store=false）",
    ]

    context = build_context_block(state)
    if context:
        parts.append(context)

    return "\n".join(parts)
```

- [ ] **Step 2: Run tests to verify**

Run: `python -m pytest tests/test_nodes.py::test_classify_and_answer -v`

Expected: PASS (prompt change should not affect test logic)

- [ ] **Step 3: Commit**

```bash
git add agent/nodes/classify_and_answer.py
git commit -m "feat: add knowledge extension guidance to classify_and_answer prompt"
```

---

## Self-Review

**1. Spec coverage:** Both spec requirements covered — Task 1 handles storage-side enrichment (`_BASE_DISTILL_PROMPT`), Task 2 handles query-side extension guidance (`classify_and_answer` system prompt). No gaps.

**2. Placeholder scan:** No TBD, TODO, or "implement later." Complete code in every step.

**3. Type consistency:** No type changes — prompt-only modifications. No schema, function signatures, or interfaces changed.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-08-knowledge-enrichment.md`. Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
