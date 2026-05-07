# Node Refactoring: Category Removal & Context Sharing

## Problem

Two independent issues in the agent graph node design:

**Issue 1 — Redundant `category` field:** `classify_and_answer` generates a `category` that is never used for persistence (the `store` node generates its own category independently). The field flows redundantly through `correct_knowledge` and `record_error` only for logging, and into `bot.py`'s message history save. This is unnecessary complexity.

**Issue 2 — Inconsistent answer quality from `regenerate`:** `regenerate` builds its prompt with only web search results + question + original answer, while `classify_and_answer` enriches its prompt with user profile, stored knowledge, conversation history, and episodic memories. This causes regenerated answers to differ in style, personalization, and context-awareness.

## Scope

**In-scope:**
- Remove `category` from graph state and all node outputs
- Extract context-building into a shared utility
- Align `regenerate`'s prompt with `classify_and_answer`'s context richness

**Out-of-scope:**
- Changing `store` node's internal category generation (knowledge classification is unchanged)
- Any other graph restructuring

## Changes

### 1. Category removal

| File | Change |
|------|--------|
| `agent/state.py` | Remove `category: str` field |
| `agent/nodes/classify_and_answer.py` | Remove `category` from `ClassifyOutput` schema and all return dicts |
| `agent/nodes/correct_knowledge.py` | Remove `state.get("category")` read and `"category"` from return |
| `agent/nodes/record_error.py` | Remove `state.get("category")` read and `"category"` from return |
| `agent/nodes/store.py` | Remove `"category": result.category` from return dict (keep `ensure_category()` and per-knowledge-point `kp.category` — these are internal to storage) |
| `server/bot.py` | Remove `"category": ""` from initial state; remove `category` param from `_save_turn`; `result.get("category", "")` → `""` |

### 2. Context sharing via `build_context_block()`

**New function in `agent/utils/agent_utils.py`:**

```python
def build_context_block(state: dict) -> str:
    """Build shared context block: user profile, stored knowledge,
    message history, episodic memories. Returns empty string if none
    available."""
```

Extracted from `classify_and_answer._build_system_prompt()` — moves the profile/knowledge/history/episodic sections into this shared function.

**Modified `classify_and_answer._build_system_prompt()`:**

```python
def _build_system_prompt(state):
    parts = [
        "你是一个专业的智能问答助手...",
        "## 分类指南",
        ...
        "## 输出要求",
    ]
    context = build_context_block(state)
    if context:
        parts.append(context)
    return "\n".join(parts)
```

**Modified `regenerate.regenerate()`:**

```python
def regenerate(state):
    search_text = "\n\n".join(state.get("search_results", []))
    if not search_text:
        # keep original, unchanged
    
    context = build_context_block(state)
    prompt = f"""
{context}

Web search results:
{search_text}

Question: {state['user_message']}
Original answer: {state.get('answer', '')}

请基于搜索结果的真实信息，结合上述背景，生成一个准确且风格一致的答案。
"""
    result = LLM.generate_structured(prompt, RegenerateOutput, use_language=False)
    return {"answer": result.answer, ...}
```

## Testing

- Existing tests should pass with no category-related assertions broken
- No new tests needed — this is a refactoring with no behavioral change for the user-facing output
