# Node Refactoring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the redundant `category` field from the agent graph state and share context-building logic between `classify_and_answer` and `regenerate`.

**Architecture:** Two independent refactors: (1) category removal touches state, 5 node files, and bot.py; (2) context sharing extracts `_build_system_prompt`'s context sections into a shared function in `agent_utils.py` consumed by both `classify_and_answer` and `regenerate`.

**Tech Stack:** Python, LangGraph, Pydantic

---

### Task 1: Remove `category` from AgentState

**Files:**
- Modify: `agent/state.py:10`

- [ ] **Step 1: Remove `category: str` from AgentState**

```python
# agent/state.py
class AgentState(TypedDict):
    user_message: str
    user_id: str
    timestamp: str
    # category: str  ← DELETE this line
    confidence: float
    needs_store: bool
    search_results: list[str]
    stored_knowledge: list[dict]
    stored_knowledge_ids: list[int]
    answer: str
    final_response: str
    reasoning_log_path: str
    contradiction_found: bool
    contradiction_details: str
    search_time: int
```

- [ ] **Step 2: Verify tests still pass (they will fail later when we update them, but state.py alone shouldn't break anything yet)**

Run: `python -c "from agent.state import AgentState; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add agent/state.py
git commit -m "refactor: remove category field from AgentState"
```

---

### Task 2: Remove category from `classify_and_answer`

**Files:**
- Modify: `agent/nodes/classify_and_answer.py`
- Modify: `tests/test_nodes.py:110` (test assertion)

- [ ] **Step 1: Remove `category` from `ClassifyOutput`**

In `ClassifyOutput`, delete the `category` field:
```python
class ClassifyOutput(BaseModel):
    reasoning_trace: str = Field(
        description="Step-by-step reasoning: why this category, why this confidence level, what knowledge was considered"
    )
    # category: str = Field(...  ← DELETE
    answer: str = Field(description="Answer to the question")
    confidence: float = Field(
        description="Confidence from 0.0 to 1.0"
    )
    needs_store: bool = Field(
        description="Whether this Q&A should be stored as knowledge. "
                    "True for factual/educational Q&A, False for casual chat/greetings"
    )
```

- [ ] **Step 2: Remove `category` from `_build_system_prompt`'s classification guide**

Remove the top-level category list (programming, life, education) and subcategory examples:
```python
def _build_system_prompt(state: dict) -> str:
    parts = [
        "你是一个专业的智能问答助手。分析问题并生成准确、有用的回答。",
        "",
        "## 输出要求",
        "- 事实性、教育性问答需要存储（needs_store=true）",
        "- 问候、闲聊、个人观点不需要存储（needs_store=false）",
    ]
    # ... rest stays the same (profile, knowledge, memory)
```

- [ ] **Step 3: Remove `category` from all return dicts in `classify_and_answer`**

In `_fallback_answer` (2 places) and `classify_and_answer` (1 place), remove `"category": ...` entries:

```python
def _fallback_answer(state: dict) -> dict:
    """..."""
    return {
        "answer": "",
        # "category": "unknown", ← DELETE
        "confidence": 0.0,
        "needs_store": False,
        ...
    }
```

And the other return in `_fallback_answer`:
```python
    return {
        # "category": result.category, ← DELETE
        "answer": result.answer,
        "confidence": result.confidence * 0.8,
        ...
    }
```

And in `classify_and_answer`:
```python
    return {
        # "category": structured.category, ← DELETE
        "answer": structured.answer,
        "confidence": structured.confidence,
        ...
    }
```

Also remove `"category"` from the logic_chain entries (3 places, all `"category": ...` lines).

- [ ] **Step 4: Update `test_classify_and_answer` to not assert on `category`**

In `tests/test_nodes.py`, remove the `assert "category" in result` line:
```python
def test_classify_and_answer():
    ...
    result = classify_and_answer({...})
    # assert "category" in result  ← DELETE
    assert "answer" in result
    assert 0 <= result["confidence"] <= 1
    assert isinstance(result["needs_store"], bool)
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_nodes.py::test_classify_and_answer tests/test_nodes.py::test_classify_and_answer_has_reasoning_trace -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add agent/nodes/classify_and_answer.py tests/test_nodes.py
git commit -m "refactor: remove category from classify_and_answer node"
```

---

### Task 3: Remove category from `correct_knowledge`

**Files:**
- Modify: `agent/nodes/correct_knowledge.py`
- Modify: `tests/test_nodes.py` (2 test inputs)

- [ ] **Step 1: Remove category read and use hardcoded `"uncategorized"` for corrected knowledge**

```python
def correct_knowledge(state: dict) -> dict:
    knowledge_ids = state.get("contradiction_knowledge_ids", [])
    correction = state.get("reflection_correction", "")
    # category = state.get("category", "uncategorized")  ← DELETE
    user_message = state.get("user_message", "")
    correction_attempts = state.get("correction_attempts", 0)
    ...
    if correction:
        ids = save_knowledge_points_bulk_with_embeddings([{
            "knowledge_text": correction,
            "source_question": f"[auto-corrected] {user_message}",
            "category": "uncategorized",  # was: category
            "tags": ["auto-corrected"],
            "status": "active",
        }])
```

- [ ] **Step 2: Remove `category` from test inputs**

In `test_nodes.py`, remove `"category": "test"` from both `test_correct_knowledge` and `test_correct_knowledge_increments_counter`:

```python
def test_correct_knowledge():
    ...
    result = correct_knowledge({
        "contradiction_knowledge_ids": [kid],
        "reflection_correction": "New correct fact",
        # "category": "test",  ← DELETE
        "user_message": "test question",
        "correction_attempts": 0,
    })

def test_correct_knowledge_increments_counter():
    ...
    result = correct_knowledge({
        "contradiction_knowledge_ids": [],
        "reflection_correction": "",
        # "category": "test",  ← DELETE
        "user_message": "test",
        "correction_attempts": 1,
    })
```

- [ ] **Step 3: Run tests**

Run: `pytest tests/test_nodes.py::test_correct_knowledge tests/test_nodes.py::test_correct_knowledge_increments_counter -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add agent/nodes/correct_knowledge.py tests/test_nodes.py
git commit -m "refactor: remove category from correct_knowledge node"
```

---

### Task 4: Remove category from `record_error`

**Files:**
- Modify: `agent/nodes/record_error.py`
- Modify: `tests/test_nodes.py` (1 test input)

- [ ] **Step 1: Remove category read and record dict entry**

```python
def record_error(state: dict) -> dict:
    correction_attempts = state.get("correction_attempts", 0)
    user_message = state.get("user_message", "")
    wrong_answer = state.get("answer", "")
    correct_answer = state.get("reflection_correction", "")
    # category = state.get("category", "")  ← DELETE
    contradiction_details = state.get("contradiction_details", "")

    record = {
        "user_message": user_message,
        "wrong_answer": wrong_answer,
        "correct_answer": correct_answer,
        # "category": category,  ← DELETE
        "contradiction_details": contradiction_details,
        "error_type": "hallucination_or_error",
    }
```

- [ ] **Step 2: Remove `category` from test input**

```python
def test_record_error():
    ...
    result = record_error({
        "user_message": "What is X?",
        "answer": "Wrong answer about X",
        "reflection_correction": "Correct answer about X",
        # "category": "test",  ← DELETE
        "contradiction_details": "X is Y, not Z",
        "correction_attempts": 0,
    })
```

- [ ] **Step 3: Run tests**

Run: `pytest tests/test_nodes.py::test_record_error -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add agent/nodes/record_error.py tests/test_nodes.py
git commit -m "refactor: remove category from record_error node"
```

---

### Task 5: Remove category from `store` node return

**Files:**
- Modify: `agent/nodes/store.py:131`
- Modify: `tests/test_nodes.py` (2 test assertions)

- [ ] **Step 1: Remove `"category": result.category` from store's return dict**

```python
    return {
        # "category": result.category,  ← DELETE
        "stored_knowledge_ids": ids,
        "logic_chain": [...],
    }
```

Keep all internal uses of `result.category` inside the function (e.g., `d.category = normalize_category_str(d.category)` and `ensure_category(d.category)` and `"category": kp.category` in the list comprehension).

- [ ] **Step 2: Update `test_store_distills_knowledge`**

Remove the category assertion — store no longer returns category:
```python
def test_store_distills_knowledge():
    from agent.nodes.store import store
    result = store({
        "user_message": "What is Redis persistence?",
        "answer": "Redis supports RDB snapshots and AOF logs for persistence.",
    })
    # assert "category" in result  ← DELETE
    # assert isinstance(result["category"], str)  ← DELETE
    # Instead, verify knowledge was stored by checking stored_knowledge_ids
    assert isinstance(result.get("stored_knowledge_ids"), list)
```

- [ ] **Step 3: Update `test_store_returns_stored_ids`**

```python
def test_store_returns_stored_ids():
    ...
    result = store({...})
    if result:  # May be empty if dedup skips everything
        # assert isinstance(result.get("category"), str)  ← DELETE
        assert isinstance(result.get("stored_knowledge_ids"), list)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_nodes.py::test_store_distills_knowledge tests/test_nodes.py::test_store_returns_stored_ids tests/test_nodes.py::test_store_empty_answer tests/test_nodes.py::test_store_skips_on_contradiction -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add agent/nodes/store.py tests/test_nodes.py
git commit -m "refactor: remove category from store node return"
```

---

### Task 6: Remove category from `bot.py`

**Files:**
- Modify: `server/bot.py:90,127,189-193`
- Modify: `tests/test_graph.py:36`

- [ ] **Step 1: Remove `"category": ""` from initial state dict**

```python
result = await asyncio.to_thread(graph.invoke, {
    "user_message": content,
    "user_id": user_id,
    "session_id": str(session["id"]),
    "message_history": context.get("history_section", ""),
    "episodic_memories": context.get("episodic_section", ""),
    "user_profile": load_profile(user_id),
    "timestamp": "",
    # "category": "",  ← DELETE
    "confidence": 0.0,
    ...
})
```

- [ ] **Step 2: Remove `category` from `_save_turn` call and method**

Change the call at line 126-128:
```python
asyncio.create_task(self._save_turn(
    session["id"], user_id, content, answer_text,
    # result.get("category", ""),  ← REMOVE this arg
))
```

Change the method signature at line 189:
```python
async def _save_turn(self, session_id: int, user_id: str, user_msg: str, asst_msg: str):
    """Persist conversation turn asynchronously (non-blocking)."""
    try:
        await asyncio.to_thread(message_history.add_message, session_id, user_id, "user", user_msg)
        await asyncio.to_thread(message_history.add_message, session_id, user_id, "assistant", asst_msg)
        ...
```

- [ ] **Step 3: Remove `category` assertion from `test_graph.py`**

```python
def test_graph_short_circuit():
    ...
    # assert "category" in result  ← DELETE
```

- [ ] **Step 4: Run graph test**

Run: `pytest tests/test_graph.py::test_graph_short_circuit -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add server/bot.py tests/test_graph.py
git commit -m "refactor: remove category from bot.py"
```

---

### Task 7: Add `build_context_block()` shared utility

**Files:**
- Modify: `agent/utils/agent_utils.py`

- [ ] **Step 1: Add `build_context_block` function**

```python
def build_context_block(state: dict) -> str:
    """Build shared context block with user profile, stored knowledge,
    message history, and episodic memories. Returns empty string if
    none available."""
    parts = []

    profile = state.get("user_profile", {})
    if profile and any(v for v in profile.values() if v):
        profile_summary = _summarize_profile(profile)
        parts.append("")
        parts.append("## 用户画像")
        parts.append(profile_summary)

    if state.get("stored_knowledge"):
        parts.append("")
        parts.append("## 相关知识")
        for k in state["stored_knowledge"]:
            parts.append(f"- {k['knowledge_text']}")

    if state.get("message_history"):
        parts.append("")
        parts.append("## 近期对话历史")
        for msg in state["message_history"]:
            if isinstance(msg, str):
                parts.append(msg)
            elif isinstance(msg, dict):
                role = "用户" if msg.get("role") == "user" else "助手"
                content = msg.get("content", "")
                parts.append(f"{role}: {content}")

    if state.get("episodic_memories"):
        if isinstance(state["episodic_memories"], list) and state["episodic_memories"]:
            parts.append("")
            parts.append("## 历史相关记忆")
            parts.extend(state["episodic_memories"] if all(isinstance(m, str) for m in state["episodic_memories"]) else [str(m) for m in state["episodic_memories"]])

    return "\n".join(parts)
```

Also need to move the `_summarize_profile` helper from `classify_and_answer.py` to `agent_utils.py` since `build_context_block` depends on it:

```python
def _summarize_profile(profile: dict) -> str:
    """Flatten profile dict into a readable summary string."""
    lines = []
    for section, data in profile.items():
        if section == "updated_at" or not data:
            continue
        if isinstance(data, dict):
            for k, v in data.items():
                if v:
                    lines.append(f"- {section}.{k}: {v}")
        elif isinstance(data, list):
            if data:
                lines.append(f"- {section}: {', '.join(str(x) for x in data)}")
    return "\n".join(lines) if lines else "（暂无用户画像信息）"
```

- [ ] **Step 2: Write a quick unit test for `build_context_block`**

Add to `tests/test_nodes.py` or create `tests/test_agent_utils.py`:

```python
def test_build_context_block_empty():
    from agent.utils.agent_utils import build_context_block
    result = build_context_block({})
    assert result == ""


def test_build_context_block_with_profile():
    from agent.utils.agent_utils import build_context_block
    result = build_context_block({
        "user_profile": {"basic": {"name": "Alice"}},
    })
    assert "Alice" in result
    assert "用户画像" in result
```

If creating a new test file, create `tests/test_agent_utils.py`:
```python
"""Tests for agent utility functions."""

from agent.utils.agent_utils import build_context_block


def test_build_context_block_empty():
    result = build_context_block({})
    assert result == ""


def test_build_context_block_with_profile():
    result = build_context_block({
        "user_profile": {"basic": {"name": "Alice"}},
    })
    assert "Alice" in result
    assert "用户画像" in result
```

- [ ] **Step 3: Run the new test**

Run: `pytest tests/test_agent_utils.py -v` (or `tests/test_nodes.py -k build_context -v`)
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add agent/utils/agent_utils.py tests/test_agent_utils.py
git commit -m "feat: add build_context_block shared utility"
```

---

### Task 8: Update `classify_and_answer._build_system_prompt` to use shared context

**Files:**
- Modify: `agent/nodes/classify_and_answer.py`

- [ ] **Step 1: Replace `_build_system_prompt` to use `build_context_block`**

```python
from agent.utils.agent_utils import build_context_block

def _build_system_prompt(state: dict) -> str:
    """Build the system prompt with classification rules and optional context."""
    parts = [
        "你是一个专业的智能问答助手。分析问题并生成准确、有用的回答。",
        "",
        "## 分类指南",
        "- programming: 编程、软件开发、算法等技术问题",
        "- life: 日常生活、健康、饮食等非技术问题",
        "- education: 学术、学习、考试等教育相关问题",
        "- 使用更细粒度的子分类，如 'programming/python'、'life/health'",
        "- 不要使用 personal 分类（个人信息由系统独立管理）",
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

Also remove `_summarize_profile` from this file since it's now in `agent_utils.py`.

- [ ] **Step 2: Run tests**

Run: `pytest tests/test_nodes.py::test_classify_and_answer tests/test_nodes.py::test_classify_and_answer_has_reasoning_trace -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add agent/nodes/classify_and_answer.py
git commit -m "refactor: use shared build_context_block in classify_and_answer"
```

---

### Task 9: Update `regenerate` to use shared context

**Files:**
- Modify: `agent/nodes/regenerate.py`

- [ ] **Step 1: Update regenerate prompt to include context block**

```python
import logging

from pydantic import BaseModel, Field

from agent.utils.agent_utils import build_context_block
from agent.utils.llm import LLM

logger = logging.getLogger(__name__)


class RegenerateOutput(BaseModel):
    reasoning_trace: str = Field(
        description="Step-by-step reasoning: how the web search results inform the answer, any corrections from the original answer"
    )
    answer: str = Field(description="The regenerated answer based on web search results")


def regenerate(state: dict) -> dict:
    search_text = "\n\n".join(state.get("search_results", []))
    if not search_text:
        logger.info("No search results, keeping original answer")
        answer = state.get("answer", "")
        return {
            "answer": answer,
            "logic_chain": [{
                "node": "regenerate",
                "action": "无搜索结果，保留原答案",
                "reasoning": "Web search returned no results, keeping original answer unchanged",
            }],
        }

    logger.info("Regenerating answer with %d search results", len(state.get("search_results", [])))

    context = build_context_block(state)

    prompt = (
        f"{context}\n\n"
        f"## 网络搜索结果\n{search_text}\n\n"
        f"## 用户问题\n{state['user_message']}\n\n"
        f"## 原答案\n{state.get('answer', '')}\n\n"
        f"请基于搜索结果的真实信息，结合上述背景，生成一个准确且风格一致的答案。"
    )
    result = LLM.generate_structured(prompt, RegenerateOutput, use_language=False)

    logger.info("Regenerated answer: %s", result.answer[:80])
    return {
        "answer": result.answer,
        "logic_chain": [{
            "node": "regenerate",
            "action": "基于搜索结果重新生成答案",
            "reasoning": result.reasoning_trace,
        }],
    }
```

- [ ] **Step 2: Run tests**

Run: `pytest tests/test_nodes.py::test_regenerate_empty_search tests/test_nodes.py::test_regenerate_with_search -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add agent/nodes/regenerate.py
git commit -m "feat: regenerate now uses shared context block for consistent style"
```

---

### Task 10: Final verification — run all tests

- [ ] **Step 1: Run full test suite**

Run: `pytest tests/ -v`
Expected: All tests pass

- [ ] **Step 2: Run graph integration test**

Run: `pytest tests/test_graph.py -v`
Expected: PASS

- [ ] **Step 3: Commit any final test fixes**

```bash
git add -A
git commit -m "test: update assertions after category removal"
```
