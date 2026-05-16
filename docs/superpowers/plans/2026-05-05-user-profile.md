# User Profile System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Separate mutable personal information from the immutable knowledge base, storing user profile in a JSON file with automatic extraction and backup history.

**Architecture:** New `storage/profile.py` handles JSON load/save/backup. New `update_profile` node extracts personal info from conversation via LLM. Parallel fan-out from `classify_and_answer` to both `update_profile` and `fact_check`. Profile is injected into classify prompt for personalization.

**Tech Stack:** Python stdlib (`json`, `shutil`, `Path`, `datetime`), existing LLM infrastructure.

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `storage/profile.py` | Create | Profile load/save/backup/field-update |
| `agent/nodes/update_profile.py` | Create | LLM extracts personal info from conversation |
| `agent/state.py` | Modify | Add `user_profile: dict` field |
| `agent/nodes/classify_and_answer.py` | Modify | Inject profile into system prompt; simplify prompt |
| `agent/nodes/store.py` | Modify | Remove personal extraction instructions; simplify prompt |
| `agent/graph.py` | Modify | Add `update_profile` node; parallel fan-out |
| `tests/test_profile.py` | Create | Tests for load/save/backup/update |
| `tests/test_update_profile.py` | Create | Tests for profile extraction |

---

### Task 1: Create storage/profile.py

**Files:**
- Create: `storage/profile.py`

- [ ] **Step 1: Write storage/profile.py**

```python
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

PROFILE_DIR = Path("data")
PROFILE_PATH = PROFILE_DIR / "profile.json"
BACKUP_DIR = PROFILE_DIR / "profile_backups"
MAX_BACKUPS = 10

_DEFAULT_PROFILE = {
    "identity": {},
    "preferences": {},
    "habits": [],
    "plans": {},
    "updated_at": "",
}


def _ensure_dirs():
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)


def load_profile() -> dict:
    """Load profile from JSON file. Returns default structure if not found or corrupt."""
    if not PROFILE_PATH.exists():
        logger.info("No profile file found, returning default")
        return dict(_DEFAULT_PROFILE)
    try:
        with open(PROFILE_PATH, encoding="utf-8") as f:
            data = json.load(f)
        # Ensure all default keys exist
        for k, v in _DEFAULT_PROFILE.items():
            data.setdefault(k, v)
        return data
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to load profile (%s), returning default", e)
        return dict(_DEFAULT_PROFILE)


def save_profile(profile: dict) -> None:
    """Backup current profile, write new profile, clean old backups."""
    _ensure_dirs()
    # Backup existing profile if it exists
    if PROFILE_PATH.exists():
        timestamp = datetime.now().strftime("%Y-%m-%dT%H%M%S")
        backup_path = BACKUP_DIR / f"{timestamp}.json"
        shutil.copy2(PROFILE_PATH, backup_path)
        _clean_old_backups()
    # Write new profile
    with open(PROFILE_PATH, "w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)
    logger.info("Profile saved to %s", PROFILE_PATH)


def _clean_old_backups():
    backups = sorted(BACKUP_DIR.glob("*.json"))
    while len(backups) > MAX_BACKUPS:
        backups[0].unlink()
        backups = sorted(BACKUP_DIR.glob("*.json"))


def update_profile_field(profile: dict, field: str, value: object) -> dict:
    """Update a field supporting dot-notation (e.g. 'plans.current_study').

    Returns the updated profile dict. Does NOT auto-save — caller must
    call save_profile() separately.
    """
    keys = field.split(".")
    obj = profile
    for key in keys[:-1]:
        if key not in obj or not isinstance(obj[key], dict):
            obj[key] = {}
        obj = obj[key]
    obj[keys[-1]] = value
    profile["updated_at"] = datetime.now().isoformat()
    return profile
```

- [ ] **Step 2: Run quick import check**

Run: `python -c "from storage.profile import load_profile, save_profile, update_profile_field; print('OK')"`  
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add storage/profile.py
git commit -m "feat: add user profile storage module"
```

---

### Task 2: Update state.py

**Files:**
- Modify: `agent/state.py`

- [ ] **Step 1: Add user_profile field**

Add `user_profile: dict` after `logic_chain`:

```python
    # Reasoning trace (accumulates across nodes via operator.add)
    logic_chain: Annotated[list[dict], operator.add]

    # User profile (loaded from profile.json)
    user_profile: dict
```

- [ ] **Step 2: Verify graph still builds**

Run: `python -c "from agent.graph import build_graph; build_graph(); print('OK'"`  
Expected: `OK` (will fail if the field is missing from initial state)

- [ ] **Step 3: Commit**

```bash
git add agent/state.py
git commit -m "feat: add user_profile field to AgentState"
```

---

### Task 3: Create update_profile node

**Files:**
- Create: `agent/nodes/update_profile.py`

**Depends on:** Task 1 (profile.py), Task 2 (state.py)

- [ ] **Step 1: Write update_profile.py**

```python
import logging

from pydantic import BaseModel, Field

from agent.utils.llm import LLM
from storage.profile import load_profile, save_profile, update_profile_field

logger = logging.getLogger(__name__)


class ProfileUpdate(BaseModel):
    """Extracted personal information from the conversation."""
    field: str = Field(
        description="Dot-notation field path, e.g. 'identity.name', 'plans.current_study', 'preferences.tech_stack'"
    )
    value: object = Field(
        description="The value to store. Use string for single values, list for multiple items."
    )
    should_update: bool = Field(
        description="True if the conversation contains new/changed personal information worth saving"
    )


def update_profile(state: dict) -> dict:
    """Extract personal information from the conversation and update profile."""
    user_message = state.get("user_message", "")
    answer = state.get("answer", "")

    if not user_message:
        return {"user_profile": state.get("user_profile", load_profile())}

    prompt = (
        f"分析以下对话，判断用户是否在谈论个人信息（姓名、职业、生活习惯、学习计划、偏好等）。\n\n"
        f"用户消息：{user_message}\n"
        f"回答：{answer}\n\n"
        f"如果包含值得记录的个人信息，提取字段路径和值。否则 should_update=false。"
    )

    try:
        result = LLM.generate_structured(prompt, ProfileUpdate, use_language=False)
    except Exception as e:
        logger.warning("Profile extraction failed: %s", e)
        return {"user_profile": state.get("user_profile", load_profile())}

    if not result.should_update:
        logger.info("No personal info detected in conversation")
        return {"user_profile": state.get("user_profile", load_profile())}

    profile = load_profile()
    profile = update_profile_field(profile, result.field, result.value)
    save_profile(profile)

    logger.info("Profile updated: %s = %s", result.field, result.value)
    return {"user_profile": profile}
```

- [ ] **Step 2: Verify import**

Run: `python -c "from agent.nodes.update_profile import update_profile; print('OK'"`  
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add agent/nodes/update_profile.py
git commit -m "feat: add update_profile node for personal info extraction"
```

---

### Task 4: Update classify_and_answer prompt

**Files:**
- Modify: `agent/nodes/classify_and_answer.py`

**Changes:**
1. Inject profile into system prompt
2. Remove personal info recording instructions (was "记录用户的生活习惯")
3. Compress search rules from 10 lines to 2 lines
4. Remove confidence evaluation rubric
5. Remove `needs_store` detailed guidelines (keep 1 line)
6. Simplify classification guide (remove personal category)

- [ ] **Step 1: Rewrite _build_system_prompt()**

Replace the current `_build_system_prompt` function:

```python
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

    # Inject user profile for personalization
    profile = state.get("user_profile", {})
    if profile and any(v for v in profile.values() if v):
        profile_summary = _summarize_profile(profile)
        parts.append("")
        parts.append(f"## 用户画像\n{profile_summary}")

    # Inject relevant stored knowledge
    if state.get("stored_knowledge"):
        parts.append("")
        parts.append("## 相关知识")
        for k in state["stored_knowledge"]:
            parts.append(f"- {k['knowledge_text']}")

    return "\n".join(parts)


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
                lines.append(f"- {section}: {', '.join(data)}")
    return "\n".join(lines) if lines else "（暂无用户画像信息）"
```

Add import at top (no new imports needed — `state` is already imported).

- [ ] **Step 2: Run existing classify tests**

Run: `python -m pytest tests/test_nodes.py::test_classify_and_answer tests/test_nodes.py::test_classify_and_answer_has_reasoning_trace -v`  
Expected: Both PASS

- [ ] **Step 3: Commit**

```bash
git add agent/nodes/classify_and_answer.py
git commit -m "refactor: inject user profile and simplify classify prompt"
```

---

### Task 5: Simplify store.py prompt

**Files:**
- Modify: `agent/nodes/store.py`

**Changes:**
1. Remove the personal extraction "额外要求" section (lines 151-161 in current file)
2. Remove the full classification hierarchy table (keep only the fixed top-level list)
3. Reduce examples to 3-4 most representative
4. Keep boundary case rules section but simplify

- [ ] **Step 1: Rewrite the distill prompt**

Replace the prompt section (currently lines 65-162) with a simplified version:

```python
    prompt = (
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
        "5. Multiple core topics → use & separator (e.g. 'programming&mathematics')\n"
        "6. Unclassifiable → 'other'\n\n"
        "## Examples\n"
        "- What is Python → programming/python\n"
        "- Django routing → programming/python/web/django\n"
        "- An-Shi Rebellion impact → history/chinese-history/tang-dynasty/an-shi-rebellion\n"
        "- Hello → other\n\n"
        f"Question: {state['user_message']}\n"
        f"Answer: {state['answer']}"
    )
```

Also remove the `personal` category mentions and the "额外要求" section. The simplified prompt uses English since the codebase already uses English for most prompts (and `use_language=False` is passed to `LLM.generate_structured`).

- [ ] **Step 2: Run store tests**

Run: `python -m pytest tests/test_nodes.py::test_store_empty_answer tests/test_nodes.py::test_store_distills_knowledge tests/test_nodes.py::test_store_skips_on_contradiction tests/test_nodes.py::test_store_returns_stored_ids -v`  
Expected: All 4 PASS

- [ ] **Step 3: Commit**

```bash
git add agent/nodes/store.py
git commit -m "refactor: simplify store prompt, remove personal extraction rules"
```

---

### Task 6: Update graph.py with parallel fan-out

**Files:**
- Modify: `agent/graph.py`

**Changes:**
1. Import `update_profile` node
2. Add `update_profile` node to the graph
3. Add parallel edges from `classify_and_answer` to both `update_profile` and `fact_check`
4. Load profile at graph entry and pass to initial state

- [ ] **Step 1: Modify graph.py**

Add import at top:
```python
from agent.nodes.update_profile import update_profile
```

Add node:
```python
builder.add_node("update_profile", update_profile)
```

Replace the sequential edge:
```python
# Remove this line:
# builder.add_edge("classify_and_answer", "fact_check")

# Replace with parallel fan-out:
builder.add_edge("classify_and_answer", "update_profile")
builder.add_edge("classify_and_answer", "fact_check")
```

No further routing needed — `update_profile` writes to state and terminates. The main flow continues via `fact_check → [reflect | store] → respond`.

- [ ] **Step 2: Verify graph builds**

Run: `python -c "from agent.graph import build_graph; build_graph(); print('OK'"`  
Expected: `OK`

- [ ] **Step 3: Run graph tests**

Run: `python -m pytest tests/test_graph.py -v`  
Expected: All 3 PASS

- [ ] **Step 4: Commit**

```bash
git add agent/graph.py
git commit -m "feat: add update_profile node with parallel fan-out"
```

---

### Task 7: Add tests

**Files:**
- Create: `tests/test_profile.py`
- Create: `tests/test_update_profile.py`

- [ ] **Step 1: Write tests/test_profile.py**

```python
import json
import pytest
from pathlib import Path


def test_load_profile_empty(tmp_path):
    """First load returns default dict when no file exists."""
    import storage.profile as pf
    pf.PROFILE_DIR = tmp_path
    pf.PROFILE_PATH = tmp_path / "profile.json"
    pf.BACKUP_DIR = tmp_path / "profile_backups"

    profile = pf.load_profile()
    assert "identity" in profile
    assert "preferences" in profile
    assert "habits" in profile
    assert "plans" in profile
    assert "updated_at" in profile
    assert profile["identity"] == {}


def test_save_and_load_profile(tmp_path):
    import storage.profile as pf
    pf.PROFILE_DIR = tmp_path
    pf.PROFILE_PATH = tmp_path / "profile.json"
    pf.BACKUP_DIR = tmp_path / "profile_backups"

    profile = pf.load_profile()
    profile["identity"]["name"] = "TestUser"
    pf.save_profile(profile)

    loaded = pf.load_profile()
    assert loaded["identity"]["name"] == "TestUser"


def test_update_field_dot_notation():
    from storage.profile import update_profile_field

    profile = {"identity": {}, "preferences": {}, "habits": [], "plans": {}, "updated_at": ""}
    profile = update_profile_field(profile, "plans.current_study", "LangChain")
    assert profile["plans"]["current_study"] == "LangChain"
    assert profile["updated_at"] != ""

    profile = update_profile_field(profile, "identity.name", "李明")
    assert profile["identity"]["name"] == "李明"


def test_backup_rotation(tmp_path):
    import storage.profile as pf
    pf.PROFILE_DIR = tmp_path
    pf.PROFILE_PATH = tmp_path / "profile.json"
    pf.BACKUP_DIR = tmp_path / "profile_backups"
    pf.MAX_BACKUPS = 3

    pf.save_profile({"identity": {"name": "u1"}, "updated_at": "t1"})
    pf.save_profile({"identity": {"name": "u2"}, "updated_at": "t2"})
    pf.save_profile({"identity": {"name": "u3"}, "updated_at": "t3"})
    pf.save_profile({"identity": {"name": "u4"}, "updated_at": "t4"})

    backups = list(pf.BACKUP_DIR.glob("*.json"))
    assert len(backups) == 3  # MAX_BACKUPS


def test_load_profile_corrupt(tmp_path):
    import storage.profile as pf
    pf.PROFILE_DIR = tmp_path
    pf.PROFILE_PATH = tmp_path / "profile.json"
    pf.BACKUP_DIR = tmp_path / "profile_backups"

    pf.PROFILE_PATH.write_text("{invalid json}", encoding="utf-8")
    profile = pf.load_profile()
    assert "identity" in profile


def test_update_field_creates_nested_dicts():
    from storage.profile import update_profile_field

    profile = {"identity": {}, "preferences": {}, "habits": [], "plans": {}, "updated_at": ""}
    profile = update_profile_field(profile, "a.b.c", "deep")
    assert profile["a"]["b"]["c"] == "deep"
```

- [ ] **Step 2: Write tests/test_update_profile.py**

```python
import pytest


def test_update_profile_no_pii():
    """No personal info → should_update=false, profile unchanged."""
    from agent.nodes.update_profile import update_profile

    result = update_profile({
        "user_message": "What is Python?",
        "answer": "Python is a programming language.",
    })
    assert "user_profile" in result


def test_update_profile_extracts_name(monkeypatch):
    """Simulate LLM extracting a name from the conversation."""
    from agent.nodes.update_profile import update_profile, ProfileUpdate

    class FakeLLM:
        @staticmethod
        def generate_structured(prompt, output_model, **kwargs):
            return ProfileUpdate(
                field="identity.name",
                value="李明",
                should_update=True,
            )

    monkeypatch.setattr("agent.nodes.update_profile.LLM", FakeLLM)

    result = update_profile({
        "user_message": "我叫李明",
        "answer": "你好李明！",
    })
    assert result["user_profile"]["identity"]["name"] == "李明"
```

- [ ] **Step 3: Run profile tests**

Run: `python -m pytest tests/test_profile.py -v`  
Expected: All 6 tests PASS

- [ ] **Step 4: Run update_profile tests**

Run: `python -m pytest tests/test_update_profile.py -v`  
Expected: All 2 tests PASS

- [ ] **Step 5: Run full test suite**

Run: `python -m pytest tests/ -v`  
Expected: All tests pass

- [ ] **Step 6: Commit**

```bash
git add tests/test_profile.py tests/test_update_profile.py
git commit -m "test: add user profile tests"
```

---

## Self-Review

**1. Spec coverage:**
- `storage/profile.py`: load/save/backup/update → Task 1 ✓
- `agent/state.py`: user_profile field → Task 2 ✓
- `agent/nodes/update_profile.py`: extraction node → Task 3 ✓
- `classify_and_answer.py`: profile injection + simplify prompt → Task 4 ✓
- `store.py`: remove personal extraction + simplify prompt → Task 5 ✓
- `graph.py`: add node + parallel fan-out → Task 6 ✓
- Tests → Task 7 ✓

**2. Placeholder scan:** No TBD, TODO, or incomplete code blocks. All code is complete.

**3. Type consistency:** `update_profile_field(profile: dict, field: str, value: object) -> dict` matches call site. `update_profile(state: dict) -> dict` matches LangGraph node signature. `user_profile: dict` in state matches profile.py dict type.

---

## Execution Handoff

Plan complete. Two execution options:

**1. Subagent-Driven (recommended)** — dispatch a fresh subagent per task, review between tasks

**2. Inline Execution** — execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
