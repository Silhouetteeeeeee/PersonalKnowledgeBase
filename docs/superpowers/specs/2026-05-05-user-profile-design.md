# User Profile System Design

> **Problem:** Personal information (habits, plans, preferences) is stored in the same knowledge base as immutable facts. When personal info changes (e.g., study plan from "Python" to "Golang"), the fact-check/reflect loop treats it as a contradiction, triggering unwanted correction cycles.

**Goal:** A dedicated user profile system that separates mutable personal information from factual knowledge, with automatic extraction, JSON-file storage, and backup history.

**Non-goal:** Multi-user support, profile sharing, complex merging, or semantic deduplication.

---

## Data Model

Single JSON file at `data/profile.json`:

```json
{
  "identity": {
    "name": "李明",
    "career": "后端工程师"
  },
  "preferences": {
    "tech_stack": ["Go", "Python"],
    "learning_interests": ["AI", "分布式系统"]
  },
  "habits": [
    "经常深夜工作",
    "喜欢喝美式咖啡"
  ],
  "plans": {
    "current_study": "学习 LangChain",
    "goals": "构建个人 AI 助手"
  },
  "updated_at": "2026-05-05T12:00:00"
}
```

- Automatically created on first access
- Overwrite semantics: new value replaces old directly
- No locking needed (single-user)

### Backup mechanism

Before each write, the current `profile.json` is copied to `data/profile_backups/`:

```
data/profile_backups/
├── 2026-05-05T120000.json
├── 2026-05-05T180000.json
└── ...  (max 10, oldest deleted first)
```

---

## Architecture

### Graph flow

```
parse → retrieve → classify_and_answer
                        │
               ┌────────┴────────┐
               ↓                  ↓
        update_profile      fact_check
               │                  │
               └────────┬─────────┘
                        ↓
                     store → respond
```

`classify_and_answer` fans out to `update_profile` and `fact_check` in parallel. LangGraph's fan-in waits for both branches to complete before `store`.

### Profile injection into classify

When `classify_and_answer` runs, profile content is injected into the system prompt:

```
User profile: {"name": "李明", "plans": {"current_study": "LangChain"}}
```

This lets the LLM personalize responses without storing personal info as knowledge.

---

## Files Changed

| File | Action | Responsibility |
|------|--------|---------------|
| `storage/profile.py` | Create | `load_profile()`, `save_profile()`, `update_profile_field()`, backup management |
| `agent/nodes/update_profile.py` | Create | Extract personal info from conversation, call `update_profile_field()` |
| `agent/nodes/classify_and_answer.py` | Modify | Inject profile into prompt; remove personal extraction rules from prompt; simplify prompt |
| `agent/nodes/store.py` | Modify | Remove personal extraction instructions from prompt; simplify classification rules |
| `agent/state.py` | Modify | Add `user_profile: dict` field |
| `agent/graph.py` | Modify | Add `update_profile` node; parallel fan-out `classify_and_answer → update_profile + fact_check` |
| `tests/test_profile.py` | Create | Tests for load/save/backup/update |
| `tests/test_update_profile.py` | Create | Tests for profile extraction |

---

## Component Details

### `storage/profile.py`

```python
PROFILE_DIR = Path("data")
PROFILE_PATH = PROFILE_DIR / "profile.json"
BACKUP_DIR = PROFILE_DIR / "profile_backups"
MAX_BACKUPS = 10


def load_profile() -> dict:
    """Load profile from JSON file. Returns default structure if not found."""


def save_profile(profile: dict) -> None:
    """Backup current profile, write new profile, clean old backups."""


def update_profile_field(profile: dict, field: str, value: any) -> dict:
    """Update a field (supports dot-notation like 'plans.current_study') and set updated_at."""
```

### `agent/nodes/update_profile.py`

- Receives `state["user_message"]` and `state["answer"]`
- LLM analyzes conversation for personal info
- Updates `profile.json` via `update_profile_field()`
- Returns updated profile dict

### Prompt simplifications

**Classify prompt** — remove:
- Personal info recording instructions ("记录用户的生活习惯")
- Redundant needs_store guidelines (kept as 1 line)
- Verbose search rules (compressed to 2 lines)
- Confidence evaluation rubric (removed entirely — LLM knows this)

**Store prompt** — remove:
- Personal extraction rules (lines 151-161)
- Full classification hierarchy table (keep: fixed top-level list only)
- Excessive examples (keep: 3-4 most representative)
- Redundant boundary case rules

---

## Edge Cases

| Case | Handling |
|------|----------|
| First run, no profile | `load_profile()` returns empty default structure |
| No personal info in message | `update_profile` returns existing profile unchanged |
| Backup dir doesn't exist | Auto-created on first backup |
| Profile file corrupted/empty | `load_profile()` returns default, old backup not deleted |
| More than 10 backups | Oldest files deleted, keeping 10 newest by filename timestamp |
| Profile field never seen before | Treated as new field, added to the appropriate section |

---

## Testing

| Test | What it verifies |
|------|-----------------|
| `test_load_profile_empty` | First load returns default dict |
| `test_save_and_load_profile` | Write then read back matches |
| `test_update_field_dot_notation` | `plans.current_study` updates nested field |
| `test_backup_rotation` | Only 10 backups kept after repeated saves |
| `test_update_profile_no_pii` | No change to profile when message has no personal info |
| `test_update_profile_extracts_name` | "我叫李明" correctly writes to identity.name |
