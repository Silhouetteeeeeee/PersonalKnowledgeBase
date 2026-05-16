# Daily Knowledge Summary Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every morning at 09:00, summarize the previous day's new/updated wiki pages into a markdown report and proactively push it to a WeChat user.

**Architecture:** A new `server/daily_summary.py` module queries yesterday's pages from SQLite with source questions, feeds them through LLM to generate a structured daily report, then pushes via `WSClient.send_message()`. An APScheduler cron job in `bot.py` triggers it daily at 09:00.

**Tech Stack:** APScheduler (async cron), LangGraph LLM (same model as Q&A), WSClient.send_message() for push.

---

### Task 1: Add source_questions table to database

**Files:**
- Modify: `storage/database.py`
- Modify: `storage/models.py` (add helper functions)

- [ ] **Step 1: Add the table creation SQL**

Insert into `init_db()` in `storage/database.py`, after the `page_relations` table:

```python
        CREATE TABLE IF NOT EXISTS source_questions (
            source_id TEXT PRIMARY KEY,
            question TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
        );
```

- [ ] **Step 2: Add helper functions to storage/models.py**

```python
from storage.database import get_connection


def save_source_question(source_id: str, question: str) -> None:
    """Store the mapping from source_id to the user question that triggered it."""
    conn = get_connection()
    conn.execute(
        "INSERT OR IGNORE INTO source_questions (source_id, question) VALUES (?, ?)",
        (source_id, question),
    )
    conn.commit()
    conn.close()


def get_source_questions(source_ids: list[str]) -> list[str]:
    """Look up questions for a list of source IDs."""
    if not source_ids:
        return []
    placeholders = ",".join("?" for _ in source_ids)
    conn = get_connection()
    rows = conn.execute(
        f"SELECT question FROM source_questions WHERE source_id IN ({placeholders})",
        source_ids,
    ).fetchall()
    conn.close()
    return [r[0] for r in rows]
```

- [ ] **Step 3: Verify the table creation**

```bash
pytest -x -q --no-header -p no:warnings -k "test_init_db" tests/ -c /dev/null 2>/dev/null || echo "no existing test for init_db, checking manually"
python -c "from storage.database import init_db, get_connection; init_db(); c=get_connection(); print(c.execute('PRAGMA table_info(source_questions)').fetchall()); c.close()"
```

Expected: columns [(0, 'source_id', 'TEXT', 1, None, 1), (1, 'question', 'TEXT', ...), ...]

- [ ] **Step 4: Commit**

```bash
git add storage/database.py storage/models.py
git commit -m "feat: add source_questions table for source_id→question mapping"
```

---

### Task 2: Save source question in store.py

**Files:**
- Modify: `agent/nodes/store.py`

- [ ] **Step 1: Add import for save_source_question**

Add to existing imports in `agent/nodes/store.py`:

```python
from storage.models import (
    upsert_page,
    update_page_relations,
    get_page_by_title,
    find_similar_pages,
    save_source_question,  # new
)
```

- [ ] **Step 2: Add the save call in store() function**

In `agent/nodes/store.py` in the `store()` function, after `source_id = _get_source_id()`:

```python
    source_id = _get_source_id()
    save_source_question(source_id, state["user_message"])  # new
    source_label = f"Question: {state['user_message']}"
```

- [ ] **Step 3: Run unit tests to verify no breakage**

```bash
python -m pytest tests/unit/test_nodes.py -v --tb=short
```

Expected: all store-related tests pass (test_store_empty_answer, etc.)

- [ ] **Step 4: Commit**

```bash
git add agent/nodes/store.py
git commit -m "feat: save user question mapping when storing wiki pages"
```

---

### Task 3: Add apscheduler to dependencies and configure

**Files:**
- Modify: `requirements.txt`
- Modify: `server/config.py`

- [ ] **Step 1: Add apscheduler to requirements.txt**

```txt
apscheduler>=3.10.0
```

Add after the existing pytest line.

- [ ] **Step 2: Add config variables to server/config.py**

```python
# Daily knowledge summary (APScheduler cron at 09:00)
DAILY_SUMMARY_ENABLED = os.getenv("DAILY_SUMMARY_ENABLED", "true").lower() == "true"
DAILY_SUMMARY_USER_ID = os.getenv("DAILY_SUMMARY_USER_ID", "")
DAILY_SUMMARY_TIME = os.getenv("DAILY_SUMMARY_TIME", "09:00")
```

- [ ] **Step 3: Install apscheduler**

```bash
pip install apscheduler
```

- [ ] **Step 4: Commit**

```bash
git add requirements.txt server/config.py
git commit -m "chore: add apscheduler dependency and daily summary config"
```

---

### Task 4: Implement daily_summary.py core logic

**Files:**
- Create: `server/daily_summary.py`
- Create: `tests/unit/test_daily_summary.py`

This is the main module. It has three internal functions and one public async function.

- [ ] **Step 1: Write the test file first (TDD)**

Create `tests/unit/test_daily_summary.py`:

```python
"""Unit tests for daily knowledge summary."""
import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, AsyncMock


@pytest.fixture
def mock_pages():
    return [
        {
            "id": 1, "title": "TCP三次握手",
            "file_path": "pages/tcp-handshake.md",
            "tags": '["TCP","网络"]',
            "sources": '["conv_001"]',
            "created_at": "2026-05-14 10:00:00",
            "updated_at": "2026-05-14 10:00:00",
            "content": "TCP三次握手是建立可靠连接的过程...",
            "source_questions": ["TCP三次握手的过程是什么"],
        },
    ]


def test_get_yesterday_pages_empty(monkeypatch, tmp_path):
    """No pages yesterday → empty list."""
    from server.daily_summary import _get_yesterday_pages

    # Monkey-patch get_connection to return empty
    monkeypatch.setattr("storage.database.DB_DIR", str(tmp_path / "data"))
    monkeypatch.setattr("storage.database.DB_PATH", str(tmp_path / "data" / "knowledge.db"))
    from storage.database import init_db
    init_db()
    
    pages = _get_yesterday_pages()
    assert pages == []


def test_get_yesterday_pages_with_data(monkeypatch, tmp_path):
    """Pages created yesterday → returned with content."""
    from server.daily_summary import _get_yesterday_pages
    from storage.database import get_connection

    monkeypatch.setattr("storage.database.DB_DIR", str(tmp_path / "data"))
    monkeypatch.setattr("storage.database.DB_PATH", str(tmp_path / "data" / "knowledge.db"))
    from storage.database import init_db
    init_db()

    conn = get_connection()
    conn.execute(
        "INSERT INTO pages (id, title, file_path, tags, sources, created_at, updated_at) "
        "VALUES (1, 'Test Page', 'pages/test.md', '[]', '[]', datetime('now','localtime','-1 day'), datetime('now','localtime','-1 day'))"
    )
    conn.commit()
    conn.close()

    # Create the wiki file
    wiki_dir = tmp_path / "wiki" / "pages"
    wiki_dir.mkdir(parents=True)
    test_page = wiki_dir / "test.md"
    test_page.write_text("---\ntitle: Test Page\ntags: []\nsources: []\ncreated: 2026-05-13\nupdated: 2026-05-13\n---\n\nTest content body")
    monkeypatch.setattr("storage.wiki_storage.WIKI_DIR", str(tmp_path / "wiki"))
    from server.daily_summary import WIKI_DIR
    monkeypatch.setattr("server.daily_summary.WIKI_DIR", str(tmp_path / "wiki"))

    pages = _get_yesterday_pages()
    assert len(pages) == 1
    assert pages[0]["title"] == "Test Page"


def test_generate_summary_text(mock_pages):
    """Verify LLM summary generation produces markdown."""
    from server.daily_summary import _generate_summary_text
    from unittest.mock import patch

    mock_summary = "📅 昨日知识汇总\n\n📌 新增页面（1篇）\n\n**1. TCP三次握手**\n用户问：TCP三次握手的过程是什么？\n→ 内容摘要"

    with patch("server.daily_summary.LLM.generate_structured") as mock_llm:
        mock_llm.return_value = MagicMock(summary=mock_summary)
        result = _generate_summary_text(mock_pages)
        assert "📅" in result
        assert "TCP三次握手" in result


@pytest.mark.asyncio
async def test_send_daily_summary_skips_when_empty():
    """No pages → no send_message call."""
    from server.daily_summary import send_daily_summary
    from unittest.mock import patch

    mock_client = MagicMock()

    with patch("server.daily_summary._get_yesterday_pages", return_value=[]):
        result = await send_daily_summary(mock_client, "user1")
        assert result is False
        mock_client.send_message.assert_not_called()


@pytest.mark.asyncio
async def test_send_daily_summary_sends(mock_pages):
    """Has pages → calls send_message with markdown."""
    from server.daily_summary import send_daily_summary
    from unittest.mock import patch, MagicMock

    mock_client = MagicMock()
    mock_summary = MagicMock(summary="📅 昨日知识汇总\n\n内容...")

    with patch("server.daily_summary._get_yesterday_pages", return_value=mock_pages):
        with patch("server.daily_summary._generate_summary_text", return_value="📅 昨日知识汇总\n\n内容..."):
            result = await send_daily_summary(mock_client, "user1")
            assert result is True
            mock_client.send_message.assert_called_once()


def test_split_new_vs_updated():
    """Verify page splitting logic: new vs updated."""
    from server.daily_summary import _split_pages

    yesterday = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)
    today_str = yesterday.strftime("%Y-%m-%d")

    pages = [
        {"id": 1, "title": "New Page", "created_at": f"{today_str} 10:00:00", "updated_at": f"{today_str} 10:00:00"},
        {"id": 2, "title": "Updated Page", "created_at": "2026-01-01 10:00:00", "updated_at": f"{today_str} 10:00:00"},
    ]

    # Set yesterday bound
    y_bound = yesterday
    t_bound = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    new, updated = _split_pages(pages, y_bound, t_bound)
    assert len(new) == 1
    assert new[0]["title"] == "New Page"
    assert len(updated) == 1
    assert updated[0]["title"] == "Updated Page"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/unit/test_daily_summary.py -v --tb=short
```

Expected: 5/5 FAILED (ImportError: no module named server.daily_summary)

- [ ] **Step 3: Implement daily_summary.py**

Create `server/daily_summary.py`:

```python
"""Daily knowledge summary: query yesterday's pages, LLM summarize, push to WeChat."""

import json
import logging
from datetime import datetime, timedelta

from pydantic import BaseModel, Field

from agent.utils.llm import LLM
from storage.database import get_connection
from storage.models import get_source_questions
from storage.wiki_storage import WIKI_DIR, parse_frontmatter

logger = logging.getLogger(__name__)


class SummaryOutput(BaseModel):
    summary: str = Field(description="Full markdown daily summary content")


def _get_yesterday_range() -> tuple[datetime, datetime]:
    """Return (yesterday_00:00, today_00:00) as datetimes."""
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    yesterday = today - timedelta(days=1)
    return yesterday, today


def _get_yesterday_pages() -> list[dict]:
    """Query all pages created or updated yesterday, with full content and source questions."""
    y_bound, t_bound = _get_yesterday_range()
    y_str = y_bound.strftime("%Y-%m-%d %H:%M:%S")
    t_str = t_bound.strftime("%Y-%m-%d %H:%M:%S")

    conn = get_connection()
    rows = conn.execute(
        """SELECT id, title, file_path, tags, sources, created_at, updated_at
           FROM pages
           WHERE status = 'active'
             AND ((created_at >= ? AND created_at < ?)
               OR (updated_at >= ? AND updated_at < ?))
           ORDER BY created_at DESC""",
        (y_str, t_str, y_str, t_str),
    ).fetchall()
    conn.close()

    if not rows:
        return []

    pages = []
    for r in rows:
        fp = r["file_path"]
        full_path = WIKI_DIR / fp
        content = ""
        if full_path.exists():
            raw = full_path.read_text(encoding="utf-8")
            _, content = parse_frontmatter(raw)

        # Look up source questions
        try:
            source_ids = json.loads(r["sources"])
        except (json.JSONDecodeError, TypeError):
            source_ids = []
        conv_ids = [s for s in source_ids if s.startswith("conv_")]
        questions = get_source_questions(conv_ids) if conv_ids else []

        pages.append({
            "id": r["id"],
            "title": r["title"],
            "file_path": r["file_path"],
            "tags": r["tags"],
            "sources": r["sources"],
            "created_at": r["created_at"],
            "updated_at": r["updated_at"],
            "content": content[:500],  # preview only
            "source_questions": questions,
        })

    return pages


def _split_pages(
    pages: list[dict],
    y_bound: datetime,
    t_bound: datetime,
) -> tuple[list[dict], list[dict]]:
    """Split pages into 'new' (created yesterday) and 'updated' (modified yesterday)."""
    new = []
    updated = []
    for p in pages:
        created = datetime.strptime(p["created_at"], "%Y-%m-%d %H:%M:%S") if p["created_at"] else t_bound
        if y_bound <= created < t_bound:
            new.append(p)
        else:
            updated.append(p)
    return new, updated


def _generate_summary_text(pages: list[dict]) -> str:
    """Use LLM to generate a structured markdown daily summary."""
    y_bound, t_bound = _get_yesterday_range()
    new_pages, updated_pages = _split_pages(pages, y_bound, t_bound)

    # Build prompt
    lines = [
        f"你是一个知识库助手。请根据以下{y_bound.date()}新增和更新的知识页面，生成一篇中文知识日报。",
        "",
        "## 输出要求",
        "- 使用markdown格式，标题用emoji：📌 新增知识、📝 更新知识",
        "- 相关主题的页面归为一组",
        "- 每个页面给出标题、摘要、以及它来自什么用户问题",
        "- 如果用户问了多个相关问题，在最后加一个「💡 知识关联」部分说明这些知识之间的联系",
        "- 语言：中文，技术术语保留英文",
        "",
    ]

    if new_pages:
        lines.append(f"## 新增页面（{len(new_pages)}篇）")
        for p in new_pages:
            lines.append(f"\n### {p['title']}")
            questions = p.get("source_questions", [])
            if questions:
                for q in questions:
                    lines.append(f"- 用户问：{q}")
            if p["content"]:
                lines.append(f"- 摘要：{p['content'][:200]}")
        lines.append("")

    if updated_pages:
        lines.append(f"## 更新页面（{len(updated_pages)}篇）")
        for p in updated_pages:
            lines.append(f"\n### {p['title']}")
            questions = p.get("source_questions", [])
            if questions:
                for q in questions:
                    lines.append(f"- 用户问：{q}")
            if p["content"]:
                lines.append(f"- 摘要：{p['content'][:200]}")
        lines.append("")

    prompt = "\n".join(lines)

    result = LLM.generate_structured(prompt, SummaryOutput)
    if result is None:
        logger.error("LLM summary generation returned None")
        return ""
    return result.summary


async def send_daily_summary(client, user_id: str) -> bool:
    """Query yesterday's pages, generate summary, push to WeChat user.

    Args:
        client: WSClient instance for sending messages.
        user_id: WeChat user ID to push to.

    Returns:
        True if message was sent, False if skipped or failed.
    """
    pages = _get_yesterday_pages()
    if not pages:
        logger.info("No new or updated pages yesterday, skipping daily summary")
        return False

    logger.info("Generating daily summary for %d pages", len(pages))
    summary = _generate_summary_text(pages)
    if not summary:
        logger.error("Failed to generate summary text")
        return False

    try:
        client.send_message(user_id, {
            "msgtype": "markdown",
            "markdown": {"content": summary},
        })
        logger.info("Daily summary pushed to user %s (%d pages)", user_id, len(pages))
        return True
    except Exception as e:
        logger.error("Failed to send daily summary to %s: %s", user_id, e)
        return False
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/unit/test_daily_summary.py -v --tb=short
```

Expected: 5/5 PASSED

- [ ] **Step 5: Commit**

```bash
git add server/daily_summary.py tests/unit/test_daily_summary.py
git commit -m "feat: implement daily knowledge summary module"
```

---

### Task 5: Wire APScheduler into bot.py

**Files:**
- Modify: `server/bot.py`

- [ ] **Step 1: Add imports and scheduler start in bot.py**

Add import at top of `server/bot.py`:

```python
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from server.config import (
    WECOM_BOT_ID, WECOM_BOT_SECRET, CLAUDE_CODE_BRIDGE_ENABLED,
    DAILY_SUMMARY_ENABLED, DAILY_SUMMARY_USER_ID,
)
from server.daily_summary import send_daily_summary
```

In `KnowledgeBot.__init__()`, add:

```python
class KnowledgeBot:
    def __init__(self):
        self.client = WSClient(WSClientOptions(
            bot_id=WECOM_BOT_ID,
            secret=WECOM_BOT_SECRET,
            max_reconnect_attempts=-1,
        ))
        self.scheduler = AsyncIOScheduler()
        self._setup_handlers()
        self._start_daily_summary_scheduler()
```

Add the method:

```python
    def _start_daily_summary_scheduler(self):
        """Schedule daily knowledge summary at 09:00 via APScheduler."""
        if not DAILY_SUMMARY_ENABLED:
            logger.info("Daily summary disabled via config")
            return
        if not DAILY_SUMMARY_USER_ID:
            logger.warning("DAILY_SUMMARY_USER_ID not set, daily summary disabled")
            return

        self.scheduler.add_job(
            send_daily_summary,
            "cron",
            hour=9,
            minute=0,
            args=[self.client, DAILY_SUMMARY_USER_ID],
            misfire_grace_time=300,
            id="daily_summary",
            replace_existing=True,
        )
        self.scheduler.start()
        logger.info(
            "Daily summary scheduler started (09:00, user=%s)",
            DAILY_SUMMARY_USER_ID,
        )
```

Also update `KnowledgeBot.run()` if needed — APScheduler runs on the same event loop, so no special handling needed.

- [ ] **Step 2: Verify the bot still loads (syntax check)**

```bash
python -c "import server.bot; print('OK')"
```

Expected: OK (might show config warnings if .env not set, but no ImportError)

- [ ] **Step 3: Commit**

```bash
git add server/bot.py
git commit -m "feat: add APScheduler for daily knowledge summary at 09:00"
```

---

### Task 6: Run all tests

- [ ] **Step 1: Run full test suite**

```bash
python -m pytest tests/unit/ -v --tb=short
```

Expected: all tests pass (existing 90+ tests + 5 new daily_summary tests)

- [ ] **Step 2: Final commit if any fixes needed**

```bash
git add -A
git commit -m "test: add daily summary unit tests"
```

---

### Spec Coverage Check

| Spec Requirement | Task |
|-----------------|------|
| Source questions table | Task 1 |
| Save source_id→question in store.py | Task 2 |
| apscheduler dependency | Task 3 |
| Config vars (ENABLED, USER_ID, TIME) | Task 3 |
| `_get_yesterday_pages()` | Task 4 |
| `_generate_summary_text()` | Task 4 |
| `send_daily_summary()` | Task 4 |
| APScheduler cron at 09:00 in bot.py | Task 5 |
| Edge cases (empty, LLM failure, send failure) | Task 4 (tests) |
| New vs updated page splitting | Task 4 |
| Cross-reference source questions | Task 2 + Task 4 |
