# Spaced Repetition Thinker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone "thinker" module that periodically pushes wiki page reviews via SM-2 spaced repetition, processes user quote-reply feedback, and weekly generates cross-page knowledge integration.

**Architecture:** A new `server/thinker.py` module with SM-2 logic + review scheduling + feedback handling, triggered by APScheduler (already in project). User quotes a `#review_xxx` marked message → bot.py routes to thinker (bypasses LangGraph). A new `review_schedule` and `sent_reviews` table track page-level SM-2 state and sent messages.

**Tech Stack:** APScheduler (async cron), LLM.generate_structured (same model as Q&A), WSClient.send_message() for push, sqlite3 for SM-2 state

---

## File Structure

| File | Change | Responsibility |
|------|--------|---------------|
| `storage/database.py` | **Modify** | Add `review_schedule` and `sent_reviews` tables in `init_db()` |
| `server/thinker.py` | **Create** | SM-2 algorithm, review push, feedback handler, weekly integration |
| `server/bot.py` | **Modify** | Add thinker route before graph invoke, add APScheduler job |
| `agent/nodes/store.py` | **Modify** | After upsert_page, insert into `review_schedule` |
| `server/config.py` | **Modify** | Add `THINKER_USER_ID`, `THINKER_CHECK_INTERVAL` config |
| `tests/unit/test_thinker.py` | **Create** | Unit tests for SM-2, review generation, feedback handling |

---

### Task 1: Add review_schedule and sent_reviews tables

**Files:**
- Modify: `storage/database.py:48-69` — add two CREATE TABLE statements

- [ ] **Step 1: Add tables to init_db()**

```python
# After the pages table block (before page_vectors), add:
CREATE TABLE IF NOT EXISTS review_schedule (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    page_id INTEGER NOT NULL UNIQUE,
    easiness_factor REAL NOT NULL DEFAULT 2.5,
    interval_days INTEGER NOT NULL DEFAULT 1,
    repetitions INTEGER NOT NULL DEFAULT 0,
    next_review_at TEXT NOT NULL,
    last_reviewed_at TEXT DEFAULT '',
    last_quality INTEGER DEFAULT -1,
    created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
    FOREIGN KEY (page_id) REFERENCES pages(id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS sent_reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    schedule_id INTEGER NOT NULL,
    page_id INTEGER NOT NULL,
    marker_id TEXT NOT NULL UNIQUE,
    sent_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
    status TEXT NOT NULL DEFAULT 'pending',
    FOREIGN KEY (schedule_id) REFERENCES review_schedule(id) ON DELETE CASCADE,
    FOREIGN KEY (page_id) REFERENCES pages(id) ON DELETE CASCADE
);
```

- [ ] **Step 2: Add helper functions in storage/models.py**

Add these query helpers to `storage/models.py`:

```python
# ── Spaced repetition helpers ──

def init_review_schedule(page_id: int, next_review_at: str | None = None) -> int:
    """Insert a new review schedule for a page. Returns schedule id."""
    from datetime import datetime, timedelta
    conn = get_connection()
    try:
        if next_review_at is None:
            next_review_at = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")
        conn.execute(
            """INSERT OR IGNORE INTO review_schedule (page_id, next_review_at)
               VALUES (?, ?)""",
            (page_id, next_review_at),
        )
        conn.commit()
        row = conn.execute(
            "SELECT id FROM review_schedule WHERE page_id = ?", (page_id,)
        ).fetchone()
        return row["id"] if row else 0
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def get_due_reviews(limit: int = 10) -> list[dict]:
    """Query all schedules where next_review_at <= now, up to limit."""
    conn = get_connection()
    try:
        rows = conn.execute(
            """SELECT rs.*, p.title, p.file_path
               FROM review_schedule rs
               JOIN pages p ON p.id = rs.page_id
               WHERE rs.next_review_at <= datetime('now', 'localtime')
                 AND p.status = 'active'
               ORDER BY rs.next_review_at ASC
               LIMIT ?""",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def update_review_schedule(
    schedule_id: int,
    easiness_factor: float,
    interval_days: int,
    repetitions: int,
    next_review_at: str,
    quality: int,
) -> None:
    """Update SM-2 parameters after a review."""
    conn = get_connection()
    try:
        conn.execute(
            """UPDATE review_schedule
               SET easiness_factor = ?, interval_days = ?, repetitions = ?,
                   next_review_at = ?, last_reviewed_at = datetime('now', 'localtime'),
                   last_quality = ?
               WHERE id = ?""",
            (easiness_factor, interval_days, repetitions, next_review_at, quality, schedule_id),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def has_pending_review(page_id: int) -> bool:
    """Check if a page has a pending (unanswered) review."""
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT 1 FROM sent_reviews WHERE page_id = ? AND status = 'pending' LIMIT 1",
            (page_id,),
        ).fetchone()
        return row is not None
    finally:
        conn.close()


def get_sent_review_by_marker(marker_id: str) -> dict | None:
    """Look up a sent_review by its marker_id."""
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM sent_reviews WHERE marker_id = ?", (marker_id,),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def mark_review_answered(sent_id: int) -> None:
    """Mark a sent_review as answered."""
    conn = get_connection()
    try:
        conn.execute(
            "UPDATE sent_reviews SET status = 'reviewed' WHERE id = ?", (sent_id,),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def get_reviewed_pages_since(days: int = 7) -> list[dict]:
    """Get pages reviewed in the last N days (for weekly integration)."""
    conn = get_connection()
    try:
        rows = conn.execute(
            """SELECT DISTINCT rs.page_id, p.title, p.file_path
               FROM review_schedule rs
               JOIN pages p ON p.id = rs.page_id
               WHERE rs.last_reviewed_at >= datetime('now', 'localtime', ?)
                 AND rs.last_quality >= 3
                 AND p.status = 'active'
               ORDER BY rs.last_reviewed_at DESC""",
            (f'-{days} days',),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def record_sent_review(schedule_id: int, page_id: int, marker_id: str) -> int:
    """Record that a review was sent to the user."""
    conn = get_connection()
    try:
        cur = conn.execute(
            "INSERT INTO sent_reviews (schedule_id, page_id, marker_id) VALUES (?, ?, ?)",
            (schedule_id, page_id, marker_id),
        )
        conn.commit()
        return cur.lastrowid
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
```

- [ ] **Step 3: Add imports to models.py**

Add these to the import section of `storage/models.py`:
```python
from datetime import datetime, timedelta
```

- [ ] **Step 4: Commit**

```bash
git add storage/database.py storage/models.py
git commit -m "feat: add review_schedule and sent_reviews tables + helpers"
```

---

### Task 2: Create server/thinker.py — SM-2 + review push + feedback + weekly

**Files:**
- Create: `server/thinker.py`

- [ ] **Step 1: Create server/thinker.py**

```python
"""Spaced-repetition thinker module.

SM-2 algorithm, review push, feedback handling, weekly integration.
Completely independent of the LangGraph Q&A flow.
"""

import logging
import re
from datetime import datetime, timedelta

from pydantic import BaseModel, Field

from agent.utils.llm import LLM
from storage.models import (
    get_due_reviews,
    update_review_schedule,
    has_pending_review,
    record_sent_review,
    get_sent_review_by_marker,
    mark_review_answered,
    get_reviewed_pages_since,
)
from storage.wiki_storage import read_page

logger = logging.getLogger(__name__)

# ── SM-2 constants ──

MIN_EF = 1.3
MAX_EF = 3.0
QUALITY_PERFECT = 5     # 记住了
QUALITY_HESITANT = 3    # 模糊
QUALITY_FORGOT = 1      # 忘了

FEEDBACK_KEYWORDS = {
    "记住了": QUALITY_PERFECT,
    "模糊": QUALITY_HESITANT,
    "忘了": QUALITY_FORGOT,
}


def _sm2_update(quality: int, easiness_factor: float, interval: int, repetitions: int):
    """Compute new SM-2 parameters given a quality rating (0-5).

    Returns (new_ef, new_interval, new_repetitions).
    """
    if quality < 3:
        repetitions = 0
        interval = 1
    else:
        if repetitions == 0:
            interval = 1
        elif repetitions == 1:
            interval = 6
        else:
            interval = round(interval * easiness_factor)
        repetitions += 1

    ef = easiness_factor + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
    ef = max(MIN_EF, min(MAX_EF, ef))

    return ef, interval, repetitions


# ── LLM output models ──

class ReviewContent(BaseModel):
    summary: str = Field(description="Review summary of the wiki page, ~100-200 chars")
    key_points: list[str] = Field(description="3-5 key knowledge points from this page")
    review_question: str = Field(description="A self-test question to check recall")


class WeeklyIntegration(BaseModel):
    title: str = Field(description="Title for the weekly knowledge integration")
    content: str = Field(description="Full markdown content for the weekly review")
    new_questions: list[str] = Field(description="New questions raised by cross-linking these topics")


# ── Review generation ──

def _generate_review_content(page_title: str, page_body: str) -> ReviewContent | None:
    """LLM generates a review summary + key points + question from wiki body."""
    prompt = (
        "You are a spaced-repetition tutor. Given a wiki page, generate review material.\n\n"
        f"## Page Title\n{page_title}\n\n"
        f"## Page Content\n{page_body[:2000]}\n\n"
        "## Output Requirements\n"
        "1. summary: A concise review (~100-200 chars) covering the core concept\n"
        "2. key_points: 3-5 bullet-point knowledge items\n"
        "3. review_question: One self-test question to verify recall\n"
        "Use Chinese for explanations, English for technical terms."
    )
    return LLM.generate_structured(prompt, ReviewContent, use_language=False)


def _generate_weekly_integration(pages: list[dict]) -> WeeklyIntegration | None:
    """LLM generates a cross-page integration from reviewed pages."""
    pages_text = ""
    for p in pages:
        file_data = read_page(p["file_path"])
        body = file_data["body"][:1000] if file_data else "(content unavailable)"
        pages_text += f"\n### {p['title']}\n{body}\n---\n"

    prompt = (
        "You are a knowledge integration expert. Given multiple wiki pages reviewed this week, "
        "generate an integrated summary that connects them, identifies patterns, "
        "and raises new questions.\n\n"
        "## Pages Reviewed This Week\n"
        f"{pages_text}\n\n"
        "## Output Requirements\n"
        "1. title: A concise title for this integration\n"
        "2. content: Full markdown (~300-500 chars) connecting the topics, "
        "pointing out relationships, contrasts, and deeper insights\n"
        "3. new_questions: 2-3 new questions that arise from combining these topics\n"
        "Use Chinese for explanations, English for technical terms."
    )
    return LLM.generate_structured(prompt, WeeklyIntegration, use_language=False)


# ── Public API ──

def get_review_marker(page_id: int) -> str:
    """Generate a unique marker for a review message, e.g. #review_42_20260517."""
    return f"#review_{page_id}_{datetime.now().strftime('%Y%m%d')}"


def check_due_reviews(client=None, user_id: str = "") -> list[dict]:
    """Check for due reviews, push messages, return list of pushed reviews.

    Called by APScheduler. Requires a WSClient instance and target user_id.

    Returns list of dicts: [{marker_id, page_id, page_title}, ...]
    """
    if client is None or not user_id:
        logger.warning("check_due_reviews: no client or user_id provided")
        return []

    due = get_due_reviews(limit=10)
    if not due:
        logger.info("Thinker: no pages due for review")
        return []

    pushed = []
    for item in due:
        page_id = item["page_id"]
        page_title = item["title"]
        file_path = item["file_path"]

        # Skip if pending review exists
        if has_pending_review(page_id):
            logger.info("Thinker: page %d already has pending review, skipping", page_id)
            continue

        # Read wiki content
        page_data = read_page(file_path)
        if page_data is None:
            logger.warning("Thinker: page %s not found on disk, skipping", file_path)
            continue

        # Generate review content
        review = _generate_review_content(page_title, page_data["body"])
        if review is None:
            logger.warning("Thinker: failed to generate review for '%s'", page_title)
            continue

        marker = get_review_marker(page_id)

        # Build message
        key_points = "\n".join(f"- {kp}" for kp in review.key_points)
        message = (
            f"🧠 **复习提醒：{page_title}**\n\n"
            f"**摘要：** {review.summary}\n\n"
            f"**关键知识点：**\n{key_points}\n\n"
            f"**自测问题：** {review.review_question}\n\n"
            f"**反馈：** 引用此消息回复「记住了」「模糊」或「忘了」\n\n"
            f"{marker}"
        )

        try:
            # We'll use asyncio.to_thread adaptor in bot.py — here just build the message
            pushed.append({
                "marker_id": marker,
                "page_id": page_id,
                "page_title": page_title,
                "message": message,
                "schedule_id": item["id"],
            })
            logger.info("Thinker: prepared review for '%s' (%s)", page_title, marker)
        except Exception as e:
            logger.error("Thinker: failed to prepare review for '%s': %s", page_title, e)
            continue

    return pushed


def handle_review_response(quote_text: str, user_feedback: str) -> str:
    """Process a user's quote reply to a review message.

    Args:
        quote_text: The quoted message text (contains #review_ marker).
        user_feedback: User's reply text ("记住了", "模糊", or "忘了").

    Returns:
        A response string to send back to the user.
    """
    # Extract marker from quoted text
    match = re.search(r'(#review_\d+_\d+)', quote_text)
    if not match:
        return "抱歉，无法识别这条复习消息。请直接引用我发送的复习消息。😅"

    marker_id = match.group(1)

    # Look up the sent review
    sent = get_sent_review_by_marker(marker_id)
    if sent is None:
        return "这条复习消息已过期或无法找到对应记录。不过没关系，继续加油学习吧！💪"

    if sent["status"] == "reviewed":
        return "这条复习你已经回复过啦！记得保持复习节奏哦～"

    # Normalize feedback
    feedback = user_feedback.strip().lower()
    quality = None
    for keyword, q in FEEDBACK_KEYWORDS.items():
        if keyword in user_feedback or keyword in feedback:
            quality = q
            break

    if quality is None:
        return "请回复「记住了」「模糊」或「忘了」来告诉我你掌握得怎么样～"

    # Get SM-2 params
    from storage.database import get_connection
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM review_schedule WHERE id = ?", (sent["schedule_id"],)
    ).fetchone()
    conn.close()

    if row is None:
        return "找不到对应的复习记录。不过没关系，继续加油！💪"

    # SM-2 update
    new_ef, new_interval, new_reps = _sm2_update(
        quality, row["easiness_factor"], row["interval_days"], row["repetitions"],
    )
    next_review = (datetime.now() + timedelta(days=new_interval)).strftime("%Y-%m-%d %H:%M:%S")
    update_review_schedule(
        schedule_id=row["id"],
        easiness_factor=new_ef,
        interval_days=new_interval,
        repetitions=new_reps,
        next_review_at=next_review,
        quality=quality,
    )
    mark_review_answered(sent["id"])

    # Build confirmation
    quality_labels = {5: "记住了 ✅", 3: "模糊 🤔", 1: "忘了 ❌"}
    label = quality_labels.get(quality, str(quality))
    next_date = next_review[:10]

    return (
        f"收到！你的反馈：**{label}**\n\n"
        f"下次复习：**{next_date}**（{new_interval} 天后）\n"
        f"坚持复习，效果更佳！📚"
    )


def generate_weekly_integration(client=None, user_id: str = "") -> None:
    """Generate and push a weekly knowledge integration.

    Called by APScheduler (e.g., every Monday 10:00).
    """
    if client is None or not user_id:
        logger.warning("generate_weekly_integration: no client or user_id provided")
        return

    pages = get_reviewed_pages_since(days=7)
    if len(pages) < 2:
        logger.info("Thinker: only %d pages reviewed this week, skipping integration", len(pages))
        return

    integration = _generate_weekly_integration(pages)
    if integration is None:
        logger.warning("Thinker: failed to generate weekly integration")
        return

    new_qs = "\n".join(f"- {q}" for q in integration.new_questions)
    message = (
        f"📚 **本周知识整合：{integration.title}**\n\n"
        f"{integration.content}\n\n"
        f"**延伸思考：**\n{new_qs}\n\n"
        f"本周共复习了 {len(pages)} 个知识点，继续保持！🎯"
    )

    try:
        import asyncio
        asyncio.create_task(client.send_message(user_id, {
            "msgtype": "markdown",
            "markdown": {"content": message},
        }))
        logger.info("Thinker: pushed weekly integration '%s'", integration.title)
    except Exception as e:
        logger.error("Thinker: failed to push weekly integration: %s", e)
```

- [ ] **Step 2: Commit**

```bash
git add server/thinker.py
git commit -m "feat: create thinker module with SM-2, review push, feedback, weekly integration"
```

---

### Task 3: Wire thinker into bot.py

**Files:**
- Modify: `server/bot.py:58-139` — add thinker route + import + scheduler

- [ ] **Step 1: Add thinker import and scheduler**

In the import section of `server/bot.py`, add:
```python
from server.config import THINKER_USER_ID, THINKER_CHECK_INTERVAL
from server.thinker import check_due_reviews, handle_review_response, get_review_marker
```

In `KnowledgeBot.__init__`, add:
```python
self._start_thinker_scheduler()
```

Add the scheduler method:
```python
def _start_thinker_scheduler(self):
    """Schedule periodic thinker review checks."""
    if not THINKER_USER_ID:
        logger.info("Thinker disabled: no THINKER_USER_ID configured")
        return

    # Check every N hours for due reviews
    self.scheduler.add_job(
        self._run_thinker_check,
        "interval",
        hours=THINKER_CHECK_INTERVAL,
        id="thinker_review_check",
        replace_existing=True,
        misfire_grace_time=600,
    )
    self.scheduler.add_job(
        self._run_weekly_integration,
        "cron",
        day_of_week="mon",
        hour=10,
        minute=0,
        id="thinker_weekly_integration",
        replace_existing=True,
        misfire_grace_time=600,
    )
    logger.info(
        "Thinker scheduler started (check every %dh, weekly Mon 10:00, user=%s)",
        THINKER_CHECK_INTERVAL, THINKER_USER_ID,
    )
```

Add the async methods:
```python
async def _run_thinker_check(self):
    """Async wrapper for thinker review check."""
    try:
        pushed = await asyncio.to_thread(check_due_reviews, None, THINKER_USER_ID)
        # Actually send the messages
        for item in pushed:
            await self.client.send_message(THINKER_USER_ID, {
                "msgtype": "markdown",
                "markdown": {"content": item["message"]},
            })
            # Record sent review
            await asyncio.to_thread(
                record_sent_review,
                item["schedule_id"], item["page_id"], item["marker_id"],
            )
            logger.info("Thinker: pushed review '%s' to %s", item["page_title"], THINKER_USER_ID)
    except Exception:
        logger.exception("Thinker check failed")

async def _run_weekly_integration(self):
    """Async wrapper for weekly integration."""
    from server.thinker import generate_weekly_integration
    try:
        await asyncio.to_thread(generate_weekly_integration, self.client, THINKER_USER_ID)
    except Exception:
        logger.exception("Weekly integration failed")
```

- [ ] **Step 2: Add thinker route in _on_text**

Right after the `/code` handler and its early return, add the thinker route:

```python
# ── Thinker review response route ──
quote = body.get('quote')
if quote and isinstance(quote, dict):
    quoted_text = quote.get('text', '')
    if '#review_' in quoted_text:
        logger.info("Thinker: user %s replied to review: %s", user_id, content[:30])
        response = handle_review_response(quoted_text, content)
        await self.client.reply(frame, {
            "msgtype": "markdown",
            "markdown": {"content": response},
        })
        logger.info("Thinker response sent to user_id=%s", user_id)
        return
```

Note: This goes before the existing `body.get("quote")` usage in the original code (line 64). Since the original code only extracts `quote` for storing, we need to check the `quote` var after it's defined. Move the existing `quote = body.get('quote')` before the `/code` handler, so it's available for both the thinker route and the existing code's logging.

- [ ] **Step 3: Add record_sent_review import**

```python
from storage.models import record_sent_review
```

- [ ] **Step 4: Commit**

```bash
git add server/bot.py
git commit -m "feat: wire thinker route and APScheduler into bot.py"
```

---

### Task 4: Initialize review_schedule when pages are created

**Files:**
- Modify: `agent/nodes/store.py:309-316` — after `upsert_page` call

- [ ] **Step 1: Add import and init call**

Add import:
```python
from storage.models import init_review_schedule
```

After `upsert_page(...)` returns a pid (line 317), add:
```python
# Initialize spaced repetition schedule for new pages
init_review_schedule(pid)
```

The full loop context:
```python
        pid = upsert_page(
            title=wp.title,
            file_path=file_path,
            tags=tags,
            sources=sources,
            checksum=checksum,
            content=full_content,
        )
        init_review_schedule(pid)  # <-- new line
        saved_ids.append(pid)
```

- [ ] **Step 2: Commit**

```bash
git add agent/nodes/store.py
git commit -m "feat: init review_schedule when pages are created"
```

---

### Task 5: Add thinker config

**Files:**
- Modify: `server/config.py`

- [ ] **Step 1: Add config vars**

```python
# Thinker module (spaced repetition)
THINKER_USER_ID = os.getenv("THINKER_USER_ID", "")
THINKER_CHECK_INTERVAL = int(os.getenv("THINKER_CHECK_INTERVAL", "4"))
```

- [ ] **Step 2: Commit**

```bash
git add server/config.py
git commit -m "feat: add thinker config (THINKER_USER_ID, THINKER_CHECK_INTERVAL)"
```

---

### Task 6: Write tests

**Files:**
- Create: `tests/unit/test_thinker.py`

- [ ] **Step 1: Write test file**

```python
"""Unit tests for spaced repetition thinker module."""
import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from storage.database import init_db


@pytest.fixture(autouse=True)
def temp_db(monkeypatch, tmp_path):
    db_dir = tmp_path / "data"
    db_dir.mkdir()
    monkeypatch.setattr("storage.database.DB_DIR", str(db_dir))
    monkeypatch.setattr("storage.database.DB_PATH", str(db_dir / "knowledge.db"))
    init_db()


@pytest.fixture
def wiki_dir(monkeypatch, tmp_path):
    wiki_dir = tmp_path / "wiki"
    pages_dir = wiki_dir / "pages"
    pages_dir.mkdir(parents=True)
    monkeypatch.setattr("storage.wiki_storage.WIKI_DIR", str(wiki_dir))
    return wiki_dir


@pytest.fixture
def sample_page(wiki_dir):
    """Insert a sample wiki page and return its id."""
    from storage.database import get_connection
    content = "---\ntitle: test page\ntags: [test]\n---\n\nTest content for SM-2 review."
    file_path = "pages/test-page.md"
    # Write the disk file
    full_path = wiki_dir / file_path
    full_path.parent.mkdir(parents=True, exist_ok=True)
    full_path.write_text(content, encoding="utf-8")

    conn = get_connection()
    conn.execute(
        "INSERT INTO pages (title, file_path, tags) VALUES (?, ?, ?)",
        ("test page", file_path, '["test"]'),
    )
    conn.commit()
    pid = conn.execute("SELECT id FROM pages WHERE title = ?", ("test page",)).fetchone()[0]
    conn.close()
    return pid


def test_sm2_perfect_recall():
    """Quality=5 → EF increases, interval grows."""
    from server.thinker import _sm2_update
    ef, interval, reps = _sm2_update(5, 2.5, 1, 0)
    assert reps == 1
    assert interval == 1
    assert ef > 2.5


def test_sm2_second_review():
    """Second perfect recall → interval = 6."""
    from server.thinker import _sm2_update
    ef, interval, reps = _sm2_update(5, 2.5, 1, 1)
    assert reps == 2
    assert interval == 6


def test_sm2_subsequent_reviews():
    """Third+ perfect recall → interval = round(prev * EF)."""
    from server.thinker import _sm2_update
    ef, interval, reps = _sm2_update(5, 2.5, 6, 2)
    assert reps == 3
    assert interval == 15  # round(6 * 2.5)


def test_sm2_forgot_resets():
    """Quality=1 → reps=0, interval=1."""
    from server.thinker import _sm2_update
    ef, interval, reps = _sm2_update(1, 2.5, 6, 3)
    assert reps == 0
    assert interval == 1
    assert ef < 2.5  # EF decreases


def test_sm2_ef_floor():
    """EF should not go below MIN_EF (1.3)."""
    from server.thinker import _sm2_update
    ef, interval, reps = _sm2_update(0, 1.3, 1, 0)
    assert ef >= 1.3


def test_sm2_ef_ceiling():
    """EF should not exceed MAX_EF (3.0)."""
    from server.thinker import _sm2_update
    ef, interval, reps = _sm2_update(5, 3.0, 1, 0)
    assert ef <= 3.0


def test_get_review_marker():
    """Marker format: #review_{pageId}_{date}."""
    from server.thinker import get_review_marker
    marker = get_review_marker(42)
    assert marker.startswith("#review_42_")
    assert len(marker) == len("#review_42_20260517")


def test_handle_review_response_bad_quote():
    """No marker in quote → friendly error."""
    from server.thinker import handle_review_response
    result = handle_review_response("some random text", "记住了")
    assert "无法识别" in result


def test_handle_review_response_no_feedback():
    """Unrecognized feedback → guidance message."""
    from server.thinker import handle_review_response
    result = handle_review_response("#review_1_20260517 hello", "huh?")
    assert "记住了" in result


def test_init_review_schedule_creates_record(sample_page):
    """init_review_schedule inserts a row with default 1-day interval."""
    from storage.models import init_review_schedule
    sid = init_review_schedule(sample_page)
    assert sid > 0

    from storage.database import get_connection
    conn = get_connection()
    row = conn.execute("SELECT * FROM review_schedule WHERE id = ?", (sid,)).fetchone()
    conn.close()
    assert row is not None
    assert row["page_id"] == sample_page
    assert row["interval_days"] == 1
    assert row["repetitions"] == 0
    assert row["easiness_factor"] == 2.5


def test_get_due_reviews_returns_due_only(sample_page):
    """Only schedules with next_review_at <= now are returned."""
    from storage.models import init_review_schedule, get_due_reviews
    # Default next_review_at = now + 1 day → not due
    init_review_schedule(sample_page)
    due = get_due_reviews(limit=10)
    assert len(due) == 0

    # Set to past → should be due
    from storage.database import get_connection
    conn = get_connection()
    conn.execute(
        "UPDATE review_schedule SET next_review_at = ? WHERE page_id = ?",
        ((datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S"), sample_page),
    )
    conn.commit()
    conn.close()

    due = get_due_reviews(limit=10)
    assert len(due) >= 1
    assert due[0]["page_id"] == sample_page


def test_update_review_schedule_persists(sample_page):
    """After update, SM-2 params are saved."""
    from storage.models import init_review_schedule, update_review_schedule, get_due_reviews
    from storage.database import get_connection

    sid = init_review_schedule(sample_page)
    # Make it due first
    conn = get_connection()
    conn.execute(
        "UPDATE review_schedule SET next_review_at = ? WHERE id = ?",
        ((datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S"), sid),
    )
    conn.commit()
    conn.close()

    update_review_schedule(sid, 2.6, 6, 1,
                           (datetime.now() + timedelta(days=6)).strftime("%Y-%m-%d %H:%M:%S"), 5)

    conn = get_connection()
    row = conn.execute("SELECT * FROM review_schedule WHERE id = ?", (sid,)).fetchone()
    conn.close()
    assert row["easiness_factor"] == 2.6
    assert row["interval_days"] == 6
    assert row["repetitions"] == 1
    assert row["last_quality"] == 5


def test_has_pending_review(sample_page):
    """has_pending_review detects unanswered sent reviews."""
    from storage.models import init_review_schedule, has_pending_review, record_sent_review
    sid = init_review_schedule(sample_page)

    # No sent reviews yet
    assert has_pending_review(sample_page) is False

    record_sent_review(sid, sample_page, "#review_1_test")
    assert has_pending_review(sample_page) is True


def test_get_reviewed_pages_since(sample_page):
    """get_reviewed_pages_since returns pages reviewed in the window."""
    from storage.models import init_review_schedule, update_review_schedule, get_reviewed_pages_since
    from storage.database import get_connection

    sid = init_review_schedule(sample_page)

    # Simulate a recent review
    from datetime import datetime
    update_review_schedule(sid, 2.5, 6, 1,
                           (datetime.now() + timedelta(days=6)).strftime("%Y-%m-%d %H:%M:%S"), 5)

    pages = get_reviewed_pages_since(days=7)
    assert len(pages) >= 1
    assert pages[0]["page_id"] == sample_page


def test_get_sent_review_by_marker(sample_page):
    """Can look up sent review by marker."""
    from storage.models import init_review_schedule, record_sent_review, get_sent_review_by_marker
    sid = init_review_schedule(sample_page)
    record_sent_review(sid, sample_page, "#review_99_test")

    sent = get_sent_review_by_marker("#review_99_test")
    assert sent is not None
    assert sent["page_id"] == sample_page


def test_handle_review_response_already_reviewed(sample_page):
    """Already reviewed → message says already answered."""
    from storage.models import init_review_schedule, record_sent_review
    from server.thinker import handle_review_response
    sid = init_review_schedule(sample_page)
    record_sent_review(sid, sample_page, "#review_1_already")

    from storage.database import get_connection
    conn = get_connection()
    conn.execute("UPDATE sent_reviews SET status = 'reviewed' WHERE marker_id = ?",
                 ("#review_1_already",))
    conn.commit()
    conn.close()

    result = handle_review_response("#review_1_already", "记住了")
    assert "已经回复过" in result


def test_handle_review_response_success(sample_page):
    """Full feedback flow: marker found → SM-2 update → confirmation."""
    from storage.models import init_review_schedule, record_sent_review
    from storage.database import get_connection
    from server.thinker import handle_review_response

    sid = init_review_schedule(sample_page)
    # Make it due
    conn = get_connection()
    conn.execute(
        "UPDATE review_schedule SET next_review_at = ? WHERE id = ?",
        (("2000-01-01 00:00:00", sid)),
    )
    conn.commit()
    conn.close()

    record_sent_review(sid, sample_page, "#review_2_success")

    result = handle_review_response("#review_2_success", "记住了")
    assert "收到" in result
    assert "下次复习" in result

    # Verify SM-2 was updated
    conn = get_connection()
    row = conn.execute("SELECT * FROM review_schedule WHERE id = ?", (sid,)).fetchone()
    conn.close()
    assert row["repetitions"] == 1
    assert row["last_quality"] == 5


def test_check_due_reviews_no_client():
    """No client provided → returns empty list."""
    from server.thinker import check_due_reviews
    result = check_due_reviews(client=None, user_id="")
    assert result == []
```

- [ ] **Step 2: Run tests**

```bash
python -m pytest tests/unit/test_thinker.py -v
```

Expected: All PASS

- [ ] **Step 3: Commit**

```bash
git add tests/unit/test_thinker.py
git commit -m "test: add thinker module unit tests (SM-2, DB helpers, feedback)"
```

---

### Task 7: Add migration script for existing pages

**Files:**
- Create: `scripts/migrate_review_schedule.py`

- [ ] **Step 1: Create migration script**

```python
"""One-time migration: initialize review_schedule for existing wiki pages.

Run: python -m scripts.migrate_review_schedule
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from storage.database import get_connection, init_db
from storage.models import init_review_schedule


def main():
    init_db()
    conn = get_connection()
    pages = conn.execute(
        "SELECT id, title FROM pages WHERE status = 'active'"
    ).fetchall()
    conn.close()

    count = 0
    for p in pages:
        if init_review_schedule(p["id"]):
            count += 1

    print(f"Initialized review schedule for {count} existing pages")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add scripts/migrate_review_schedule.py
git commit -m "feat: add migration script for existing pages review_schedule"
```

---

## Self-Review

**1. Spec coverage:** All spec items covered — SM-2 algorithm (Task 2), DB tables (Task 1), review push (Task 2+3), feedback handling (Task 2+3), weekly integration (Task 2+3), store.py init (Task 4), config (Task 5). No gaps.

**2. Placeholder scan:** No TBD, TODO, or "implement later." Complete code in every step.

**3. Type consistency:** All function signatures match across tasks. `init_review_schedule(pid)` in Task 4 matches `def init_review_schedule(page_id: int)` in Task 1. `handle_review_response(quote_text, user_feedback)` in Task 2 matches the call in bot.py Task 3. `check_due_reviews(client, user_id)` in Task 2 matches the call in bot.py Task 3.

**4. Bot.py quote handling note:** The existing bot.py line 64 already defines `quote = body.get('quote')`. The thinker route in Task 3 Step 2 reuses this variable — no duplication.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-17-spaced-repetition-thinker.md`. Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
