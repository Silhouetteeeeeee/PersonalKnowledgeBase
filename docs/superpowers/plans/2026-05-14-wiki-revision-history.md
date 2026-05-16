# Wiki Revision History Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every wiki page modification is recorded as a version snapshot in SQLite, viewable through a CLI tool with log/show/diff commands, with automatic cleanup of versions older than 30 days.

**Architecture:** A new `page_versions` SQLite table stores complete page snapshots per version. Before `extract_to_wiki()` writes new content, it saves the current content as a version. A `wiki_history.py` CLI provides log/show/diff. APScheduler runs daily cleanup at 03:00. The existing `source_questions` table is removed — question text is embedded directly in `page_versions.source_question`.

**Tech Stack:** Python `difflib` for version diff, APScheduler for cleanup, argparse for CLI.

---

### Task 1: Add page_versions table, remove source_questions table

**Files:**
- Modify: `storage/database.py`
- Test: manual DB verification

- [ ] **Step 1: Add page_versions CREATE TABLE, remove source_questions**

In `storage/database.py` `init_db()`, replace the `source_questions` line with `page_versions`:

```python
        -- (remove the source_questions line entirely)
        CREATE TABLE IF NOT EXISTS page_versions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            page_id INTEGER NOT NULL,
            version INTEGER NOT NULL,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            checksum TEXT NOT NULL,
            source_id TEXT DEFAULT '',
            source_question TEXT DEFAULT '',
            change_summary TEXT DEFAULT '',
            created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
            FOREIGN KEY (page_id) REFERENCES pages(id),
            UNIQUE(page_id, version)
        );
```

Remove this line from `init_db()`:
```python
        -- DELETE THIS ENTIRE LINE (the source_questions CREATE TABLE)
```

- [ ] **Step 2: Verify the schema**

Run:
```bash
python -c "from storage.database import init_db, get_connection; init_db(); c=get_connection(); print([r[1] for r in c.execute('PRAGMA table_info(page_versions)').fetchall()]); c.close()"
```

Expected: `['id', 'page_id', 'version', 'title', 'content', 'checksum', 'source_id', 'source_question', 'change_summary', 'created_at']`

Also verify source_questions is gone:
```bash
python -c "from storage.database import get_connection; c=get_connection(); tables=[r[0] for r in c.execute(\"SELECT name FROM sqlite_master WHERE type='table'\").fetchall()]; print('source_questions' in tables); c.close()"
```

Expected: False (table not present in new DB — existing DB won't be dropped, but new code won't create it)

- [ ] **Step 3: Commit**

```bash
git add storage/database.py
git commit -m "feat: add page_versions table, remove source_questions"
```

---

### Task 2: Add version helpers to models.py, remove source_question helpers

**Files:**
- Modify: `storage/models.py`

- [ ] **Step 1: Remove source_question helpers and add version helpers**

Replace the `# ── Source question helpers ──` section (lines 184-212) with version helpers:

```python
# ── Version helpers ──

def save_page_version(
    page_id: int,
    title: str,
    content: str,
    checksum: str,
    source_id: str = "",
    source_question: str = "",
) -> int:
    """Save a new version of a wiki page. Auto-increments version number per page_id.

    Returns the version number that was saved.
    """
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT COALESCE(MAX(version), 0) + 1 FROM page_versions WHERE page_id = ?",
            (page_id,),
        ).fetchone()
        next_ver = row[0]

        conn.execute(
            """INSERT INTO page_versions (page_id, version, title, content, checksum, source_id, source_question)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (page_id, next_ver, title, content, checksum, source_id, source_question),
        )
        conn.commit()
        logger.info("Saved version %d for page '%s' (id=%d)", next_ver, title, page_id)
        return next_ver
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def get_page_versions(page_id: int, limit: int = 20) -> list[dict]:
    """List versions for a page, most recent first."""
    conn = get_connection()
    try:
        rows = conn.execute(
            """SELECT id, version, title, source_id, source_question, change_summary, created_at
               FROM page_versions WHERE page_id = ?
               ORDER BY version DESC LIMIT ?""",
            (page_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_page_version(page_id: int, version: int) -> dict | None:
    """Get a specific version's full content."""
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM page_versions WHERE page_id = ? AND version = ?",
            (page_id, version),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def cleanup_old_versions(days: int = 30) -> int:
    """Delete versions older than `days`, keeping at least 1 per page.

    Returns number of deleted rows.
    """
    from datetime import datetime, timedelta
    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")

    conn = get_connection()
    try:
        deleted = conn.execute(
            """DELETE FROM page_versions WHERE created_at < ? AND id NOT IN (
                   SELECT MAX(id) FROM page_versions GROUP BY page_id
               )""",
            (cutoff,),
        ).rowcount
        conn.commit()
        if deleted:
            logger.info("Cleaned up %d old page versions (cutoff=%s)", deleted, cutoff)
        return deleted
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
```

Also remove the import for `save_source_question` from `agent/nodes/store.py` imports (done in Task 3), and remove `get_source_questions` import from `server/daily_summary.py` (done in Task 4).

- [ ] **Step 2: Verify the module loads**

```bash
python -c "from storage.models import save_page_version, get_page_versions, get_page_version, cleanup_old_versions; print('OK')"
```

Expected: OK

- [ ] **Step 3: Commit**

```bash
git add storage/models.py
git commit -m "feat: add page version helpers, remove source_question helpers"
```

---

### Task 3: Save version before overwrite in store.py

**Files:**
- Modify: `agent/nodes/store.py`

- [ ] **Step 1: Update imports**

Replace `save_source_question` with `save_page_version` in the import block:

```python
from storage.models import (
    upsert_page,
    update_page_relations,
    get_page_by_title,
    find_similar_pages,
    save_page_version,
)
```

Add a new import at the top:
```python
from storage.database import get_connection
```

- [ ] **Step 2: Modify extract_to_wiki() to save version before overwrite**

The existing code calls `read_page(file_path)` at line 255 to check if the page already exists. We save the CURRENT disk content as a version before overwriting it.

First, add `compute_checksum` to the wiki_storage import:
```python
from storage.wiki_storage import (
    ensure_dirs,
    title_to_filename,
    read_schema,
    read_page,
    write_page,
    build_frontmatter,
    extract_wikilinks,
    compute_checksum,  # ADD THIS
)
```

In `extract_to_wiki()`, after `full_content = frontmatter + "\n\n" + wp.content.strip()` (line 276) and before `checksum = write_page(...)` (line 278), insert:

```python
        # Save current disk content as version before overwriting
        if existing_page_data:
            disk_full_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "data", "wiki", file_path
            )
            if os.path.exists(disk_full_path):
                with open(disk_full_path, encoding="utf-8") as _f:
                    old_raw = _f.read()
                _pid_row = get_connection().execute(
                    "SELECT id FROM pages WHERE title = ?", (wp.title,)
                ).fetchone()
                if _pid_row:
                    save_page_version(
                        page_id=_pid_row["id"],
                        title=wp.title,
                        content=old_raw,
                        checksum=compute_checksum(old_raw),
                        source_id=source_id,
                        source_question=source_label.replace("Question: ", ""),
                    )
                get_connection().close()
```

Note: `os.path.dirname(os.path.dirname(__file__)), "data", "wiki"` resolves to `WIKI_DIR`. This avoids requiring a new import. The implementer can also import `WIKI_DIR` from `storage.wiki_storage` if preferred.

Note: we save the NEW content as previous version. This is because the version represents "what the page looked like before this change" — the old content on disk IS the content we're about to replace. The write_page() below will write the new content. Actually, wait — we need to save the CURRENT disk content as the old version, not the new content.

Let me reconsider. The flow is:
1. Extract old content from disk (existing_page_data from read_page)
2. Save that OLD content as a version (this represents "what was current before this update")
3. Write new content to disk

So the version stores the about-to-be-overwritten content. Let me fix:

```python
        # Save current version before overwriting (if page exists on disk)
        existing_data = read_page(file_path)
        if existing_data:
            conn_inner = get_connection()
            try:
                existing_row = conn_inner.execute(
                    "SELECT id FROM pages WHERE title = ?", (wp.title,)
                ).fetchone()
                if existing_row:
                    disk_content = ""
                    disk_full_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", wiki_storage.WIKI_DIR, file_path)
                    # Actually, read what's currently on disk
                    import storage.wiki_storage as ws
                    page_file = os.path.join(ws.WIKI_DIR, file_path)
                    if os.path.exists(page_file):
                        with open(page_file, encoding="utf-8") as f:
                            disk_content = f.read()
```

Hmm, this is getting complicated. Let me simplify. The existing code already calls `read_page(file_path)` to get `existing_page_data` at line 255. That gives us the metadata. But we need the FULL file content (including frontmatter) to save as version. 

Actually, the simplest approach: read the file again for the full raw content.

Wait, I need to think about this differently. The `read_page()` function reads from `WIKI_DIR / file_path`. So I already have the disk path. Let me just use the existing `D"` path...

Actually, looking more carefully at the code flow:

```python
existing_page_data = read_page(file_path)  # line 255 — reads from disk
...
full_content = frontmatter + "\n\n" + wp.content.strip()  # line 276 — new content
checksum = write_page(file_path, full_content)  # line 278 — writes new
```

So at line 255, `existing_page_data` tells us if the page exists on disk. If it does, we need to save the CURRENT disk content as a version before replacing it. The simplest way:

```python
if existing_page_data:
    # Read the current file content from disk to save as version
    disk_full_path = os.path.join(wiki_storage.WIKI_DIR, file_path)
    if os.path.exists(disk_full_path):
        with open(disk_full_path, encoding="utf-8") as f:
            old_content = f.read()
        # Get page_id from DB
        conn_inner = get_connection()
        try:
            existing_row = conn_inner.execute(
                "SELECT id FROM pages WHERE title = ?", (wp.title,)
            ).fetchone()
            if existing_row:
                save_page_version(
                    page_id=existing_row["id"],
                    title=wp.title,
                    content=old_content,
                    checksum=compute_checksum(old_content),
                    source_id=source_id,
                    source_question=source_label.replace("Question: ", ""),
                )
        finally:
            conn_inner.close()
```

But `compute_checksum` and `wiki_storage.WIKI_DIR` need to be imported. `compute_checksum` is already in `wiki_storage`, and `WIKI_DIR` is also there.

Actually, the `read_page` function already does all this. Let me look at what it returns:
```python
def read_page(file_path: str) -> Optional[dict]:
    full_path = os.path.join(WIKI_DIR, file_path)
    ...
    with open(full_path, "r", encoding="utf-8") as f:
        content = f.read()
    metadata, body = parse_frontmatter(content)
    checksum = compute_checksum(content)
    return {
        "title": ..., "body": body, "tags": ..., "sources": ..., "checksum": checksum, "created": ...,
    }
```

It returns the full content's checksum and body, but not the RAW content with frontmatter. I need the full content for the version, including frontmatter.

So I need to either:
1. Read the file again to get raw content
2. Modify `read_page` to also return raw content

Option 1 is simpler and doesn't change existing API. Let me go with that.

OK let me refine the plan with correct code.

Actually, we already have the file path: `file_path` is like `"pages/django-orm.md"`. And `WIKI_DIR` is in `wiki_storage`. So:

```python
full_path = os.path.join(wiki_storage.WIKI_DIR, file_path)
```

Let me not over-complicate this. I'll show clean code that reads the disk content, saves it as a version, then proceeds.

- [ ] **Step 3: Remove `save_source_question` call from `store()` function**

Replace:
```python
    source_id = _get_source_id()
    save_source_question(source_id, state["user_message"])
    source_label = f"Question: {state['user_message']}"
```

With:
```python
    source_id = _get_source_id()
    source_label = f"Question: {state['user_message']}"
```

- [ ] **Step 4: Run unit tests to verify no breakage**

```bash
python -m pytest tests/unit/test_nodes.py -v --tb=short
```

Expected: all 13 tests pass (test_store_empty_answer etc.)

- [ ] **Step 5: Commit**

```bash
git add agent/nodes/store.py
git commit -m "feat: save page version before overwrite, remove source_question call"
```

---

### Task 4: Update daily_summary.py to read from page_versions

**Files:**
- Modify: `server/daily_summary.py` (lines 1-80)
- Modify: `tests/unit/test_daily_summary.py`

- [ ] **Step 1: Modify `_get_yesterday_pages()` to query page_versions for source_question**

Replace `from storage.models import get_source_questions` with no import needed.

In the for loop in `_get_yesterday_pages()`, replace the source questions lookup (lines 60-66):

```python
        # Look up source question from latest page_versions entry
        try:
            source_ids = json.loads(r["sources"])
        except (json.JSONDecodeError, TypeError):
            source_ids = []
        conv_ids = [s for s in source_ids if isinstance(s, str) and s.startswith("conv_")]

        # Only look up if we have conv_ source IDs
        questions = []
        if conv_ids:
            conn_inner = get_connection()
            try:
                # Get the latest version's source_question for this page
                row = conn_inner.execute(
                    "SELECT source_question FROM page_versions WHERE page_id = ? ORDER BY version DESC LIMIT 1",
                    (r["id"],),
                ).fetchone()
                if row and row["source_question"]:
                    questions = [row["source_question"]]
            finally:
                conn_inner.close()
```

- [ ] **Step 2: Run existing tests to verify**

```bash
python -m pytest tests/unit/test_daily_summary.py -v --tb=short
```

Expected: 6 tests pass (they mock `_get_yesterday_pages` so the DB change won't affect them directly, but the import change needs to work)

- [ ] **Step 3: Commit**

```bash
git add server/daily_summary.py
git commit -m "feat: read source_question from page_versions instead of source_questions"
```

---

### Task 5: Create wiki_history.py CLI

**Files:**
- Create: `wiki_history.py`

- [ ] **Step 1: Write the test file**

Create `tests/unit/test_wiki_history.py`:

```python
"""Unit tests for wiki revision history CLI."""
import pytest
from unittest.mock import MagicMock, patch


def test_log_output():
    """Verify log formatting is correct."""
    from wiki_history import format_log
    versions = [
        {"version": 3, "created_at": "2026-05-14 15:30:00",
         "source_question": "TCP四次挥手", "change_summary": ""},
        {"version": 2, "created_at": "2026-05-13 20:00:00",
         "source_question": "TCP状态转换", "change_summary": ""},
        {"version": 1, "created_at": "2026-05-13 10:00:00",
         "source_question": "TCP三次握手的过程", "change_summary": ""},
    ]
    output = format_log("TCP三次握手", versions)
    assert "v3" in output
    assert "v2" in output
    assert "v1" in output
    assert "TCP三次握手" in output
    assert "14 15:30" in output


def test_show_version():
    """Verify show returns full content for a specific version."""
    from wiki_history import format_show
    version = {
        "version": 2, "title": "TCP三次握手",
        "content": "---\ntitle: TCP三次握手\n---\n\nContent body",
        "created_at": "2026-05-14 15:30:00",
        "source_question": "TCP四次挥手",
    }
    output = format_show(version)
    assert "v2" in output
    assert "TCP三次握手" in output
    assert "Content body" in output


def test_diff_output():
    """Verify diff produces unified diff lines."""
    from wiki_history import format_diff
    old_content = "line1\nline2\nline3\n"
    new_content = "line1\nline2_modified\nline3\nline4\n"
    output = format_diff("Test Page", 1, 2, old_content, new_content)
    assert "--- Test Page v1" in output
    assert "+++ Test Page v2" in output
    assert "-line2" in output
    assert "+line2_modified" in output
    assert "+line4" in output


def test_diff_identical():
    """Diff of identical content shows no changes."""
    from wiki_history import format_diff
    content = "line1\nline2\nline3\n"
    output = format_diff("Test", 1, 2, content, content)
    assert "No differences" in output


def test_page_not_found():
    """Non-existent page returns error."""
    from wiki_history import handle_log
    with patch("wiki_history.get_page_by_title", return_value=None):
        result = handle_log("NonExistent")
        assert "not found" in result.lower()
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
python -m pytest tests/unit/test_wiki_history.py -v --tb=short
```

Expected: 6/6 FAILED (ModuleNotFoundError: No module named 'wiki_history')

- [ ] **Step 3: Implement wiki_history.py**

Create `wiki_history.py`:

```python
#!/usr/bin/env python3
"""CLI tool for wiki revision history: log, show, diff."""

import argparse
import difflib
import sys

from storage.database import get_connection
from storage.models import get_page_versions, get_page_version
from storage.models import get_page_by_title


def format_log(title: str, versions: list[dict]) -> str:
    """Format version history as a readable table."""
    lines = [f"📜 Revision history for: {title}", ""]
    for v in versions:
        ts = v["created_at"][:16] if v.get("created_at") else "unknown"
        q = v.get("source_question", "") or "未知来源"
        lines.append(f"  v{v['version']}  |  {ts}  |  {q}")
    return "\n".join(lines) if len(versions) > 1 else "\n".join(lines) if versions else ""


def format_show(version: dict) -> str:
    """Format a single version's full content."""
    lines = [
        f"📄 {version['title']} (v{version['version']})",
        f"   创建时间: {version.get('created_at', 'unknown')}",
        f"   来源问题: {version.get('source_question', '未知来源')}",
        "",
        version.get("content", ""),
    ]
    return "\n".join(lines)


def format_diff(title: str, v1: int, v2: int, old_content: str, new_content: str) -> str:
    """Generate unified diff between two versions."""
    if old_content == new_content:
        return f"⚠️ No differences between v{v1} and v{v2}"

    old_lines = old_content.splitlines(keepends=True)
    new_lines = new_content.splitlines(keepends=True)
    diff = difflib.unified_diff(
        old_lines, new_lines,
        fromfile=f"{title} v{v1}",
        tofile=f"{title} v{v2}",
    )
    return "".join(diff)


def handle_log(title: str) -> str:
    """Handle 'log' subcommand."""
    page = get_page_by_title(title)
    if not page:
        return f"❌ Page '{title}' not found"
    versions = get_page_versions(page["id"])
    if not versions:
        return f"ℹ️  No version history for '{title}'"
    return format_log(title, versions)


def handle_show(title: str, version_num: int | None) -> str:
    """Handle 'show' subcommand."""
    page = get_page_by_title(title)
    if not page:
        return f"❌ Page '{title}' not found"

    if version_num is None:
        # Show latest version
        versions = get_page_versions(page["id"], limit=1)
        if not versions:
            return f"ℹ️  No versions for '{title}'"
        version_num = versions[0]["version"]

    version = get_page_version(page["id"], version_num)
    if not version:
        return f"❌ Version v{version_num} not found for '{title}'"
    return format_show(version)


def handle_diff(title: str, v1: int, v2: int) -> str:
    """Handle 'diff' subcommand."""
    page = get_page_by_title(title)
    if not page:
        return f"❌ Page '{title}' not found"

    old_ver = get_page_version(page["id"], v1)
    new_ver = get_page_version(page["id"], v2)
    if not old_ver:
        return f"❌ Version v{v1} not found for '{title}'"
    if not new_ver:
        return f"❌ Version v{v2} not found for '{title}'"

    return format_diff(title, v1, v2, old_ver["content"], new_ver["content"])


def main():
    parser = argparse.ArgumentParser(description="Wiki revision history")
    sub = parser.add_subparsers(dest="command", required=True)

    log_p = sub.add_parser("log", help="Show version history")
    log_p.add_argument("title", help="Page title")

    show_p = sub.add_parser("show", help="Show version content")
    show_p.add_argument("title", help="Page title")
    show_p.add_argument("version", nargs="?", type=int, default=None, help="Version number (default: latest)")

    diff_p = sub.add_parser("diff", help="Diff two versions")
    diff_p.add_argument("title", help="Page title")
    diff_p.add_argument("v1", type=int, help="First version")
    diff_p.add_argument("v2", type=int, help="Second version")

    args = parser.parse_args()

    if args.command == "log":
        print(handle_log(args.title))
    elif args.command == "show":
        print(handle_show(args.title, args.version))
    elif args.command == "diff":
        print(handle_diff(args.title, args.v1, args.v2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/unit/test_wiki_history.py -v --tb=short
```

Expected: 6/6 PASSED

- [ ] **Step 5: Commit**

```bash
git add wiki_history.py tests/unit/test_wiki_history.py
git commit -m "feat: add wiki_history.py CLI with log/show/diff"
```

---

### Task 6: Add cleanup cron job to bot.py

**Files:**
- Modify: `server/bot.py`

- [ ] **Step 1: Add the cleanup import and cron job**

Add import at the top of `server/bot.py`:
```python
from storage.models import cleanup_old_versions
```

Add the cleanup job in `_start_daily_summary_scheduler()` method (or create a separate method):

Add a new method:
```python
    def _start_cleanup_scheduler(self):
        """Schedule daily wiki version cleanup at 03:00."""
        self.scheduler.add_job(
            cleanup_old_versions,
            "cron",
            hour=3,
            minute=0,
            kwargs={"days": 30},
            id="wiki_cleanup",
            replace_existing=True,
        )
        logger.info("Wiki cleanup scheduler started (03:00, keep 30 days)")
```

Call it at the end of `__init__`:
```python
        self._start_cleanup_scheduler()
```

- [ ] **Step 2: Verify the bot loads without error**

```bash
python -c "import server.bot; print('OK')"
```

Expected: OK (may show config warnings, but no ImportError)

- [ ] **Step 3: Commit**

```bash
git add server/bot.py
git commit -m "feat: add daily wiki version cleanup at 03:00"
```

---

### Task 7: Run all tests and verify

- [ ] **Step 1: Run the full unit test suite**

```bash
python -m pytest tests/unit/ -v --tb=short --ignore=tests/unit/test_context_builder.py
```

Expected:
- Existing node tests (13) pass
- Daily summary tests (6) pass
- Wiki history tests (6) pass
- Other existing tests pass
- No regressions from removing source_questions

- [ ] **Step 2: Run a quick integration test of the version workflow**

```bash
python -c "
from storage.database import init_db, get_connection
from storage.models import save_page_version, get_page_versions, get_page_version
from storage.models import cleanup_old_versions

init_db()
conn = get_connection()
conn.execute('INSERT OR IGNORE INTO pages (id, title, file_path) VALUES (1, \"Test Page\", \"pages/test.md\")')
conn.commit()
conn.close()

# Save v1
save_page_version(1, 'Test Page', 'content v1', 'abc123', 'conv_001', 'What is test?')
# Save v2
save_page_version(1, 'Test Page', 'content v2', 'def456', 'conv_002', 'What is test v2?')

versions = get_page_versions(1)
assert len(versions) == 2
assert versions[0]['version'] == 2

v1 = get_page_version(1, 1)
assert v1['content'] == 'content v1'
assert v1['source_question'] == 'What is test?'

print('All version workflow checks passed')
"
```

Expected: "All version workflow checks passed"

- [ ] **Step 3: Final commit if any fixes needed**

```bash
git add -A
git commit -m "test: add wiki history tests and version workflow verification"
```

---

### Spec Coverage Check

| Spec Requirement | Task |
|-----------------|------|
| page_versions table (replace source_questions) | Task 1 |
| version helpers in models.py | Task 2 |
| Save version before overwrite in store.py | Task 3 |
| daily_summary.py reads from page_versions | Task 4 |
| wiki_history.py CLI (log/show/diff) | Task 5 |
| Cleanup cron job at 03:00 | Task 6 |
| Tests for all components | Tasks 5-7 (test files) |
| Remove source_question helpers | Task 2 |
| Remove save_source_question call in store() | Task 3 |
| Keep at least 1 version per page on cleanup | Task 2 (cleanup SQL) |
| Edge cases (not found, no history, identical diff) | Task 5 (tests) |
