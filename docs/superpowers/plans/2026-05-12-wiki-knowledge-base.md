# Wiki Knowledge Base Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace isolated `knowledge_points` fragments with wiki-style topic pages stored as markdown files, using two-step CoT extraction for higher quality and `[[wikilink]]` for cross-references.

**Architecture:** Two-step CoT extraction (analyze → generate) writes wiki pages to `data/wiki/pages/<title>.md`. SQLite `pages` table is metadata-only (index + vector search). `page_relations` manages `[[wikilink]]` connections. Retrieval reads full page files instead of 50-char snippets.

**Tech Stack:** Python 3.12, SQLite + sqlite-vec, LangGraph, DeepSeek-Chat

---

### Task 1: Create `data/wiki/` directory structure and SCHEMA.md

**Files:**
- Create: `data/wiki/SCHEMA.md`
- Create: `data/wiki/index.md` (empty initial)

- [ ] **Step 1: Create the directory and initial files**

Run:
```bash
mkdir -p "D:/Programming/LangChain-Learning/data/wiki/pages"
```

- [ ] **Step 2: Create `data/wiki/SCHEMA.md`**

```markdown
# Wiki Schema

## Directory Layout
- `data/wiki/pages/` — page files, one `.md` per topic
- `data/wiki/index.md` — full index, auto-maintained
- `data/wiki/SCHEMA.md` — this file

## Page Format
Every page must start with YAML frontmatter (between `---` delimiters):

```
---
title: Django ORM
tags: [python, django, database]
sources: [conv_20250601_001]
created: 2025-06-01
updated: 2025-06-02
---
```

Required frontmatter fields:
- `title`: string, page title, match target for [[wikilink]]
- `tags`: list of strings, at least one tag (Chinese preferred)
- `sources`: list of strings, source conversation IDs

Auto-managed fields (do NOT write in prompt):
- `created`, `updated` — set by system on write

## Body Structure
- Start with a concise overview paragraph
- Use `##` for major sections, `###` for subsections
- End with a "与其他概念的关系" section listing [[wikilink]] entries
- Format: `- [[页面标题]] —— 关系描述`

## [[wikilink]] Rules
- Full page title only, no aliases
- Section reference: `[[title#heading]]`
- Keep links meaningful, don't force connections
- Broken links are acceptable (page may be created later)

## Style Guidelines
- Chinese preferred, English for technical terms
- Self-contained: readable without external context
- Precise and concise, not verbose
```

- [ ] **Step 3: Create empty `data/wiki/index.md`**

```markdown
# Wiki Index

| 页面 | 标签 | 来源 | 最后更新 |
|------|------|------|----------|
```

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "feat(wiki): add wiki directory structure and SCHEMA.md"
```

---

### Task 2: Add wiki tables to database schema

**Files:**
- Modify: `storage/database.py` (add create table statements)

- [ ] **Step 1: Add new table creation to `init_db()`**

Add after the existing error_vectors creation in `storage/database.py:init_db()`:

```python
    # Wiki pages (index-only, content stored as markdown files)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS pages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL UNIQUE,
            file_path TEXT NOT NULL,
            tags TEXT NOT NULL DEFAULT '[]',
            sources TEXT NOT NULL DEFAULT '[]',
            status TEXT NOT NULL DEFAULT 'active',
            checksum TEXT DEFAULT '',
            created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
            updated_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
        );
        CREATE VIRTUAL TABLE IF NOT EXISTS page_vectors USING vec0(
            embedding float[512] distance_metric=cosine
        );
        CREATE TABLE IF NOT EXISTS page_relations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_id INTEGER NOT NULL,
            target_title TEXT NOT NULL,
            relation_type TEXT NOT NULL DEFAULT 'wikilink',
            FOREIGN KEY (source_id) REFERENCES pages(id)
        );
    """)
```

- [ ] **Step 2: Run a quick test to verify tables are created**

```python
# Run this in Python:
from storage.database import init_db
init_db()
conn = get_connection()
tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name IN ('pages', 'page_relations')").fetchall()
print(tables)  # Should show both tables
```

Run:
```bash
python -c "from storage.database import init_db, get_connection; init_db(); c=get_connection(); print(c.execute(\"SELECT name FROM sqlite_master WHERE type='table' AND name IN ('pages','page_relations')\").fetchall())"
```

Expected:
```
[('pages',), ('page_relations',)]
```

- [ ] **Step 3: Commit**

```bash
git add storage/database.py
git commit -m "feat(wiki): add pages, page_vectors, page_relations tables"
```

---

### Task 3: Create WikiStorage utility (file I/O, frontmatter, wikilinks)

**Files:**
- Create: `storage/wiki_storage.py`

This module handles all filesystem operations for wiki pages. No database interaction.

- [ ] **Step 1: Create `storage/wiki_storage.py`**

```python
"""Filesystem operations for wiki pages: read/write files, parse frontmatter,
extract [[wikilinks]], title↔filename conversion, checksum."""

import hashlib
import logging
import os
import re
from typing import Optional

logger = logging.getLogger(__name__)

WIKI_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "wiki")
PAGES_DIR = os.path.join(WIKI_DIR, "pages")
INDEX_PATH = os.path.join(WIKI_DIR, "index.md")
SCHEMA_PATH = os.path.join(WIKI_DIR, "SCHEMA.md")


def ensure_dirs() -> None:
    os.makedirs(PAGES_DIR, exist_ok=True)


def title_to_filename(title: str) -> str:
    """Convert a page title to a filename.

    Examples:
        "Django ORM"     -> "django-orm.md"
        "Python 基础"    -> "python基础.md"
        "HTTP/2 协议"    -> "http2协议.md"
    """
    name = title.lower().strip()
    name = name.replace(" ", "-")
    # Remove characters unsafe for filenames
    name = re.sub(r'[<>:"/\\|?*]', "", name)
    return name + ".md"


def read_schema() -> str:
    """Read SCHEMA.md content. Returns empty string if file doesn't exist."""
    if not os.path.exists(SCHEMA_PATH):
        return ""
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        return f.read()


def parse_frontmatter(content: str) -> tuple[dict, str]:
    """Parse YAML-like frontmatter from markdown content.

    Handles the simple subset used by wiki pages (strings, lists, dates).
    Returns (metadata_dict, body_string).
    """
    match = re.match(r'^---\n(.*?)\n---\n(.*)', content, re.DOTALL)
    if not match:
        return {}, content.strip()

    raw = match.group(1)
    body = match.group(2).strip()
    metadata = _parse_simple_yaml(raw)
    return metadata, body


def _parse_simple_yaml(raw: str) -> dict:
    """Parse a simplified YAML subset: strings and lists only."""
    result = {}
    for line in raw.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        m = re.match(r'(\w+):\s*(.*)', line)
        if not m:
            continue
        key = m.group(1)
        val = m.group(2).strip()

        if val.startswith("[") and val.endswith("]"):
            # List: [a, b, c]
            items = [x.strip().strip("\"'") for x in val[1:-1].split(",") if x.strip()]
            result[key] = items
        elif val and val[0] in ("'", '"') and val[-1] == val[0]:
            result[key] = val[1:-1]
        else:
            result[key] = val
    return result


def build_frontmatter(title: str, tags: list[str], sources: list[str],
                      created: str = "", updated: str = "") -> str:
    """Build YAML frontmatter string."""
    lines = ["---"]
    lines.append(f"title: {title}")
    lines.append(f"tags: [{', '.join(tags)}]")
    lines.append(f"sources: [{', '.join(sources)}]")
    if created:
        lines.append(f"created: {created}")
    if updated:
        lines.append(f"updated: {updated}")
    lines.append("---")
    return "\n".join(lines)


def read_page(file_path: str) -> Optional[dict]:
    """Read a wiki page from disk.

    Returns dict with keys: title, body, tags, sources, checksum
    Returns None if file does not exist.
    """
    full_path = os.path.join(WIKI_DIR, file_path)
    if not os.path.exists(full_path):
        return None

    with open(full_path, "r", encoding="utf-8") as f:
        content = f.read()

    metadata, body = parse_frontmatter(content)
    checksum = compute_checksum(content)

    return {
        "title": metadata.get("title", ""),
        "body": body,
        "tags": metadata.get("tags", []),
        "sources": metadata.get("sources", []),
        "checksum": checksum,
    }


def write_page(file_path: str, content: str) -> str:
    """Write a wiki page to disk. Returns SHA256 checksum of written content."""
    full_path = os.path.join(WIKI_DIR, file_path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(content)
    checksum = compute_checksum(content)
    logger.info("Wrote wiki page: %s (%d bytes, checksum=%s)",
                file_path, len(content.encode("utf-8")), checksum[:12])
    return checksum


def compute_checksum(content: str) -> str:
    """Compute SHA256 checksum of page content."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def extract_wikilinks(content: str) -> list[str]:
    """Extract unique [[link]] targets from content.

    Handles both [[title]] and [[title#section]] forms.
    Returns unique target titles (without section suffix).
    """
    links = re.findall(r'\[\[(.+?)\]\]', content)
    # Strip section references: [[title#section]] -> [[title]]
    titles = [link.split("#")[0].strip() for link in links]
    seen = set()
    result = []
    for t in titles:
        if t and t not in seen:
            seen.add(t)
            result.append(t)
    return result


def read_index() -> str:
    """Read index.md content. Returns empty string if it doesn't exist."""
    if not os.path.exists(INDEX_PATH):
        return ""
    with open(INDEX_PATH, "r", encoding="utf-8") as f:
        return f.read()


def write_index(content: str) -> None:
    """Write index.md content."""
    os.makedirs(WIKI_DIR, exist_ok=True)
    with open(INDEX_PATH, "w", encoding="utf-8") as f:
        f.write(content)
    logger.info("Updated index.md (%d bytes)", len(content.encode("utf-8")))
```

- [ ] **Step 2: Run a quick smoke test**

Run:
```bash
python -c "
from storage.wiki_storage import *
ensure_dirs()
print('title_to_filename:', repr(title_to_filename('Django ORM')))
print('title_to_filename cn:', repr(title_to_filename('Python 基础')))
content = '---\ntitle: Test\ntags: [a, b]\nsources: [c]\n---\n\nBody text with [[link]]'
meta, body = parse_frontmatter(content)
print('parse:', meta, body)
print('wikilinks:', extract_wikilinks(content))
print('checksum:', compute_checksum(content))
"
```

Expected:
```
title_to_filename: 'django-orm.md'
title_to_filename cn: 'python基础.md'
parse: {'title': 'Test', 'tags': ['a', 'b'], 'sources': ['c']} Body text with [[link]]
wikilinks: ['link']
checksum: <64-char hex>
```

- [ ] **Step 3: Commit**

```bash
git add storage/wiki_storage.py
git commit -m "feat(wiki): add WikiStorage file I/O and frontmatter utilities"
```

---

### Task 4: Add wiki DB operations to models.py

**Files:**
- Modify: `storage/models.py` (add page CRUD, relation queries)

- [ ] **Step 1: Add imports and helper functions for page DB operations**

Add these imports at the top of `storage/models.py`:

```python
# ── Wiki page helpers ──

def _page_row_to_dict(row: sqlite3.Row) -> dict:
    """Convert a pages table row to a dict with parsed JSON fields."""
    d = dict(row)
    if isinstance(d.get("tags"), str):
        d["tags"] = json.loads(d["tags"])
    if isinstance(d.get("sources"), str):
        d["sources"] = json.loads(d["sources"])
    return d
```

- [ ] **Step 2: Add `upsert_page()` function**

```python
def upsert_page(title: str, file_path: str, tags: list[str],
                sources: list[str], checksum: str,
                content: str) -> int:
    """Insert or update a wiki page record. Returns page id."""
    conn = get_connection()
    try:
        # Check if page exists
        existing = conn.execute(
            "SELECT id FROM pages WHERE title = ?", (title,)
        ).fetchone()

        if existing:
            pid = existing["id"]
            conn.execute(
                """UPDATE pages SET file_path=?, tags=?, sources=?,
                   checksum=?, updated_at=datetime('now')
                   WHERE id=?""",
                (file_path, json.dumps(tags), json.dumps(sources),
                 checksum, pid),
            )
            logger.info("Updated page index: '%s' (id=%d)", title, pid)
        else:
            cur = conn.execute(
                """INSERT INTO pages (title, file_path, tags, sources, checksum)
                   VALUES (?, ?, ?, ?, ?)""",
                (title, file_path, json.dumps(tags), json.dumps(sources), checksum),
            )
            pid = cur.lastrowid
            logger.info("Created page index: '%s' (id=%d)", title, pid)

        # Update embedding
        from sqlite_vec import serialize_float32
        embedding = generate_embedding(content)
        conn.execute(
            "INSERT OR REPLACE INTO page_vectors(rowid, embedding) VALUES (?, ?)",
            (pid, serialize_float32(embedding)),
        )

        conn.commit()
        return pid
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
```

- [ ] **Step 3: Add `update_page_relations()` function**

```python
def update_page_relations(page_id: int, linked_titles: list[str]) -> None:
    """Replace page_relations for a given page with fresh [[wikilink]] data."""
    conn = get_connection()
    try:
        conn.execute(
            "DELETE FROM page_relations WHERE source_id = ?", (page_id,)
        )
        for title in linked_titles:
            conn.execute(
                """INSERT INTO page_relations (source_id, target_title)
                   VALUES (?, ?)""",
                (page_id, title),
            )
        conn.commit()
        logger.info("Updated %d relations for page id=%d", len(linked_titles), page_id)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
```

- [ ] **Step 4: Add `find_similar_pages()` function**

```python
def find_similar_pages(query: str, threshold: float = 0.6, limit: int = 5) -> list[dict]:
    """Search wiki pages by semantic similarity. Returns list of page dicts with distance."""
    embedding = generate_embedding(query)
    conn = get_connection()
    try:
        rows = conn.execute(
            """SELECT p.*, v.distance
               FROM (
                   SELECT rowid, distance
                   FROM page_vectors
                   WHERE embedding MATCH ?
                     AND k = ?
               ) v
               JOIN pages p ON p.id = v.rowid
               WHERE p.status = 'active'
                 AND v.distance <= ?
               ORDER BY v.distance""",
            (serialize_float32(embedding), limit * 4, threshold),
        ).fetchall()
        results = [_page_row_to_dict(r) for r in rows][:limit]
        logger.info("Page semantic search: %d results for '%s'", len(results), query[:30])
        return results
    finally:
        conn.close()
```

Note: needs import for `serialize_float32` if not already present at top of file. Add to existing import:

```python
from sqlite_vec import serialize_float32
```

- [ ] **Step 5: Add `get_related_pages()` function**

```python
def get_related_pages(page_id: int) -> list[dict]:
    """Get pages linked via page_relations to the given page."""
    conn = get_connection()
    try:
        rows = conn.execute(
            """SELECT p.* FROM pages p
               JOIN page_relations r ON r.target_title = p.title
               WHERE r.source_id = ? AND p.status = 'active'
               UNION
               SELECT p.* FROM pages p
               JOIN page_relations r ON r.source_id = p.id
               WHERE r.target_title = (SELECT title FROM pages WHERE id = ?)
                 AND p.status = 'active'
               """,
            (page_id, page_id),
        ).fetchall()
        return [_page_row_to_dict(r) for r in rows]
    finally:
        conn.close()
```

- [ ] **Step 6: Add `get_all_pages_index()` function**

```python
def get_all_pages_index() -> list[dict]:
    """Get all active pages for building index.md."""
    conn = get_connection()
    try:
        rows = conn.execute(
            """SELECT id, title, tags, sources, updated_at
               FROM pages WHERE status = 'active'
               ORDER BY updated_at DESC LIMIT 50"""
        ).fetchall()
        return [_page_row_to_dict(r) for r in rows]
    finally:
        conn.close()
```

- [ ] **Step 7: Add `get_page_by_title()` function**

```python
def get_page_by_title(title: str) -> Optional[dict]:
    """Look up a page by exact title match."""
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM pages WHERE title = ? AND status = 'active'",
            (title,),
        ).fetchone()
        return _page_row_to_dict(row) if row else None
    finally:
        conn.close()
```

- [ ] **Step 8: Quick test**

Run:
```bash
python -c "
from storage.models import upsert_page, find_similar_pages, get_all_pages_index
pid = upsert_page('Test Page', 'wiki/pages/test.md', ['test'], ['conv_001'], 'abc123', 'Test content about [[Python]]')
print('Page ID:', pid)
idx = get_all_pages_index()
print('Index entries:', len(idx))
similar = find_similar_pages('test', threshold=1.0)
print('Similar found:', len(similar))
"
```

Expected: no errors, page created and found.

- [ ] **Step 9: Commit**

```bash
git add storage/models.py
git commit -m "feat(wiki): add page DB operations (upsert, search, relations, index)"
```

---

### Task 5: Create WikiIndex helper for index.md management

**Files:**
- Create: `storage/wiki_index.py`

- [ ] **Step 1: Create `storage/wiki_index.py`**

```python
"""Manage data/wiki/index.md: build from DB, format for prompt injection."""

import logging
import os
from datetime import datetime

from storage.wiki_storage import WIKI_DIR, INDEX_PATH, read_index, write_index
from storage.models import get_all_pages_index

logger = logging.getLogger(__name__)


_INDEX_HEADER = """# Wiki Index

| 页面 | 标签 | 来源 | 最后更新 |
|------|------|------|----------|
"""


def rebuild_index() -> None:
    """Read all active pages from DB and rewrite index.md."""
    pages = get_all_pages_index()
    lines = [_INDEX_HEADER]
    for p in pages:
        tags_str = ", ".join(p.get("tags", [])) if isinstance(p.get("tags"), list) else p.get("tags", "")
        src_count = len(p.get("sources", [])) if isinstance(p.get("sources"), list) else 0
        updated = p.get("updated_at", "")[:10]  # YYYY-MM-DD
        title = p["title"]
        lines.append(f"| [[{title}]] | {tags_str} | {src_count} 条对话 | {updated} |\n")

    write_index("".join(lines))
    logger.info("Rebuilt index.md with %d pages", len(pages))


def get_index_for_prompt(max_entries: int = 50) -> str:
    """Get a condensed index text for LLM prompt injection.

    Returns a plain list format that fits within prompt limits.
    """
    pages = get_all_pages_index()
    if not pages:
        return "(当前 Wiki 为空，暂无页面)"

    lines = ["当前 Wiki 页面索引："]
    for p in pages[:max_entries]:
        title = p["title"]
        tags_str = ", ".join(p.get("tags", [])[:3])
        lines.append(f"  - {title} [{tags_str}]")
    if len(pages) > max_entries:
        lines.append(f"  ...及其他 {len(pages) - max_entries} 个页面")

    return "\n".join(lines)
```

- [ ] **Step 2: Quick test**

Run:
```bash
python -c "
from storage.wiki_index import rebuild_index, get_index_for_prompt
rebuild_index()
print(get_index_for_prompt())
"
```

Expected: index.md written, prompt text printed (may be empty if no pages).

- [ ] **Step 3: Commit**

```bash
git add storage/wiki_index.py
git commit -m "feat(wiki): add WikiIndex for index.md management"
```

---

### Task 6: Rewrite store.py with two-step CoT extraction

**Files:**
- Modify: `agent/nodes/store.py` (full rewrite)
- Modify: `agent/state.py` (update return type if needed)

- [ ] **Step 1: Update `agent/state.py` — add wiki-related optional fields**

```python
# In AgentState, add after stored_knowledge_ids:
    wiki_page_ids: list[int]  # new, IDs of wiki pages created/updated
```

- [ ] **Step 2: Rewrite `agent/nodes/store.py`**

The new store node implements two-step CoT:

```python
"""Wiki page extraction: two-step CoT (analyze → generate).

Step 1 (Analyze): LLM reads Q&A + index + SCHEMA, outputs analysis
  (topics, actions, related pages, contradictions).

Step 2 (Generate): LLM reads analysis + existing pages, outputs wiki page content.
  System writes to disk, updates SQLite index, rebuilds index.md.
"""

import logging
import os
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from agent.utils.llm import LLM
from storage.models import (
    upsert_page,
    update_page_relations,
    get_page_by_title,
)
from storage.wiki_storage import (
    ensure_dirs,
    title_to_filename,
    read_schema,
    read_page,
    write_page,
    extract_wikilinks,
    PAGES_DIR,
)
from storage.wiki_index import rebuild_index, get_index_for_prompt

logger = logging.getLogger(__name__)


# ── Pydantic models for Step 1 (Analysis) ──

class AnalysisAction(BaseModel):
    topic: str = Field(description="Topic extracted from the Q&A")
    action: str = Field(description="'create' for new page, 'update' for existing")
    target: str = Field(description="Existing page title to update, or empty string for new")


class AnalysisOutput(BaseModel):
    topics: list[str] = Field(description="Topics covered in this Q&A")
    actions: list[AnalysisAction] = Field(
        description="Per-topic actions: create new page or update existing"
    )
    related_pages: list[str] = Field(
        description="Titles of existing pages related to this content"
    )
    contradictions: list[str] = Field(
        description="Contradictions between new content and existing knowledge"
    )


# ── Pydantic model for Step 2 (Generation) ──

class WikiPageOutput(BaseModel):
    title: str = Field(description="Page title")
    content: str = Field(
        description="Full page markdown content INCLUDING YAML frontmatter "
                    "(title and tags only, do not add created/updated)"
    )
    tags: list[str] = Field(description="Tags for this page")
    sources: list[str] = Field(description="Source conversation IDs")


class WikiBatchOutput(BaseModel):
    pages: list[WikiPageOutput] = Field(
        description="All wiki pages to create or update (one per topic action)"
    )


# ── Prompt templates ──

def _build_analysis_prompt(user_message: str, answer: str) -> str:
    schema_content = read_schema()
    page_index = get_index_for_prompt()

    return (
        f"{schema_content}\n\n"
        f"## 当前 Wiki 页面索引\n\n"
        f"{page_index}\n\n"
        f"## 需要分析的问答\n\n"
        f"Question: {user_message}\n"
        f"Answer: {answer}\n\n"
        f"## 分析要求\n\n"
        f"1. 识别这段问答涉及的所有主题(topics)\n"
        f"2. 对每个主题，判断应该创建新页面还是更新已有页面(actions)\n"
        f"3. 列出与当前内容相关的现有页面(related_pages)\n"
        f"4. 如果新内容与已有知识矛盾，标注出来(contradictions)\n\n"
        f"注意：多个主题可以对应不同的操作。例如一条问答可能同时涉及\n"
        f"更新「Django 框架」页面和创建「ORM 查询优化」新页面。"
    )


def _build_generation_prompt(
    analysis: AnalysisOutput,
    user_message: str,
    answer: str,
    existing_page_contents: list[dict],
) -> str:
    schema_content = read_schema()

    # Build existing page context
    existing_text = ""
    if existing_page_contents:
        existing_text = "## 已有页面内容（需要更新的页面）\n\n"
        for p in existing_page_contents:
            existing_text += f"### 页面: {p['title']}\n\n"
            existing_text += f"当前文件路径: {p.get('file_path', '')}\n\n"
            existing_text += f"{p.get('body', '')}\n\n---\n\n"

    actions_text = "\n".join(
        f"- {a.topic}: {'创建新页面' if a.action == 'create' else '更新「' + a.target + '」'}"
        for a in analysis.actions
    )

    return (
        f"{schema_content}\n\n"
        f"## 分析报告\n\n"
        f"主题: {', '.join(analysis.topics)}\n"
        f"操作:\n{actions_text}\n"
        f"相关页面: {', '.join(analysis.related_pages) if analysis.related_pages else '无'}\n"
        f"矛盾: {', '.join(analysis.contradictions) if analysis.contradictions else '未发现'}\n\n"
        f"{existing_text}"
        f"## 原始问答\n\n"
        f"Question: {user_message}\n"
        f"Answer: {answer}\n\n"
        f"## 生成要求\n\n"
        f"根据以上分析，生成 Wiki 页面内容：\n"
        f"1. 每页必须包含 YAML frontmatter（title, tags, sources）\n"
        f"2. 正文用中文，技术名词保留英文\n"
        f"3. 页面间使用 [[页面标题]] 交叉引用\n"
        f"4. tags 字段只放标签，不要放 category 信息\n"
        f"5. 不要添加 created/updated 字段（系统自动管理）\n"
        f"6. 如果是更新已有页面，输出完整页面内容（不是增量）"
    )


def _get_source_id() -> str:
    """Generate a source conversation ID from timestamp."""
    return f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def _read_existing_pages(actions: list[AnalysisAction]) -> list[dict]:
    """Read full content of pages that need updating."""
    existing = []
    for a in actions:
        if a.action != "update" or not a.target:
            continue
        # Look up file_path from DB
        page = get_page_by_title(a.target)
        if page:
            file_page = read_page(page["file_path"])
            if file_page:
                existing.append({
                    "title": page["title"],
                    "file_path": page["file_path"],
                    "body": file_page["body"],
                })
    return existing


def store(state: dict) -> dict:
    """Two-step CoT extraction: analyze → generate → write."""
    if not state.get("needs_store", True):
        logger.info("Skipping store: needs_store=False")
        return {}

    if not state.get("answer"):
        logger.info("Skipping store: no answer")
        return {}

    if state.get("contradiction_found"):
        logger.info("Skipping store: contradiction detected")
        return {}

    ensure_dirs()
    user_msg = state["user_message"]
    answer = state["answer"]
    source_id = _get_source_id()

    # ── Step 1: Analysis ──
    logger.info("Step 1: Analyzing Q&A for wiki content...")
    analysis_prompt = _build_analysis_prompt(user_msg, answer)
    analysis = LLM.generate_structured(analysis_prompt, AnalysisOutput, use_language=False)
    if analysis is None:
        logger.error("Analysis LLM returned None")
        return {}
    logger.info("Analysis complete: %d topics, %d actions", len(analysis.topics), len(analysis.actions))

    # ── Step 2: Generation ──
    existing_contents = _read_existing_pages(analysis.actions)
    logger.info("Step 2: Generating wiki page(s)...")
    gen_prompt = _build_generation_prompt(analysis, user_msg, answer, existing_contents)
    batch = LLM.generate_structured(gen_prompt, WikiBatchOutput, use_language=False)
    if batch is None or not batch.pages:
        logger.error("Generation LLM returned None or empty pages")
        return {}

    # ── Write to filesystem + update SQLite ──
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    saved_ids = []

    for wp in batch.pages:
        # Ensure frontmatter has title, tags, sources (system manages created/updated)
        frontmatter_tags = wp.tags
        frontmatter_sources = wp.sources
        if source_id not in frontmatter_sources:
            frontmatter_sources.append(source_id)

        # Check if page already exists → preserve created date, append source
        filename = title_to_filename(wp.title)
        file_path = os.path.join("wiki", "pages", filename)
        existing_page = read_page(file_path)

        created_str = existing_page["title"] if existing_page else ""

        # Build final content with proper frontmatter
        from storage.wiki_storage import build_frontmatter
        frontmatter = build_frontmatter(
            title=wp.title,
            tags=frontmatter_tags,
            sources=frontmatter_sources,
            created=created_str,
            updated=now,
        )
        full_content = frontmatter + "\n\n" + wp.content.lstrip("\n")

        # Write to file
        checksum = write_page(file_path, full_content)

        # Update SQLite index
        pid = upsert_page(
            title=wp.title,
            file_path=file_path,
            tags=frontmatter_tags,
            sources=frontmatter_sources,
            checksum=checksum,
            content=full_content,
        )
        saved_ids.append(pid)

        # Extract [[wikilinks]] and update relations
        links = extract_wikilinks(full_content)
        if links:
            update_page_relations(pid, links)

    # ── Rebuild index.md ──
    rebuild_index()
    logger.info("Stored %d wiki pages from this Q&A", len(saved_ids))

    return {
        "stored_knowledge_ids": saved_ids,
        "logic_chain": [{
            "node": "store",
            "action": f"Wiki: 存储 {len(saved_ids)} 个页面",
            "reasoning": f"页面: {[wp.title for wp in batch.pages]}, "
                         f"操作: {[a.action for a in analysis.actions]}",
        }],
    }
```

- [ ] **Step 3: Run a quick test that the module imports cleanly**

Run:
```bash
python -c "from agent.nodes.store import store; print('store module loaded')"
```

- [ ] **Step 4: Commit**

```bash
git add agent/nodes/store.py agent/state.py
git commit -m "feat(wiki): rewrite store with two-step CoT wiki page extraction"
```

---

### Task 7: Update retrieve.py for page-level retrieval

**Files:**
- Modify: `agent/nodes/retrieve.py`

- [ ] **Step 1: Rewrite `agent/nodes/retrieve.py`**

```python
"""Wiki page retrieval: vector search → read full pages → expand relations.

Returns page content instead of knowledge_point fragments.
Falls back to old knowledge_points search if no wiki pages exist.
"""

import logging

from storage.models import find_similar_pages, get_related_pages, search_knowledge_points_semantic, rerank_knowledge
from storage.wiki_storage import read_page

logger = logging.getLogger(__name__)


def retrieve(state: dict) -> dict:
    query = state.get("search_query") or state["user_message"]
    logger.info("Wiki retrieval for: '%s'", query[:40])

    # Step 1: Vector search for pages
    try:
        pages = find_similar_pages(query, threshold=0.6, limit=5)
    except Exception as e:
        logger.warning("Page semantic search failed: %s, falling back to knowledge_points", e)
        pages = []

    if not pages:
        # Fallback: use existing knowledge_points
        logger.info("No pages found, falling back to knowledge_points retrieval")
        try:
            candidates = search_knowledge_points_semantic(query, threshold=0.6, limit=20)
            if len(candidates) > 3:
                results = rerank_knowledge(query, candidates, top_k=5)
            else:
                results = candidates[:5]
        except Exception:
            results = []
        return {"stored_knowledge": results}

    # Step 2: Read full page content from disk
    results = []
    for p in pages:
        file_page = read_page(p["file_path"])
        if not file_page:
            continue
        results.append({
            "type": "wiki_page",
            "page_id": p["id"],
            "title": p["title"],
            "content": file_page["body"],
            "tags": file_page["tags"],
            "distance": p.get("distance", 0),
        })

    # Step 3: Expand with related pages (second pass)
    related_titles = set()
    for r in results:
        related = get_related_pages(r["page_id"])
        for rp in related:
            if rp["title"] not in {x["title"] for x in results}:
                related_titles.add(rp["title"])

    for title in related_titles:
        # Must look up by title, then read file
        from storage.models import get_page_by_title
        page = get_page_by_title(title)
        if page:
            file_page = read_page(page["file_path"])
            if file_page:
                results.append({
                    "type": "wiki_page",
                    "page_id": page["id"],
                    "title": title,
                    "content": file_page["body"],
                    "tags": file_page["tags"],
                    "distance": 0,  # relation-expanded, not direct match
                })

    logger.info("Retrieved %d wiki pages (including %d relation-expanded)",
                len(results), len(results) - len(pages))
    return {"stored_knowledge": results}
```

- [ ] **Step 2: Quick import test**

Run:
```bash
python -c "from agent.nodes.retrieve import retrieve; print('retrieve module loaded')"
```

- [ ] **Step 3: Commit**

```bash
git add agent/nodes/retrieve.py
git commit -m "feat(wiki): update retrieve with page-level search and relation expansion"
```

---

### Task 8: Test the full pipeline

**Files:**
- Create: `tests/test_wiki.py`

- [ ] **Step 1: Create `tests/test_wiki.py`**

```python
"""Tests for wiki knowledge base."""
import json
import os
import sys
import tempfile

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from storage.wiki_storage import (
    title_to_filename,
    parse_frontmatter,
    build_frontmatter,
    extract_wikilinks,
    compute_checksum,
)
from storage.wiki_index import get_index_for_prompt


class TestWikiStorage:
    def test_title_to_filename(self):
        assert title_to_filename("Django ORM") == "django-orm.md"
        assert title_to_filename("Python 基础") == "python基础.md"
        assert title_to_filename("HTTP/2 协议") == "http2协议.md"
        # Should handle special chars
        result = title_to_filename("C++ 模板元编程")
        assert result.endswith(".md")
        assert " " not in result

    def test_parse_frontmatter(self):
        content = """---
title: Test
tags: [a, b, c]
sources: [conv_001]
---

Body text here
"""
        meta, body = parse_frontmatter(content)
        assert meta["title"] == "Test"
        assert meta["tags"] == ["a", "b", "c"]
        assert meta["sources"] == ["conv_001"]
        assert body == "Body text here"

    def test_parse_frontmatter_no_frontmatter(self):
        content = "Just some text"
        meta, body = parse_frontmatter(content)
        assert meta == {}
        assert body == "Just some text"

    def test_build_frontmatter(self):
        result = build_frontmatter("Test", ["a"], ["conv_001"], "2025-01-01", "2025-01-02")
        assert "title: Test" in result
        assert "tags: [a]" in result
        assert "created: 2025-01-01" in result

    def test_extract_wikilinks(self):
        content = "See [[Django]] and [[Python]] and [[Django#ORM]]"
        links = extract_wikilinks(content)
        assert "Django" in links
        assert "Python" in links
        assert len(links) == 2  # Django should appear once

    def test_extract_wikilinks_empty(self):
        assert extract_wikilinks("No links here") == []

    def test_compute_checksum(self):
        c1 = compute_checksum("hello")
        c2 = compute_checksum("hello")
        c3 = compute_checksum("world")
        assert c1 == c2
        assert c1 != c3
        assert len(c1) == 64  # SHA256 hex


class TestWikiIndex:
    def test_get_index_for_prompt(self):
        # Should not crash, returns string
        result = get_index_for_prompt()
        assert isinstance(result, str)


class TestWikiDB:
    def test_upsert_and_search_page(self):
        from storage.models import upsert_page, find_similar_pages, get_page_by_title

        pid = upsert_page(
            title="Test Wiki Page",
            file_path="wiki/pages/test-wiki-page.md",
            tags=["test", "wiki"],
            sources=["conv_test_001"],
            checksum="abc123",
            content="This is a test wiki page about testing wiki pages.",
        )
        assert pid > 0

        # Should be findable
        found = get_page_by_title("Test Wiki Page")
        assert found is not None
        assert found["title"] == "Test Wiki Page"

        # Semantic search
        results = find_similar_pages("test wiki", threshold=1.0, limit=5)
        titles = [r["title"] for r in results]
        assert "Test Wiki Page" in titles

    def test_page_relations(self):
        from storage.models import upsert_page, update_page_relations, get_related_pages

        pid1 = upsert_page("Page A", "wiki/pages/page-a.md", ["a"], ["conv_001"], "abc", "Content about [[Page B]]")
        pid2 = upsert_page("Page B", "wiki/pages/page-b.md", ["b"], ["conv_001"], "def", "Content")

        update_page_relations(pid1, ["Page B"])

        related = get_related_pages(pid1)
        titles = [r["title"] for r in related]
        assert "Page B" in titles
```

- [ ] **Step 2: Run tests**

Run:
```bash
pytest tests/test_wiki.py -v
```

Expected: all tests pass.

- [ ] **Step 3: Run existing tests to verify no regressions**

Run:
```bash
pytest -v
```

Expected: existing tests still pass.

- [ ] **Step 4: Commit**

```bash
git add tests/test_wiki.py
git commit -m "test(wiki): add tests for wiki storage, index, and DB operations"
```

---

## Self-Review

**1. Spec coverage:**
- SCHEMA.md created: Task 1 ✅
- Wiki directory structure: Task 1 ✅
- DB tables (pages, page_vectors, page_relations): Task 2 ✅
- File I/O utilities (read/write, frontmatter, checksum): Task 3 ✅
- `[[wikilink]]` extraction and relations: Tasks 3 + 4 ✅
- index.md management: Task 5 ✅
- Two-step CoT extraction: Task 6 ✅
- Page-level retrieval with relation expansion: Task 7 ✅
- Fallback to old knowledge_points: Task 7 ✅
- Migration compatibility (keep old tables): Task 2 ✅ (old tables untouched)
- Tests: Task 8 ✅

**2. Placeholders:** None. All steps contain complete code.

**3. Type consistency:** `upsert_page()` returns `int` page id, used consistently. `get_page_by_title()` returns `Optional[dict]`. `find_similar_pages()` returns `list[dict]`. All function signatures match across tasks.
