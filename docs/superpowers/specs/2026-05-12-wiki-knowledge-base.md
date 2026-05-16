# Wiki Knowledge Base — Personal Knowledge Agent v2

Date: 2026-05-12
Status: Draft

## Motivation

Current knowledge extraction stores isolated 50-150 word fragments (`knowledge_points`). This leads to:

- **Quality issues**: single-pass LLM extraction produces noisy/inaccurate points that degrade future answers
- **No inter-knowledge relations**: fragments are independent, no cross-references, no knowledge graph
- **Limited context at retrieval**: reading a 50-word snippet vs. a full topic page gives LLM vastly different context
- **Pure append/no merge**: new knowledge never updates existing entries, old data goes `deprecated`

## Design

### Core Concept

Replace isolated `knowledge_points` with **Wiki pages** — structured, interconnected topic pages stored as markdown files, written and maintained by LLM. SQLite serves as index + vector search layer only, the file system is the source of truth for content. This follows the **LLM Wiki** pattern (Karpathy, 2025): persistent, compounding knowledge compiled once and kept current, not re-derived on every query.

### Storage Architecture

```
data/wiki/
  index.md                     ← auto-maintained full index
  pages/
    django-orm.md              ← one file per page
    python基础.md
    restful-api.md
    ...
```

- Each page is a standalone `.md` file (self-contained with frontmatter)
- `pages/` directory is flat (no subdirectories per category — tags handle classification)
- File named from `title`: lowercase + hyphenated, preserve unicode
- `index.md` auto-updated on each write
- The `data/wiki/` directory is the single source of truth for content

### Schema Layer (`data/wiki/SCHEMA.md`)

A dedicated schema document that defines wiki conventions for the LLM. Injected into LLM prompts at extraction time — keeps format rules out of Python code, single source of truth for page structure.

```markdown
# Wiki Schema

## Directory Layout
- `data/wiki/pages/` — page files, one `.md` per topic
- `data/wiki/index.md` — full index, auto-maintained
- `data/wiki/SCHEMA.md` — this file

## Page Format
Every page must start with YAML frontmatter (between `---` delimiters):

---{...}---

Required frontmatter fields:
- `title`: string, page title, match target for [[wikilink]]
- `tags`: list of strings, at least one tag
- `sources`: list of strings, conversation IDs

Auto-managed (do not write in prompt):
- `created`, `updated` — set by system

## Body Structure
- Start with a concise overview paragraph
- Use ## headings for major sections
- Use ### for subsections when needed
- End with "与其他概念的关系" listing [[links]]

## [[wikilink]] Rules
- Full page title only, no aliases
- For section references: [[title#heading]]
- Keep links meaningful — don't force connections
- Broken links are acceptable (page may be created later)

## Style Guidelines
- Chinese preferred, keep English for technical terms
- Each page should be self-contained: readable without context
- Be precise, not verbose
```

### Page Format

```markdown
---
title: Django ORM
tags: [python, django, database]
sources: [conv_20250601_001, conv_20250602_003]
created: 2025-06-01
updated: 2025-06-02
---

Django ORM 是 Django 框架内置的 [[对象关系映射]] 工具...

## 核心内容
展开讲解，可以用多级标题

## 与其他概念的关系
- [[Django 框架]] —— ORM 是其核心组件
- [[SQLAlchemy]] —— Python 中另一个 ORM 方案

## 参考来源（可选）
原始对话中的关键引用
```

### Frontmatter Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| title | string | Yes | Page title, exact match target for `[[wikilink]]` |
| tags | list[string] | Yes | At least one tag, Chinese preferred |
| sources | list[string] | Yes | Source conversation IDs |
| created | date | Auto | Set by system |
| updated | date | Auto | Set by system on each update |

`category` field removed — `tags` is more flexible (many-to-many, no fixed hierarchy).

### `[[wikilink]]` Syntax

- Basic: `[[页面标题]]` — page-level reference
- Precise: `[[页面标题#段落标题]]` — paragraph-level reference
- Save-time resolution: parser extracts all `[[links]]` from content → `page_relations` table
- Broken links allowed: saved as warnings, may trigger page creation later
- Dedup: multiple occurrences of same link in one page count as one

### Index (`data/wiki/index.md`)

Auto-maintained markdown file, updated on each page write. Serves as the LLM's entry point for navigation:

```markdown
# Wiki Index

| 页面 | 标签 | 来源 | 最后更新 |
|------|------|------|----------|
| [[Django ORM]] | python, django, database | 2 条对话 | 2025-06-02 |
| [[Python 基础]] | python, programming | 5 条对话 | 2025-05-28 |
```

- Limits to prevent bloat: only shows pages with `status = active`
- At extraction time, system reads `index.md` and injects into prompt (truncate at ~50 entries)

### SQLite Tables (Index-Only)

SQLite no longer stores page content. `pages` table shrinks to pure metadata:

```sql
CREATE TABLE pages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL UNIQUE,
    file_path TEXT NOT NULL,        -- relative: wiki/pages/django-orm.md
    tags TEXT DEFAULT '[]',         -- JSON array, cached from file frontmatter
    sources TEXT DEFAULT '[]',      -- JSON array
    status TEXT DEFAULT 'active',
    checksum TEXT,                  -- SHA256 of file content, for change detection
    created_at TEXT,
    updated_at TEXT
);

CREATE VIRTUAL TABLE page_vectors USING vec0(
    embedding float[512] distance_metric=cosine
);

CREATE TABLE page_relations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id INTEGER NOT NULL,
    target_title TEXT NOT NULL,     -- [[link]] target title
    relation_type TEXT DEFAULT 'wikilink',
    FOREIGN KEY (source_id) REFERENCES pages(id)
);
```

### Two-Step CoT Extraction (replaces current single-pass)

#### Step 1: Analysis (LLM call #1)

Input:
- User question + bot answer
- `data/wiki/SCHEMA.md` (page format rules)
- `data/wiki/index.md` (existing page index)

Output:
```json
{
  "topics": ["Django ORM", "查询优化"],
  "actions": [
    {"topic": "Django ORM", "action": "update", "target": "Django 框架"},
    {"topic": "查询优化", "action": "create"}
  ],
  "related_pages": ["Django 框架", "数据库性能"],
  "contradictions": []
}
```

#### Step 2: Generation (LLM call #2)

Input:
- Analysis report from Step 1
- `data/wiki/SCHEMA.md` (page format rules)
- Full content of involved existing pages (read from files at `file_path`)
- User question + bot answer

Output:
- For new pages: complete page content (frontmatter + body)
- For updates: new paragraphs + full updated page text
- `[[wikilink]]` embedded in body

### Write Consistency

Since LangGraph processes messages serially, no concurrent write conflicts. Flow:

1. LLM returns page content (markdown string)
2. Write to file at `data/wiki/pages/<title>.md`
3. Read back file, compute SHA256 checksum → verify integrity
4. Compute embedding from content
5. Update SQLite: `pages` + `page_vectors` + `page_relations`
6. Update `data/wiki/index.md`

If step 2 fails → abort, no SQLite update. If SQLite update fails → file on disk but index stale (next extraction will warn and re-sync).

### Reading Pages for Retrieval

When answering a question:
1. Vector search on `page_vectors` → get page IDs + scores
2. Query `page_relations` for related page IDs
3. Read files from `file_path` for all matched + related pages
4. Combine full content → feed to LLM

### Page File Naming Rule

```
title → filename:
  "Django ORM"     → "django-orm.md"
  "Python 基础"    → "python基础.md"
  "HTTP/2 协议"    → "http2协议.md"
  "C++ 模板元编程" → "c++模板元编程.md"

Rules:
  - lowercase
  - spaces → hyphens
  - preserve unicode (CJK characters kept as-is)
  - no extension override: always .md
```

### Migration Plan

1. Create `data/wiki/` directory, init with empty index
2. Existing `knowledge_points` table kept as-is for transition
3. New extractions → wiki pages. Old data still queryable as fallback
4. Optional: one-time script to consolidate old kps into wiki pages
5. After validation period, `knowledge_points` can be deprecated

### Files Changed

| File | Change |
|------|--------|
| `storage/database.py` | Add `pages`, `page_vectors`, `page_relations` tables (no content field) |
| `storage/models.py` | Add file I/O: `save_page_file()`, `read_page_file()`, `compute_checksum()` |
| `storage/models.py` | Add DB operations: `upsert_page_index()`, `find_similar_pages()`, relation queries |
| `data/wiki/SCHEMA.md` | **New**: Schema layer defining page format and conventions |
| `storage/wiki_index.py` | **New**: `WikiIndex` class managing `data/wiki/index.md` reads/updates |
| `agent/nodes/store.py` | Rewrite: two-step CoT, file-based page storage, inject SCHEMA.md into prompts |
| `agent/nodes/retrieve.py` | Add page-level retrieval: vector → file read → LLM input |
| Prompt template (in store.py) | Updated for wiki page generation |

### Out of Scope (Phase 2)

- Web UI for browsing wiki pages
- `wikilink` graph visualization
- Louvain community detection on page relations
- Conflict review queue with human-in-loop
- Obsidian integration guide

---

## Self-Review

- **Placeholders**: None. All sections filled.
- **Consistency**: File storage + SQLite index + two-step CoT all aligned. `file_path` is the single link between filesystem and DB.
- **Scope**: Focused on core store+retrieve pipeline. Web UI, graph viz deferred.
- **Ambiguity**: Write consistency model specified (serial, verify after write). File naming rules explicit. Index format defined.
