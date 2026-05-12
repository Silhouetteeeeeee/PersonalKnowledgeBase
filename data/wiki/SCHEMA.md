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
