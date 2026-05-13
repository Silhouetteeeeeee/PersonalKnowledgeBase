# File-to-Wiki Pipeline — Upload Documents as Raw Material

Date: 2026-05-13
Status: Draft

## Motivation

The Wiki Knowledge Base v2 rewrite (`2026-05-12`) replaced the old `knowledge_points` system with two-step CoT wiki page extraction, but only covered the Q&A flow (graph's `store` node). The file upload pipeline in `bot.py:_process_and_store_file` was left behind — it still references deleted functions (`_BASE_DISTILL_PROMPT`, `_distill_and_save`) and the old knowledge_points distillation approach.

Users need to upload documents (PDF, DOCX, TXT, images via OCR) as raw materials that get analyzed and split into multi-topic wiki pages, the same way Q&A content does.

## Design

### Core Concept

Extract the two-step CoT wiki extraction from `store()` into a shared `extract_to_wiki()` function, so both the graph's store node and the file upload handler use the same pipeline. A file upload goes through: save locally → extract text → dedup by hash → analyze topics → generate wiki pages → write to disk + index.

### Data Flow

```
File Upload (WeChat)
  │
  ├─ 1. Download & save: data/files/{hash}.ext
  ├─ 2. extract_text_from_file() → raw text
  ├─ 3. Hash dedup check (skip if already processed)
  ├─ 4. Length check (MAX_FILE_CHARS = 8000)
  │
  └─ 5. extract_to_wiki(text, file_hash, f"From file: {filename}")
       │
       ├─ Step 1 (Analyze): LLM reads text + wiki index →
       │    {topics, actions[], related_pages[], contradictions[]}
       │
       ├─ Step 2 (Generate): LLM reads analysis →
       │    {pages[{title, content, tags, sources}]}
       │
       ├─ Write each page to data/wiki/pages/{slug}.md
       ├─ Update SQLite pages + page_vectors + page_relations
       └─ Rebuild data/wiki/index.md
```

### Shared Function: `extract_to_wiki()`

Location: `agent/nodes/store.py`

```python
def extract_to_wiki(
    source_text: str,     # Full text to analyze (answer text or file extracted text)
    source_id: str,       # Unique source identifier (conv_... or file_{hash})
    source_label: str,    # Short descriptor for prompt context ("Question: xxx" or "From file: xxx")
) -> dict:
    """Two-step CoT: analyze → generate → write → index.

    Returns: {"page_ids": list[int], "logic_chain": list[dict]}
    Both Q&A store() and file upload call this.
    """
```

- `source_id` is used as source identifier in wiki page metadata
- `source_label` is a short context descriptor inserted into the analysis prompt header (replaces the hardcoded "Q&A" format)
- Internally the prompt combines `source_label` and `source_text`: `{source_label}\n\n{source_text}`
- The function internally calls `ensure_dirs()`, loops through generated pages, writes files, updates SQLite, and rebuilds the index

### `store()` Node — Thin Wrapper

```python
def store(state: dict) -> dict:
    if not state.get("needs_store", True):
        return {}
    if not state.get("answer"):
        return {}
    if state.get("contradiction_found"):
        return {}

    source_id = _get_source_id()
    source_label = f"Question: {state['user_message']}"
    return extract_to_wiki(state["answer"], source_id, source_label)
```

- All guards (`needs_store`, answer exists, contradiction) stay in `store()`
- `extract_to_wiki` is unaware of graph state — pure function

### File Upload Handler — `bot.py:_process_and_store_file`

Current broken code (lines 219, 248-257):

```python
from agent.nodes.store import _BASE_DISTILL_PROMPT, _distill_and_save  # DELETED
...
saved = _distill_and_save(prompt=..., source_question=...)            # DELETED
```

Replacement:

```python
from agent.nodes.store import extract_to_wiki

MAX_FILE_CHARS = 8000

# After text extraction, before wiki processing:
if len(text) > MAX_FILE_CHARS:
    return f"文件「{filename}」内容过长（{len(text)} 字符），暂不支持处理超过 {MAX_FILE_CHARS} 字符的文档。"

# Replace the old distillation call:
saved = extract_to_wiki(
    source_text=text,
    source_id=f"file_{file_hash}",
    source_label=f"From file: {filename}",
)
```

### Prompt Changes

All prompt templates in `store.py` replace hardcoded "Q&A" language:

| Current | Changed to |
|---|---|
| `"## Q&A to Analyze"` | `"## Source Content to Analyze"` |
| `"Question: {user_msg}\nAnswer: {answer}"` | Replaced by `source_label` + `source_text` parameters |
| `"Identify all topics covered in this Q&A"` | `"Identify all topics covered in this content"` |
| `"## Original Q&A"` | `"## Original Content"` |
| Requirement 5 in generation prompt: unchanged | Still requires complete content on update |

### Length Limit

- `MAX_FILE_CHARS = 8000` (~2000 tokens for Chinese text)
- Checked after `extract_text_from_file()` succeeds
- Rejects with a clear message; no automatic chunking for now
- Future enhancement: implement chunking + merge when needed

### Files Changed

| File | Change |
|---|---|
| `agent/nodes/store.py` | Extract `extract_to_wiki()`, generalize prompts, `store()` becomes thin wrapper |
| `server/bot.py` | Replace broken imports/calls with `extract_to_wiki()`, add length check |
| `tests/test_nodes.py` | Add `test_extract_to_wiki_creates_pages` |
| `tests/test_file_processor.py` | Add `test_process_file_creates_wiki_pages`, `test_process_file_dedup`, `test_process_file_too_large` |
| `tests/test_graph.py` | (optional) Graph-level test for store node still works |

### Files NOT Changed

| File | Reason |
|---|---|
| `storage/file_processor.py` | Text extraction unchanged |
| `storage/models.py` | `save_file_record` keeps existing signature |
| `storage/wiki_storage.py` | File operations unchanged |
| `storage/wiki_index.py` | Index rebuilding unchanged |
| `agent/state.py` | No new state fields needed |

### Testing

```python
# store.py
def test_extract_to_wiki_creates_pages():
    """Call extract_to_wiki with simple text → creates wiki pages with correct structure."""
    result = extract_to_wiki("Python dict is a key-value store", "test_001", "## Source\n\ntest")
    assert "page_ids" in result
    assert len(result["page_ids"]) >= 1

def test_extract_to_wiki_creates_no_duplicate_pages():
    """Same content twice → second call should detect and skip via hash."""
    ...

# bot.py (via _process_and_store_file)
def test_process_file_creates_wiki_pages(tmp_path):
    """Upload a .txt file → text extracted → wiki pages created."""
    ...

def test_process_file_dedup(tmp_path):
    """Same file hash → second upload returns 'already processed'."""
    ...

def test_process_file_too_large(tmp_path):
    """File with >8000 chars → rejected with length warning."""
    ...
```
