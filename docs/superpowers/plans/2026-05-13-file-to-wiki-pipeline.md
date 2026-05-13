# File-to-Wiki Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make uploaded documents (PDF, DOCX, TXT, images) go through the same two-step CoT wiki extraction as Q&A content, via a shared `extract_to_wiki()` function.

**Architecture:** Extract the core wiki extraction logic from `store()` into a reusable `extract_to_wiki(source_text, source_id, source_label)` function. Generalize prompt templates from Q&A-specific to generic source content. `store()` becomes a thin wrapper that calls `extract_to_wiki()` with `source_label="Question: {msg}"`. The bot's file handler calls the same function with `source_label="From file: {filename}"`.

**Tech Stack:** Python 3.10, LangGraph, DeepSeek-Chat, store.py, bot.py

---

### Task 1: Refactor store.py — extract `extract_to_wiki()`, generalize prompts

**Files:**
- Modify: `agent/nodes/store.py` (entire file, ~270 lines)
- Test: `tests/test_nodes.py`

**Changes:**

1. Rename `_build_analysis_prompt(user_message, answer)` to `_build_analysis_prompt(source_text, source_label)`:
   - `source_label` replaces `f"Question: {user_message}"` as the context descriptor
   - `source_text` replaces `answer` as the body
   - Header: `"## Q&A to Analyze"` → `"## Source Content to Analyze"`
   - Requirement 1: `"in this Q&A"` → `"in this content"`

2. Rename `_build_generation_prompt(analysis, user_message, answer, existing)` to `_build_generation_prompt(analysis, source_text, source_label, existing)`:
   - `"## Original Q&A"` → `"## Original Content"`
   - `f"Question: {user_message}\nAnswer: {answer}"` → `f"{source_label}\n\n{source_text}"`

3. Add `extract_to_wiki(source_text, source_id, source_label)` function — extract the loop body from `store()`:

```python
def extract_to_wiki(
    source_text: str,
    source_id: str,
    source_label: str,
) -> dict:
    """Two-step CoT: analyze -> generate -> write -> index.

    Args:
        source_text: Full text to analyze (answer text or file text).
        source_id: Unique identifier for this extraction (conv_... or file_{hash}).
        source_label: Short descriptor for prompt context e.g. "Question: What is X?".

    Returns:
        dict with "page_ids" (list[int]) and "logic_chain" (list[dict]).
    """
    ensure_dirs()

    # Step 1: Analysis
    logger.info("Step 1: Analyzing content for wiki extraction...")
    analysis_prompt = _build_analysis_prompt(source_text, source_label)
    analysis = LLM.generate_structured(analysis_prompt, AnalysisOutput, use_language=False)
    if analysis is None:
        logger.error("Analysis LLM returned None")
        return {}
    logger.info("Analysis complete: %d topics, %d actions",
                len(analysis.topics), len(analysis.actions))

    # Step 2: Generation
    existing_contents = _read_existing_pages(analysis.actions)
    logger.info("Step 2: Generating wiki page(s)...")
    gen_prompt = _build_generation_prompt(analysis, source_text, source_label, existing_contents)
    batch = LLM.generate_structured(gen_prompt, WikiBatchOutput, use_language=False)
    if batch is None or not batch.pages:
        logger.error("Generation LLM returned None or empty pages")
        return {}

    # Write to filesystem + update SQLite
    now = datetime.now().strftime("%Y-%m-%d")
    saved_ids = []

    for wp in batch.pages:
        tags = wp.tags
        sources = wp.sources
        if source_id not in sources:
            sources.append(source_id)

        filename = title_to_filename(wp.title)
        file_path = os.path.join("wiki", "pages", filename)

        existing_page_data = read_page(file_path)
        created_str = existing_page_data.get("created", "") if existing_page_data else ""

        frontmatter = build_frontmatter(
            title=wp.title,
            tags=tags,
            sources=sources,
            created=created_str,
            updated=now,
        )
        full_content = frontmatter + "\n\n" + wp.content.strip()

        checksum = write_page(file_path, full_content)

        pid = upsert_page(
            title=wp.title,
            file_path=file_path,
            tags=tags,
            sources=sources,
            checksum=checksum,
            content=full_content,
        )
        saved_ids.append(pid)

        links = extract_wikilinks(full_content)
        if links:
            update_page_relations(pid, links)

    rebuild_index()
    logger.info("Stored %d wiki pages", len(saved_ids))

    return {
        "page_ids": saved_ids,
        "logic_chain": [{
            "node": "store",
            "action": f"Wiki: stored {len(saved_ids)} pages",
            "reasoning": (
                f"Pages: {[wp.title for wp in batch.pages]}, "
                f"Actions: {[a.action for a in analysis.actions]}"
            ),
        }],
    }
```

4. Rewrite `store(state)` as a thin wrapper:

```python
def store(state: dict) -> dict:
    """Two-step CoT extraction: analyze -> generate -> write."""
    if not state.get("needs_store", True):
        logger.info("Skipping store: needs_store=False")
        return {}

    if not state.get("answer"):
        logger.info("Skipping store: no answer")
        return {}

    if state.get("contradiction_found"):
        logger.info("Skipping store: contradiction detected")
        return {}

    source_id = _get_source_id()
    source_label = f"Question: {state['user_message']}"
    result = extract_to_wiki(state["answer"], source_id, source_label)

    if not result.get("page_ids"):
        return {}

    return {
        "stored_knowledge_ids": result["page_ids"],
        "wiki_page_ids": result["page_ids"],
        "logic_chain": result.get("logic_chain", []),
    }
```

5. Update `_get_source_id()` — already exists, no changes needed.

6. Remove the duplicate topic logging lines:
   - Keep line 193-194 (original)
   - Remove lines 195-196 (the two filtered topic lists that were recently added as debug helpers)

- [ ] **Step 1: Write the failing test for `extract_to_wiki`**

Add to `tests/test_nodes.py`:

```python
def test_extract_to_wiki_creates_pages(monkeypatch, tmp_path):
    """Call extract_to_wiki with simple text -> creates wiki pages."""
    from storage.database import DB_DIR, DB_PATH

    monkeypatch.setattr("storage.database.DB_DIR", str(tmp_path / "data"))
    monkeypatch.setattr("storage.database.DB_PATH", str(tmp_path / "data" / "knowledge.db"))
    monkeypatch.setattr("storage.wiki_storage.WIKI_DIR", str(tmp_path / "wiki"))

    from storage.database import init_db
    init_db()

    from agent.nodes.store import extract_to_wiki
    result = extract_to_wiki(
        "Python dict is a key-value store",
        "test_001",
        "Question: What is Python dict?",
    )
    assert "page_ids" in result
    assert len(result["page_ids"]) >= 1
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_nodes.py::test_extract_to_wiki_creates_pages -v`
Expected: FAIL — `extract_to_wiki` not defined or renamed functions.

- [ ] **Step 3: Implement refactoring in store.py**

Apply all changes described above: rename prompt builders, add `extract_to_wiki()`, rewrite `store()`, fix `_build_generation_prompt` to use new signature.

Key signatures to change:

```python
# Old:
def _build_analysis_prompt(user_message: str, answer: str) -> str:
# New:
def _build_analysis_prompt(source_text: str, source_label: str) -> str:
```

```python
# Old:
def _build_generation_prompt(analysis, user_message, answer, existing_page_contents):
# New:
def _build_generation_prompt(analysis, source_text, source_label, existing_page_contents):
```

Prompt content changes inside `_build_analysis_prompt`:
- `"## Q&A to Analyze"` → `"## Source Content to Analyze"`
- `f"Question: {user_message}\nAnswer: {answer}"` → `f"Context: {source_label}\n\n{source_text}"`
- `"Identify all topics covered in this Q&A"` → `"Identify all topics covered in this content"`

Prompt content changes inside `_build_generation_prompt`:
- `"## Original Q&A"` → `"## Original Content"`
- `f"Question: {user_message}\nAnswer: {answer}"` → `f"Context: {source_label}\n\n{source_text}"`

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_nodes.py::test_extract_to_wiki_creates_pages -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add agent/nodes/store.py tests/test_nodes.py
git commit -m "feat: extract extract_to_wiki() shared function, generalize prompts"
```

---

### Task 2: Fix bot.py file upload handler

**Files:**
- Modify: `server/bot.py:211-270`
- Test: `tests/test_file_processor.py`

- [ ] **Step 1: Write failing tests for file-to-wiki flow**

Add to `tests/test_file_processor.py`:

```python
def test_process_file_creates_wiki_pages(monkeypatch, tmp_path):
    """Upload a .txt file -> text extracted -> wiki pages created."""
    from storage.database import DB_DIR, DB_PATH

    monkeypatch.setattr("storage.database.DB_DIR", str(tmp_path / "data"))
    monkeypatch.setattr("storage.database.DB_PATH", str(tmp_path / "data" / "knowledge.db"))
    monkeypatch.setattr("storage.wiki_storage.WIKI_DIR", str(tmp_path / "wiki"))

    from storage.database import init_db
    init_db()

    from server.bot import _process_and_store_file
    content = b"Python dict is a key-value store. List is an ordered collection."
    result = _process_and_store_file(content, "test.txt", "user1")
    assert "wiki" in result.lower() or "页面" in result


def test_process_file_too_large(monkeypatch, tmp_path):
    """File with >8000 chars -> rejected with length warning."""
    from server.bot import _process_and_store_file, MAX_FILE_CHARS
    large_content = b"A" * (MAX_FILE_CHARS + 1)
    result = _process_and_store_file(large_content, "large.txt", "user1")
    assert "过长" in result
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_file_processor.py::test_process_file_creates_wiki_pages tests/test_file_processor.py::test_process_file_too_large -v`
Expected: FAIL — broken import or missing `MAX_FILE_CHARS`.

- [ ] **Step 3: Fix `_process_and_store_file` in bot.py**

Replace the broken logic (lines 211-270) with:

```python
MAX_FILE_CHARS = 8000


def _process_and_store_file(file_bytes: bytes, filename: str, user_id: str) -> str:
    """保存文件、提取文字、两步 CoT 提取 wiki 页面。返回回复文本。"""
    from storage.database import DB_DIR
    from storage.file_processor import extract_text_from_file, compute_file_hash
    from storage.models import (
        save_file_record,
        get_file_record_by_hash,
    )
    from agent.nodes.store import extract_to_wiki

    file_hash = compute_file_hash(file_bytes)
    ext = os.path.splitext(filename)[1].lower() or ".bin"

    # 去重：检查是否已处理过
    existing = get_file_record_by_hash(file_hash)
    if existing:
        logger.info("File already processed: %s (hash=%s)", filename, file_hash)
        return f"文件「{filename}」已处理过，无需重复处理。"

    # 保存到本地 data/files/
    files_dir = os.path.join(DB_DIR, "files")
    os.makedirs(files_dir, exist_ok=True)
    safe_name = f"{file_hash}{ext}"
    file_path = os.path.join(files_dir, safe_name)
    with open(file_path, "wb") as f:
        f.write(file_bytes)
    logger.info("File saved to %s", file_path)

    # 提取文字内容
    text = extract_text_from_file(file_path)
    if not text.strip():
        logger.warning("No text extracted from %s", filename)
        return f"未能从文件「{filename}」中提取到文字内容。"

    logger.info("Extracted %d chars from %s", len(text), filename)

    # 长度检查
    if len(text) > MAX_FILE_CHARS:
        logger.warning("File too long: %d chars (max %d)", len(text), MAX_FILE_CHARS)
        return f"文件「{filename}」内容过长（{len(text)} 字符），暂不支持处理超过 {MAX_FILE_CHARS} 字符的文档。"

    # 两步 CoT 提取为 wiki 页面
    saved = extract_to_wiki(
        source_text=text,
        source_id=f"file_{file_hash}",
        source_label=f"From file: {filename}",
    )

    if not saved.get("page_ids"):
        return f"未能从文件「{filename}」中提取到 wiki 页面。"

    # 记录文件处理记录
    save_file_record(filename, ext, file_hash, text, saved["page_ids"], user_id)

    reply = (
        f"已从文件「{filename}」中提取并创建了 {len(saved['page_ids'])} 篇 wiki 页面"
    )
    logger.info("File processed: %s -> %d wiki pages", filename, len(saved['page_ids']))
    return reply
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_file_processor.py::test_process_file_creates_wiki_pages tests/test_file_processor.py::test_process_file_too_large -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add server/bot.py tests/test_file_processor.py
git commit -m "feat: fix file upload to use extract_to_wiki, add length limit"
```

---

### Task 3: Verify full test suite

**Files:** All

- [ ] **Step 1: Run all tests**

Run: `pytest -v`  
Expected: All tests pass (no regressions in existing tests).

- [ ] **Step 2: Run the file-creation test with real LLM to confirm end-to-end**

Run: `pytest tests/test_file_processor.py::test_process_file_creates_wiki_pages -v -s`  
Expected: PASS (may be slow, 30-60s for LLM calls).

If tests pass, implementation is complete. If any regressions, fix them before proceeding.
