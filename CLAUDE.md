# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Personal Knowledge Base Agent — a WeChat Work smart bot powered by LangGraph. It ingests user messages, classifies them, retrieves/stores knowledge in a local SQLite DB with vector embeddings, and handles contradictions through a reflection loop.

## Commands

```bash
# Run unit tests (fast, ~18s)
pytest -v

# Run integration tests (slow, real LLM calls)
pytest tests/integration/ -v

# Run all tests
pytest tests/unit/ tests/integration/ -v

# Run single test
pytest tests/unit/test_nodes.py -v -k test_parse

# Run the bot
python main.py

# Environment setup
pip install -r requirements.txt
```

## Architecture

### LangGraph Agent (agent/)

A `StateGraph` with these nodes in order:

1. **parse** — strips/validates input
2. **rewrite_query** — rewrites the user query into a standalone search query (with conversation context)
3. **retrieve** — vector search (BAAI/bge-small-zh-v1.5) for wiki page retrieval
4. **classify_and_answer** — ReAct agent with web search tool, produces structured output (answer, confidence, needs_store)
5. **update_profile** — updates user profile from conversation (parallel with fact_check)
6. **fact_check** — checks answer against stored knowledge for contradictions
7. **reflect** — if contradiction found, determines whether stored knowledge or new answer is wrong
8. **record_error** — records the error for future reference
9. **search_web** — fallback web search when needed
10. **regenerate** — regenerates answer with web search results
11. **store** — LLM distills Q&A into wiki pages via two-step CoT extraction (analyze → generate)
12. **respond** — builds final response, writes reasoning log

**Contradiction loop**: fact_check → reflect → record_error → search_web → fact_check (max 2 cycles)

### State (agent/state.py)

Central `AgentState` TypedDict holds all runtime data: message, user context, knowledge, reflection fields, control flags, and a `logic_chain` list that accumulates reasoning traces across nodes via `operator.add`.

### Storage Layer (storage/)

- **SQLite** (`knowledge.db`) with `sqlite-vec` for vector search (512-dim cosine distance)
- Tables: `pages`, `page_vectors` (vec0 virtual), `page_relations`, `file_records`, `error_records`, `error_vectors`
- **Profile** per user as JSON files in `data/profiles/` with backup rotation
- **File processing** via `file_processor.py`: PDF (PyMuPDF), DOCX, XMind, images (PaddleOCR), plain text

### Memory System (memory/)

Three-tier memory per user:

1. **User profile** — persistent JSON, loaded per request
2. **Conversation history** — recent messages in the current session (30-min timeout)
3. **Episodic memory** — cross-session summarized memories with vector search

Tables: `sessions`, `messages` in the same `knowledge.db`.

### Bot Interface (server/)

- **bot.py** — `WSClient` from wecom-aibot-python-sdk, handles text/image/file messages asynchronously
- **claude_bridge.py** — optional `/code` command that pipes messages to `claude --print` for remote coding
- **config.py** — env-based configuration (DeepSeek model, WeChat credentials, optional Baidu search API)

### LLM Interface (agent/utils/llm.py)

Unified `LLM` class wrapping `ChatDeepSeek` (langchain-deepseek). Supports structured output (Pydantic), per-task model overrides, language instruction injection, and retry logic.

### Key Design Decisions

- **DeepSeek-chat** as the default LLM (configurable via `.env`)
- **fastembed** (BAAI/bge-small-zh-v1.5) for embedding — lazy-loaded singleton, thread-safe
- **Wiki pages** as the only knowledge store — page content lives in markdown files on disk, SQLite serves as index for vector search
- Output language: controlled by `OUTPUT_LANGUAGE` env var (default English, no extra prompt tokens)
