# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Personal Knowledge Base Agent — a WeChat Work smart bot powered by LangGraph. It ingests user messages, classifies them, retrieves/stores knowledge in a local SQLite DB with vector embeddings, and handles contradictions through a reflection loop.

## Commands

```bash
# Run tests (single)
pytest tests/test_nodes.py -v -k test_parse
pytest tests/test_graph.py -v
pytest tests/test_storage.py -v

# Run all tests
pytest -v

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
3. **retrieve** — vector search (BAAI/bge-small-zh-v1.5) → cross-encoder rerank (BAAI/bge-reranker-v2-m3) to find relevant knowledge
4. **classify_and_answer** — ReAct agent with web search tool, produces structured output (answer, confidence, needs_store)
5. **update_profile** — updates user profile from conversation (parallel with fact_check)
6. **fact_check** — checks answer against stored knowledge for contradictions
7. **reflect** — if contradiction found, determines whether stored knowledge or new answer is wrong
8. **correct_knowledge** / **record_error** — fixes knowledge or records the error
9. **search_web** — fallback web search when needed
10. **regenerate** — regenerates answer with web search results
11. **store** — LLM distills Q&A into knowledge points, deduplicates via vector similarity, saves with embeddings
12. **respond** — builds final response, writes reasoning log

**Contradiction loop**: fact_check → reflect → correct_knowledge/record_error → search_web → fact_check (max 2 cycles)

### State (agent/state.py)

Central `AgentState` TypedDict holds all runtime data: message, user context, knowledge, reflection fields, control flags, and a `logic_chain` list that accumulates reasoning traces across nodes via `operator.add`.

### Storage Layer (storage/)

- **SQLite** (`knowledge.db`) with `sqlite-vec` for vector search (512-dim cosine distance)
- Tables: `categories`, `knowledge_points`, `knowledge_vectors` (vec0 virtual), `file_records`, `error_records`, `error_vectors`
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
- **Cross-encoder reranker** (BAAI/bge-reranker-v2-m3) for precision improvement when >3 candidates
- Category normalization: lowercase, unified separators, 4-level max hierarchy
- Knowledge dedup: vector similarity check (cosine distance < 0.25) before insertion
- Output language: controlled by `OUTPUT_LANGUAGE` env var (default English, no extra prompt tokens)
