# Personal Knowledge Base Agent — Design Spec

## Overview

A WeChat Work-based personal knowledge agent built with Python + LangGraph. Receives daily questions, classifies them, stores distilled knowledge points locally, and autonomously searches the web when needed. All data stays on the user's machine in SQLite.

## Architecture

```
[WeChat Work] ←→ [Flask Webhook Server]
                      ↓
               [LangGraph Pipeline] ←→ [Web Search (DuckDuckGo)]
                      ↓
               [SQLite Database]
```

## WeChat Work Integration

- Flask server listens on localhost; exposed via a tunnel (frp/ngrok) for WeChat Work webhook callbacks
- Webhook parses incoming messages and feeds them into the LangGraph pipeline
- Final response is sent back through WeChat Work's send-message API
- No ICP filing or public server required (WeChat Work personal bot webhooks)

## LangGraph Pipeline

### State

```python
class AgentState(TypedDict):
    user_message: str
    user_id: str
    timestamp: str
    category: str | None
    confidence: float
    needs_search: bool
    search_results: list | None
    stored_knowledge: list
    final_response: str
```

### Nodes

1. **parse** — extract user message + metadata from WeChat Work payload
2. **retrieve** — query SQLite (FTS5 / keyword match) for past knowledge points in related categories
3. **classify_and_answer** — single LLM call: categorize question AND attempt to answer. Outputs: `category`, `answer`, `confidence`, `needs_search`
4. **[conditional]** if `needs_search` → **search_web** (DuckDuckGo) + **regenerate** (LLM rewrites answer with web context); if confident → skip
5. **store** — LLM distills `(question, answer)` into 1+ knowledge points, then saves to SQLite
6. **respond** — format and return response to WeChat Work

### Flow

```
parse → retrieve → classify_and_answer → [needs_search?]
                                         ↙           ↘
                                       yes            no
                                       ↓              ↓
                                 search_web       store + respond
                                 + regenerate
                                       ↓
                                 store + respond
```

The LLM's structured output includes a `needs_search` flag. The graph uses a LangGraph conditional edge to branch.

## Storage Schema (SQLite)

### `knowledge_points`

| Column | Type | Notes |
|---|---|---|
| id | INTEGER PK | auto-increment |
| knowledge_text | TEXT | The distilled knowledge point |
| source_question | TEXT | Original user question |
| category | TEXT | hierarchical, e.g. "databases/redis" |
| tags | TEXT | JSON array |
| created_at | TEXT | ISO timestamp |
| updated_at | TEXT | for future edits |

### `categories`

| Column | Type | Notes |
|---|---|---|
| id | INTEGER PK | |
| name | TEXT | hierarchical category name |
| parent_id | INTEGER | nullable FK to self |
| description | TEXT | |

### Knowledge Distillation

The `store` node calls the LLM to transform raw Q&A into concise, standalone knowledge points:

```
Input Q: "What is Redis persistence?"
Input A: "Redis supports RDB snapshots and AOF logs..."
                     ↓ (LLM distill)
Knowledge 1: "Redis RDB — point-in-time snapshots, configured by save intervals"
Knowledge 2: "Redis AOF — append-only log of write ops, more durable, supports rewrite"
Category: "databases/redis"
Tags: ["persistence", "rdb", "aof"]
```

## Directory Structure

```
langchain-learning/
├── agent/
│   ├── __init__.py
│   ├── graph.py          # LangGraph graph definition
│   ├── state.py          # AgentState TypedDict
│   ├── nodes/
│   │   ├── __init__.py
│   │   ├── parse.py
│   │   ├── retrieve.py
│   │   ├── classify_and_answer.py
│   │   ├── search_web.py
│   │   ├── regenerate.py
│   │   ├── store.py
│   │   └── respond.py
│   └── tools/
│       ├── __init__.py
│       └── web_search.py
├── server/
│   ├── __init__.py
│   ├── webhook.py        # Flask webhook for WeChat Work
│   └── config.py
├── storage/
│   ├── __init__.py
│   ├── database.py       # SQLite connection + schema init
│   └── models.py         # Data access functions
├── tests/
│   ├── test_nodes.py
│   ├── test_graph.py
│   └── test_storage.py
├── requirements.txt
├── main.py
└── .env                  # API keys (not committed)
```

## Error Handling

- **Web search failure** (network/DNS): LLM responds with "I'm not fully sure, but..." and flags the knowledge point as `unverified`
- **WeChat Work webhook failure**: log error, no retry (messages are idempotent from user perspective)
- **LLM call failure**: retry once, then return friendly error to user
- **SQLite errors**: log + return error to user; no data loss (writes are atomic)

## Testing

- `pytest` for all tests
- Unit tests for each graph node: mock LLM calls, test classification + search + storage logic in isolation
- Integration test for the full pipeline with a real SQLite database (temp file, cleaned up after)
- Webhook handler tests using Flask test client

## Phase 2 (Future)

- Document/image upload memory (new graph node)
- Independent reasoning / fact-checking against stored knowledge
- Mind-map visualization of the knowledge base (query `categories` + `knowledge_points`, output to a visualization tool)
- User correction feedback loop (when user corrects a stored point, the agent learns from the mistake)

## Non-Goals (Phase 1)

- No user authentication beyond WeChat Work's built-in identity
- No cloud sync or backup
- No multi-user support
- No image/voice processing
