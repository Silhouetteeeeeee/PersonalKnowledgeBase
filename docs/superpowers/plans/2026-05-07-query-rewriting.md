# 对话式 RAG 查询重写实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 retrieve 之前用 LLM 将依赖上下文的追问重写为独立查询，解决多轮对话中 RAG 检索丢失上文语境的问题。

**Architecture:** 在 `parse → retrieve` 之间插入 `rewrite_query` 节点，读取 `MessageHistory` 获取最近对话，如历史充分则调用 LLM 改写，结果写入 `state["search_query"]`。`retrieve` 节点改用 `search_query` 替代 `user_message`。同时引入 `LLM.get_model_for(task)` 支持按任务选择模型。

**Tech Stack:** LangGraph (StateGraph), ChatDeepSeek, sqlite-vec, sentence-transformers

---

## 文件结构

| 文件 | 操作 | 职责 |
|---|---|---|
| `server/config.py:14` | + 配置 | 添加 `TASK_MODEL_MAP` 字典 |
| `agent/utils/llm.py:38-44` | + 方法 | 添加 `get_model_for(task)` 类方法 |
| `agent/state.py:7` | + 字段 | 添加 `search_query: str` |
| `agent/nodes/rewrite_query.py` | **新建** | LLM 查询改写节点 |
| `agent/nodes/retrieve.py:9` | 改 1 行 | `query = state.get("search_query") or state["user_message"]` |
| `agent/graph.py:77-79` | + 2 行 | 注册 `rewrite_query` 节点+边 |
| `tests/test_rewrite_query.py` | **新建** | rewrite_query 节点测试 |
| `tests/test_nodes.py:33-40` | + 测试 | retrieve 使用 search_query 的测试 |

---

### Task 1: 模型选择机制（config + LLM class）

**Files:**
- Modify: `server/config.py:14` — 加 `TASK_MODEL_MAP`
- Modify: `agent/utils/llm.py:38-44` — 加 `get_model_for(task)`

- [ ] **Step 1: 在 config 中添加 TASK_MODEL_MAP**

```python
# server/config.py — 在 LLM_TEMPERATURE 配置之后追加

TASK_MODEL_MAP: dict[str, str] = {
    "default": LLM_MODEL,
    "rewrite": LLM_MODEL,  # 后续可换为专用轻量模型
}
```

- [ ] **Step 2: 在 LLM 类中添加 get_model_for 方法**

```python
# agent/utils/llm.py — 在 get_model 方法之后添加

@classmethod
def get_model_for(cls, task: str, temperature: float | None = None) -> ChatDeepSeek:
    """根据任务名获取模型实例。

    在 TASK_MODEL_MAP 中配置 task → model_name 映射。
    未注册的任务回退到 LLM_MODEL。
    """
    from server.config import TASK_MODEL_MAP
    model_name = TASK_MODEL_MAP.get(task, LLM_MODEL)
    if temperature is not None:
        return ChatDeepSeek(model=model_name, temperature=temperature)
    if cls._default_model is None:
        cls._default_model = ChatDeepSeek(model=model_name, temperature=LLM_TEMPERATURE)
    return cls._default_model
```

- [ ] **Step 3: 提交**

```bash
git add server/config.py agent/utils/llm.py
git commit -m "feat: add TASK_MODEL_MAP and LLM.get_model_for(task)"
```

---

### Task 2: search_query 字段 + rewrite_query 节点

**Files:**
- Modify: `agent/state.py:7` — 加 `search_query`
- Create: `agent/nodes/rewrite_query.py` — 新节点

- [ ] **Step 1: 在 AgentState 中添加 search_query 字段**

```python
# agent/state.py — 在 user_message 之后插入
    search_query: str  # LLM 重写后的独立查询语句（检索用）
```

初始 invoke 时传空字符串，`parse` 节点不需要动（它只处理 `user_message`）。

- [ ] **Step 2: 创建 rewrite_query.py**

```python
"""Rewrite user query with conversation context for standalone retrieval."""
import logging

from memory.message_history import MessageHistory
from agent.utils.llm import LLM

logger = logging.getLogger(__name__)

REWRITE_PROMPT = (
    "你是一个查询改写助手。根据对话历史，将用户的最新问题改写为一个"
    "不需要上下文就能理解的独立问题。\n\n"
    "要求：\n"
    "- 补全指代（如"它"→"Python dict"、"区别呢"→"A和B的区别"）\n"
    "- 补全省略的部分\n"
    "- 不要添加不存在的信息\n"
    "- 如果问题已经是独立的，保持原文\n"
    "- 只输出改写后的文本，不要任何解释\n\n"
    "对话历史：\n{history}\n\n"
    "用户最新消息：{message}"
)


def rewrite_query(state: dict) -> dict:
    """Rewrite the user message as a standalone query using conversation history.

    Falls back to the original user_message when:
    - Less than 2 prior messages exist (new or short conversation)
    - LLM call fails or returns empty
    """
    user_message = state["user_message"]
    session_id = int(state["session_id"])

    history = MessageHistory.get_recent(session_id)
    if len(history) < 2:
        logger.debug("Skipping rewrite: only %d history messages", len(history))
        return {"search_query": user_message}

    lines = []
    for msg in history:
        role = "用户" if msg["role"] == "user" else "助手"
        lines.append(f"{role}：{msg['content'][:200]}")
    history_text = "\n".join(lines)

    prompt = REWRITE_PROMPT.format(history=history_text, message=user_message)
    try:
        model = LLM.get_model_for("rewrite")
        rewritten = model.invoke(prompt)
        if hasattr(rewritten, "content"):
            rewritten = rewritten.content
        rewritten = rewritten.strip()
        if not rewritten:
            logger.warning("Rewrite returned empty, falling back to original")
            return {"search_query": user_message}
        logger.info("Rewrote query: '%s' → '%s'", user_message[:40], rewritten[:60])
        return {"search_query": rewritten}
    except Exception as e:
        logger.warning("Rewrite query failed: %s, falling back to original", e)
        return {"search_query": user_message}
```

- [ ] **Step 3: 提交**

```bash
git add agent/state.py agent/nodes/rewrite_query.py
git commit -m "feat: add search_query field and rewrite_query node"
```

---

### Task 3: 修改 retrieve 节点使用 search_query

**Files:**
- Modify: `agent/nodes/retrieve.py:9` — 改 1 行

- [ ] **Step 1: 修改 query 来源**

```python
# agent/nodes/retrieve.py — line 9 将 query = state["user_message"] 改为：
    query = state.get("search_query") or state["user_message"]
```

- [ ] **Step 2: 提交**

```bash
git add agent/nodes/retrieve.py
git commit -m "feat: retrieve node uses search_query when available"
```

---

### Task 4: Graph 中注册 rewrite_query 节点

**Files:**
- Modify: `agent/graph.py`

- [ ] **Step 1: 在 graph.py 中注册节点和边**

```python
# agent/graph.py — 在 import 部分追加
from agent.nodes.rewrite_query import rewrite_query

# 在 build_graph() 中，add_node 区域追加
    builder.add_node("rewrite_query", rewrite_query)

# 修改边的连接：
# 将：
#     builder.add_edge("parse", "retrieve")
# 改为：
    builder.add_edge("parse", "rewrite_query")
    builder.add_edge("rewrite_query", "retrieve")
```

- [ ] **Step 2: 提交**

```bash
git add agent/graph.py
git commit -m "feat: wire rewrite_query node into graph (parse → rewrite_query → retrieve)"
```

---

### Task 5: rewrite_query 节点测试

**Files:**
- Create: `tests/test_rewrite_query.py`

- [ ] **Step 1: 写 rewrite_query 的完整测试**

```python
"""Tests for the rewrite_query node."""
import pytest


class FakeMessage:
    """Duck-typed message dict matching MessageHistory return format."""
    def __init__(self, role, content):
        self.role = role
        self.content = content

    def __getitem__(self, key):
        return getattr(self, key)


@pytest.fixture(autouse=True)
def fixture_temp_db(monkeypatch, tmp_path):
    """Minimal DB init needed for MessageHistory.get_recent."""
    from storage.database import init_db
    db_dir = tmp_path / "data"
    db_dir.mkdir()
    monkeypatch.setattr("storage.database.DB_DIR", str(db_dir))
    monkeypatch.setattr("storage.database.DB_PATH", str(db_dir / "knowledge.db"))
    init_db()


def test_skip_rewrite_when_no_history(monkeypatch):
    """Conversation with <2 messages → returns original unchanged."""
    monkeypatch.setattr(
        "agent.nodes.rewrite_query.MessageHistory.get_recent",
        lambda session_id, limit=12: [],
    )
    from agent.nodes.rewrite_query import rewrite_query
    result = rewrite_query({
        "user_message": "Java Map 和它的区别？",
        "session_id": "1",
    })
    assert result["search_query"] == "Java Map 和它的区别？"


def test_skip_rewrite_when_only_one_message(monkeypatch):
    """Only 1 prior message → returns original unchanged."""
    monkeypatch.setattr(
        "agent.nodes.rewrite_query.MessageHistory.get_recent",
        lambda session_id, limit=12: [
            {"role": "user", "content": "Python dict 是什么？"},
        ],
    )
    from agent.nodes.rewrite_query import rewrite_query
    result = rewrite_query({
        "user_message": "Java Map 和它的区别？",
        "session_id": "1",
    })
    assert result["search_query"] == "Java Map 和它的区别？"


def test_rewrite_with_full_history(monkeypatch):
    """Has 2+ history messages → LLM called, rewritten query returned."""
    monkeypatch.setattr(
        "agent.nodes.rewrite_query.MessageHistory.get_recent",
        lambda session_id, limit=12: [
            {"role": "user", "content": "Python dict 是什么？"},
            {"role": "assistant", "content": "Python dict 是键值对集合"},
        ],
    )

    fake_response = "Java Map 和 Python dict 的区别"
    class FakeModel:
        def invoke(self, prompt):
            assert "Python dict" in prompt
            assert "Java Map" in prompt
            return type("AIMessage", (), {"content": fake_response})()

    monkeypatch.setattr(
        "agent.nodes.rewrite_query.LLM.get_model_for",
        lambda task, temperature=None: FakeModel(),
    )

    from agent.nodes.rewrite_query import rewrite_query
    result = rewrite_query({
        "user_message": "Java Map 和它的区别？",
        "session_id": "1",
    })
    assert result["search_query"] == "Java Map 和 Python dict 的区别"


def test_fallback_when_llm_returns_empty(monkeypatch):
    """LLM returns empty → falls back to original."""
    monkeypatch.setattr(
        "agent.nodes.rewrite_query.MessageHistory.get_recent",
        lambda session_id, limit=12: [
            {"role": "user", "content": "Python dict"},
            {"role": "assistant", "content": "Key-value store"},
        ],
    )

    class FakeModel:
        def invoke(self, prompt):
            return type("AIMessage", (), {"content": ""})()

    monkeypatch.setattr(
        "agent.nodes.rewrite_query.LLM.get_model_for",
        lambda task, temperature=None: FakeModel(),
    )

    from agent.nodes.rewrite_query import rewrite_query
    result = rewrite_query({
        "user_message": "Java Map 和它的区别？",
        "session_id": "1",
    })
    assert result["search_query"] == "Java Map 和它的区别？"


def test_fallback_when_llm_raises(monkeypatch):
    """LLM exception → falls back to original."""
    monkeypatch.setattr(
        "agent.nodes.rewrite_query.MessageHistory.get_recent",
        lambda session_id, limit=12: [
            {"role": "user", "content": "Python dict"},
            {"role": "assistant", "content": "Key-value store"},
        ],
    )

    class FakeModel:
        def invoke(self, prompt):
            raise RuntimeError("API timeout")

    monkeypatch.setattr(
        "agent.nodes.rewrite_query.LLM.get_model_for",
        lambda task, temperature=None: FakeModel(),
    )

    from agent.nodes.rewrite_query import rewrite_query
    result = rewrite_query({
        "user_message": "Java Map 和它的区别？",
        "session_id": "1",
    })
    assert result["search_query"] == "Java Map 和它的区别？"


def test_rewrite_converts_session_id(monkeypatch):
    """session_id conversion from string to int works."""
    captured = {}

    def fake_get_recent(session_id, limit=12):
        captured["sid"] = session_id
        return []

    monkeypatch.setattr(
        "agent.nodes.rewrite_query.MessageHistory.get_recent",
        fake_get_recent,
    )

    from agent.nodes.rewrite_query import rewrite_query
    rewrite_query({
        "user_message": "test",
        "session_id": "42",
    })
    assert captured["sid"] == 42
    assert isinstance(captured["sid"], int)
```

- [ ] **Step 2: 运行测试确认通过**

```bash
pytest tests/test_rewrite_query.py -v
Expected: 6 passed
```

- [ ] **Step 3: 提交**

```bash
git add tests/test_rewrite_query.py
git commit -m "test: add rewrite_query node tests"
```

---

### Task 6: retrieve 使用 search_query 的测试

**Files:**
- Modify: `tests/test_nodes.py` — 追加测试

- [ ] **Step 1: 在 test_nodes.py 末尾追加两个测试**

```python
# tests/test_nodes.py — 在文件末尾追加

def test_retrieve_uses_search_query():
    """retrieve should prefer search_query over user_message."""
    from storage.models import save_knowledge_point
    from agent.nodes.retrieve import retrieve

    save_knowledge_point("Python dict is a key-value store", "What is Python dict?", "programming/python", ["python"])

    # user_message 不包含 "Python"，但 search_query 包含
    result = retrieve({
        "user_message": "Java Map and its differences",
        "search_query": "Python dict differences",
    })
    assert len(result["stored_knowledge"]) >= 1
    assert "Python" in result["stored_knowledge"][0]["knowledge_text"]


def test_retrieve_falls_back_to_user_message():
    """retrieve should fall back to user_message when search_query is missing."""
    from storage.models import save_knowledge_point
    from agent.nodes.retrieve import retrieve

    save_knowledge_point("Java Map is a collection interface", "What is Java Map?", "programming/java", ["java"])

    result = retrieve({
        "user_message": "Tell me about Java Map",
        # no search_query in state
    })
    assert len(result["stored_knowledge"]) >= 1
    assert "Java" in result["stored_knowledge"][0]["knowledge_text"]
```

- [ ] **Step 2: 运行测试确认通过**

```bash
pytest tests/test_nodes.py::test_retrieve_uses_search_query tests/test_nodes.py::test_retrieve_falls_back_to_user_message -v
Expected: 2 passed
```

- [ ] **Step 3: 提交**

```bash
git add tests/test_nodes.py
git commit -m "test: add retrieve search_query tests"
```

---

### Task 7: Graph 集成测试

**Files:**
- Modify: `tests/test_graph.py` — 追加测试

- [ ] **Step 1: 在 test_graph.py 末尾追加集成测试**

```python
# tests/test_graph.py — 在文件末尾追加

def test_graph_has_rewrite_query_node():
    """Graph should include the rewrite_query node."""
    from agent.graph import build_graph

    g = build_graph()
    assert "rewrite_query" in g.nodes


def test_graph_rewrite_query_edge():
    """parse should connect to rewrite_query, not directly to retrieve."""
    from agent.graph import build_graph

    g = build_graph()
    edges = [(s, t) for s, t in g.edges if s == "parse"]
    assert ("parse", "rewrite_query") in edges
    assert ("parse", "retrieve") not in edges


def test_graph_search_query_flows_to_retrieve(monkeypatch):
    """search_query from rewrite_query should be used by retrieve."""
    from storage.models import save_knowledge_point

    save_knowledge_point("Python dict is a key-value store", "What is Python dict?", "programming/python", ["python"])

    monkeypatch.setattr(
        "agent.nodes.rewrite_query.MessageHistory.get_recent",
        lambda session_id, limit=12: [],
    )

    # Mock LLM to return a rewritten standalone query
    class FakeModel:
        def invoke(self, prompt):
            return type("AIMessage", (), {"content": "Python dict differences"})()

    monkeypatch.setattr(
        "agent.nodes.rewrite_query.LLM.get_model_for",
        lambda task, temperature=None: FakeModel(),
    )

    from agent.graph import build_graph
    g = build_graph()

    result = g.invoke({
        "user_message": "Java Map and its differences",
        "user_id": "test_user",
        "session_id": "1",
        "search_query": "",
        "timestamp": "2026-05-02T12:00:00",
    })

    stored = result.get("stored_knowledge", [])
    assert len(stored) >= 1
    assert "Python" in stored[0]["knowledge_text"]
```

- [ ] **Step 2: 运行测试确认通过**

```bash
pytest tests/test_graph.py::test_graph_has_rewrite_query_node tests/test_graph.py::test_graph_rewrite_query_edge tests/test_graph.py::test_graph_search_query_flows_to_retrieve -v
Expected: 3 passed
```

- [ ] **Step 3: 提交**

```bash
git add tests/test_graph.py
git commit -m "test: add graph integration tests for rewrite_query"
```

---

## 自检验证

- [ ] **Spec 覆盖验证**：
  - Task 1 → 模型选择机制 (`TASK_MODEL_MAP` + `LLM.get_model_for(task)`) ✓
  - Task 2 → `search_query` 字段 + `rewrite_query` 节点 ✓
  - Task 3 → retrieve 改用 `search_query` ✓
  - Task 4 → Graph 注册和边 ✓
  - Task 5,6,7 → 测试覆盖 ✓
  - Episodic memory 原文标注"本次暂不改"→ 没有对应 task ✓
  - Error_records 标注"本次不涉及"→ 没有对应 task ✓
  - `bot.py` 标注"不改"→ 没有对应 task ✓
