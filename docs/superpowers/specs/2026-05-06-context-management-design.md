# 三层混合记忆系统 — 上下文管理设计

**Problem:** 当前对话机器人无状态，每次 `graph.invoke()` 都创建全新的 `AgentState`。用户无法使用"展开讲讲"、"我刚才说了什么"这类短距离指代，也无法跨 session 记住用户偏好和习惯。

**Solution:** 分三层记忆系统 — 工作记忆（最近 6 轮对话）、情景记忆（跨 session 向量检索）、核心记忆（用户画像）。通过 30 分钟无活动超时划分 session 边界。

**Non-goal:** 实时流式输出协作、多模态上下文。

---

## Architecture

```
企业微信 → bot.py
               │
               ├── 1. session_manager.lookup(user_id) → 创建/恢复 session
               ├── 2. context_builder.build(session, content) → 三层装配
               │       ├── 核心记忆 (user_profile per-user)
               │       ├── 情景记忆 (vector search 跨 session)
               │       └── 工作记忆 (最近 6 轮对话)
               ├── 3. graph.invoke(state + context)
               ├── 4. asyncio.create_task(save_turn(...)) → 异步持久化
               └── 5. 回复企业微信
```

### Key decisions

- **三层分离**: 工作记忆保短期精确、情景记忆保长期关联、核心记忆保用户画像，互不干扰
- **会话超时**: 30 分钟无活动自动归档，不手动删除记录，向量检索仍在
- **异步持久化**: 情景摘要提取和 embedding 在主流程返回后异步执行，不阻塞回复
- **LangGraph 无 checkpoint**: 不依赖 LangGraph persistence 层，自行管理 session 状态，更灵活可控

---

## Components

### New: `memory/session_manager.py`

```python
class SessionManager:
    def lookup(self, user_id: str) -> dict: ...
    def refresh(self, session_id: int): ...
    def archive_stale(self): ...
```

| 方法 | 职责 |
|------|------|
| `lookup()` | 查找 active 且 30 分钟内活跃的 session；没有则归档旧的、创建新的 |
| `refresh()` | 更新 `last_active_at` |
| `archive_stale()` | 定时清理超时 session（惰性：在 lookup 中顺便处理） |

### New: `memory/message_history.py`

```python
class MessageHistory:
    def add_message(self, session_id: int, user_id: str, role: str, content: str): ...
    def get_recent(self, session_id: int, limit: int = 12) -> list[dict]: ...
    def get_session_messages(self, session_id: int) -> list[dict]: ...
```

| 方法 | 职责 |
|------|------|
| `add_message()` | 插入一条消息到 `messages` 表 |
| `get_recent()` | 取最近 N 条消息用于工作记忆组装 |
| `get_session_messages()` | 取 session 全量消息（用于异步摘要生成） |

### New: `memory/episodic.py`

```python
class EpisodicMemory:
    def summarize_and_embed(self, session_id: int, user_id: str):
        """后台异步：取最新一轮对话 → LLM 摘要 → embedding 存储"""
        session_messages = message_history.get_session_messages(session_id)
        last_turn = session_messages[-2:]  # (user, assistant)
        user_msg = last_turn[0]["content"][:500] if last_turn else ""
        asst_msg = last_turn[1]["content"][:500] if len(last_turn) > 1 else ""
        if not user_msg and not asst_msg:
            return
        summarize_prompt = (
            f"请将以下对话浓缩为一句话摘要，保留关键信息（主题、结论、用户偏好）：\n"
            f"用户：{user_msg}\n助手：{asst_msg}"
        )
        summary = LLM.generate(summarize_prompt, use_language=False)
        embedding = embed_text(summary)
        save_embedding(session_id, user_id, summary, embedding)

    def search(self, user_id: str, query: str, limit: int = 3) -> list[dict]:
        """向量检索 top-3 情景记忆"""
        ...
```

| 方法 | 职责 |
|------|------|
| `summarize_and_embed()` | 后台异步：LLM 摘要 + embedding 存储 |
| `search()` | 向量检索 top-3 情景记忆 |

### New: `memory/context_builder.py`

```python
class ContextBuilder:
    def build(self, user_id: str, session_id: int, content: str) -> dict:
        # 1. 核心记忆 ← load_profile(user_id)
        # 2. 工作记忆 ← message_history.get_recent(session_id, 12)
        # 3. 情景记忆 ← episodic.search(user_id, content, 3)
        # 4. 拼接为三段式 context
```

输出结构：

```python
{
    "profile_section": "<user_profile>...</user_profile>",
    "history_section": "<conversation_history>...</conversation_history>",
    "episodic_section": "<your_long_term_memories>...</your_long_term_memories>",
}
```

### New: `memory/models.py`

初始化新表：

```sql
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
    last_active_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
);
CREATE INDEX IF NOT EXISTS idx_sessions_user_status ON sessions(user_id, status);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES sessions(id),
    user_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    category TEXT DEFAULT '',
    embedding BLOB,
    created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, created_at);
CREATE INDEX IF NOT EXISTS idx_messages_user ON messages(user_id, created_at);
```

### Existing: `storage/database.py`

调用 `memory/models.py` 的表初始化，`init_db()` 末尾新增 `init_memory_tables()` 调用。

### Existing: `storage/profile.py`

`load_profile()` 改为接受 `user_id` 参数，读写 `data/profiles/<user_id>.json`。`save_profile()` 同理。

### Existing: `server/bot.py`

```python
# Before (无状态):
graph.invoke({"user_message": content, "user_id": user_id, "user_profile": load_profile()})

# After (有状态):
session = session_manager.lookup(user_id)
context = context_builder.build(session["id"], content)
result = graph.invoke({
    "user_message": content,
    "user_id": user_id,
    "session_id": session["id"],
    "user_profile": load_profile(user_id),
})
asyncio.create_task(save_turn(session["id"], user_id, content, result))
```

### Existing: `agent/state.py`

新增字段：

```python
class AgentState(TypedDict):
    # ... 已有字段 ...
    session_id: str
    message_history: list[dict]
    episodic_memories: list[str]
```

### Existing: `agent/nodes/classify_and_answer.py`

在 system prompt 末尾追加三段式 context，通过 `state["user_profile"]`、`state["message_history"]`、`state["episodic_memories"]` 注入：

```python
# 注入核心记忆
if state.get("user_profile"):
    profile = json.dumps(state["user_profile"], ensure_ascii=False)
    context_parts.append(f"<user_profile>\n{profile}\n</user_profile>")

# 注入工作记忆
if state.get("message_history"):
    history_lines = []
    for msg in state["message_history"]:
        role = "user" if msg["role"] == "user" else "assistant"
        history_lines.append(f"{role}: {msg['content']}")
    context_parts.append("<conversation_history>\n" + "\n".join(history_lines[-12:]) + "\n</conversation_history>")

# 注入情景记忆
if state.get("episodic_memories"):
    memories = "\n".join(f"[{m['date']}] {m['summary']}" for m in state["episodic_memories"])
    context_parts.append(f"<your_long_term_memories>\n{memories}\n</your_long_term_memories>")

full_prompt = base_prompt + "\n\n" + "\n\n".join(context_parts)
```

### Existing: `agent/nodes/update_profile.py`

改为按 `user_id` 读写 profile 文件 `data/profiles/<user_id>.json`。

---

## Data Flow

### 正常流程

```
1. 用户发送消息 → bot.py _on_text
2. session_manager.lookup("user_123")
   - 查找 active + 30min 内活跃的 session
   - 有 → 复用；无 → 归档旧的 + 新建
3. context_builder.build(session_id, "快速排序怎么写")
   - load_profile("user_123") → <user_profile>
   - message_history.get_recent(session_id, 12) → <conversation_history>
   - episodic.search("user_123", "快速排序", 3) → <your_long_term_memories>
4. graph.invoke(state + 三层 context)
5. 回复用户
6. 异步:
   - message_history.add_message(session_id, role="user", content="...")
   - message_history.add_message(session_id, role="assistant", content="...")
   - episodic.summarize_and_embed(session_id, "user_123")  [后台 LLM 摘要]
```

### 边界情况

| 情况 | 处理 |
|------|------|
| 首次用户 | session_manager 创建新 session，三层记忆均为空 |
| 30 分钟内再次发言 | session 复用，工作记忆包含上轮对话 |
| 超过 30 分钟 | session 归档，新 session 工作记忆为空，情景记忆仍可检索 |
| 情景检索无结果 | 空 `<your_long_term_memories>` 段，不影响流程 |
| 工作记忆不足 6 轮 | 有多少取多少 |
| 异步摘要失败 | 仅丢失该轮的情景记忆条目，不影响主流程 |
| profile 文件不存在 | 返回空 profile，不报错 |
| 并发写入 | session_manager 加锁或使用 SQLite WAL 模式（已启用） |

---

## Schema 迁移

现有 `knowledge.db` 中新增表，不修改现有表结构。

```python
# memory/models.py
def init_memory_tables():
    conn = get_connection()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'active',
            created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
            last_active_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
        );
        CREATE INDEX IF NOT EXISTS idx_sessions_user_status ON sessions(user_id, status);
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL REFERENCES sessions(id),
            user_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            category TEXT DEFAULT '',
            embedding BLOB,
            created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
        );
        CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, created_at);
        CREATE INDEX IF NOT EXISTS idx_messages_user ON messages(user_id, created_at);
    """)
    conn.commit()
    conn.close()
```

在 `database.py` 的 `init_db()` 末尾调用 `init_memory_tables()`。

---

## Files Changed

| File | Action | Responsibility |
|------|--------|---------------|
| `memory/session_manager.py` | Create | Session CRUD + 30min timeout |
| `memory/message_history.py` | Create | Message storage + sliding window |
| `memory/episodic.py` | Create | Episodic memory embed + search |
| `memory/context_builder.py` | Create | Assemble 3-tier context |
| `memory/models.py` | Create | DB table initialization |
| `storage/database.py` | Modify | Call `init_memory_tables()` in `init_db()` |
| `storage/profile.py` | Modify | `load_profile(user_id)` per-user path |
| `server/bot.py` | Modify | Use session_manager + context_builder |
| `agent/state.py` | Modify | Add `session_id`, `message_history`, `episodic_memories` |
| `agent/nodes/classify_and_answer.py` | Modify | Inject 3-tier context into prompt |
| `agent/nodes/update_profile.py` | Modify | Per-user profile write |
| `tests/test_session_manager.py` | Create | Session CRUD + timeout tests |
| `tests/test_message_history.py` | Create | Message add + get_recent tests |
| `tests/test_episodic.py` | Create | Summarize + search tests |
| `tests/test_context_builder.py` | Create | Context assembly tests |

---

## Testing

| 测试 | 内容 |
|------|------|
| `test_lookup_new_user` | 首次查找创建新 session |
| `test_lookup_reuses_active` | 30 分钟内返回同一 session |
| `test_lookup_archives_stale` | 超过 30 分钟归档旧 session 并新建 |
| `test_add_and_get_recent` | 写入后取回最近 N 条 |
| `test_empty_session` | 无消息时 get_recent 返回空列表 |
| `test_episodic_summarize` | 模拟 LLM 返回摘要，验证写入 embedding |
| `test_episodic_search` | 写入后语义检索返回结果 |
| `test_context_builder_empty` | 首次用户，三层均为空 |
| `test_context_builder_with_data` | 三层均有内容时正确拼接 |
| `test_profile_per_user` | 多用户 profile 隔离 |
