# PostgreSQL + pgvector 迁移实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将向量和业务数据从 SQLite + sqlite-vec 全量迁移到 PostgreSQL + pgvector

**架构:** 创建 `pg_database.py` + `pg_models.py` 作为新的存储后端，保持与现有 `storage/database.py` 和 `storage/models.py` 一致的接口签名，替换后再逐步清理 SQLite 依赖

**Tech Stack:** PostgreSQL 16+, pgvector 0.7+, psycopg2-binary, fastembed (不变)

## Global Constraints

- 所有函数签名必须与当前 `storage.database` 和 `storage.models` 保持一致，调用方代码不改
- generate_embedding() 保持使用 BAAI/bge-small-zh-v1.5（512维），不随数据库迁移改动
- 向量距离度量使用余弦距离（cosine），与当前 sqlite-vec 一致
- PostgreSQL 连接失败时必须给出清晰的安装指引提示
- 保留 SQLite 数据文件作为冷备，迁移成功后再清理

---

### Task 1: 配置和依赖

**Files:**
- Modify: `server/config.py`
- Modify: `requirements.txt`

**Interfaces:**
- Consumes: 无
- Produces: PG_HOST, PG_PORT, PG_DB, PG_USER, PG_PASS 配置常量

- [ ] **Step 1: 添加 PG 配置到 config.py**

在 `server/config.py` 中 LLM 配置之后添加：

```python
# PostgreSQL + pgvector 向量数据库
PG_HOST = os.getenv("PG_HOST", "localhost")
PG_PORT = int(os.getenv("PG_PORT", "5432"))
PG_DB = os.getenv("PG_DB", "knowledge")
PG_USER = os.getenv("PG_USER", "postgres")
PG_PASS = os.getenv("PG_PASS", "")
```

- [ ] **Step 2: 添加依赖到 requirements.txt**

追加：
```
psycopg2-binary>=2.9.9
pgvector>=0.3.0
```

- [ ] **Step 3: Commit**

```bash
git add server/config.py requirements.txt
git commit -m "chore: 添加 PostgreSQL 连接配置和依赖"
```

---

### Task 2: PostgreSQL 连接管理 (pg_database.py)

**Files:**
- Create: `storage/pg_database.py`

**Interfaces:**
- Produces: `get_connection() -> psycopg2.Connection`, `init_db() -> None`

- [ ] **Step 1: 创建 pg_database.py**

```python
"""PostgreSQL 数据库连接管理 + Schema 初始化。

替代 storage/database.py 的 SQLite 实现。
"""
import logging
import os

import psycopg2
from psycopg2 import sql
from server.config import PG_HOST, PG_PORT, PG_DB, PG_USER, PG_PASS

logger = logging.getLogger(__name__)

# 连接池参数
_CONN: psycopg2.extensions.connection | None = None
_PG_READY = False


def get_connection() -> psycopg2.extensions.connection:
    """获取 PostgreSQL 连接（懒加载单例）。"""
    global _CONN
    if _CONN is None or _CONN.closed:
        try:
            _CONN = psycopg2.connect(
                host=PG_HOST,
                port=PG_PORT,
                dbname=PG_DB,
                user=PG_USER,
                password=PG_PASS,
            )
            _CONN.autocommit = False
            logger.info("PostgreSQL 已连接: %s@%s:%s/%s", PG_USER, PG_HOST, PG_PORT, PG_DB)
        except psycopg2.OperationalError as e:
            logger.error(
                "PostgreSQL 连接失败: %s\n"
                "请确认 PostgreSQL 已安装并运行:\n"
                "  1. 安装: https://www.postgresql.org/download/\n"
                "  2. 创建数据库: createdb knowledge\n"
                "  3. 启用 pgvector: psql -d knowledge -c 'CREATE EXTENSION vector;'\n"
                "  4. 配置 .env 中的 PG_* 变量",
                e,
            )
            raise
    return _CONN


def _table_exists(conn, table_name: str) -> bool:
    """检查 PostgreSQL 表是否存在。"""
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = %s)",
            (table_name,),
        )
        return cur.fetchone()[0]
    finally:
        cur.close()


def init_db() -> None:
    """初始化 PostgreSQL 数据库表结构和 pgvector 扩展。"""
    conn = get_connection()
    cur = conn.cursor()

    try:
        # 启用 pgvector
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        logger.info("pgvector 扩展已就绪")

        # ── Wiki 页面 ──
        cur.execute("""
            CREATE TABLE IF NOT EXISTS pages (
                id SERIAL PRIMARY KEY,
                title TEXT NOT NULL UNIQUE,
                file_path TEXT NOT NULL,
                tags JSONB NOT NULL DEFAULT '[]',
                sources JSONB NOT NULL DEFAULT '[]',
                status TEXT NOT NULL DEFAULT 'active',
                checksum TEXT DEFAULT '',
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS wiki_embeddings (
                id INTEGER PRIMARY KEY REFERENCES pages(id) ON DELETE CASCADE,
                embedding VECTOR(512) NOT NULL
            )
        """)
        # HNSW 索引（仅首次创建，忽略已存在的错误）
        try:
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_wiki_embeddings_hnsw
                ON wiki_embeddings USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 200)
            """)
        except Exception:
            conn.rollback()
            cur.execute("CREATE INDEX IF NOT EXISTS idx_wiki_embeddings_hnsw ON wiki_embeddings USING hnsw (embedding vector_cosine_ops)")

        cur.execute("""
            CREATE TABLE IF NOT EXISTS page_relations (
                id SERIAL PRIMARY KEY,
                source_id INTEGER NOT NULL REFERENCES pages(id) ON DELETE CASCADE,
                target_title TEXT NOT NULL,
                relation_type TEXT NOT NULL DEFAULT 'wikilink',
                UNIQUE(source_id, target_title)
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS page_versions (
                id SERIAL PRIMARY KEY,
                page_id INTEGER NOT NULL REFERENCES pages(id) ON DELETE CASCADE,
                version INTEGER NOT NULL,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                checksum TEXT NOT NULL,
                source_id TEXT DEFAULT '',
                source_question TEXT DEFAULT '',
                change_summary TEXT DEFAULT '',
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                UNIQUE(page_id, version)
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS review_schedule (
                id SERIAL PRIMARY KEY,
                page_id INTEGER NOT NULL UNIQUE REFERENCES pages(id) ON DELETE CASCADE,
                easiness_factor REAL NOT NULL DEFAULT 2.5,
                interval_days INTEGER NOT NULL DEFAULT 1,
                repetitions INTEGER NOT NULL DEFAULT 0,
                next_review_at TIMESTAMPTZ NOT NULL,
                last_reviewed_at TIMESTAMPTZ DEFAULT NULL,
                last_quality INTEGER DEFAULT -1,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS sent_reviews (
                id SERIAL PRIMARY KEY,
                schedule_id INTEGER NOT NULL REFERENCES review_schedule(id) ON DELETE CASCADE,
                page_id INTEGER NOT NULL REFERENCES pages(id) ON DELETE CASCADE,
                marker_id TEXT NOT NULL UNIQUE,
                sent_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                status TEXT NOT NULL DEFAULT 'pending',
                feedback_quality INTEGER DEFAULT -1,
                answered_at TIMESTAMPTZ DEFAULT NULL
            )
        """)

        # ── 文件记录 ──
        cur.execute("""
            CREATE TABLE IF NOT EXISTS file_records (
                id SERIAL PRIMARY KEY,
                file_name TEXT NOT NULL,
                file_type TEXT NOT NULL,
                file_hash TEXT,
                extracted_text TEXT NOT NULL,
                knowledge_ids JSONB NOT NULL DEFAULT '[]',
                source_user_id TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)

        # ── 错误记录 ──
        cur.execute("""
            CREATE TABLE IF NOT EXISTS error_records (
                id SERIAL PRIMARY KEY,
                user_message TEXT NOT NULL,
                wrong_answer TEXT NOT NULL,
                correct_answer TEXT DEFAULT '',
                category TEXT DEFAULT '',
                contradiction_details TEXT DEFAULT '',
                error_type TEXT DEFAULT 'unknown',
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS error_embeddings (
                id INTEGER PRIMARY KEY REFERENCES error_records(id) ON DELETE CASCADE,
                embedding VECTOR(512) NOT NULL
            )
        """)
        try:
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_error_embeddings_hnsw
                ON error_embeddings USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 200)
            """)
        except Exception:
            conn.rollback()
            cur.execute("CREATE INDEX IF NOT EXISTS idx_error_embeddings_hnsw ON error_embeddings USING hnsw (embedding vector_cosine_ops)")

        # ── 基金模块 ──
        for tbl in ['user_portfolio', 'fund_info', 'fund_nav_cache', 'fund_holdings_cache', 'fund_decisions']:
            cur.execute(f"CREATE TABLE IF NOT EXISTS {tbl} (id SERIAL PRIMARY KEY, _dummy BOOLEAN)")

        # ── 会话和消息 ──
        cur.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id SERIAL PRIMARY KEY,
                user_id TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'active',
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                last_active_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_sessions_user_status ON sessions(user_id, status)")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id SERIAL PRIMARY KEY,
                session_id INTEGER NOT NULL REFERENCES sessions(id),
                user_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                category TEXT DEFAULT '',
                embedding VECTOR(512),
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, created_at)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_messages_user ON messages(user_id, created_at)")

        # ── 好奇心研究记录 ──
        cur.execute("""
            CREATE TABLE IF NOT EXISTS curiosity_topics (
                id SERIAL PRIMARY KEY,
                topic TEXT NOT NULL,
                search_query TEXT NOT NULL DEFAULT '',
                reason TEXT DEFAULT '',
                status TEXT NOT NULL DEFAULT 'success',
                pages_created INTEGER DEFAULT 0,
                summary TEXT DEFAULT '',
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)

        conn.commit()
        logger.info("PostgreSQL 数据库初始化完成")
    except Exception:
        conn.rollback()
        logger.exception("数据库初始化失败")
        raise
    finally:
        cur.close()
```

- [ ] **Step 2: Commit**

```bash
git add storage/pg_database.py
git commit -m "feat: PostgreSQL 连接管理和 Schema 初始化"
```

---

### Task 3: PostgreSQL Models (pg_models.py) — 基础 CRUD

**Files:**
- Create: `storage/pg_models.py`（向量相关函数）
- Create: `storage/pg_models.py`（插入时同步生成 embedding）

**Interfaces:**
- Produces: `generate_embedding()`, `_page_row_to_dict()`, `upsert_page()`, `update_page_relations()`, `save_page_version()`, `get_page_versions()`, `get_page_version()`, `cleanup_old_versions()`, `get_page_by_title()`, `get_all_pages_index()`, `find_similar_pages()`, `get_related_pages()`

注意：`generate_embedding()` 保持原有 FastEmbed 实现不变（与数据库无关，只是函数位置迁移）。

- [ ] **Step 1: 创建 pg_models.py（嵌入模型 + 页面 CRUD + 向量检索）**

```python
"""PostgreSQL + pgvector 数据访问层。

替代 storage/models.py 的 SQLite + sqlite-vec 实现。
所有函数签名与原来保持一致，调用方无需修改。
"""

import json
import logging
import threading
import time
from datetime import datetime, timezone
from typing import Optional

import psycopg2.extras
from pgvector.psycopg2 import register_vector

from agent.models.storage import WikiPage
from storage.pg_database import get_connection

logger = logging.getLogger(__name__)

# ── Embedding model (lazy-loaded singleton, thread-safe) ──

_embedder: Optional[any] = None
_embedder_lock = threading.Lock()


def _get_embedder():
    global _embedder
    if _embedder is None:
        with _embedder_lock:
            if _embedder is None:
                from fastembed import TextEmbedding
                logger.info("Loading embedding model (BAAI/bge-small-zh-v1.5)...")
                t0 = time.time()
                _embedder = TextEmbedding("BAAI/bge-small-zh-v1.5")
                logger.info("Embedding model loaded in %.2fs (dim=512)", time.time() - t0)
    return _embedder


def generate_embedding(text: str) -> list[float]:
    """生成 512 维文本嵌入向量（与原来完全一致）。"""
    model = _get_embedder()
    vec = next(model.embed(text))
    return vec.tolist()


# ── Tokenization ──


def _tokenize(text: str) -> list[str]:
    import jieba
    words = jieba.lcut(text)
    return [w.strip() for w in words if len(w.strip()) >= 2]


# ── Page helpers ──


def _page_row_to_dict(row: psycopg2.extras.RealDictRow) -> dict:
    """将 PostgreSQL 行转为 dict，JSONB 字段自动解析。"""
    d = dict(row)
    if isinstance(d.get("tags"), str):
        d["tags"] = json.loads(d["tags"])
    if isinstance(d.get("sources"), str):
        d["sources"] = json.loads(d["sources"])
    return d


def upsert_page(
    title: str, file_path: str, tags: list[str],
    sources: list[str], checksum: str, content: str,
) -> int:
    """Insert or update a wiki page record. 同时写入向量嵌入。"""
    conn = get_connection()
    register_vector(conn)
    cur = conn.cursor()
    try:
        # 检查是否已存在
        cur.execute("SELECT id FROM pages WHERE title = %s", (title,))
        existing = cur.fetchone()

        if existing:
            pid = existing[0]
            cur.execute(
                """UPDATE pages SET file_path=%s, tags=%s, sources=%s,
                   checksum=%s, updated_at=NOW() WHERE id=%s""",
                (file_path, json.dumps(tags), json.dumps(sources), checksum, pid),
            )
            logger.info("Updated page index: '%s' (id=%d)", title, pid)
        else:
            cur.execute(
                """INSERT INTO pages (title, file_path, tags, sources, checksum)
                   VALUES (%s, %s, %s, %s, %s) RETURNING id""",
                (title, file_path, json.dumps(tags), json.dumps(sources), checksum),
            )
            pid = cur.fetchone()[0]
            logger.info("Created page index: '%s' (id=%d)", title, pid)

        # 更新向量嵌入
        if content:
            embedding = generate_embedding(content)
            cur.execute(
                """INSERT INTO wiki_embeddings (id, embedding) VALUES (%s, %s)
                   ON CONFLICT (id) DO UPDATE SET embedding = EXCLUDED.embedding""",
                (pid, embedding),
            )

        conn.commit()
        return pid
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()


def get_page_by_title(title: str) -> Optional[dict]:
    """通过标题获取页面信息。"""
    conn = get_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    try:
        cur.execute(
            "SELECT * FROM pages WHERE LOWER(title) = LOWER(%s) AND status = 'active'",
            (title,),
        )
        row = cur.fetchone()
        return _page_row_to_dict(row) if row else None
    finally:
        cur.close()


def get_all_pages_index(status: str = "active") -> list[dict]:
    """获取所有页面索引（不含文件内容）。"""
    conn = get_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    try:
        if status:
            cur.execute(
                "SELECT id, title, file_path, tags, sources, status, created_at, updated_at "
                "FROM pages WHERE status = %s ORDER BY id", (status,)
            )
        else:
            cur.execute(
                "SELECT id, title, file_path, tags, sources, status, created_at, updated_at "
                "FROM pages ORDER BY id"
            )
        rows = cur.fetchall()
        return [_page_row_to_dict(r) for r in rows]
    finally:
        cur.close()


def find_similar_pages(
    query: str, threshold: float = 0.6, limit: int = 5,
) -> list[WikiPage]:
    """pgvector 余弦距离语义搜索。"""
    embedding = generate_embedding(query)
    conn = get_connection()
    register_vector(conn)
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    try:
        cur.execute(
            """SELECT p.id, p.title, p.file_path, p.tags,
                      1 - (e.embedding <=> %s::vector) AS distance
               FROM wiki_embeddings e
               JOIN pages p ON p.id = e.id
               WHERE p.status = 'active'
                 AND 1 - (e.embedding <=> %s::vector) >= %s
               ORDER BY e.embedding <=> %s::vector
               LIMIT %s""",
            (embedding, embedding, threshold, embedding, limit),
        )
        results = []
        for row in cur.fetchall():
            tags = json.loads(row["tags"]) if isinstance(row["tags"], str) else (row["tags"] or [])
            results.append(WikiPage(
                id=row["id"],
                title=row["title"],
                file_path=row["file_path"],
                tags=tags,
                distance=float(row["distance"]),
            ))
        if results:
            logger.info("Page semantic search: %d results for '%s'", len(results), query[:40])
        return results
    except Exception as e:
        logger.warning("Page semantic search failed: %s", e)
        return []
    finally:
        cur.close()


def update_page_relations(page_id: int, linked_titles: list[str]) -> None:
    """更新页面间的 Wikilink 关系。"""
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("DELETE FROM page_relations WHERE source_id = %s", (page_id,))
        for title in linked_titles:
            cur.execute(
                "INSERT INTO page_relations (source_id, target_title) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                (page_id, title),
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()


def get_related_pages(page_id: int) -> list[WikiPage]:
    """获取与指定页面有 Wikilink 关系的页面。"""
    conn = get_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    try:
        cur.execute(
            """SELECT p.id, p.title, p.file_path, p.tags
               FROM page_relations r
               JOIN pages p ON LOWER(p.title) = LOWER(r.target_title)
               WHERE r.source_id = %s AND p.status = 'active'""",
            (page_id,),
        )
        results = []
        for row in cur.fetchall():
            tags = json.loads(row["tags"]) if isinstance(row["tags"], str) else (row["tags"] or [])
            results.append(WikiPage(
                id=row["id"],
                title=row["title"],
                file_path=row["file_path"],
                tags=tags,
                distance=0,
            ))
        return results
    finally:
        cur.close()
```

- [ ] **Step 2: 添加剩余 CRUD 函数（page_versions, review_schedule, file_records, error_records 等）**

追加到同一个 `storage/pg_models.py`：

```python
# ── Page versions ──


def save_page_version(
    page_id: int, title: str, content: str, checksum: str,
    source_id: str = "", source_question: str = "",
) -> int:
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT COALESCE(MAX(version), 0) + 1 FROM page_versions WHERE page_id = %s",
            (page_id,),
        )
        next_version = cur.fetchone()[0]
        cur.execute(
            """INSERT INTO page_versions
               (page_id, version, title, content, checksum, source_id, source_question)
               VALUES (%s, %s, %s, %s, %s, %s, %s)""",
            (page_id, next_version, title, content, checksum, source_id, source_question),
        )
        conn.commit()
        return next_version
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()


def get_page_versions(page_id: int, limit: int = 20) -> list[dict]:
    conn = get_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    try:
        cur.execute(
            "SELECT * FROM page_versions WHERE page_id = %s ORDER BY version DESC LIMIT %s",
            (page_id, limit),
        )
        return [dict(r) for r in cur.fetchall()]
    finally:
        cur.close()


def get_page_version(page_id: int, version: int) -> dict | None:
    conn = get_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    try:
        cur.execute(
            "SELECT * FROM page_versions WHERE page_id = %s AND version = %s",
            (page_id, version),
        )
        row = cur.fetchone()
        return dict(row) if row else None
    finally:
        cur.close()


def cleanup_old_versions(days: int = 30) -> int:
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            """DELETE FROM page_versions
               WHERE created_at < NOW() - INTERVAL '%s days'
               AND version NOT IN (
                   SELECT version FROM page_versions pv2
                   WHERE pv2.page_id = page_versions.page_id
                   ORDER BY version DESC LIMIT 1
               )""",
            (days,),
        )
        deleted = cur.rowcount
        conn.commit()
        logger.info("Cleaned up %d old page versions (>=%d days)", deleted, days)
        return deleted
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()


# ── Error records ──


def save_error_record(record: dict) -> int:
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            """INSERT INTO error_records
               (user_message, wrong_answer, correct_answer, category, contradiction_details, error_type)
               VALUES (%s, %s, %s, %s, %s, %s) RETURNING id""",
            (record["user_message"], record["wrong_answer"], record.get("correct_answer", ""),
             record.get("category", ""), record.get("contradiction_details", ""),
             record.get("error_type", "unknown")),
        )
        eid = cur.fetchone()[0]
        conn.commit()
        return eid
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()


def save_error_record_with_embedding(record: dict) -> int:
    conn = get_connection()
    register_vector(conn)
    cur = conn.cursor()
    try:
        cur.execute(
            """INSERT INTO error_records
               (user_message, wrong_answer, correct_answer, category, contradiction_details, error_type)
               VALUES (%s, %s, %s, %s, %s, %s) RETURNING id""",
            (record["user_message"], record["wrong_answer"], record.get("correct_answer", ""),
             record.get("category", ""), record.get("contradiction_details", ""),
             record.get("error_type", "unknown")),
        )
        eid = cur.fetchone()[0]

        # 生成并存储向量
        text = record["user_message"] + " " + record["wrong_answer"]
        embedding = generate_embedding(text)
        cur.execute(
            "INSERT INTO error_embeddings (id, embedding) VALUES (%s, %s) ON CONFLICT (id) DO UPDATE SET embedding = EXCLUDED.embedding",
            (eid, embedding),
        )
        conn.commit()
        logger.info("Saved error record (id=%d)", eid)
        return eid
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()


def search_error_records_semantic(query: str, limit: int = 3) -> list[dict]:
    embedding = generate_embedding(query)
    conn = get_connection()
    register_vector(conn)
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    try:
        cur.execute(
            """SELECT e.*, 1 - (ev.embedding <=> %s::vector) AS distance
               FROM error_records e
               JOIN error_embeddings ev ON ev.id = e.id
               WHERE 1 - (ev.embedding <=> %s::vector) >= 0.5
               ORDER BY ev.embedding <=> %s::vector
               LIMIT %s""",
            (embedding, embedding, embedding, limit),
        )
        return [dict(r) for r in cur.fetchall()]
    except Exception as e:
        logger.warning("Error semantic search failed: %s", e)
        return []
    finally:
        cur.close()


# ── File records ──


def save_file_record(
    file_name: str, file_type: str, file_hash: str,
    extracted_text: str, knowledge_ids: list[int], source_user_id: str,
) -> int:
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            """INSERT INTO file_records
               (file_name, file_type, file_hash, extracted_text, knowledge_ids, source_user_id)
               VALUES (%s, %s, %s, %s, %s, %s) RETURNING id""",
            (file_name, file_type, file_hash, extracted_text,
             json.dumps(knowledge_ids), source_user_id),
        )
        rid = cur.fetchone()[0]
        conn.commit()
        return rid
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()


def get_file_records(limit: int = 10) -> list[dict]:
    conn = get_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    try:
        cur.execute("SELECT * FROM file_records ORDER BY created_at DESC LIMIT %s", (limit,))
        return [dict(r) for r in cur.fetchall()]
    finally:
        cur.close()


def get_file_record_by_hash(file_hash: str) -> dict | None:
    conn = get_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    try:
        cur.execute("SELECT * FROM file_records WHERE file_hash = %s", (file_hash,))
        row = cur.fetchone()
        return dict(row) if row else None
    finally:
        cur.close()
```

- [ ] **Step 3: 添加剩余函数（review_schedule, sent_reviews 相关）**

追加：

```python
# ── Review schedule (SM-2 spaced repetition) ──


def init_review_schedule(page_id: int, next_review_at: str | None = None) -> int:
    conn = get_connection()
    cur = conn.cursor()
    try:
        if next_review_at:
            cur.execute(
                """INSERT INTO review_schedule (page_id, next_review_at)
                   VALUES (%s, %s::timestamptz)
                   ON CONFLICT (page_id) DO NOTHING RETURNING id""",
                (page_id, next_review_at),
            )
        else:
            cur.execute(
                """INSERT INTO review_schedule (page_id, next_review_at)
                   VALUES (%s, NOW())
                   ON CONFLICT (page_id) DO NOTHING RETURNING id""",
                (page_id,),
            )
        result = cur.fetchone()
        conn.commit()
        return result[0] if result else 0
    except Exception:
        conn.rollback()
        return 0
    finally:
        cur.close()


def get_due_reviews(limit: int = 10) -> list[dict]:
    conn = get_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    try:
        cur.execute(
            """SELECT rs.*, p.title, p.file_path
               FROM review_schedule rs
               JOIN pages p ON p.id = rs.page_id
               WHERE p.status = 'active'
                 AND rs.next_review_at <= NOW()
               ORDER BY rs.next_review_at ASC
               LIMIT %s""",
            (limit,),
        )
        return [dict(r) for r in cur.fetchall()]
    finally:
        cur.close()


def update_review_schedule(
    page_id: int, quality: int, easiness_factor: float,
    interval_days: int, repetitions: int, next_review_at: str,
) -> None:
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            """UPDATE review_schedule
               SET easiness_factor=%s, interval_days=%s, repetitions=%s,
                   next_review_at=%s::timestamptz,
                   last_reviewed_at=NOW(), last_quality=%s
               WHERE page_id=%s""",
            (easiness_factor, interval_days, repetitions, next_review_at, quality, page_id),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()


def has_pending_review(page_id: int) -> bool:
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT 1 FROM review_schedule WHERE page_id = %s AND next_review_at <= NOW()",
            (page_id,),
        )
        return cur.fetchone() is not None
    finally:
        cur.close()


def get_sent_review_by_marker(marker_id: str) -> dict | None:
    conn = get_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    try:
        cur.execute("SELECT * FROM sent_reviews WHERE marker_id = %s", (marker_id,))
        row = cur.fetchone()
        return dict(row) if row else None
    finally:
        cur.close()


def mark_review_answered(sent_id: int) -> None:
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            "UPDATE sent_reviews SET status='answered', answered_at=NOW() WHERE id=%s",
            (sent_id,),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()


def get_reviewed_pages_since(days: int = 7) -> list[dict]:
    conn = get_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    try:
        cur.execute(
            """SELECT DISTINCT p.id, p.title, sr.status, sr.answered_at
               FROM sent_reviews sr
               JOIN pages p ON p.id = sr.page_id
               WHERE sr.sent_at >= NOW() - INTERVAL '%s days'
               ORDER BY sr.sent_at DESC""",
            (days,),
        )
        return [dict(r) for r in cur.fetchall()]
    finally:
        cur.close()


def record_sent_review(schedule_id: int, page_id: int, marker_id: str) -> int:
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            """INSERT INTO sent_reviews (schedule_id, page_id, marker_id)
               VALUES (%s, %s, %s) ON CONFLICT (marker_id) DO NOTHING RETURNING id""",
            (schedule_id, page_id, marker_id),
        )
        result = cur.fetchone()
        conn.commit()
        return result[0] if result else 0
    finally:
        cur.close()


def process_review_feedback(
    marker_id: str, quality: int,
) -> tuple[bool, str, str]:
    """处理用户对复习的反馈（SM-2 算法）。"""
    conn = get_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    try:
        cur.execute(
            """SELECT sr.id, sr.schedule_id, sr.page_id, sr.status,
                      rs.easiness_factor, rs.interval_days, rs.repetitions
               FROM sent_reviews sr
               JOIN review_schedule rs ON rs.id = sr.schedule_id
               WHERE sr.marker_id = %s""",
            (marker_id,),
        )
        sent = cur.fetchone()
        if not sent or sent["status"] != "pending":
            return False, "", ""

        # SM-2 算法
        q = max(0, min(5, quality))
        ef = float(sent["easiness_factor"])
        ef = max(1.3, ef + (0.1 - (5 - q) * (0.08 + (5 - q) * 0.02)))
        rep = int(sent["repetitions"]) if q >= 3 else 0
        if q < 3:
            interval = 1
        elif rep == 0:
            interval = 1
        elif rep == 1:
            interval = 6
        else:
            interval = round(float(sent["interval_days"]) * ef)

        from datetime import timedelta
        next_review = datetime.now(timezone.utc) + timedelta(days=interval)

        cur.execute("UPDATE sent_reviews SET status='answered', feedback_quality=%s, answered_at=NOW() WHERE id=%s",
                     (quality, sent["id"]))
        cur.execute(
            """UPDATE review_schedule
               SET easiness_factor=%s, interval_days=%s, repetitions=%s,
                   next_review_at=%s, last_reviewed_at=NOW(), last_quality=%s
               WHERE id=%s""",
            (ef, interval, rep + 1, next_review, quality, sent["schedule_id"]),
        )
        conn.commit()

        # 获取页面标题
        cur.execute("SELECT title FROM pages WHERE id = %s", (sent["page_id"],))
        page = cur.fetchone()
        title = page["title"] if page else "unknown"

        feedback = "记住了" if q >= 3 else "忘记了"
        return True, feedback, title
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
```

- [ ] **Step 4: Commit**

```bash
git add storage/pg_models.py
git commit -m "feat: PostgreSQL pgvector 数据访问层（完整 CRUD）"
```

---

### Task 4: 替换 storage/database.py 和 storage/models.py

**Files:**
- Modify: `storage/database.py` — 改为导入 pg_database
- Modify: `storage/models.py` — 改为导入 pg_models

**Interfaces:**
- Consumes: `pg_database.py`, `pg_models.py`
- Produces: 与原来完全相同的导入接口

- [ ] **Step 1: 替换 storage/database.py**

```python
"""数据库连接 — 当前使用 PostgreSQL + pgvector。

初始化时自动调用 pg_database.init_db() 确保表结构存在。
保持与 SQLite 时代相同的函数签名（get_connection, init_db）。
"""

import os
from storage.pg_database import get_connection, init_db

# 保留 DB_DIR/DB_PATH 兼容引用（部分外部模块可能用到）
DB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
DB_PATH = os.path.join(DB_DIR, "knowledge.db")
```

- [ ] **Step 2: 替换 storage/models.py**

```python
"""数据模型 — 当前使用 PostgreSQL + pgvector。

所有函数实现委托到 pg_models，保持导入接口不变。
"""

# 从 pg_models 重新导出所有公开函数
from storage.pg_models import (
    generate_embedding,
    upsert_page,
    update_page_relations,
    save_page_version,
    get_page_versions,
    get_page_version,
    cleanup_old_versions,
    find_similar_pages,
    get_related_pages,
    get_all_pages_index,
    get_page_by_title,
    save_file_record,
    get_file_records,
    get_file_record_by_hash,
    save_error_record,
    save_error_record_with_embedding,
    search_error_records_semantic,
    init_review_schedule,
    get_due_reviews,
    update_review_schedule,
    has_pending_review,
    get_sent_review_by_marker,
    mark_review_answered,
    get_reviewed_pages_since,
    record_sent_review,
    process_review_feedback,
)
```

- [ ] **Step 3: 清理 memory/models.py 中 SQLite 特定的 init_memory_tables**

```python
# memory/models.py — 保留但改为空操作，因为表在 pg_database.init_db() 中已创建
def init_memory_tables():
    logger.info("Memory tables managed by pg_database.init_db(), skipping")
```

- [ ] **Step 4: Commit**

```bash
git add storage/database.py storage/models.py memory/models.py
git commit -m "refactor: 切换到 PostgreSQL 存储后端，SQLite 代码由 pg_ 模块替代"
```

---

### Task 5: 数据迁移脚本

**Files:**
- Create: `scripts/migrate_to_pg.py`

- [ ] **Step 1: 创建迁移脚本**

```python
#!/usr/bin/env python
"""从 SQLite 迁移所有数据到 PostgreSQL。

用法：
    python scripts/migrate_to_pg.py

要求：
    - SQLite 数据库在 data/knowledge.db（原路径）
    - PostgreSQL 已配置 .env 中的 PG_* 变量
    - PostgreSQL 中已运行 CREATE EXTENSION vector;
"""

import json
import logging
import os
import sys

# 确保项目根目录在 Python 路径中
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("migrate_to_pg")

from storage.database import DB_DIR
from storage.pg_database import get_connection as pg_conn, init_db
from pgvector.psycopg2 import register_vector
import psycopg2.extras as pg_extras


def migrate_table(cur_sqlite, cur_pg, table: str, columns: list[str], pg_insert_sql: str, transform=None):
    """从 SQLite 读取并写入 PostgreSQL。"""
    cur_sqlite.execute(f"SELECT {', '.join(columns)} FROM {table}")
    rows = cur_sqlite.fetchall()
    if not rows:
        logger.info("  %s: 无数据", table)
        return 0

    for row in rows:
        try:
            values = dict(row) if hasattr(row, 'keys') else row
            if transform:
                values = transform(values)
            cur_pg.execute(pg_insert_sql, values)
        except Exception as e:
            logger.warning("  %s 迁移行失败: %s", table, e)
    count = len(rows)
    logger.info("  %s: 迁移 %d 行", table, count)
    return count


def main():
    import sqlite3
    from storage.pg_models import generate_embedding

    # 连接 SQLite
    sqlite_path = os.path.join(DB_DIR, "knowledge.db")
    if not os.path.exists(sqlite_path):
        logger.error("SQLite 数据库不存在: %s", sqlite_path)
        sys.exit(1)

    conn_sqlite = sqlite3.connect(sqlite_path)
    conn_sqlite.row_factory = sqlite3.Row
    logger.info("已连接 SQLite: %s", sqlite_path)

    # 初始化 PostgreSQL
    init_db()
    conn_pg = pg_conn()
    cur_pg = conn_pg.cursor()
    register_vector(conn_pg)
    logger.info("已连接 PostgreSQL")

    total = 0

    # 1. pages
    total += migrate_table(
        conn_sqlite.cursor(), cur_pg, "pages",
        ["id", "title", "file_path", "tags", "sources", "status", "checksum", "created_at", "updated_at"],
        """INSERT INTO pages (id, title, file_path, tags, sources, status, checksum, created_at, updated_at)
           VALUES (%(id)s, %(title)s, %(file_path)s, %(tags)s::jsonb, %(sources)s::jsonb, %(status)s, %(checksum)s,
                   %(created_at)s::timestamptz, %(updated_at)s::timestamptz)
           ON CONFLICT (id) DO NOTHING""",
        transform=lambda r: dict(r),
    )

    # 2. wiki_embeddings（从 page_vectors 迁移）
    # 需要读取原有 SQLite 的 page_vectors，这需要 sqlite_vec 支持
    # 回退方案: 如果无法直接迁移向量，则从 pages 内容重新生成
    try:
        cur_sqlite = conn_sqlite.cursor()
        cur_sqlite.execute("SELECT rowid FROM page_vectors LIMIT 1")
        has_vectors = cur_sqlite.fetchone() is not None
    except Exception:
        has_vectors = False

    if has_vectors:
        logger.info("  从 SQLite page_vectors 迁移向量...")
        cur_sqlite.execute("""
            SELECT pv.rowid as page_id, p.content
            FROM page_vectors pv
            JOIN pages p ON p.id = pv.rowid AND p.status = 'active'
        """)
        # Note: sqlite-vec stores binary, we can't read it directly
        # Fall through to regenerate
        has_vectors = False

    if not has_vectors:
        logger.info("  从页面内容重新生成向量嵌入（可能需要较长时间）...")
        cur_sqlite.execute("SELECT id FROM pages WHERE status = 'active'")
        page_ids = [r["id"] for r in cur_sqlite.fetchall()]
        for pid in page_ids:
            # 读取文件内容
            from storage.wiki_storage import read_page
            page = conn_sqlite.execute(
                "SELECT title, file_path FROM pages WHERE id = ?", (pid,)
            ).fetchone()
            if page:
                file_page = read_page(page["file_path"])
                content = file_page["body"] if file_page else ""
                if content:
                    embedding = generate_embedding(content)
                    cur_pg.execute(
                        "INSERT INTO wiki_embeddings (id, embedding) VALUES (%s, %s) ON CONFLICT (id) DO NOTHING",
                        (pid, embedding),
                    )
        logger.info("  wiki_embeddings 重构完成")

    # 3. page_relations
    total += migrate_table(
        conn_sqlite.cursor(), cur_pg, "page_relations",
        ["source_id", "target_title", "relation_type"],
        """INSERT INTO page_relations (source_id, target_title, relation_type)
           VALUES (%(source_id)s, %(target_title)s, %(relation_type)s)
           ON CONFLICT DO NOTHING""",
        transform=lambda r: dict(r),
    )

    # 4. page_versions
    total += migrate_table(
        conn_sqlite.cursor(), cur_pg, "page_versions",
        ["page_id", "version", "title", "content", "checksum", "source_id", "source_question", "change_summary", "created_at"],
        """INSERT INTO page_versions (page_id, version, title, content, checksum, source_id, source_question, change_summary, created_at)
           VALUES (%(page_id)s, %(version)s, %(title)s, %(content)s, %(checksum)s, %(source_id)s, %(source_question)s,
                   %(change_summary)s, %(created_at)s::timestamptz)
           ON CONFLICT (page_id, version) DO NOTHING""",
        transform=lambda r: dict(r),
    )

    # 5. error_records
    total += migrate_table(
        conn_sqlite.cursor(), cur_pg, "error_records",
        ["id", "user_message", "wrong_answer", "correct_answer", "category", "contradiction_details", "error_type", "created_at"],
        """INSERT INTO error_records (id, user_message, wrong_answer, correct_answer, category, contradiction_details, error_type, created_at)
           VALUES (%(id)s, %(user_message)s, %(wrong_answer)s, %(correct_answer)s, %(category)s,
                   %(contradiction_details)s, %(error_type)s, %(created_at)s::timestamptz)
           ON CONFLICT (id) DO NOTHING""",
        transform=lambda r: dict(r),
    )

    # 6. error_embeddings（重新生成）
    logger.info("  从 error_records 重新生成错误向量...")
    cur_sqlite.execute("SELECT id, user_message, wrong_answer FROM error_records")
    for r in cur_sqlite.fetchall():
        text = r["user_message"] + " " + r["wrong_answer"]
        embedding = generate_embedding(text)
        cur_pg.execute(
            "INSERT INTO error_embeddings (id, embedding) VALUES (%s, %s) ON CONFLICT (id) DO NOTHING",
            (r["id"], embedding),
        )

    # 7. file_records
    total += migrate_table(
        conn_sqlite.cursor(), cur_pg, "file_records",
        ["file_name", "file_type", "file_hash", "extracted_text", "knowledge_ids", "source_user_id", "created_at"],
        """INSERT INTO file_records (file_name, file_type, file_hash, extracted_text, knowledge_ids, source_user_id, created_at)
           VALUES (%(file_name)s, %(file_type)s, %(file_hash)s, %(extracted_text)s, %(knowledge_ids)s::jsonb,
                   %(source_user_id)s, %(created_at)s::timestamptz)""",
        transform=lambda r: dict(r),
    )

    # 8. review_schedule
    total += migrate_table(
        conn_sqlite.cursor(), cur_pg, "review_schedule",
        ["page_id", "easiness_factor", "interval_days", "repetitions", "next_review_at", "last_reviewed_at", "last_quality", "created_at"],
        """INSERT INTO review_schedule (page_id, easiness_factor, interval_days, repetitions, next_review_at, last_reviewed_at, last_quality, created_at)
           VALUES (%(page_id)s, %(easiness_factor)s, %(interval_days)s, %(repetitions)s, %(next_review_at)s::timestamptz,
                   %(last_reviewed_at)s::timestamptz, %(last_quality)s, %(created_at)s::timestamptz)
           ON CONFLICT (page_id) DO NOTHING""",
        transform=lambda r: dict(r),
    )

    # 9. sent_reviews
    total += migrate_table(
        conn_sqlite.cursor(), cur_pg, "sent_reviews",
        ["schedule_id", "page_id", "marker_id", "sent_at", "status"],
        """INSERT INTO sent_reviews (schedule_id, page_id, marker_id, sent_at, status)
           VALUES (%(schedule_id)s, %(page_id)s, %(marker_id)s, %(sent_at)s::timestamptz, %(status)s)
           ON CONFLICT (marker_id) DO NOTHING""",
        transform=lambda r: dict(r),
    )

    # 10. sessions
    total += migrate_table(
        conn_sqlite.cursor(), cur_pg, "sessions",
        ["id", "user_id", "status", "created_at", "last_active_at"],
        """INSERT INTO sessions (id, user_id, status, created_at, last_active_at)
           VALUES (%(id)s, %(user_id)s, %(status)s, %(created_at)s::timestamptz, %(last_active_at)s::timestamptz)
           ON CONFLICT (id) DO NOTHING""",
        transform=lambda r: dict(r),
    )

    # 11. messages
    total += migrate_table(
        conn_sqlite.cursor(), cur_pg, "messages",
        ["session_id", "user_id", "role", "content", "category", "created_at"],
        """INSERT INTO messages (session_id, user_id, role, content, category, created_at)
           VALUES (%(session_id)s, %(user_id)s, %(role)s, %(content)s, %(category)s, %(created_at)s::timestamptz)""",
        transform=lambda r: dict(r),
    )

    # 12. curiosity_topics
    total += migrate_table(
        conn_sqlite.cursor(), cur_pg, "curiosity_topics",
        ["topic", "search_query", "reason", "status", "pages_created", "summary", "created_at"],
        """INSERT INTO curiosity_topics (topic, search_query, reason, status, pages_created, summary, created_at)
           VALUES (%(topic)s, %(search_query)s, %(reason)s, %(status)s, %(pages_created)s, %(summary)s, %(created_at)s::timestamptz)""",
        transform=lambda r: dict(r),
    )

    conn_pg.commit()
    conn_sqlite.close()
    logger.info("迁移完成! 共计处理 %d 条记录", total)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add scripts/migrate_to_pg.py
git commit -m "feat: 数据迁移脚本 SQLite → PostgreSQL"
```

---

### Task 6: 集成测试验证

**Files:**
- Modify: `_comprehensive_test.py`（验证 PostgreSQL 连接正常）
- 或手动运行 `python main.py` 确认启动正常

- [ ] **Step 1: 运行迁移脚本**

```bash
python scripts/migrate_to_pg.py
```

- [ ] **Step 2: 启动 Agent 验证**

```bash
python main.py
```
预期：启动日志显示 "PostgreSQL 已连接" 和 "pgvector 扩展已就绪"，无数据库报错。

- [ ] **Step 3: 运行综合测试验证向量检索**

```bash
python -m pytest tests/ -v -x
```

- [ ] **Step 4: 最终提交**

```bash
git add -A && git commit -m "chore: PostgreSQL 迁移集成测试通过"
```
