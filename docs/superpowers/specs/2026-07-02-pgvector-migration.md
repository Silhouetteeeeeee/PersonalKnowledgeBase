# PostgreSQL + pgvector 向量数据库迁移方案

Date: 2026-07-02
Status: Design

## Motivation

当前使用 sqlite-vec 作为向量搜索引擎，存在以下限制：

- **sqlite-vec 非主流**：社区较小，文档和生态不如 PostgreSQL/pgvector 成熟
- **性能瓶颈**：SQLite 的并发写入锁在请求量增大时成为瓶颈
- **缺乏混合过滤**：无法同时做向量相似度搜索和标量条件过滤（如按标签筛选）
- **单一数据库耦合**：向量和业务数据在同一個文件中，扩展不灵活

迁移到 PostgreSQL + pgvector 后：

- **主流技术栈**：pgvector 是 PostgreSQL 官方生态的一部分，社区活跃，生产验证充分
- **HNSW 索引**：支持 Hierarchical Navigable Small World 索引，大幅提升检索性能
- **混合搜索**：支持 `WHERE` 条件 + `ORDER BY vector <-> query` 的组合查询
- **统一数据平台**：所有业务数据（用户、会话、页面索引、向量）集中在 PostgreSQL

## 架构变化

### 当前架构

```
knowledge.db (SQLite)
├── pages (页面元数据)
├── page_vectors (vec0 虚拟表, 512维)
├── page_relations
├── page_versions
├── error_records
├── error_vectors (vec0 虚拟表, 512维)
├── review_schedule
├── file_records
├── sessions / messages
└── curiosity_topics
```

### 目标架构

```
PostgreSQL 数据库
├── 所有表从 SQLite 迁移到 PostgreSQL
├── wiki_embeddings  (pgvector, 替代 page_vectors)
├── error_embeddings (pgvector, 替代 error_vectors)
├── HNSW 索引加速向量检索
└── 业务查询支持 WHERE + ORDER BY 混合过滤
```

## 环境变量

新增 `.env` 配置：

```env
PG_HOST=localhost
PG_PORT=5432
PG_DB=knowledge
PG_USER=postgres
PG_PASS=your_password
```

如果 PostgreSQL 连接失败，系统应给出清晰的错误提示并退出。

## 数据库 Schema

### 1. pages 表（已有，从 SQLite 迁移）

```sql
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
);
```

### 2. wiki_embeddings（pgvector，替代 page_vectors）

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS wiki_embeddings (
    id INTEGER PRIMARY KEY REFERENCES pages(id),
    embedding VECTOR(512) NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_wiki_embeddings_hnsw
    ON wiki_embeddings USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 200);
```

### 3. 其他表

所有现有 SQLite 表（page_relations, page_versions, error_records, review_schedule, file_records, sessions, messages, curiosity_topics）迁移到对应的 PostgreSQL 表，字段类型适配 PostgreSQL 规范（INTEGER → SERIAL/INT, TEXT → TEXT, JSON → JSONB, datetime → TIMESTAMPTZ）。

### 4. error_embeddings（pgvector，替代 error_vectors）

```sql
CREATE TABLE IF NOT EXISTS error_embeddings (
    id INTEGER PRIMARY KEY REFERENCES error_records(id),
    embedding VECTOR(512) NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_error_embeddings_hnsw
    ON error_embeddings USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 200);
```

## 核心函数变更

### 替换 `find_similar_pages()`

当前：
```python
# SQLite: sqlite-vec 虚拟表查询
conn.execute(
    "SELECT rowid, distance FROM page_vectors WHERE embedding MATCH ? "
    "AND k = ? AND distance < ?",
    (serialize_float32(embedding), limit, 1 - threshold),
)
```

改为：
```python
# PostgreSQL: pgvector 余弦距离查询
conn.execute(
    "SELECT p.id, p.title, p.file_path, p.tags, 1 - (e.embedding <=> %s::vector) AS distance "
    "FROM wiki_embeddings e "
    "JOIN pages p ON p.id = e.id "
    "WHERE p.status = 'active' "
    "  AND 1 - (e.embedding <=> %s::vector) >= %s "
    "ORDER BY e.embedding <=> %s::vector "
    "LIMIT %s",
    (embedding, embedding, threshold, embedding, limit),
)
```

### 替换 `save_error_record_with_embedding()`

当前：INSERT INTO error_records + INSERT INTO error_vectors
改为：INSERT INTO PostgreSQL error_records + INSERT INTO error_embeddings

## 数据流

### 写入流程

```
upsert_page(title, file_path, tags, sources, checksum, content)
  ├→ INSERT INTO pages (...) ON CONFLICT (title) DO UPDATE
  ├→ generate_embedding(content)
  └→ INSERT INTO wiki_embeddings (id, embedding) VALUES (...) ON CONFLICT (id) DO UPDATE
```

### 查询流程

```
find_similar_pages(query, threshold=0.6, limit=5)
  ├→ generate_embedding(query)
  └→ SELECT ... FROM wiki_embeddings e JOIN pages p
       WHERE 1 - (e.embedding <=> %s) >= %s
       ORDER BY e.embedding <=> %s
       LIMIT %s
```

## pgvector 的优势

| 特性 | sqlite-vec | pgvector |
|------|-----------|----------|
| 索引类型 | 暴力搜索（Brute Force） | HNSW + IVFFlat |
| 混合过滤 | ❌ 不支持 | ✅ `WHERE tags ? 'python'` + 向量搜索 |
| 并发写入 | 🔴 SQLite 写锁 | 🟢 MVCC 多版本并发 |
| 社区生态 | ⚪ 小众 | 🟢 PostgreSQL 生态 |
| 生产部署 | ⚪ 本地文件 | 🟢 可主从、可备份 |

## 迁移步骤

### Step 1: 依赖安装

```bash
# PostgreSQL 安装（Windows: https://www.postgresql.org/download/windows/）
pip install psycopg2-binary pgvector
```

### Step 2: 创建 PostgreSQL 数据库

```bash
createdb knowledge
psql -d knowledge -c "CREATE EXTENSION vector;"
```

### Step 3: 数据迁移脚本

编写 `scripts/migrate_to_pg.py`，将现有 SQLite 数据全量迁移到 PostgreSQL：

1. 连接 SQLite 读取所有表数据
2. 在 PostgreSQL 创建对应表结构
3. 逐表迁移数据（pages → pages, page_vectors → wiki_embeddings 等）
4. 重建 pgvector HNSW 索引

### Step 4: 修改 storage 层

1. 新增 `storage/pg_database.py` — PostgreSQL 连接管理
2. 新增 `storage/pg_models.py` — 使用 psycopg2 的增删改查（替代 database.py + models.py）
3. 修改 `storage/models.py` — 导入 pg_models 中的函数，保持对外接口一致

### Step 5: 测试验证

1. 运行向量检索，确认距离计算结果一致
2. 运行全链路 Graph 测试，确认业务正常
3. 性能对比：HNSW 索引 vs 暴力搜索

## 风险与回退

- **风险**：PostgreSQL 未安装时 Agent 无法启动 → 启动时检查连接，失败则给出清晰安装指引
- **回退**：保留迁移前的 SQLite 作为冷备，切换后发现重大问题可切回
- **兼容性**：生成嵌入向量的逻辑不变（仍用 BAAI/bge-small-zh-v1.5），迁移后查询结果应一致
