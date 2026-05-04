# Reranker 优化检索质量 设计文档

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 通过在向量检索后引入 cross-encoder reranker，解决同类知识点中角度/粒度不匹配带来的噪声问题。

**Architecture:** 在 `retrieve` 节点中，将向量搜索的召回量从 top 5 扩大到 top 20，然后用 `BAAI/bge-reranker-v2-m3` cross-encoder 对 (query, doc) 逐对打分，重排后取 top 5。模型加载采用与现有 embedder 相同的惰加载单例模式。

**Tech Stack:** `FlagEmbedding` + `BAAI/bge-reranker-v2-m3` + `sqlite-vec`

---

## 流程

```
用户消息 → vector search (top 20) → cross-encoder rerank → top 5
             ↓ empty                    ↓ model fail
        keyword fallback           vector 原始顺序（降级）
```

- vector search 从现有 top 5 扩大到 top 20，增加召回候选
- rerank 后仍返回 top 5，对外接口不变（`stored_knowledge` 仍是 5 条）
- 候选 ≤3 时跳过 rerank（样本太少意义不大）
- reranker 加载或评分失败时降级到原始 vector 顺序

## 修改文件

| 文件 | 变更 |
|------|------|
| `storage/models.py` | 新增 `_get_reranker()` 单例 + `rerank_knowledge()` 函数 |
| `agent/nodes/retrieve.py` | 向量搜索后调用 reranker，调整 limit |
| `tests/test_nodes.py` | 新增 rerank 相关测试 |
| `requirements.txt` | 新增 `FlagEmbedding` 依赖 |

## 实现细节

### Reranker 加载（storage/models.py）

```python
_reranker: Optional[any] = None
_reranker_lock = threading.Lock()

def _get_reranker():
    global _reranker
    if _reranker is None:
        with _reranker_lock:
            if _reranker is None:
                from FlagEmbedding import FlagReranker
                _reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=False)
    return _reranker

def rerank_knowledge(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    pairs = [(query, doc["knowledge_text"]) for doc in candidates]
    model = _get_reranker()
    scores = model.compute_score(pairs)
    scored = list(zip(candidates, scores))
    scored.sort(key=lambda x: x[1], reverse=True)
    for doc, score in scored:
        doc["relevance_score"] = score
    return [doc for doc, _ in scored[:top_k]]
```

### 检索节点（agent/nodes/retrieve.py）

```python
def retrieve(state: dict) -> dict:
    query = state["user_message"]
    try:
        candidates = search_knowledge_points_semantic(query, limit=20)
    except Exception as e:
        logger.warning("Semantic search failed: %s", e)
        candidates = []

    if len(candidates) > 3:
        try:
            results = rerank_knowledge(query, candidates, top_k=5)
        except Exception as e:
            logger.warning("Reranker failed: %s", e)
            results = candidates[:5]
    elif candidates:
        results = candidates[:5]
    else:
        results = search_knowledge_points(query, limit=5)

    return {"stored_knowledge": results}
```

## 错误处理

| 场景 | 行为 |
|------|------|
| Reranker 模型加载失败 | 跳过 rerank，使用 vector 原始顺序 |
| compute_score 返回空/NaN | 跳过对应候选项 |
| Vector search 返回 0 | 走 keyword fallback（不变） |
| 候选 ≤ 3 | 跳过 rerank（样本太少） |

## 测试

- 正常 rerank：mock compute_score 返回固定分数，验证排序
- 降级：reranker 抛异常，验证回退到原始顺序
- 少量结果：≤3 候选项，验证跳过 rerank
- 关键路径：集成测试验证 retrieve 返回格式不变

## 不变的部分

- 图的拓扑结构：`parse → retrieve → classify_and_answer → ...`
- AgentState 字段
- 其他节点（classify_and_answer / fact_check / reflect 等）
- 其他存储函数
