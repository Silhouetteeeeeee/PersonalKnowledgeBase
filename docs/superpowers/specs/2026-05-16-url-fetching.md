# URL 网页内容抓取与知识入库

**Goal:** 用户发送含 URL 的消息时，bot 自动抓取网页内容、LLM 总结、并提取为 wiki 知识页面。

**Architecture:** 不改变现有消息路由，URL 作为 graph state 字段 `url_contents` 流过 parse → rewrite_query → classify_and_answer → store 节点。parse 节点负责并发提取和清洗，classify_and_answer 节点将 URL 内容注入 prompt。

**Tech Stack:** trafilatura（静态提取）、tavily（动态后备）、concurrent.futures（并发）、DeepSeek（LLM 总结）

---

## 场景

用户可发送纯文本或含 URL 的混合消息，例如：

> 总结这个文章 https://xxx.com/article 和 https://yyy.com/post，然后告诉我它们的共同点

## 数据流

### 1. Graph State 新增字段

`agent/state.py` 新增：

```python
url_contents: list[dict]  # 初始值 []
```

每个元素结构：

```python
{
    "url": str,        # 原始 URL
    "title": str | None,  # 页面标题（可能取不到）
    "content": str,    # 清洗后的正文
}
```

### 2. parse 节点 —— URL 提取 + 并发抓取

`agent/nodes/parse.py`：

- 从 `user_message` 中正则提取所有 `https?://` 开头的 URL
- 如果没有 URL：`url_contents` 保持 `[]`，行为完全不变
- 如果有 URL：
  - 去重（相同 URL 只抓一次）
  - `ThreadPoolExecutor(max_workers=5)` 并发抓取
  - 单条 URL 失败不影响其他 URL
  - 单条抓取流程：

```
fetch_url_text(url):
  ① requests.get(url) + trafilatura.extract(html)
  ② 如果内容 < 200 chars → tavily_client.extract(url)
  ③ clean_content() 清洗
  ④ 尝试提取 title（<title> 或 h1）
  ⑤ 返回 {"url", "title", "content"}
```

### 3. rewrite_query 节点 —— 检索策略

- 如果 `url_contents` 为空：原逻辑不变
- 如果 `url_contents` 非空：
  - **有 title 的 URL → 用 title 作为 query 候选**
  - **无 title 的 URL → 取正文第一句话（截到第一个句号 `.。!?`，或前 100 字）**
  - 如果用户有剩余问题文字（去掉 URL 后剩余 ≥ 10 字）→ **优先用用户问题**作为 query
  - 如果用户消息中**只包含 URL**（去掉 URL 后为空或只剩空白）→ 用 title 或首句作为 query，归类为"总结意图"
  - 有多个 URL 时，取所有 query 候选拼接，最多 300 字

### 4. classify_and_answer 节点 —— 注入 prompt

- 如果 `url_contents` 非空，在 prompt 中加入网页内容块：

```markdown
## 用户提供的网页内容

### URL: https://xxx.com/article
> 标题：xxx
> 前 500 字摘要：
> xxx

全文：
xxx（完整正文）
```

- LLM 看到 URL 内容 + 知识库检索结果 + 用户问题，综合作答
- **特殊处理：纯 URL 消息（无额外文字）**
  - prompt 末尾追加指令：
    > "用户只发送了网页链接，没有附加问题。请直接总结这篇文章的核心内容，用中文输出。"
  - LLM 直接输出中文文章总结

### 5. store 节点 —— 不修改

- 从最终 `answer` 正常提取 wiki 页面
- `source_label` 标注来源 URL

## 新增文件

### `server/url_processor.py`

三个核心函数：

```python
def fetch_url_text(url: str) -> dict:
    """下载单个 URL → 提取正文 → 清洗 → 返回结构化结果。

    使用两层策略：trafilatura 静态提取，不足 200 字则 tavily 后备。
    单 URL 超时 30 秒。
    """

def fetch_urls_concurrent(urls: list[str]) -> list[dict]:
    """多线程并发抓取，max_workers=5。单条失败不影响整体。"""

def clean_content(raw: str) -> str:
    """清洗网页内容：
    - 删除 ![xxx](url) 图片引用
    - 删除只有 # ## ### 等无文字的标题行
    - 删除连续分隔线 --- ___
    - 压缩连续空行为单行
    - 去除首尾空白
    """
```

### 错误处理

- 单条 URL 抓取失败：跳过该 URL，不影响其他 URL，返回 `{"url": url, "title": None, "content": "[抓取失败]"}`
- 全部 URL 都失败：`url_contents` 为空列表，走普通文本流程
- 网络超时：统一 30 秒超时

## 修改文件

| 文件 | 改动 |
|------|------|
| `agent/state.py` | 加 `url_contents` 字段 |
| `agent/nodes/parse.py` | URL 检测 + 并发抓取 + 清洗 |
| `agent/nodes/rewrite_query.py` | URL 场景检索策略 |
| `agent/nodes/classify_and_answer.py` | prompt 注入 url_contents |
| `server/config.py` | 读 `TAVILY_API_KEY` 环境变量 |
| `.env` | 加 `TAVILY_API_KEY` |
| `requirements.txt` | 加 `trafilatura`、`tavily-python` |

## 不修改的文件

- `bot.py` — 消息路由不变
- `agent/graph.py` — parse 已在首位，无需改连接
- `agent/nodes/store.py` — 知识提取逻辑不变
- `storage/` — 不新增表，不涉及去重逻辑（后续可按需加）

## 测试要点

1. `test_fetch_url_text()` — mock requests + tavily，验证静态/动态降级路径
2. `test_fetch_urls_concurrent()` — 多 URL 并发，单条失败不影响
3. `test_clean_content()` — 图片移除、空标题、分隔线、空行压缩
4. `test_parse_with_urls()` — 含 URL 的 parse 输出正确 url_contents
5. `test_parse_without_urls()` — 无 URL 时行为不变
6. `test_rewrite_query_with_urls()` — title 优先、无 title 取首句、用户问题优先
7. `test_classify_and_answer_url_injection()` — prompt 中包含 URL 内容
