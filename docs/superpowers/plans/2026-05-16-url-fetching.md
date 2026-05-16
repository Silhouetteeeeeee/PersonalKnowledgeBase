# URL 网页内容抓取与知识入库 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用户发送含 URL 的消息时，bot 自动抓取网页内容、LLM 总结、并提取为 wiki 知识页面。

**Architecture:** URL 作为 graph state 字段 `url_contents` 流过 parse → rewrite_query → classify_and_answer → store。parse 节点用 ThreadPoolExecutor 并发抓取和清洗，classify_and_answer 将 URL 内容注入 prompt。

**Tech Stack:** trafilatura（静态提取）、tavily-python（动态后备）、concurrent.futures（并发）、DeepSeek（LLM）

---

## File Structure

| File | Responsibility |
|------|---------------|
| `server/url_processor.py` (new) | URL 抓取、HTML 提取、内容清洗、并发编排 |
| `agent/state.py` | 新增 `url_contents: list[dict]` 字段 |
| `agent/nodes/parse.py` | 检测消息中的 URL，调用 url_processor 并发抓取注入 state |
| `agent/nodes/rewrite_query.py` | URL 场景下的检索策略：title/首句优先 |
| `agent/utils/agent_utils.py` | `build_context_block` 增加 URL 内容块 |
| `agent/nodes/classify_and_answer.py` | 无额外修改（context_block 已包含 URL 内容） |
| `server/config.py` | 读取 `TAVILY_API_KEY` |
| `.env` | 添加 `TAVILY_API_KEY` |
| `requirements.txt` | 添加 `trafilatura`、`tavily-python` |
| `server/bot.py` | graph invoke 增加 `url_contents=[]` 默认值 |
| `tests/unit/test_url_processor.py` (new) | url_processor 单元测试 |
| `tests/unit/test_nodes.py` | parse 节点 URL 场景测试追加 |

---

### Task 1: 依赖安装 + 配置

**Files:**
- Modify: `requirements.txt`
- Modify: `.env`
- Modify: `server/config.py`

- [ ] **Step 1: 添加依赖到 requirements.txt**

末尾追加：
```
trafilatura>=2.0.0
tavily-python>=0.7.0
```

- [ ] **Step 2: .env 添加 TAVILY_API_KEY**

```env
TAVILY_API_KEY=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
```

- [ ] **Step 3: config.py 添加读取逻辑**

在 `server/config.py` 末尾添加：
```python
# URL fetching
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
```

- [ ] **Step 4: 安装依赖**

```bash
pip install trafilatura tavily-python
```

- [ ] **Step 5: 验证安装**

```bash
python -c "import trafilatura; from tavily import TavilyClient; print('OK')"
```
Expected: `OK`

- [ ] **Step 6: Commit**

```bash
git add requirements.txt .env server/config.py
git commit -m "chore: add trafilatura and tavily-python dependencies"
```

---

### Task 2: url_processor.py — 核心函数

**Files:**
- Create: `server/url_processor.py`

- [ ] **Step 1: 写 clean_content 测试**

```python
# tests/unit/test_url_processor.py
import pytest
from server.url_processor import clean_content


def test_clean_content_removes_images():
    raw = "开头\n\n![cover_image](https://mmbiz.qpic.cn/xxx)\n\n正文内容"
    assert "![cover_image]" not in clean_content(raw)
    assert "正文内容" in clean_content(raw)


def test_clean_content_removes_empty_headings():
    raw = "#\n\n##\n\n###\n\n正文"
    result = clean_content(raw)
    assert "#\n" not in result
    assert "正文" in result


def test_clean_content_removes_separators():
    raw = "上面\n\n---\n\n下面"
    result = clean_content(raw)
    assert "---" not in result
    assert "上面" in result
    assert "下面" in result


def test_clean_content_collapses_blank_lines():
    raw = "第一行\n\n\n\n\n\n第二行"
    result = clean_content(raw)
    assert result == "第一行\n\n第二行"


def test_clean_content_strips_whitespace():
    raw = "  \n内容\n  \n"
    assert clean_content(raw) == "内容"


def test_clean_content_removes_image_variants():
    raw = "![图1](url1) 文字 ![图2](url2) 结尾"
    result = clean_content(raw)
    assert "![图1]" not in result
    assert "![图2]" not in result
    assert "文字" in result
    assert "结尾" in result
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
pytest tests/unit/test_url_processor.py::test_clean_content_removes_images -v
```
Expected: `FAILED (ImportError: cannot import name 'clean_content')`

- [ ] **Step 3: 实现 clean_content**

```python
# server/url_processor.py
import re

def clean_content(raw: str) -> str:
    """清洗网页内容：移除图片引用、空标题行、分隔线、压缩空行。"""
    # 移除 ![alt](url) 图片引用
    text = re.sub(r'!\[.*?\]\(.*?\)', '', raw)
    # 移除仅有 # ## ### 等无文字的标题行
    text = re.sub(r'^#{1,6}\s*$', '', text, flags=re.MULTILINE)
    # 移除连续分隔线 --- 或 ___
    text = re.sub(r'^[---]{3,}\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^[___]{3,}\s*$', '', text, flags=re.MULTILINE)
    # 压缩连续空行为单行
    text = re.sub(r'\n{3,}', '\n\n', text)
    # 去除首尾空白
    return text.strip()
```

- [ ] **Step 4: 运行测试，验证通过**

```bash
pytest tests/unit/test_url_processor.py::test_clean_content_removes_images -v
pytest tests/unit/test_url_processor.py::test_clean_content_removes_empty_headings -v
pytest tests/unit/test_url_processor.py::test_clean_content_removes_separators -v
pytest tests/unit/test_url_processor.py::test_clean_content_collapses_blank_lines -v
pytest tests/unit/test_url_processor.py::test_clean_content_strips_whitespace -v
pytest tests/unit/test_url_processor.py::test_clean_content_removes_image_variants -v
```
Expected: All PASS

- [ ] **Step 5: 写 fetch_url_text 测试**

```python
def test_fetch_url_text_static_success(mocker):
    """静态页面提取成功，trafilatura 返回内容 > 200 chars。"""
    mocker.patch('trafilatura.fetch_url', return_value='<html><body><p>静态内容正文...（超过200字）'
                                                       '</p></body></html>')
    mocker.patch('trafilatura.extract', return_value='静态内容正文。' * 50)
    from server.url_processor import fetch_url_text
    result = fetch_url_text('https://example.com')
    assert result['url'] == 'https://example.com'
    assert len(result['content']) > 200
    assert result['title'] is None or isinstance(result['title'], str)


def test_fetch_url_text_fallback_to_tavily(mocker):
    """trafilatura 提取内容不足 200 chars → 降级到 tavily。"""
    mocker.patch('trafilatura.fetch_url', return_value='<html></html>')
    mocker.patch('trafilatura.extract', return_value='短内容')
    mock_tavily = mocker.patch('server.url_processor.TavilyClient')
    mock_tavily.return_value.extract.return_value = {
        'results': [{'content': 'Tavily 提取的长篇正文内容' * 50}]
    }
    from server.url_processor import fetch_url_text
    result = fetch_url_text('https://example.com')
    assert 'Tavily' in result['content']
    assert len(result['content']) > 200


def test_fetch_url_text_tavily_returns_none(mocker):
    """trafilatura 和 tavily 都失败时返回错误信息。"""
    mocker.patch('trafilatura.fetch_url', return_value=None)
    mocker.patch('trafilatura.extract', return_value='')
    mock_tavily = mocker.patch('server.url_processor.TavilyClient')
    mock_tavily.return_value.extract.return_value = {'results': []}
    from server.url_processor import fetch_url_text
    result = fetch_url_text('https://example.com')
    assert result['content'] == '[抓取失败]'
    assert result['title'] is None


def test_fetch_url_text_extracts_title(mocker):
    """从 HTML <title> 标签提取标题。"""
    mocker.patch('trafilatura.fetch_url',
                 return_value='<html><head><title>测试文章标题</title></head><body><p>内容</p></body></html>')
    mocker.patch('trafilatura.extract', return_value='内容内容' * 50)
    from server.url_processor import fetch_url_text
    result = fetch_url_text('https://example.com')
    assert result['title'] == '测试文章标题'


def test_fetch_url_text_timeout(mocker):
    """网络超时场景。"""
    mocker.patch('trafilatura.fetch_url', side_effect=Exception('Timeout'))
    from server.url_processor import fetch_url_text
    result = fetch_url_text('https://example.com')
    assert result['content'] == '[抓取失败]'
```

- [ ] **Step 6: 运行测试（应失败）**

```bash
pytest tests/unit/test_url_processor.py::test_fetch_url_text_static_success -v
```
Expected: FAIL

- [ ] **Step 7: 实现 fetch_url_text**

```python
import logging
import re
import requests
import trafilatura
from tavily import TavilyClient
from server.config import TAVILY_API_KEY

logger = logging.getLogger(__name__)


def fetch_url_text(url: str) -> dict:
    """下载单个 URL → 提取正文 → 清洗 → 返回结构化结果。

    使用两层策略：trafilatura 静态提取，不足 200 字则 tavily 后备。
    单 URL 超时 30 秒。
    """
    result = {"url": url, "title": None, "content": ""}
    try:
        # 尝试提取 title
        try:
            resp = requests.get(url, timeout=15, headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            })
            if resp.status_code == 200:
                m = re.search(r'<title>(.*?)</title>', resp.text, re.IGNORECASE | re.DOTALL)
                if m:
                    title = m.group(1).strip()
                    if title:
                        result["title"] = title[:200]

            # trafilatura 提取正文
            downloaded = trafilatura.fetch_url(url)
            if downloaded:
                text = trafilatura.extract(downloaded)
                if text and len(text) >= 200:
                    result["content"] = clean_content(text)
                    return result
        except Exception as e:
            logger.warning("Trafilatura extraction failed for %s: %s", url, e)

        # 后备：tavily extract
        if TAVILY_API_KEY:
            try:
                client = TavilyClient(api_key=TAVILY_API_KEY)
                tavily_result = client.extract(url)
                if tavily_result and tavily_result.get("results"):
                    content = tavily_result["results"][0].get("content", "")
                    if content:
                        result["content"] = clean_content(content)
                        return result
            except Exception as e:
                logger.warning("Tavily extraction failed for %s: %s", url, e)

        # 全部失败
        result["content"] = "[抓取失败]"
        return result

    except Exception as e:
        logger.error("Unexpected error fetching %s: %s", url, e)
        result["content"] = "[抓取失败]"
        return result
```

- [ ] **Step 8: 运行测试，验证通过**

```bash
pytest tests/unit/test_url_processor.py::test_fetch_url_text_static_success -v
pytest tests/unit/test_url_processor.py::test_fetch_url_text_fallback_to_tavily -v
pytest tests/unit/test_url_processor.py::test_fetch_url_text_tavily_returns_none -v
pytest tests/unit/test_url_processor.py::test_fetch_url_text_extracts_title -v
pytest tests/unit/test_url_processor.py::test_fetch_url_text_timeout -v
```
Expected: All PASS

- [ ] **Step 9: 写 fetch_urls_concurrent 测试**

```python
def test_fetch_urls_concurrent_all_success(mocker):
    """多个 URL 全部成功。"""
    mocker.patch('server.url_processor.fetch_url_text', side_effect=[
        {"url": "https://a.com", "title": "A", "content": "内容A"},
        {"url": "https://b.com", "title": "B", "content": "内容B"},
    ])
    from server.url_processor import fetch_urls_concurrent
    results = fetch_urls_concurrent(["https://a.com", "https://b.com"])
    assert len(results) == 2
    assert results[0]["title"] == "A"
    assert results[1]["title"] == "B"


def test_fetch_urls_concurrent_partial_failure(mocker):
    """部分 URL 失败，不影响其他。"""
    mocker.patch('server.url_processor.fetch_url_text', side_effect=[
        {"url": "https://a.com", "title": "A", "content": "内容A"},
        {"url": "https://b.com", "title": None, "content": "[抓取失败]"},
        {"url": "https://c.com", "title": "C", "content": "内容C"},
    ])
    from server.url_processor import fetch_urls_concurrent
    results = fetch_urls_concurrent(["https://a.com", "https://b.com", "https://c.com"])
    assert len(results) == 3
    assert results[0]["content"] == "内容A"
    assert results[1]["content"] == "[抓取失败]"
    assert results[2]["content"] == "内容C"


def test_fetch_urls_concurrent_empty():
    """空列表返回空列表。"""
    from server.url_processor import fetch_urls_concurrent
    assert fetch_urls_concurrent([]) == []


def test_fetch_urls_concurrent_deduplicates(mocker):
    """重复 URL 只抓取一次。"""
    mock_fn = mocker.patch('server.url_processor.fetch_url_text', return_value={
        "url": "https://a.com", "title": "A", "content": "内容A"
    })
    from server.url_processor import fetch_urls_concurrent
    results = fetch_urls_concurrent(["https://a.com", "https://a.com"])
    assert len(results) == 1
    mock_fn.assert_called_once()
```

- [ ] **Step 10: 实现 fetch_urls_concurrent**

```python
from concurrent.futures import ThreadPoolExecutor, as_completed


def fetch_urls_concurrent(urls: list[str]) -> list[dict]:
    """多线程并发抓取，max_workers=5。单条失败不影响整体。"""
    if not urls:
        return []

    # 去重，保持顺序
    seen = set()
    unique_urls = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            unique_urls.append(u)

    results = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(fetch_url_text, url): url for url in unique_urls}
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                url = futures[future]
                logger.error("Concurrent fetch failed for %s: %s", url, e)
                results.append({"url": url, "title": None, "content": "[抓取失败]"})

    # 按原始顺序排序
    order = {url: i for i, url in enumerate(unique_urls)}
    results.sort(key=lambda r: order.get(r.get("url", ""), 999))
    return results
```

- [ ] **Step 11: 运行全部 url_processor 测试**

```bash
pytest tests/unit/test_url_processor.py -v
```
Expected: All PASS

- [ ] **Step 12: Commit**

```bash
git add server/url_processor.py tests/unit/test_url_processor.py
git commit -m "feat: add url_processor with fetch, clean, and concurrent support"
```

---

### Task 3: Graph State 添加 url_contents 字段

**Files:**
- Modify: `agent/state.py`
- Modify: `server/bot.py`

- [ ] **Step 1: state.py 添加 url_contents**

```python
# 在 class AgentState(TypedDict) 中添加：
url_contents: list[dict]  # URL 抓取结果列表，每个元素 {"url", "title", "content"}
```

- [ ] **Step 2: bot.py 添加默认值**

在 `tests/unit/test_url_processor.py` 的 graph.invoke() 调用中，state 字典里加：
```python
"url_contents": [],
```

在 `server/bot.py` 的 graph.invoke() 调用中（约第 89 行），state 字典里加：
```python
"url_contents": [],
```

- [ ] **Step 3: 运行没有 URL 的 parse 测试，确认行为不变**

```bash
pytest tests/unit/test_nodes.py::test_parse -v
```
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add agent/state.py server/bot.py
git commit -m "feat: add url_contents field to graph state"
```

---

### Task 4: parse 节点集成 URL 抓取

**Files:**
- Modify: `agent/nodes/parse.py`
- Test: `tests/unit/test_nodes.py`（追加 URL 测试）

- [ ] **Step 1: 写 parse 的 URL 测试**

```python
# 追加到 tests/unit/test_nodes.py

def test_parse_with_urls(mocker):
    """含 URL 的消息 → parse 输出 url_contents。"""
    mocker.patch('server.url_processor.fetch_urls_concurrent', return_value=[
        {"url": "https://example.com", "title": "测试页面", "content": "正文"},
    ])
    from agent.nodes.parse import parse
    result = parse({
        "user_message": "总结 https://example.com 的内容",
        "user_id": "user1",
        "timestamp": "",
    })
    assert len(result["url_contents"]) == 1
    assert result["url_contents"][0]["title"] == "测试页面"


def test_parse_without_urls():
    """无 URL 的消息 → url_contents 为空列表。"""
    from agent.nodes.parse import parse
    result = parse({
        "user_message": "你好，今天天气怎么样",
        "user_id": "user1",
        "timestamp": "",
    })
    assert result["url_contents"] == []


def test_parse_with_multiple_urls(mocker):
    """多个 URL 全部提取。"""
    mocker.patch('server.url_processor.fetch_urls_concurrent', return_value=[
        {"url": "https://a.com", "title": "A", "content": "内容A"},
        {"url": "https://b.com", "title": "B", "content": "内容B"},
    ])
    from agent.nodes.parse import parse
    result = parse({
        "user_message": "比较 https://a.com 和 https://b.com",
        "user_id": "user1",
        "timestamp": "",
    })
    assert len(result["url_contents"]) == 2


def test_parse_with_only_url(mocker):
    """消息纯 URL，没有附加文字。"""
    mocker.patch('server.url_processor.fetch_urls_concurrent', return_value=[
        {"url": "https://example.com", "title": None, "content": "正文内容"},
    ])
    from agent.nodes.parse import parse
    result = parse({
        "user_message": "https://example.com",
        "user_id": "user1",
        "timestamp": "",
    })
    assert len(result["url_contents"]) == 1
    assert result["url_contents"][0]["title"] is None
```

- [ ] **Step 2: 运行测试（应失败）**

```bash
pytest tests/unit/test_nodes.py::test_parse_with_urls -v
```
Expected: FAIL（parse 还没有 URL 提取逻辑）

- [ ] **Step 3: 修改 parse.py**

```python
import logging
import re

from server.url_processor import fetch_urls_concurrent

logger = logging.getLogger(__name__)

_URL_PATTERN = re.compile(r'https?://[^\s]+')


def parse(state: dict) -> dict:
    user_message = state["user_message"].strip()
    logger.info("Parsed message from %s: '%s'", state.get("user_id", "unknown"), user_message[:60])

    result = {
        "user_message": user_message,
        "user_id": state.get("user_id", "unknown"),
        "timestamp": state.get("timestamp", ""),
    }

    # URL 提取 + 并发抓取
    urls = _URL_PATTERN.findall(user_message)
    if urls:
        url_contents = fetch_urls_concurrent(urls)
        result["url_contents"] = url_contents
        logger.info("Fetched %d URLs from message", len(url_contents))
    else:
        result["url_contents"] = []

    return result
```

- [ ] **Step 4: 运行测试，验证通过**

```bash
pytest tests/unit/test_nodes.py::test_parse_with_urls -v
pytest tests/unit/test_nodes.py::test_parse_without_urls -v
pytest tests/unit/test_nodes.py::test_parse_with_multiple_urls -v
pytest tests/unit/test_nodes.py::test_parse_with_only_url -v
```
Expected: All PASS

- [ ] **Step 5: 验证无 URL 的原有测试仍然通过**

```bash
pytest tests/unit/test_nodes.py::test_parse -v
```
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add agent/nodes/parse.py tests/unit/test_nodes.py
git commit -m "feat: integrate URL fetching into parse node"
```

---

### Task 5: rewrite_query URL 场景检索策略

**Files:**
- Modify: `agent/nodes/rewrite_query.py`

- [ ] **Step 1: 写 rewrite_query URL 测试**

```python
# 追加到 tests/unit/test_nodes.py 或新文件 tests/unit/test_rewrite_query_url.py
import pytest
from unittest.mock import patch


def test_rewrite_query_uses_title_when_available(mocker):
    """有 title 时用 title 做 query。"""
    mocker.patch('memory.message_history.MessageHistory.get_recent', return_value=[])
    from agent.nodes.rewrite_query import rewrite_query
    result = rewrite_query({
        "user_message": "总结 https://example.com",
        "session_id": "1",
        "url_contents": [
            {"url": "https://example.com", "title": "Python入门教程", "content": "正文..."}
        ],
    })
    assert "Python入门教程" in result["search_query"]


def test_rewrite_query_uses_first_sentence_when_no_title(mocker):
    """无 title 时用正文第一句做 query。"""
    mocker.patch('memory.message_history.MessageHistory.get_recent', return_value=[])
    from agent.nodes.rewrite_query import rewrite_query
    result = rewrite_query({
        "user_message": "总结 https://example.com",
        "session_id": "1",
        "url_contents": [
            {"url": "https://example.com", "title": None,
             "content": "这是文章的第一句话。这是第二句。第三句。"}
        ],
    })
    assert "第一句话" in result["search_query"]
    assert "第二句" not in result["search_query"]


def test_rewrite_query_prefers_user_question(mocker):
    """用户有附加问题时优先用用户问题。"""
    mocker.patch('memory.message_history.MessageHistory.get_recent', return_value=[])
    from agent.nodes.rewrite_query import rewrite_query
    result = rewrite_query({
        "user_message": "这个文章提到了哪些设计模式 https://example.com",
        "session_id": "1",
        "url_contents": [
            {"url": "https://example.com", "title": "设计模式详解", "content": "正文..."}
        ],
    })
    assert "设计模式" in result["search_query"]
    assert result["search_query"].startswith("这个文章提到了哪些设计模式")


def test_rewrite_query_url_only_no_additional_text(mocker):
    """纯 URL 消息 → 用 title 或首句。"""
    mocker.patch('memory.message_history.MessageHistory.get_recent', return_value=[])
    from agent.nodes.rewrite_query import rewrite_query
    result = rewrite_query({
        "user_message": "https://example.com",
        "session_id": "1",
        "url_contents": [
            {"url": "https://example.com", "title": "Python入门", "content": "正文..."}
        ],
    })
    assert "Python入门" in result["search_query"]


def test_rewrite_query_no_urls_normal_behavior(mocker):
    """无 URL 时行为不变。"""
    mocker.patch('memory.message_history.MessageHistory.get_recent',
                 return_value=[{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}])
    from agent.nodes.rewrite_query import rewrite_query
    result = rewrite_query({
        "user_message": "Python和Java的区别",
        "session_id": "1",
        "url_contents": [],
    })
    # 应该走正常 rewrite 逻辑（有历史记录）
    assert "Python" in result["search_query"]
```

- [ ] **Step 2: 运行测试（应失败）**

```bash
pytest tests/unit/test_nodes.py::test_rewrite_query_uses_title_when_available -v
```
Expected: FAIL

- [ ] **Step 3: 修改 rewrite_query.py**

```python
def rewrite_query(state: dict) -> dict:
    user_message = state["user_message"]
    url_contents = state.get("url_contents", [])

    # ── URL 场景检索策略 ──
    if url_contents:
        # 提取纯问题文字（去掉 URL）
        url_pattern = re.compile(r'https?://[^\s]+')
        question_only = url_pattern.sub('', user_message).strip()

        # 构建 query 候选
        candidates = []
        for uc in url_contents:
            if uc.get("title"):
                candidates.append(uc["title"])
            else:
                # 取正文第一句话
                content = uc.get("content", "")
                first_sentence = re.split(r'[。.!?]', content)[0].strip()
                candidates.append(first_sentence[:100] if first_sentence else content[:100])

        # 用户有足够多的问题文字 → 优先用用户问题
        if len(question_only) >= 10:
            return {"search_query": question_only}

        # 纯 URL 或问题文字太少 → 用 URL 的 title/首句
        query = "；".join(candidates)[:300]
        logger.info("URL query (no user question): '%s'", query[:60])
        return {"search_query": query if query else user_message}

    # ── 原有逻辑（无 URL）──
    # ... 保持不变 ...
```

完整 rewrite_query.py 应为：

```python
"""Rewrite user query with conversation context for standalone retrieval."""
import logging
import re

from memory.message_history import MessageHistory
from agent.utils.llm import LLM

logger = logging.getLogger(__name__)

REWRITE_PROMPT = (
    '你是一个查询改写助手。根据对话历史，将用户的最新问题改写为一个'
    '不需要上下文就能理解的独立问题。\n\n'
    '要求：\n'
    '- 补全指代（如"它"→"Python dict"、"区别呢"→"A和B的区别"）\n'
    '- 补全省略的部分\n'
    '- 不要添加不存在的信息\n'
    '- 如果问题已经是独立的，保持原文\n'
    '- 只输出改写后的文本，不要任何解释\n\n'
    '对话历史：\n{history}\n\n'
    '用户最新消息：{message}'
)


def rewrite_query(state: dict) -> dict:
    user_message = state["user_message"]
    url_contents = state.get("url_contents", [])

    # ── URL 场景检索策略 ──
    if url_contents:
        url_pattern = re.compile(r'https?://[^\s]+')
        question_only = url_pattern.sub('', user_message).strip()

        candidates = []
        for uc in url_contents:
            if uc.get("title"):
                candidates.append(uc["title"])
            else:
                content = uc.get("content", "")
                first_sentence = re.split(r'[。.!?]', content)[0].strip()
                candidates.append(first_sentence[:100] if first_sentence else content[:100])

        if len(question_only) >= 10:
            return {"search_query": question_only}

        query = "；".join(candidates)[:300]
        logger.info("URL query (no user question): '%s'", query[:60])
        return {"search_query": query if query else user_message}

    # ── 原有逻辑（无 URL）──
    session_id_raw = state.get("session_id")
    if session_id_raw is None:
        logger.debug("Skipping rewrite: no session_id in state")
        return {"search_query": user_message}
    session_id = int(session_id_raw)

    history = MessageHistory.get_recent(session_id)
    if len(history) < 2:
        logger.debug("Skipping rewrite: only %d history messages", len(history))
        return {"search_query": user_message}

    lines = []
    for msg in history:
        role = "用户" if msg["role"] == "user" else "助手"
        content = (msg.get("content") or msg.get("content", ""))[:200]
        lines.append(f"{role}：{content}")
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

- [ ] **Step 4: 运行测试，验证通过**

```bash
pytest tests/unit/test_nodes.py::test_rewrite_query_uses_title_when_available -v
pytest tests/unit/test_nodes.py::test_rewrite_query_uses_first_sentence_when_no_title -v
pytest tests/unit/test_nodes.py::test_rewrite_query_prefers_user_question -v
pytest tests/unit/test_nodes.py::test_rewrite_query_url_only_no_additional_text -v
pytest tests/unit/test_nodes.py::test_rewrite_query_no_urls_normal_behavior -v
```
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add agent/nodes/rewrite_query.py
git commit -m "feat: URL-aware retrieval query strategy in rewrite_query"
```

---

### Task 6: classify_and_answer 注入 url_contents

**Files:**
- Modify: `agent/utils/agent_utils.py`

- [ ] **Step 1: 写 build_context_block URL 内容测试**

```python
# 追加到 tests/unit/test_agent_utils.py

def test_build_context_block_with_url_contents():
    """url_contents 非空时包含网页内容块。"""
    result = build_context_block({
        "url_contents": [
            {"url": "https://example.com", "title": "测试标题", "content": "正文内容"}
        ],
    })
    assert "网页内容" in result
    assert "测试标题" in result
    assert "正文内容" in result


def test_build_context_block_with_multiple_urls():
    """多个 URL 时全部包含。"""
    result = build_context_block({
        "url_contents": [
            {"url": "https://a.com", "title": "文章A", "content": "内容A"},
            {"url": "https://b.com", "title": None, "content": "内容B的内容"},
        ],
    })
    assert "文章A" in result
    assert "内容B" in result


def test_build_context_block_empty_url_contents():
    """url_contents 为空列表不影响输出。"""
    result = build_context_block({
        "url_contents": [],
        "user_profile": {"basic": {"name": "test"}},
    })
    assert "网页内容" not in result
    assert "test" in result


def test_build_context_block_with_url_only_message():
    """纯 URL 消息含 '请直接总结' 指令标记。"""
    result = build_context_block({
        "url_contents": [
            {"url": "https://example.com", "title": "文章", "content": "正文"}
        ],
        "user_message": "https://example.com",
    })
    assert "请直接总结" in result
    assert "没有附加问题" in result
```

- [ ] **Step 2: 修改 build_context_block**

在 `agent/utils/agent_utils.py` 的 `build_context_block` 函数中，在最后 return 前追加 URL 内容块：

```python
    # ── URL 网页内容 ──
    url_contents = state.get("url_contents", [])
    if url_contents:
        parts.append("")
        parts.append("## 用户提供的网页内容")
        for uc in url_contents:
            parts.append("")
            parts.append(f"### URL: {uc.get('url', '')}")
            if uc.get("title"):
                parts.append(f"> 标题：{uc['title']}")
            content = uc.get("content", "")
            if content:
                preview = content[:500]
                if len(content) > 500:
                    preview += "..."
                parts.append(f"> 前 500 字摘要：")
                parts.append(f"> {preview}")
                parts.append("")
                parts.append(f"全文：")
                parts.append(f"{content}")

        # 纯 URL 消息（无附加文字）→ 追加总结指令
        user_message = state.get("user_message", "")
        url_pattern = re.compile(r'https?://[^\s]+')
        if not url_pattern.sub('', user_message).strip():
            parts.append("")
            parts.append("用户只发送了网页链接，没有附加问题。请直接总结这篇文章的核心内容，用中文输出。")
```

在文件头部 `import logging` 下方添加：
```python
import re
```

- [ ] **Step 3: 运行测试**

```bash
pytest tests/unit/test_agent_utils.py::test_build_context_block_with_url_contents -v
pytest tests/unit/test_agent_utils.py::test_build_context_block_with_multiple_urls -v
pytest tests/unit/test_agent_utils.py::test_build_context_block_empty_url_contents -v
pytest tests/unit/test_agent_utils.py::test_build_context_block_with_url_only_message -v
pytest tests/unit/test_agent_utils.py::test_build_context_block_empty -v
```
Expected: All PASS

- [ ] **Step 4: 确认现有 classify_and_answer 测试通过**

```bash
pytest tests/unit/test_nodes.py::test_respond_normal -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add agent/utils/agent_utils.py tests/unit/test_agent_utils.py
git commit -m "feat: inject url_contents into context block with summary instruction"
```

---

### Task 7: 完整回归测试

- [ ] **Step 1: 运行全部单元测试**

```bash
pytest tests/unit/ -v
```
Expected: 全部通过（原有 3 个预先存在的失败不影响）

- [ ] **Step 2: 最终提交**

```bash
git add -A
git commit -m "feat: complete URL fetching pipeline — fetch, clean, concurrent, query, inject"
```
