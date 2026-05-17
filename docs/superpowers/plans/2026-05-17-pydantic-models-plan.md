# Pydantic 实体模型重构实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 消除所有裸 dict 类型，用 Pydantic BaseModel 替代，提供类型安全和 IDE 支持。

**Architecture:** 构建时用 Pydantic，传递时用 dict。节点内部用 `XxxResult(...).model_dump()` 转换到 LangGraph 边界。消费方保持 `.get()` 不变。

**Tech Stack:** Pydantic v2, Python 3.10+ (via `X | None` syntax)

---

## 文件结构

```
agent/models/
  __init__.py        # 空，使 agent/models 成为包
  value_objects.py   # UrlContent, LogicChainStep, StoredKnowledge, ContradictionInfo, ReflectionInfo
  nodes.py           # NodeResult + 各节点返回值模型
  storage.py         # WikiPage, ErrorRecord, FileRecord, PageVersion, Relation
```

**修改文件（20 个）：** `agent/state.py`, `agent/utils/agent_utils.py`, `agent/nodes/*.py` (13), `server/url_processor.py`, `storage/models.py`, `server/daily_summary.py`, `tests/unit/test_nodes.py`, `tests/integration/test_nodes.py`

---

### Task 1: 创建模型基础设施

**Files:**
- Create: `agent/models/__init__.py`
- Create: `agent/models/value_objects.py`
- Create: `agent/models/nodes.py`
- Create: `agent/models/storage.py`

- [ ] **Step 1: 创建目录和包文件**

```bash
mkdir -p agent/models
```

- [ ] **Step 2: 创建 `agent/models/__init__.py`**

```python
# agent/models - Pydantic entity models for type safety
```

- [ ] **Step 3: 创建 `agent/models/value_objects.py`**

```python
from pydantic import BaseModel


class UrlContent(BaseModel):
    url: str
    title: str | None = None
    content: str = ""


class LogicChainStep(BaseModel):
    node: str
    action: str
    reasoning: str = ""
    confidence: float | None = None
    needs_store: bool | None = None
    search_performed: bool | None = None
    fallback: bool | None = None
    severity: str | None = None


class StoredKnowledge(BaseModel):
    type: str = "wiki_page"
    page_id: int = 0
    title: str = ""
    content: str = ""
    tags: list[str] = []
    distance: float = 0.0


class ContradictionInfo(BaseModel):
    found: bool = False
    details: str = ""
    severity: str = ""
    knowledge_ids: list[int] = []
    knowledge_texts: list[str] = []


class ReflectionInfo(BaseModel):
    result: str = ""
    reasoning: str = ""
    correction: str = ""
```

- [ ] **Step 4: 创建 `agent/models/nodes.py`**

```python
from pydantic import BaseModel
from agent.models.value_objects import LogicChainStep, UrlContent, StoredKnowledge


class NodeResult(BaseModel):
    logic_chain: list[LogicChainStep] = []


class ClassifyResult(NodeResult):
    answer: str = ""
    confidence: float = 0.0
    needs_store: bool = False


class FactCheckResult(NodeResult):
    contradiction_found: bool = False
    contradiction_details: str = ""
    contradiction_severity: str = ""
    contradiction_knowledge_ids: list[int] = []
    contradiction_knowledge_texts: list[str] = []


class ReflectResult(NodeResult):
    reflection_result: str = ""
    reflection_reasoning: str = ""
    reflection_correction: str = ""
    force_web_search: bool = False


class ParseResult(NodeResult):
    user_message: str = ""
    user_id: str = ""
    timestamp: str = ""
    url_contents: list[UrlContent] = []


class RetrieveResult(NodeResult):
    stored_knowledge: list[StoredKnowledge] = []


class SearchWebResult(NodeResult):
    search_results: list[str] = []


class RegenerateResult(NodeResult):
    answer: str = ""


class RecordErrorResult(NodeResult):
    correction_attempts: int = 0
    error_recorded: bool = False


class UpdateProfileResult(NodeResult):
    user_profile: dict = {}


class RewriteResult(NodeResult):
    search_query: str = ""


class RespondResult(BaseModel):
    final_response: str = ""
```

- [ ] **Step 5: 创建 `agent/models/storage.py`**

```python
from pydantic import BaseModel


class WikiPage(BaseModel):
    id: int = 0
    title: str = ""
    file_path: str = ""
    tags: str = ""
    sources: str = ""
    checksum: str = ""
    status: str = "active"
    created_at: str = ""
    updated_at: str = ""
    distance: float = 0.0


class ErrorRecord(BaseModel):
    id: int = 0
    user_message: str = ""
    wrong_answer: str = ""
    correct_answer: str = ""
    contradiction_details: str = ""
    error_type: str = ""
    created_at: str = ""


class FileRecord(BaseModel):
    id: int = 0
    filename: str = ""
    extension: str = ""
    file_hash: str = ""
    text_content: str = ""
    page_ids: str = ""
    user_id: str = ""
    created_at: str = ""


class PageVersion(BaseModel):
    id: int = 0
    page_id: int = 0
    title: str = ""
    content: str = ""
    checksum: str = ""
    source_id: str = ""
    source_question: str = ""
    created_at: str = ""


class Relation(BaseModel):
    id: int = 0
    page_id: int = 0
    related_page_id: int = 0
    relation_type: str = ""
    created_at: str = ""
```

- [ ] **Step 6: 验证导入**

Run: `python -c "from agent.models import *; print('Models imported successfully')"`
Expected: `Models imported successfully`

- [ ] **Step 7: 提交**

```bash
git add agent/models/ && git commit -m "feat: create agent/models with Pydantic entity types"
```

---

### Task 2: 接入 Value Objects（state + url_processor + agent_utils）

**Files:**
- Modify: `agent/state.py`
- Modify: `server/url_processor.py`
- Modify: `agent/utils/agent_utils.py`

- [ ] **Step 1: 修改 `agent/state.py` — 细化 4 个字段类型**

```python
from typing import Annotated
from typing_extensions import TypedDict
import operator
from agent.models.value_objects import UrlContent, LogicChainStep, StoredKnowledge


class AgentState(TypedDict):
    user_message: str
    search_query: str
    user_id: str
    timestamp: str
    confidence: float
    needs_store: bool
    search_results: list[str]
    stored_knowledge: list[StoredKnowledge]          # 之前 list[dict]
    stored_knowledge_ids: list[int]
    wiki_page_ids: list[int]
    answer: str
    final_response: str
    contradiction_found: bool
    contradiction_details: str
    search_time: int
    contradiction_severity: str
    contradiction_knowledge_ids: list[int]
    contradiction_knowledge_texts: list[str]
    reflection_result: str
    reflection_reasoning: str
    reflection_correction: str
    force_web_search: bool
    correction_attempts: int
    error_recorded: bool
    logic_chain: Annotated[list[LogicChainStep], operator.add]   # 之前 list[dict]
    user_profile: dict
    session_id: str
    message_history: list[dict]
    episodic_memories: list[str]
    url_contents: list[UrlContent]                               # 之前 list[dict]
```

- [ ] **Step 2: 修改 `server/url_processor.py` — `fetch_url_text` 返回 `UrlContent`**

```python
from agent.models.value_objects import UrlContent

def fetch_url_text(url: str) -> UrlContent:
    result = {"url": url, "title": None, "content": ""}
    # ... 原有下载 + 提取逻辑不变 ...
    return UrlContent(**result)

def fetch_urls_concurrent(urls: list[str]) -> list[UrlContent]:
    # ... 逻辑不变，仅返回类型由 list[dict] 改为 list[UrlContent] ...
    results = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(fetch_url_text, url): url for url in unique_urls}
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                results.append(UrlContent(url=futures[future], content="[抓取失败]"))
    order = {url: i for i, url in enumerate(unique_urls)}
    results.sort(key=lambda r: order.get(r.url, 999))
    return results
```

注意：内部的 `fetch_url_text` 仍然构造 dict，在 return 时 `UrlContent(**result)` 转换。或者直接构造 `UrlContent(url=url, title=..., content=...)`。选择一种一致的方式。

- [ ] **Step 3: 修改 `agent/utils/agent_utils.py` — `build_url_context` 签名**

```python
def build_url_context(urls: list[UrlContent], chars: int = -1) -> str:
```

函数体内保持 `uc.get()` 写法不变（因为 state 传递时仍然是 dict），或者改为 `uc.url`/`uc.title`/`uc.content`。建议改为点号访问，因为这里是唯一的格式化函数，后续不会有运行时问题。

```python
def build_url_context(urls: list[UrlContent], chars: int = -1) -> str:
    parts = [f"爬取内容如下（摘要{chars if chars > 0 else '全部'}字符）"]
    for uc in urls:
        parts.append("")
        parts.append(f"### URL: {uc.url}")
        if uc.title:
            parts.append(f"> 标题：{uc.title}")
        content = uc.content
        if content:
            parts.append(f"全文：")
            parts.append(content[:chars] if chars > 0 else content)
    return "\n".join(parts)
```

- [ ] **Step 4: 验证**

```bash
python -c "from agent.state import AgentState; print('state OK')"
python -c "from server.url_processor import fetch_url_text; print('url_processor OK')"
python -c "from agent.utils.agent_utils import build_url_context; print('agent_utils OK')"
```

- [ ] **Step 5: 提交**

```bash
git add agent/state.py server/url_processor.py agent/utils/agent_utils.py
git commit -m "feat: wire UrlContent/LogicChainStep/StoredKnowledge into state and consumers"
```

---

### Task 3: 转换 parse + rewrite_query + retrieve

**Files:**
- Modify: `agent/nodes/parse.py`
- Modify: `agent/nodes/rewrite_query.py`
- Modify: `agent/nodes/retrieve.py`

- [ ] **Step 1: `parse.py` — 返回 `ParseResult`**

原代码：
```python
result = {
    "user_message": user_message,
    "user_id": state.get("user_id", "unknown"),
    "timestamp": state.get("timestamp", ""),
    "url_contents": url_contents,
    "logic_chain": [{"node": "parse", ...}],
}
```

改为：
```python
from agent.models.nodes import ParseResult
from agent.models.value_objects import UrlContent, LogicChainStep

result = ParseResult(
    user_message=user_message,
    user_id=state.get("user_id", "unknown"),
    timestamp=state.get("timestamp", ""),
    url_contents=url_contents,
    logic_chain=[LogicChainStep(node="parse", ...)],
).model_dump()
```

- [ ] **Step 2: `rewrite_query.py` — 返回 `RewriteResult`**

```python
from agent.models.nodes import RewriteResult

return RewriteResult(
    search_query=question_only,
    logic_chain=[LogicChainStep(node="rewrite_query", ...)],
).model_dump()
```

其他 6 个 return 点同样改为 `RewriteResult(...).model_dump()`。

- [ ] **Step 3: `retrieve.py` — 返回 `RetrieveResult`**

```python
from agent.models.nodes import RetrieveResult

return RetrieveResult(
    stored_knowledge=[StoredKnowledge(type="wiki_page", page_id=p["id"], ...) for p in pages],
    logic_chain=[LogicChainStep(node="retrieve", ...)],
).model_dump()
```

注意：`stored_knowledge` 现在需要 `list[StoredKnowledge]` 而非 `list[dict]`，构造时显式映射。

- [ ] **Step 4: 验证**

Run: `pytest tests/unit/test_nodes.py -v -k "test_parse or test_store"` — 至少确认 parse 和 retrieve 不崩溃（store 不测试，它不经过 graph）。

Expected: 现有测试通过或改动相关的测试报错（还没改 tests，暂时允许报错）

- [ ] **Step 5: 提交**

```bash
git add agent/nodes/parse.py agent/nodes/rewrite_query.py agent/nodes/retrieve.py agent/models/nodes.py
git commit -m "feat: convert parse/rewrite_query/retrieve to Pydantic returns"
```

---

### Task 4: 转换 classify_and_answer + search_web + regenerate

**Files:**
- Modify: `agent/nodes/classify_and_answer.py`
- Modify: `agent/nodes/search_web.py`
- Modify: `agent/nodes/regenerate.py`

- [ ] **Step 1: `classify_and_answer.py` — 返回 `ClassifyResult`**

原代码（4 个 return 点，每个都是裸 dict）：
```python
return {
    "answer": structured.answer,
    "confidence": structured.confidence,
    "needs_store": structured.needs_store,
    "logic_chain": [{"node": "classify_and_answer", ...}],
}
```

改为：
```python
from agent.models.nodes import ClassifyResult
from agent.models.value_objects import LogicChainStep

return ClassifyResult(
    answer=structured.answer,
    confidence=structured.confidence,
    needs_store=structured.needs_store,
    logic_chain=[LogicChainStep(node="classify_and_answer", ...)],
).model_dump()
```

对其他 3 个 return 点（line 73, 84, 114）同样处理。确保原来返回的额外字段（`search_performed`, `fallback` 等）被正确映射。`ClassifyResult` 没有 `search_performed` 字段——可以在 `LogicChainStep` 上通过 `.search_performed=True` 传递，因为 `logic_chain` 最终被 store 的 `_save_reasoning_log` 消费时会读 `step.get("search_performed")`。

- [ ] **Step 2: `search_web.py` — 返回 `SearchWebResult`**

```python
from agent.models.nodes import SearchWebResult

return SearchWebResult(
    search_results=results,
    logic_chain=[LogicChainStep(node="search_web", ...)],
).model_dump()
```

- [ ] **Step 3: `regenerate.py` — 返回 `RegenerateResult`**

```python
from agent.models.nodes import RegenerateResult

return RegenerateResult(
    answer=...,
    logic_chain=[LogicChainStep(node="regenerate", ...)],
).model_dump()
```

- [ ] **Step 4: 提交**

```bash
git add agent/nodes/classify_and_answer.py agent/nodes/search_web.py agent/nodes/regenerate.py
git commit -m "feat: convert classify/search/regenerate to Pydantic returns"
```

---

### Task 5: 转换 fact_check + reflect + record_error

**Files:**
- Modify: `agent/nodes/fact_check.py`
- Modify: `agent/nodes/reflect.py`
- Modify: `agent/nodes/record_error.py`

- [ ] **Step 1: `fact_check.py` — 返回 `FactCheckResult`**

```python
from agent.models.nodes import FactCheckResult

# 早期 return（无矛盾）
return FactCheckResult(
    contradiction_found=False,
    contradiction_details="",
).model_dump()

# 完整 return（有矛盾）
return FactCheckResult(
    contradiction_found=True,
    contradiction_details=result.details,
    contradiction_severity=result.severity,
    contradiction_knowledge_ids=[p["id"] for p in pages],
    contradiction_knowledge_texts=[p.get("content", "") for p in pages],
    logic_chain=[LogicChainStep(node="fact_check", ...)],
).model_dump()
```

- [ ] **Step 2: `reflect.py` — 返回 `ReflectResult`**

```python
from agent.models.nodes import ReflectResult

return ReflectResult(
    reflection_result=output.result,
    reflection_reasoning=output.reasoning,
    reflection_correction=output.correction,
    force_web_search=output.result != "stored_knowledge_wrong",
    logic_chain=[LogicChainStep(node="reflect", ...)],
).model_dump()
```

- [ ] **Step 3: `record_error.py` — 返回 `RecordErrorResult`**

```python
from agent.models.nodes import RecordErrorResult

return RecordErrorResult(
    correction_attempts=state.get("correction_attempts", 0) + 1,
    error_recorded=True,
    logic_chain=[LogicChainStep(node="record_error", ...)],
).model_dump()
```

- [ ] **Step 4: 验证**

Run: `pytest tests/unit/test_nodes.py -v -k "fact_check or record_error or reflect"` — 确认语法正确。

- [ ] **Step 5: 提交**

```bash
git add agent/nodes/fact_check.py agent/nodes/reflect.py agent/nodes/record_error.py
git commit -m "feat: convert fact_check/reflect/record_error to Pydantic returns"
```

---

### Task 6: 转换 update_profile + respond + store（背景函数）

**Files:**
- Modify: `agent/nodes/update_profile.py`
- Modify: `agent/nodes/respond.py`
- Modify: `agent/nodes/store.py`

- [ ] **Step 1: `update_profile.py` — 返回 `UpdateProfileResult`**

```python
from agent.models.nodes import UpdateProfileResult

# 无更新
return UpdateProfileResult(user_profile=state.get("user_profile", {})).model_dump()

# 有更新
return UpdateProfileResult(
    user_profile=profile,
    logic_chain=[LogicChainStep(node="update_profile", ...)],
).model_dump()
```

- [ ] **Step 2: `respond.py` — 返回 `RespondResult`**

```python
from agent.models.nodes import RespondResult

return RespondResult(final_response=answer).model_dump()
```

- [ ] **Step 3: `store.py` — 背景函数中的 `extract_to_wiki` 返回**

`extract_to_wiki` 返回 `{"page_ids": [...], "logic_chain": [...]}`：
```python
from agent.models.value_objects import LogicChainStep

# 在 _sync_background_store 中的 extend 部分
result = extract_to_wiki(state["answer"], source_id, source_label)
logic_chain = state.get("logic_chain", [])
logic_chain.extend(result.get("logic_chain", []))
```

这个不需要改，因为 `extract_to_wiki` 是中间函数，不是 graph 节点。

但 `_save_reasoning_log` 中读取 `logic_chain` 部分需要确认兼容性——它读 `state.get("logic_chain", [])`，每个元素是 `LogicChainStep.model_dump()` 后的 dict（因为节点返回时调了 `.model_dump()`），所以 `step.get("node")` 仍然有效，无需修改。

- [ ] **Step 4: 提交**

```bash
git add agent/nodes/update_profile.py agent/nodes/respond.py agent/nodes/store.py
git commit -m "feat: convert update_profile/respond/store to Pydantic returns"
```

---

### Task 7: 转换存储层

**Files:**
- Modify: `storage/models.py`
- Modify: `server/daily_summary.py`

- [ ] **Step 1: `storage/models.py` — 8 个函数改为返回 Pydantic 模型**

对每个返回 `list[dict]` 的函数：

```python
from agent.models.storage import WikiPage, ErrorRecord, FileRecord, PageVersion, Relation

# find_similar_pages
def find_similar_pages(...) -> list[WikiPage]:
    rows = conn.execute(...).fetchall()
    return [WikiPage(**dict(r)) for r in rows]

# get_page_by_title
def get_page_by_title(...) -> WikiPage | None:
    row = conn.execute(...).fetchone()
    return WikiPage(**dict(row)) if row else None

# get_related_pages
def get_related_pages(...) -> list[WikiPage]:
    rows = conn.execute(...).fetchall()
    return [WikiPage(**dict(r)) for r in rows]

# get_all_pages_index
def get_all_pages_index(...) -> list[WikiPage]:
    rows = conn.execute(...).fetchall()
    return [WikiPage(**dict(r)) for r in rows]

# save_error_record_with_embedding — 参数改为 ErrorRecord
def save_error_record_with_embedding(record: dict) -> int:
    # 内部保持使用 record["key"] 访问，或者改为 ErrorRecord 字段访问
    # 建议保持 dict 参数不变，仅改返回类型
    ...
```

`_page_row_to_dict` 函数可以删除（不再需要，直接 `WikiPage(**dict(row))`）。

- [ ] **Step 2: `server/daily_summary.py` — 使用 `WikiPage`**

```python
from agent.models.storage import WikiPage

# _get_yesterday_pages 返回 list[WikiPage] 而非 list[dict]
def _get_yesterday_pages() -> list[WikiPage]:
    ...
    return [WikiPage(**dict(r)) for r in rows]

# _split_pages 参数和返回改为 list[WikiPage]
def _split_pages(pages: list[WikiPage], max_chars: int = 3000) -> list[list[WikiPage]]:
    ...

# _generate_summary_text 参数改为 list[WikiPage]
def _generate_summary_text(pages: list[WikiPage]) -> str:
    ...
    for group in groups:
        for p in group:
            content += f"## {p.title}\n\n{p.content}\n\n"
```

- [ ] **Step 3: 验证**

```bash
python -c "from storage.models import find_similar_pages; print('storage OK')"
python -c "from server.daily_summary import _split_pages; print('daily_summary OK')"
```

- [ ] **Step 4: 提交**

```bash
git add storage/models.py server/daily_summary.py
git commit -m "feat: convert storage layer to return Pydantic models"
```

---

### Task 8: 适配测试

**Files:**
- Modify: `tests/unit/test_nodes.py`
- Modify: `tests/integration/test_nodes.py`

- [ ] **Step 1: 运行全部测试，识别失败用例**

```bash
cd /workspace && /c/Users/dudu/software/miniconda/envs/agent/python -m pytest tests/unit/test_nodes.py -v 2>&1 | tail -30
```

- [ ] **Step 2: 修复 `test/unit/test_nodes.py` 中因 dict→model 变化导致的失败**

每个测试用例用类似模式修复：

```python
# 之前：节点返回 dict，测试直接 assert result["key"]
# 之后：节点返回 dict（因为 model_dump()），assert 方式不变
```

节点返回的依然是 dict（`XxxResult(...).model_dump()`），所以测试的 assert 方式不变。主要需要确认的是：
- `test_parse_with_urls` — `url_contents` 现在是 `UrlContent.model_dump()` 后的 dict，shape 不变
- `test_store_empty_answer` — store 背景函数不经过 graph，没问题

运行测试：
```bash
/c/Users/dudu/software/miniconda/envs/agent/python -m pytest tests/unit/test_nodes.py -v
```

修复任何 `KeyError`（字段名变化导致）或 `AssertionError`，直到全部通过。

- [ ] **Step 3: 运行并修复集成测试**

```bash
/c/Users/dudu/software/miniconda/envs/agent/python -m pytest tests/integration/test_nodes.py -v 2>&1 | tail -30
```

修复类似问题，确保通过。

- [ ] **Step 4: 最终验证**

```bash
/c/Users/dudu/software/miniconda/envs/agent/python -m pytest tests/ -v
```

Expected: 原有测试全部通过（可能有已知失败的测试，如 async respond，不变）

- [ ] **Step 5: 提交**

```bash
git add tests/ && git commit -m "test: adapt tests for Pydantic model types"
```
