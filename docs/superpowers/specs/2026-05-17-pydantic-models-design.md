# Pydantic 实体模型重构设计

> **目标:** 消除项目中所有原始 dict 类型，使用 Pydantic BaseModel 替代隐含的 dict 形状，提供编译期类型检查和 IDE 支持。
>
> **原则:** 构建时用 Pydantic，传递时用 dict。节点内部构造使用 Pydantic 模型，到 LangGraph 边界 `.model_dump()`。
>
> **消费方现状:** 从 state 读取的仍然是 dict（LangGraph 运行时行为），`uc.get("url")` 保持不变，主要收益在构建端。

---

## 架构

```
agent/models/
  __init__.py
  value_objects.py   # UrlContent, LogicChainStep, StoredKnowledge, ContradictionInfo, ReflectionInfo
  nodes.py           # NodeResult + 各节点返回值模型
  state.py           # AgentState TypedDict（list[dict] 细化为 list[UrlContent] 等）
  storage.py         # WikiPage, ErrorRecord, FileRecord, PageVersion 等 DB 映射
```

### 依赖关系

```
nodes.py ──→ value_objects.py  （节点返回值包含 LogicChainStep、UrlContent 等）
state.py ──→ value_objects.py   （AgentState 字段引用具体模型）
storage.py        （独立，仅映射 DB 行）
```

---

## value_objects.py

每个模型 3-5 个字段，无业务方法，纯数据容器。

```python
from pydantic import BaseModel


class UrlContent(BaseModel):
    """URL 抓取结果，由 url_processor.py 生产，parse/rewrite/classify/context 消费。"""
    url: str
    title: str | None = None
    content: str = ""


class LogicChainStep(BaseModel):
    """推理链路中的一个步骤，所有节点各追加一条。"""
    node: str
    action: str
    reasoning: str = ""
    # 以下为可选扩展字段，仅部分节点设置
    confidence: float | None = None
    needs_store: bool | None = None
    search_performed: bool | None = None
    fallback: bool | None = None
    severity: str | None = None


class StoredKnowledge(BaseModel):
    """检索结果中的单条知识。"""
    type: str = "wiki_page"
    page_id: int = 0
    title: str = ""
    content: str = ""
    tags: list[str] = []
    distance: float = 0.0


class ContradictionInfo(BaseModel):
    """矛盾检测结果（合并原 contradiction_* 字段语义）。"""
    found: bool = False
    details: str = ""
    severity: str = ""
    knowledge_ids: list[int] = []
    knowledge_texts: list[str] = []


class ReflectionInfo(BaseModel):
    """反思节点输出。"""
    result: str = ""
    reasoning: str = ""
    correction: str = ""
```

---

## nodes.py

每个节点返回值对应一个 Pydantic 模型，继承 `NodeResult` 基类。

```python
from pydantic import BaseModel
from agent.models.value_objects import LogicChainStep, UrlContent, StoredKnowledge


class NodeResult(BaseModel):
    """所有节点返回的基类。调用 .model_dump() 后传给 LangGraph。"""
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


class RespondResult(BaseModel):
    final_response: str = ""
```

**使用模式** — 每个节点的 return 改为：

```python
# 之前
return {
    "answer": structured.answer,
    "confidence": structured.confidence,
    "logic_chain": [{"node": "classify_and_answer", "action": "xxx", "reasoning": "yyy"}],
}

# 之后
return ClassifyResult(
    answer=structured.answer,
    confidence=structured.confidence,
    logic_chain=[LogicChainStep(node="classify_and_answer", action="xxx", reasoning="yyy")],
).model_dump()
```

---

## state.py

AgentState 保持 TypedDict（LangGraph 最成熟的支持），仅细化字段类型：

```python
from typing import Annotated, TypedDict
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

注意：`message_history` 和 `user_profile` 保持 `dict`（SQLite 行转 dict / JSON 自由结构），后续再建模。

---

## storage.py

Maps SQLite rows to Pydantic models. All functions in `storage/models.py` change return type from `list[dict]` to `list[WikiPage]` etc.

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

---

## 文件变更清单

| 操作 | 文件 | 说明 |
|------|------|------|
| 创建 | `agent/models/__init__.py` | 空文件 |
| 创建 | `agent/models/value_objects.py` | ~50 行 |
| 创建 | `agent/models/nodes.py` | ~80 行 |
| 创建 | `agent/models/storage.py` | ~60 行 |
| 修改 | `agent/state.py` | `list[dict]` → 具体类型 |
| 修改 | `agent/nodes/*.py` (13 个) | 返回 dict → 构造 Pydantic + `.model_dump()` |
| 修改 | `agent/utils/agent_utils.py` | `build_url_context` 参数改为 `list[UrlContent]` |
| 修改 | `storage/models.py` | 8 个函数返回类型改为 Pydantic 模型 |
| 修改 | `storage/wiki_storage.py` | `read_page()` 返回 `WikiPage` 类似模型（或保持 dict） |
| 修改 | `server/url_processor.py` | `fetch_url_text()` 返回 `UrlContent` |
| 修改 | `server/bot.py` | 初始 state 适配合法构造 |
| 修改 | `server/daily_summary.py` | `list[dict]` → `list[WikiPage]` |
| 修改 | `tests/unit/test_nodes.py` | 适配新构造方式 |
| 修改 | `tests/integration/test_nodes.py` | 适配新构造方式 |

总计约 25 个文件变更。

---

## 执行策略

建议分批实施，每批可独立测试：

1. **Batch 1: 基础设施** — 创建 `agent/models/` 目录和 4 个模型文件，不修改任何业务代码
2. **Batch 2: Value Objects** — `agent/state.py` + `server/url_processor.py` + `agent/utils/agent_utils.py` 接入 `UrlContent`
3. **Batch 3: Node returns** — 所有 13 个节点改为 Pydantic 构造 + `.model_dump()`
4. **Batch 4: Storage models** — `storage/models.py` + `daily_summary.py` 接入 `WikiPage` 等
5. **Batch 5: Tests** — 适配所有测试用例

---

## 不变的部分

- 图结构（graph.py）完全不修改
- LangGraph 的 invoke/state 机制完全不修改
- `_save_reasoning_log` 不修改（它已经接受 `state["logic_chain"]` 的 dict list）
- `build_context_block` 等消费函数保持 `uc.get("url")`，仅生产端改
