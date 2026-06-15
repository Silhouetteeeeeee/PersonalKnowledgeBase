# FundBot 意图识别重构 — LLM-based Intent Classification

## 背景

FundBot 当前使用正则表达式进行意图识别（`parse_intent()`），用户说法稍有变化就无法匹配，回退到默认搜索。参考 ragent 项目的纯 LLM 意图分类模式进行重构。

## 架构

```
用户消息
  │
  ▼
classify() ─── 一次 LLM 调用，对所有 IntentNode 评分
  │
  ├─ score < 0.35 → fallback fund_search
  │
  ├─ score >= 0.35 → 路由到对应 handler
  │
  └─ 同时返回 params（LLM 按 ParamSchema 定义提取）
```

## 数据模型

```python
class IntentKind(str, Enum):
    FUND = "fund"
    PORTFOLIO = "portfolio"
    SYSTEM = "system"

class ParamSchema(BaseModel):
    name: str
    type: Literal["fund_code", "fund_name", "shares", "cost", "query"]
    description: str
    required: bool = True

class IntentNode(BaseModel):
    id: str
    name: str
    description: str
    examples: list[str]
    kind: IntentKind
    params: list[ParamSchema] = []
```

## 意图清单

| id | kind | params | 示例 |
|----|------|--------|------|
| fund_analyze | fund | query | "分析 110011"、"帮我看看易方达蓝筹" |
| fund_status | fund | query | "110011 怎么样了"、"查一下净值" |
| fund_search | fund | query | "搜索新能源"、"有哪些半导体基金" |
| portfolio_overview | portfolio | 无 | "我的持仓"、"看看我的组合" |
| add_holding | portfolio | fund_code, shares, cost | "添加基金 110011 1000份 1.5" |
| remove_holding | portfolio | fund_code | "删除 110011"、"移除易方达蓝筹" |
| greeting | system | 无 | "你好"、"在吗"、"hello" |
| help | system | 无 | "你能做什么"、"怎么用" |

## 分类器

单次 LLM 调用 + `with_structured_output(IntentClassification)`。prompt 包含：
- 所有 IntentNode（id、描述、示例、参数说明）
- 评分标准（>0.8 强匹配 / 0.35-0.8 中等 / <0.35 不匹配）
- 输出 JSON schema

## 参数提取

和分类在同一轮 LLM 调用中完成。LLM 根据每个 IntentNode 的 `params` 定义提取对应字段。分类器拿到 `{id, score, reason, params}` 后，校验必填参数是否齐全。

## 文件结构

```
fund/intent/
  __init__.py
  schemas.py        # IntentKind, ParamSchema, IntentNode, INTENTS 列表
  classifier.py     # classify() + IntentResult + 参数校验
```

`fund/bot.py` 改动：
- 删除 `parse_intent()` 和 `_lookup_fund_code()`
- 导入 `classify()` 和 `INTENTS`
- `_on_text` 中调用 `classify()` 替代 `parse_intent()`
- handler 签名统一为 `_handle_xxx(frame, user_id, params)`
