# Fund Bot: 个人基金组合管理微信机器人

## Context

当前项目 (`LangChain-Learning`) 已有一个基于 WeCom 企业微信的 `KnowledgeBot`，提供知识问答、间隔复习、文件处理等功能。现需新增一个专门用于个人基金组合管理的微信机器人，借鉴 TradingAgents 的多智能体架构设计。

### TradingAgents 可复用模式

- 分层智能体协作（分析师 → 研究员辩论 → 裁决者）
- 结构化输出 (Pydantic Schema + `with_structured_output`)
- 5 级评分体系 (Buy → Sell)
- 双 LLM 级别 (deep_think + quick_think)
- 延迟反射评估 (deferred reflection)
- LangGraph SqliteSaver checkpoint 恢复

---

## Architecture

### 核心概念

个人基金组合管理的核心是**用户的持仓组合**:

```
用户(微信ID) → 持有 N 只基金(代码+份额+成本价)
                  → 组合维度: 整体收益、风险分散、再平衡
                  → 单基维度: 业绩追踪、调仓建议、风险评估
```

### Bot 层: 双机器人同进程运行

新增 `FUND_BOT_ID` / `FUND_BOT_SECRET` 配置:

```
main.py
  ├── init_db()          # 补充基金组合相关表
  ├── KnowledgeBot.run() # 现有知识问答机器人
  └── FundBot.run()      # 新基金管理机器人(同一事件循环)
```

共享: LLM 客户端、SQLite 数据库(新增 fund 表)、配置系统。

### 模块结构

```
fund/
  __init__.py
  bot.py                      # FundBot 类 — WSClient + 消息路由 + 调度器
  state.py                    # FundAgentState (TypedDict)
  graph.py                    # LangGraph StateGraph 定义
  schemas.py                  # Pydantic 结构化输出 schema
  checkpointer.py             # SqliteSaver checkpoint (per-user DB)
  agents/
    analysts/
      portfolio_analyst.py    # 组合整体分析 Agent
      holdings_analyst.py     # 持仓分析 Agent
      performance_analyst.py  # 业绩分析 Agent
      risk_analyst.py         # 风险分析 Agent
    researchers/
      bull_researcher.py      # 看多研究员
      bear_researcher.py      # 看空研究员
    managers/
      portfolio_manager.py    # 组合经理(最终决策+建议)
  utils/
    fund_data_tools.py        # 基金数据获取工具(akshare)
    portfolio_tools.py        # 组合管理工具(持仓CRUD、收益计算)
    graph_helpers.py          # bind_structured + invoke_structured_or_freetext
    memory.py                 # 基金决策日志 + 延迟反射
```

---

## 数据源

### 数据维度覆盖

| 数据维度 | 用途 | 来源 |
|----------|------|------|
| 基金净值历史 | 收益计算、回撤分析 | akshare `fund_open_fund_info_em` |
| 基金基本信息 | 展示基金概览 | akshare |
| 基金持仓 | 持仓分析、风格漂移 | akshare `fund_portfolio_hold_em` |
| 同类排名 | 业绩对比 | akshare `fund_open_fund_rank_em` |
| 最大回撤/波动率 | 风险评估 | akshare + 计算 |
| 基金经理变动 | 稳定性分析 | akshare |
| 市场指数 | 市场环境分析 | akshare `stock_zh_index_daily` |
| QDII/海外基金 | 海外配置分析 | yfinance |

**主数据源**: akshare (A股基金全覆盖)  
**补充**: yfinance (QDII/海外基金)

---

## 数据库表

全部建在现有 `knowledge.db` 中:

### `user_portfolio` — 用户基金持仓

```sql
CREATE TABLE user_portfolio (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     TEXT NOT NULL,
    fund_code   TEXT NOT NULL,
    fund_name   TEXT,
    shares      REAL,
    cost_price  REAL,
    notes       TEXT,
    added_at    TEXT,
    updated_at  TEXT,
    UNIQUE(user_id, fund_code)
);
```

### `fund_info` — 基金基本信息缓存

```sql
CREATE TABLE fund_info (
    code             TEXT PRIMARY KEY,
    name             TEXT,
    fund_type        TEXT,
    company          TEXT,
    established_date TEXT,
    fund_size        REAL,
    manager          TEXT,
    nav              REAL,
    total_nav        REAL,
    nav_date         TEXT,
    updated_at       TEXT
);
```

### `fund_nav_cache` — 基金净值缓存 (TTL 1h)

```sql
CREATE TABLE fund_nav_cache (
    fund_code    TEXT,
    date         TEXT,
    nav          REAL,
    total_nav    REAL,
    daily_return REAL,
    PRIMARY KEY (fund_code, date)
);
```

### `fund_holdings_cache` — 基金持仓快照 (TTL 7d)

```sql
CREATE TABLE fund_holdings_cache (
    fund_code     TEXT,
    report_date   TEXT,
    holdings_json TEXT,
    sectors_json  TEXT,
    PRIMARY KEY (fund_code, report_date)
);
```

### `fund_decisions` — 基金决策日志

```sql
CREATE TABLE fund_decisions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id         TEXT,
    fund_code       TEXT,
    decision_date   TEXT,
    rating          TEXT,
    decision_text   TEXT,
    nav_at_decision REAL,
    raw_return      REAL,
    alpha_return    REAL,
    reflection      TEXT,
    status          TEXT DEFAULT 'pending',
    created_at      TEXT,
    resolved_at     TEXT
);
```

### 缓存 TTL 策略

| 表 | TTL | 说明 |
|----|-----|------|
| `fund_info` | 24h | 基金基本信息变化慢 |
| `fund_nav_cache` | 1h | 净值每日更新 |
| `fund_holdings_cache` | 7d | 持仓按季度披露 |

请求时: 先查缓存 → 命中且未过期 → 直接用; 否则 → 刷新。

---

## 消息路由 & 交互设计

### 意图识别

```python
用户消息 → parse_intent()
  |
  |-- "我的持仓" / "组合" → portfolio_overview
  |     直查 DB + 格式化返回
  |
  |-- "分析 110011" / "看看易方达蓝筹" → fund_analyze
  |     进 LangGraph pipeline
  |
  |-- "加仓 110011" / "定投 110011" → action_advise
  |     进 LangGraph pipeline
  |
  |-- "110011 怎么样了" → fund_status
  |     直查缓存 API
  |
  |-- "添加基金 110011 1000份 1.5" → add_holding
  |     直写 DB
  |
  |-- 默认 → fund_search
```

解析方式: LLM 单次分类 + 简单前缀匹配。

### FundBot 消息处理

```python
async def _on_text(frame):
    content, user_id = extract_content(frame)
    intent = parse_intent(content)

    if intent == "add_holding":
        add_to_portfolio(user_id, fund_code, shares, cost)
        await self.client.reply(frame, markdown_success)

    elif intent == "fund_status":
        data = fetch_fund_status(fund_code)
        await self.client.reply(frame, markdown(data))

    elif intent == "fund_analyze":
        memory.resolve_pending(user_id, fund_code)  # Phase B
        result = await asyncio.to_thread(graph.invoke, state)
        await self.client.reply(frame, markdown(result["final_response"]))
        asyncio.create_task(memory.store_decision(...))  # Phase A

    elif intent == "portfolio_overview":
        portfolio = load_portfolio(user_id)
        await self.client.reply(frame, markdown(portfolio))
```

---

## LangGraph Pipeline

### 图结构

```
User Input → Parse Intent (首节点)
  |
  |-- [quick intent] → Respond (不走图)
  |
  |-- [analyze] → START
                   │
                   ▼
             Portfolio Analyst
              - load_user_holding()
              - calc_profit_loss()          quick_llm | tool-calling
              - get_alternative_funds()
                   │
                   ▼
             Holdings Analyst
              - get_fund_holdings()         quick_llm | tool-calling
              - get_sector_allocation()
              - get_manager_info()
                   │
                   ▼
             Performance Analyst
              - get_fund_nav_history()      quick_llm | tool-calling
              - get_fund_rankings()
              - 自行计算收益/回撤/夏普
                   │
                   ▼
             Risk Analyst
              - get_risk_metrics()          quick_llm | tool-calling
              - get_max_drawdown()
              - get_manager_changes()
                   │
                   ▼
             Bull Researcher ↔ Bear Researcher (辩论循环)
              - 收到 4 份报告 + 历史 + 对方论点  quick_llm | invoke
              - 标记 Bull/Bear Analyst: 前缀
              - count >= 2 * max_debate_rounds → 结束
                   │
                   ▼
             Portfolio Manager
              - 综合报告 + 辩论 + past_context  deep_llm | structured
              - 输出 FundDecision
                   │
                   ▼
             END → final_decision
```

### 分析师序列

1. **Portfolio Analyst** — 分析用户在该基金上的持仓成本、盈亏、在组合中的占比，与同类基金对比
2. **Holdings Analyst** — 基金底层持仓结构、行业分布、重仓股变化、风格漂移
3. **Performance Analyst** — 各阶段收益率、同类排名、超额收益、最大回撤
4. **Risk Analyst** — 波动率、下行风险、集中度风险、基金经理稳定性

### 关键设计点

| 环节 | 设计 | 原因 |
|------|------|------|
| 分析师顺序 | Portfolio → Holdings → Performance → Risk | 先组合背景再逐步深入 |
| 辩论轮次 | `max_debate_rounds=1`（默认可配） | 个人场景快速出结论 |
| LLM 分配 | 分析师+Bull/Bear → quick_llm; PM → deep_llm | 与 TradingAgents 一致 |
| 分析师工具 | `bind_tools` → ToolNode 回转 | 复用现有模式 |
| 快速查询 | 不走图，直接回复 | 避免不必要的 LLM 开销 |

### 状态设计

```python
class FundDebateState(TypedDict):
    bull_history: str
    bear_history: str
    history: str
    current_response: str
    count: int

class FundAgentState(TypedDict):
    user_message: str
    user_id: str
    fund_code: str
    fund_name: str
    intent: str                     # analyze | status | portfolio | add_holding
    user_holding: dict              # {"shares": 1000, "cost": 1.5, ...}

    portfolio_report: str           # 组合分析报告
    holdings_report: str            # 持仓分析报告
    performance_report: str         # 业绩分析报告
    risk_report: str                # 风险分析报告

    debate_state: FundDebateState
    final_decision: str
    past_context: str               # 记忆注入
```

---

## 结构化 Schema

```python
class FundRating(str, Enum):
    STRONG_BUY = "Strong Buy"   # 强烈建议加仓
    BUY = "Buy"                 # 建议加仓/定投
    HOLD = "Hold"               # 继续持有
    REDUCE = "Reduce"           # 建议减仓
    SELL = "Sell"               # 建议清仓

class FundDecision(BaseModel):
    rating: FundRating = Field(
        description="基金评级，基于分析师的报告和辩论结果"
    )
    summary: str = Field(
        description="一句话结论，用户能快速理解"
    )
    analysis: str = Field(
        description="详细分析：持仓背景、业绩表现、风险评估"  
    )
    action_advice: str = Field(
        description="具体操作建议：加仓/定投/持有/减仓/清仓及理由"
    )
    risk_note: str = Field(
        description="关键风险提示"
    )

def render_fund_decision(d: FundDecision) -> str:
    """渲染为 markdown，下游系统直接消费"""
    return "\n".join([
        f"**评级**: {d.rating.value}",
        "",
        f"**结论**: {d.summary}",
        "",
        f"**分析**: {d.analysis}",
        "",
        f"**操作建议**: {d.action_advice}",
        "",
        f"**风险提示**: {d.risk_note}",
    ])
```

使用 `invoke_structured_or_freetext` 模式：

```python
structured_llm = bind_structured(llm, FundDecision, "Portfolio Manager")
result = invoke_structured_or_freetext(
    structured_llm, plain_llm, prompt, render_fund_decision, "Portfolio Manager",
)
```

---

## Checkpointer (TradingAgents 模式移植)

### 原理

使用 LangGraph 的 `SqliteSaver`，每完成一个节点自动保存 state。崩溃后可根据 `thread_id` 恢复从最后完成的节点继续执行。

### 设计差异

| 项目 | TradingAgents | 基金 Bot |
|------|--------------|----------|
| DB 粒度 | per-ticker | per-user |
| thread_id | `hash(ticker:date)[:16]` | `hash(user:fund:date)[:16]` |
| 同一用户同基金同日期 | — | 可恢复 |

### 核心代码

```python
# fund/checkpointer.py

import hashlib, sqlite3
from contextlib import contextmanager
from pathlib import Path
from langgraph.checkpoint.sqlite import SqliteSaver


def _db_path(data_dir: str, user_id: str) -> Path:
    p = Path(data_dir) / "fund_checkpoints"
    p.mkdir(parents=True, exist_ok=True)
    return p / f"user_{user_id}.db"


def thread_id(user_id: str, fund_code: str, date: str) -> str:
    raw = f"{user_id}:{fund_code}:{date}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


@contextmanager
def get_checkpointer(data_dir: str, user_id: str):
    db = _db_path(data_dir, user_id)
    conn = sqlite3.connect(str(db), check_same_thread=False)
    try:
        saver = SqliteSaver(conn)
        saver.setup()
        yield saver
    finally:
        conn.close()


def has_checkpoint(data_dir, user_id, fund_code, date) -> bool:
    db = _db_path(data_dir, user_id)
    if not db.exists():
        return False
    tid = thread_id(user_id, fund_code, date)
    with get_checkpointer(data_dir, user_id) as saver:
        return saver.get_tuple({"configurable": {"thread_id": tid}}) is not None


def clear_checkpoint(data_dir, user_id, fund_code, date):
    db = _db_path(data_dir, user_id)
    if not db.exists():
        return
    tid = thread_id(user_id, fund_code, date)
    conn = sqlite3.connect(str(db))
    try:
        for table in ("writes", "checkpoints"):
            conn.execute(f"DELETE FROM {table} WHERE thread_id = ?", (tid,))
        conn.commit()
    except sqlite3.OperationalError:
        pass
    finally:
        conn.close()
```

### 使用方式

```python
# bot.py
if config.get("checkpoint_enabled"):
    with get_checkpointer(data_dir, user_id) as saver:
        graph = workflow.compile(checkpointer=saver)
        result = await asyncio.to_thread(graph.invoke, state, config)
        clear_checkpoint(data_dir, user_id, fund_code, date)
else:
    result = await asyncio.to_thread(graph.invoke, state)
```

配置开关: `FUND_CHECKPOINT_ENABLED` (默认 false)

---

## 记忆与反思系统

三阶段设计，完全移植 TradingAgents 的延迟反射模式。

### Phase A — 运行时记录

```python
def store_decision(self, user_id, fund_code, rating, decision_text, nav):
    db.execute(
        """INSERT INTO fund_decisions 
           (user_id, fund_code, decision_date, rating, decision_text, 
            nav_at_decision, status, created_at)
           VALUES (?, ?, date('now'), ?, ?, ?, 'pending', datetime('now'))""",
        user_id, fund_code, rating, decision_text, nav,
    )
```

### Phase B — 延迟收益评估

触发时机: 下次用户分析同一基金时，先于 graph 执行。

```python
def resolve_pending(self, user_id, fund_code):
    pending = db.query(
        "SELECT * FROM fund_decisions WHERE fund_code=? AND user_id=? AND status='pending'",
        fund_code, user_id
    )
    for p in pending:
        today_nav = fetch_nav(fund_code, date.today())
        raw_return = (today_nav - p.nav_at_decision) / p.nav_at_decision
        bench_return = fetch_benchmark_return(p.decision_date, date.today())
        alpha_return = raw_return - bench_return
        reflection = self._reflect(p.decision_text, raw_return, alpha_return)
        db.execute(
            """UPDATE fund_decisions 
               SET status='resolved', raw_return=?, alpha_return=?, 
                   reflection=?, resolved_at=datetime('now')
               WHERE id=?""",
            raw_return, alpha_return, reflection, p.id
        )
```

Reflector prompt (TradingAgents 移植):

```python
def _reflect(self, decision_text, raw_return, alpha_return) -> str:
    prompt = (
        "You are a fund analyst reviewing your own past decision now that the outcome is known.\n"
        "Write exactly 2-4 sentences of plain prose (no bullets, no headers, no markdown).\n\n"
        "Cover in order:\n"
        "1. Was the rating call correct? (cite the return figure)\n"
        "2. Which part of the analysis held or failed?\n"
        "3. One concrete lesson to apply to the next similar fund analysis.\n\n"
        "Be specific and terse. Every word must earn its place."
    )
    messages = [
        ("system", prompt),
        ("human", 
         f"Raw return: {raw_return:+.1%}\n"
         f"Alpha vs benchmark: {alpha_return:+.1%}\n\n"
         f"Decision:\n{decision_text}"),
    ]
    return self.llm.invoke(messages).content
```

### Phase C — 记忆注入

```python
def get_past_context(self, user_id, fund_code, n_same=3, n_cross=2) -> str:
    resolved = db.query(
        "SELECT * FROM fund_decisions WHERE user_id=? AND status='resolved' ORDER BY resolved_at DESC",
        user_id
    )
    same = [d for d in resolved if d.fund_code == fund_code][:n_same]
    cross = [d for d in resolved if d.fund_code != fund_code][:n_cross]

    parts = []
    if same:
        parts.append(f"过去对 {fund_code} 的分析（近期优先）:")
        for s in same:
            parts.append(
                f"[{s.decision_date} | {s.rating} | {s.raw_return:+.1%} | {s.alpha_return:+.1%}]\n"
                f"反思: {s.reflection}"
            )
    if cross:
        parts.append("\n其他基金的经验教训:")
        for c in cross:
            parts.append(f"[{c.fund_code} | {c.decision_date} | {c.rating}] {c.reflection}")
    return "\n\n".join(parts)
```

注入 Portfolio Manager prompt:

```
**决策历史:**
过去对 110011 的分析（近期优先）:
[2026-04-15 | Buy | +3.2% | +1.8%]
反思: ...
```

---

## 主动推送

### 每周组合回顾

APScheduler cron，周六 10:00 推送。

```python
self.scheduler.add_job(
    self._run_weekly_review,
    "cron", day_of_week="sat", hour=10, minute=0,
    id="fund_weekly_review",
    misfire_grace_time=300,
)
```

执行流程:

1. 查 `user_portfolio` 获取所有有持仓的用户
2. 对每位用户，拉取持仓基金最新净值
3. 计算各基金收益、组合总资产、累计收益
4. 组装 markdown 周报（含基金表现表格 + LLM 市场简评）
5. `client.send_message()` 主动推送

周报格式:

```
# 本周组合回顾 (2026-05-18 ~ 2026-05-22)

组合总资产: ¥125,800 | 总投入: ¥120,000 | 累计收益: +4.83%

| 基金 | 本周收益 | 持仓占比 | 操作建议 |
|------|---------|---------|---------|
| 易方达蓝筹(110011) | +2.1% | 40% | 持有 |
| ...

市场简评:
本周A股震荡上行...
```

---

## 新增配置项

```python
# Fund Bot
FUND_BOT_ENABLED = os.getenv("FUND_BOT_ENABLED", "false").lower() == "true"
FUND_BOT_ID = os.getenv("FUND_BOT_ID", "")
FUND_BOT_SECRET = os.getenv("FUND_BOT_SECRET", "")

# Checkpointer
FUND_CHECKPOINT_ENABLED = os.getenv("FUND_CHECKPOINT_ENABLED", "false").lower() == "true"

# Cache
FUND_DATA_CACHE_DIR = os.getenv("FUND_DATA_CACHE_DIR", "data/fund_cache")
```

---

## 与现有系统的关系

| 层面 | KnowledgeBot | FundBot |
|------|-------------|---------|
| WSClient | `WECOM_BOT_ID` | `FUND_BOT_ID` |
| LangGraph | `agent/graph.py` Q&A | `fund/graph.py` 基金分析 |
| 分析师 | 4个知识分析师 | 4个基金分析师 |
| 辩论 | 矛盾反射 | Bull/Bear辩论 |
| Checkpoint | 无 | SqliteSaver per-user |
| LLM | 共享 | 共享 |
| DB | `knowledge.db` | 共用 + fund 表 |
| 记忆 | 三层记忆 | 基金决策日志 |
| 定时 | 日报/Thinker/清理 | 每周组合回顾 |

---

## 实施步骤

1. **配置 & 数据层**
   - `server/config.py`: 添加 fund bot 环境变量
   - `storage/database.py`: 新增 5 张 fund 表
   - 安装 `akshare` 依赖

2. **数据获取工具**
   - `fund/utils/fund_data_tools.py`: akshare 封装（净值/持仓/排名/风险指标）
   - `fund/utils/portfolio_tools.py`: 组合管理（持仓CRUD、盈亏计算）
   - 缓存层: DB 读写 + TTL 检查

3. **结构化 Schema**
   - `fund/schemas.py`: FundRating, FundDecision, render_fund_decision
   - `fund/utils/graph_helpers.py`: bind_structured, invoke_structured_or_freetext

4. **Agent Pipeline**
   - 4 个分析师 Agent (tool-calling, quick_llm)
   - Bull/Bear 研究员 (invoke, quick_llm)
   - Portfolio Manager (structured, deep_llm)

5. **图编排**
   - `fund/state.py`: FundAgentState, FundDebateState
   - `fund/graph.py`: StateGraph + 条件路由
   - `fund/checkpointer.py`: SqliteSaver

6. **记忆系统**
   - `fund/utils/memory.py`: store_decision / resolve_pending / get_past_context

7. **Bot 集成**
   - `fund/bot.py`: FundBot 类
   - `main.py`: 双 bot 启动

8. **测试**
   - 单元测试: 数据工具、schema 渲染、Reflector
   - 集成测试: akshare → agent → 回复
