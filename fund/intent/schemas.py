"""Intent node definitions for LLM-based intent classification."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class IntentKind:
    FUND = "fund"
    PORTFOLIO = "portfolio"
    SYSTEM = "system"


class ParamSchema(BaseModel):
    """Parameter expected by an intent node."""
    name: str
    type: Literal["fund_code", "fund_name", "shares", "cost", "query"]
    description: str
    required: bool = True


class IntentNode(BaseModel):
    """A single intent that the classifier can recognize."""
    id: str
    name: str
    description: str
    examples: list[str]
    kind: str  # fund | portfolio | system
    params: list[ParamSchema] = []


# ── Intent registry ────────────────────────────────────────────────

INTENTS: list[IntentNode] = [
    IntentNode(
        id="fund_analyze",
        name="深度分析",
        description="对指定基金进行多维度深度分析，返回持仓分析、业绩评估、风险评估和操作建议",
        kind=IntentKind.FUND,
        params=[ParamSchema(name="query", type="query", description="基金代码或名称", required=True)],
        examples=[
            "分析 110011",
            "帮我看看易方达蓝筹怎么样",
            "评估一下我的招商中证白酒",
            "分析所有新能源基金最近表现",
        ],
    ),
    IntentNode(
        id="fund_status",
        name="快速查询",
        description="快速查询基金最新净值、涨跌幅、规模和基本信息",
        kind=IntentKind.FUND,
        params=[ParamSchema(name="query", type="query", description="基金代码或名称", required=True)],
        examples=[
            "110011 怎么样了",
            "查一下招商中证白酒的净值",
            "易方达蓝筹今日净值多少",
        ],
    ),
    IntentNode(
        id="fund_search",
        name="搜索基金",
        description="搜索基金代码或名称关键字，返回匹配基金列表",
        kind=IntentKind.FUND,
        params=[ParamSchema(name="query", type="query", description="搜索关键词", required=True)],
        examples=[
            "搜索新能源",
            "有哪些半导体基金",
            "帮我找找医药基金",
        ],
    ),
    IntentNode(
        id="portfolio_overview",
        name="持仓概览",
        description="查看用户自己的基金持仓列表和组合概况",
        kind=IntentKind.PORTFOLIO,
        examples=[
            "我的持仓",
            "看看我的组合",
            "我买了哪些基金",
            "持仓情况",
        ],
    ),
    IntentNode(
        id="add_holding",
        name="添加持仓",
        description="添加一只基金到用户的自选持仓中",
        kind=IntentKind.PORTFOLIO,
        params=[
            ParamSchema(name="fund_code", type="fund_code", description="基金代码", required=True),
            ParamSchema(name="shares", type="shares", description="持有份额", required=True),
            ParamSchema(name="cost", type="cost", description="成本单价", required=False),
        ],
        examples=[
            "添加基金 110011 1000份 1.5",
            "加仓招商中证白酒 500份",
            "添加 005827 2000",
        ],
    ),
    IntentNode(
        id="remove_holding",
        name="删除持仓",
        description="从用户持仓中移除一只基金",
        kind=IntentKind.PORTFOLIO,
        params=[
            ParamSchema(name="fund_code", type="fund_code", description="基金代码", required=True),
        ],
        examples=[
            "删除 110011",
            "移除易方达蓝筹",
            "清仓招商中证白酒",
        ],
    ),
    IntentNode(
        id="greeting",
        name="问候",
        description="用户打招呼或寒暄",
        kind=IntentKind.SYSTEM,
        examples=[
            "你好",
            "早上好",
            "在吗",
            "hello",
            "嗨",
        ],
    ),
    IntentNode(
        id="help",
        name="帮助",
        description="用户询问支持的功能或使用方法",
        kind=IntentKind.SYSTEM,
        examples=[
            "你能做什么",
            "怎么用",
            "帮助",
            "支持哪些功能",
            "介绍一下自己",
        ],
    ),
]
