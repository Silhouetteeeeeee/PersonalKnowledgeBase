"""Fund Bot — WeChat Work bot for personal fund portfolio management."""

import asyncio
import logging
import re
from datetime import date

from aibot import WSClient, WSClientOptions
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from server.config import (
    FUND_BOT_ID, FUND_BOT_SECRET, FUND_CHECKPOINT_ENABLED,
)
from fund.graph import build_fund_graph
from fund.state import FundAgentState, FundDebateState
from fund.utils.portfolio_tools import add_holding, get_portfolio, get_holding, remove_holding
from fund.utils.fund_data_tools import search_fund, get_fund_info, get_fund_nav
from fund.utils.memory import FundMemory
from fund.checkpointer import get_checkpointer, clear_checkpoint
from storage.database import get_connection

logger = logging.getLogger(__name__)


def parse_intent(content: str) -> dict:
    """Parse user message into intent and extracted params."""
    content = content.strip()

    m = re.match(r"(?:添加基金|加仓|添加)\s*(\w+)\s*([\d.]+)\s*份?\s*([\d.]*)", content)
    if m:
        return {"intent": "add_holding", "fund_code": m.group(1),
                "shares": float(m.group(2)), "cost": float(m.group(3)) if m.group(3) else 0.0}

    m = re.match(r"(?:删除基金|移除|删除)\s*(\w+)", content)
    if m:
        return {"intent": "remove_holding", "fund_code": m.group(1)}

    if content in ("我的持仓", "持仓", "组合", "我的组合"):
        return {"intent": "portfolio_overview"}

    m = re.match(r"(?:分析|看看|评估)\s*(.+)", content)
    if m:
        return {"intent": "fund_analyze", "query": m.group(1).strip()}

    m = re.match(r"(?:查一下|查询|查看)\s*(\w+)", content)
    if m:
        return {"intent": "fund_status", "fund_code": m.group(1)}

    m = re.match(r"^(\d{6})$", content)
    if m:
        return {"intent": "fund_status", "fund_code": m.group(1)}

    return {"intent": "fund_search", "query": content}


def _lookup_fund_code(query: str) -> tuple:
    """Try to resolve a query to (fund_code, fund_name)."""
    info = get_fund_info(query)
    if info:
        return info["code"], info.get("name", "")
    results = search_fund(query)
    if results:
        r = results[0]
        return r.get("基金代码", query), r.get("基金简称", query)
    return query, query


class FundBot:
    def __init__(self):
        self.client = WSClient(WSClientOptions(
            bot_id=FUND_BOT_ID,
            secret=FUND_BOT_SECRET,
            max_reconnect_attempts=-1,
        ))
        self.graph = build_fund_graph()
        self.scheduler = AsyncIOScheduler()
        self.memory = FundMemory()
        self._setup_handlers()
        self._setup_schedulers()

    def _setup_handlers(self):
        @self.client.on("connected")
        def _on_connected():
            logger.info("FundBot connected to WeChat Work WebSocket")

        @self.client.on("authenticated")
        def _on_auth():
            logger.info("FundBot authenticated successfully")

        @self.client.on("message.text")
        async def _on_text(frame):
            body = frame.get("body", {})
            content = body.get("text", {}).get("content", "").strip()
            user_id = body.get("from", {}).get("userid", "unknown")

            if not content:
                return

            logger.info("FundBot received from %s: %s", user_id, content[:60])
            parsed = parse_intent(content)
            intent = parsed["intent"]

            try:
                if intent == "add_holding":
                    await self._handle_add_holding(frame, user_id, parsed)
                elif intent == "remove_holding":
                    await self._handle_remove_holding(frame, user_id, parsed)
                elif intent == "portfolio_overview":
                    await self._handle_portfolio_overview(frame, user_id)
                elif intent == "fund_status":
                    await self._handle_fund_status(frame, user_id, parsed)
                elif intent == "fund_analyze":
                    await self._handle_fund_analyze(frame, user_id, parsed)
                else:
                    await self._handle_search(frame, user_id, parsed)
            except Exception:
                logger.exception("FundBot error handling message from %s", user_id)
                await self.client.reply(frame, {
                    "msgtype": "markdown",
                    "markdown": {"content": "⚠️ 处理消息时出错，请稍后再试。"},
                })

        @self.client.on("error")
        def _on_error(error):
            logger.error("FundBot client error: %s", error)

    async def _handle_add_holding(self, frame, user_id, parsed):
        fund_code = parsed["fund_code"]
        info = get_fund_info(fund_code)
        fund_name = info["name"] if info else fund_code
        result = add_holding(user_id, fund_code, fund_name, parsed["shares"], parsed["cost"])
        await self.client.reply(frame, {
            "msgtype": "markdown",
            "markdown": {"content": result["message"]},
        })

    async def _handle_remove_holding(self, frame, user_id, parsed):
        result = remove_holding(user_id, parsed["fund_code"])
        await self.client.reply(frame, {
            "msgtype": "markdown",
            "markdown": {"content": result["message"]},
        })

    async def _handle_portfolio_overview(self, frame, user_id):
        portfolio = get_portfolio(user_id)
        if not portfolio:
            await self.client.reply(frame, {
                "msgtype": "markdown",
                "markdown": {"content": "你的基金持仓为空。\n使用 `添加基金 代码 份额 成本价` 添加。"},
            })
            return

        lines = ["## 我的基金持仓\n", "| 基金代码 | 基金名称 | 持有份额 | 成本价 | 备注 |", "|---------|---------|---------|-------|------|"]
        total_cost = 0.0
        for h in portfolio:
            lines.append(f"| {h['fund_code']} | {h['fund_name'] or '-'} | {h['shares']} | {h['cost_price']} | {h['notes'] or ''} |")
            total_cost += h['shares'] * h['cost_price']
        lines.append(f"\n**总投入**: ¥{total_cost:,.2f}")
        lines.append("\n使用 `分析 基金代码` 进行深度分析。")

        await self.client.reply(frame, {
            "msgtype": "markdown",
            "markdown": {"content": "\n".join(lines)},
        })

    async def _handle_fund_status(self, frame, user_id, parsed):
        fund_code = parsed.get("fund_code", "")
        if not fund_code:
            await self.client.reply(frame, {
                "msgtype": "markdown",
                "markdown": {"content": "请提供基金代码，如 `110011`"},
            })
            return

        info = get_fund_info(fund_code)
        navs = get_fund_nav(fund_code, days=30)

        lines = [f"## {info['name'] if info else fund_code} ({fund_code})\n"]
        if info:
            lines.append(f"- 类型: {info.get('fund_type', '-')}")
            lines.append(f"- 规模: {info.get('fund_size', '-')}")
            lines.append(f"- 基金经理: {info.get('manager', '-')}")

        if navs:
            current = navs[0]
            lines.append(f"\n- 最新净值: {current['nav']:.4f} ({current['date']})")
            lines.append(f"- 累计净值: {current['total_nav']:.4f}")
            if len(navs) > 1:
                change = (current['nav'] - navs[-1]['nav']) / navs[-1]['nav'] * 100
                lines.append(f"- 30天涨跌: {change:+.2f}%")

        lines.append("\n使用 `分析 代码` 查看详细分析。")
        await self.client.reply(frame, {
            "msgtype": "markdown",
            "markdown": {"content": "\n".join(lines)},
        })

    async def _handle_fund_analyze(self, frame, user_id, parsed):
        query = parsed.get("query", "")
        if not query:
            await self.client.reply(frame, {
                "msgtype": "markdown",
                "markdown": {"content": "请指定要分析的基金，如 `分析 110011`"},
            })
            return

        fund_code, fund_name = _lookup_fund_code(query)
        logger.info("FundBot analyze: %s -> %s %s", query, fund_code, fund_name)

        # Phase B: resolve past decisions
        self.memory.resolve_pending(user_id, fund_code)

        # Phase C: get past context
        past_context = self.memory.get_past_context(user_id, fund_code)

        # Get user holding context
        holding = get_holding(user_id, fund_code) or {}

        # Build initial state
        today_str = date.today().isoformat()
        state = {
            "user_message": query,
            "user_id": user_id,
            "fund_code": fund_code,
            "fund_name": fund_name,
            "intent": "fund_analyze",
            "user_holding": holding,
            "portfolio_report": "",
            "holdings_report": "",
            "performance_report": "",
            "risk_report": "",
            "debate_state": {
                "bull_history": "",
                "bear_history": "",
                "history": "",
                "current_response": "",
                "count": 0,
            },
            "final_decision": "",
            "past_context": past_context,
        }

        # Invoke graph (with optional checkpointer)
        if FUND_CHECKPOINT_ENABLED:
            with get_checkpointer(user_id) as saver:
                result = await asyncio.to_thread(
                    self.graph.compile(checkpointer=saver).invoke,
                    state,
                    {"configurable": {"thread_id": f"fund_{user_id}_{fund_code}_{today_str}"}},
                )
                clear_checkpoint(user_id, fund_code, today_str)
        else:
            result = await asyncio.to_thread(self.graph.compile().invoke, state)

        response = result.get("final_decision", "分析完成，但无法生成建议。")

        # Phase A: store decision
        rating = "Hold"
        for line in response.split("\n"):
            if "评级" in line:
                for r in ["Strong Buy", "Buy", "Hold", "Reduce", "Sell"]:
                    if r in line:
                        rating = r
                        break

        navs = get_fund_nav(fund_code, days=1)
        current_nav = navs[0]["nav"] if navs else 0.0
        asyncio.create_task(asyncio.to_thread(
            self.memory.store_decision, user_id, fund_code, rating, response, current_nav
        ))

        await self.client.reply(frame, {
            "msgtype": "markdown",
            "markdown": {"content": response},
        })

    async def _handle_search(self, frame, user_id, parsed):
        query = parsed.get("query", "")
        results = search_fund(query)
        if not results:
            await self.client.reply(frame, {
                "msgtype": "markdown",
                "markdown": {"content": f"未找到与「{query}」相关的基金。"},
            })
            return
        lines = [f"## 搜索结果: {query}\n", "| 代码 | 名称 | 类型 |", "|------|------|------|"]
        for r in results[:10]:
            lines.append(f"| {r.get('基金代码', '-')} | {r.get('基金简称', '-')} | {r.get('基金类型', '-')} |")
        lines.append("\n使用 `分析 代码` 查看详细分析。")
        await self.client.reply(frame, {
            "msgtype": "markdown",
            "markdown": {"content": "\n".join(lines)},
        })

    def _setup_schedulers(self):
        self.scheduler.add_job(
            self._run_weekly_review,
            "cron", day_of_week="sat", hour=10, minute=0,
            id="fund_weekly_review",
            misfire_grace_time=300,
        )
        logger.info("FundBot weekly review scheduler started (Sat 10:00)")

    async def _run_weekly_review(self):
        """Generate and push weekly portfolio review to all users with holdings."""
        try:
            conn = get_connection()
            try:
                user_ids = conn.execute(
                    "SELECT DISTINCT user_id FROM user_portfolio"
                ).fetchall()
            finally:
                conn.close()

            for (uid,) in user_ids:
                portfolio = get_portfolio(uid)
                if not portfolio:
                    continue

                lines = ["## 本周组合回顾\n", "| 基金 | 持有份额 | 成本价 | 备注 |", "|------|---------|-------|------|"]
                total_value = 0.0
                for h in portfolio:
                    navs = get_fund_nav(h["fund_code"], days=1)
                    current_nav = navs[0]["nav"] if navs else h["cost_price"]
                    value = h["shares"] * current_nav
                    total_value += value
                    lines.append(f"| {h['fund_code']} {h['fund_name'] or ''} | {h['shares']} | {h['cost_price']} | ¥{value:,.2f} |")

                lines.append(f"\n**组合总市值**: ¥{total_value:,.2f}")
                lines.append("\n使用 `分析 基金代码` 获取详细建议。")

                await self.client.send_message(uid, {
                    "msgtype": "markdown",
                    "markdown": {"content": "\n".join(lines)},
                })
                logger.info("FundBot weekly review sent to %s (%d funds)", uid, len(portfolio))
        except Exception:
            logger.exception("FundBot weekly review failed")

    def run(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self.client.connect())
            loop.call_soon(self.scheduler.start)
            loop.run_forever()
        except KeyboardInterrupt:
            self.client.disconnect()
        finally:
            self.scheduler.shutdown(wait=False)
            loop.close()
