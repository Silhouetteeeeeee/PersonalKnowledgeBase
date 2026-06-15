"""
FundBot fund_analyze demo — 执行单只基金的完整多智能体分析。

Usage:
    python fund/demo_analyze.py
"""

import asyncio
import logging
from datetime import date

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

from fund.graph import build_fund_graph
from fund.state import FundAgentState, FundDebateState
from fund.utils.portfolio_tools import get_holding
from fund.utils.fund_data_tools import get_fund_nav
from fund.utils.memory import FundMemory
from fund.intent.classifier import classify
from fund.intent.schemas import INTENTS
from agent.utils.llm import LLM


async def main():
    user_id = "demo_user"
    fund_code = "023639"

    # ── Step 1: Intent Classification ──
    print("=" * 60)
    print("Step 1: Intent Classification")
    print("-" * 60)

    messages = [
        "分析 023639",
        "帮我看看国泰电力设备怎么样",
    ]

    for msg in messages:
        result = classify(msg, INTENTS)
        if result:
            print(f"  [{msg}] -> id={result.id} score={result.score:.2f} params={result.params}")
        else:
            print(f"  [{msg}] -> no intent matched")

    # ── Step 2: Phase B — resolve past decisions (none for first run) ──
    print("\nStep 2: Phase B — Resolve past decisions")
    print("-" * 60)
    memory = FundMemory()
    memory.resolve_pending(user_id, fund_code)
    print("  Done.")

    # ── Step 3: Phase C — Load past context ──
    print("\nStep 3: Phase C — Load decision history")
    print("-" * 60)
    past_context = memory.get_past_context(user_id, fund_code)
    print(f"  Past context: {past_context or '(empty)'}")

    # ── Step 4: Load user holding ──
    print("\nStep 4: Load user holding")
    print("-" * 60)
    holding = get_holding(user_id, fund_code) or {}
    if holding:
        shares = holding.get("shares", 0)
        cost = holding.get("cost_price", 0)
        navs = get_fund_nav(fund_code, days=1)
        current_nav = navs[0]["nav"] if navs else 0
        print(f"  持有: {shares} 份 @ {cost}")
        print(f"  最新净值: {current_nav}")
        print(f"  浮动盈亏: {(current_nav - cost) * shares:+.2f}")
    else:
        print("  无持仓数据")

    # ── Step 5: Build state and run graph ──
    print("\nStep 5: Running multi-agent pipeline...")
    print("=" * 60)

    today_str = date.today().isoformat()
    state: FundAgentState = {
        "user_message": f"分析 {fund_code}",
        "user_id": user_id,
        "fund_code": fund_code,
        "fund_name": holding.get("fund_name", fund_code),
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

    graph = build_fund_graph()
    compiled = graph.compile()

    print("  Invoking agents (4 analysts → bull/bear debate → portfolio manager)...")
    result = await asyncio.to_thread(compiled.invoke, state)

    # ── Step 6: Print result ──
    print("\n" + "=" * 60)
    print("Step 6: Portfolio Manager Decision")
    print("=" * 60)
    print(result.get("final_decision", "No decision generated."))

    # ── Step 7: Phase A — Store decision ──
    print("\n" + "=" * 60)
    print("Step 7: Phase A — Store decision for future reflection")
    print("-" * 60)

    response = result.get("final_decision", "")
    rating = "Hold"
    for line in response.split("\n"):
        if "评级" in line:
            for r in ["Strong Buy", "Buy", "Hold", "Reduce", "Sell"]:
                if r in line:
                    rating = r
                    break

    navs = get_fund_nav(fund_code, days=1)
    current_nav = navs[0]["nav"] if navs else 0.0
    memory.store_decision(user_id, fund_code, rating, response, current_nav)
    print(f"  评级: {rating}")
    print(f"  净值: {current_nav}")
    print("  Done.")


if __name__ == "__main__":
    asyncio.run(main())
