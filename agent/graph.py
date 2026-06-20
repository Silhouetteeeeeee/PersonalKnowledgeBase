import logging

from langgraph.graph import StateGraph
from agent.state import AgentState
from agent.nodes.parse import parse
from agent.nodes.retrieve import retrieve
from agent.nodes.rewrite_query import rewrite_query
from agent.nodes.classify_and_answer import classify_and_answer
from agent.nodes.fact_check import fact_check
from agent.nodes.search_web import search_web_node
from agent.nodes.regenerate import regenerate
from agent.nodes.reflect import reflect
from agent.nodes.record_error import record_error
from agent.nodes.respond import respond
from agent.nodes.update_profile import update_profile

logger = logging.getLogger(__name__)

# 矛盾修正循环的最大次数（超过则强制输出）
MAX_CORRECTION_ATTEMPTS = 2


def fact_check_router(state: dict) -> str:
    """
    矛盾检测路由：判断是否进入矛盾处理分支。

    返回:
      - "reflect": 检测到矛盾，进入反思环节
      - "respond": 无矛盾，直接输出回答
    """
    if state.get("contradiction_found") and state.get("contradiction_details"):
        logger.info("路由: 检测到矛盾 → reflect")
        return "reflect"
    logger.info("路由: 无矛盾 → respond")
    return "respond"


def reflect_router(state: dict) -> str:
    """
    反思结果路由：根据反思结论决定下一步。

    返回:
      - "record_error": 知识或回答有误 → 记录错误 → 搜索纠正
      - "respond": 无法确定或已达上限 → 直接输出
    """
    attempts = state.get("correction_attempts", 0)
    if attempts >= MAX_CORRECTION_ATTEMPTS:
        logger.info("路由: 已达最大修正次数（%d）→ respond", MAX_CORRECTION_ATTEMPTS)
        return "respond"

    result = state.get("reflection_result", "unresolved")

    if result == "stored_knowledge_wrong":
        logger.info("路由: 已有知识错误 → record_error")
        return "record_error"
    elif result == "answer_wrong":
        logger.info("路由: 回答错误 → record_error")
        return "record_error"
    else:
        logger.info("路由: 无法确定 → respond")
        return "respond"


def post_error_router(state: dict) -> str:
    """错误记录后的路由：直接触发网络搜索以获取正确信息"""
    logger.info("路由: 错误已记录，开始网络搜索")
    return "search_web"


def build_graph() -> StateGraph:
    """
    构建 LangGraph StateGraph。

    执行流程:
      入口 → parse → rewrite_query → retrieve → classify_and_answer
      ├─→ update_profile（并行，不阻塞主流程）
      └─→ fact_check → [reflect → record_error → search_web → regenerate → fact_check（循环） | respond]

    矛盾修正循环最多 2 次，超过则强制输出。
    """
    builder = StateGraph(AgentState)

    # ── 核心处理节点 ──
    builder.add_node("parse", parse)
    builder.add_node("rewrite_query", rewrite_query)
    builder.add_node("retrieve", retrieve)
    builder.add_node("classify_and_answer", classify_and_answer)
    builder.add_node("fact_check", fact_check)
    builder.add_node("search_web", search_web_node)
    builder.add_node("regenerate", regenerate)
    builder.add_node("respond", respond)

    # ── 矛盾处理节点 ──
    builder.add_node("reflect", reflect)
    builder.add_node("record_error", record_error)
    builder.add_node("update_profile", update_profile)

    # ── 入口 ──
    builder.set_entry_point("parse")

    # ── 正向链路 ──
    builder.add_edge("parse", "rewrite_query")
    builder.add_edge("rewrite_query", "retrieve")
    builder.add_edge("retrieve", "classify_and_answer")

    # classify_and_answer → [update_profile, fact_check] 并行 fan-out
    builder.add_edge("classify_and_answer", "update_profile")
    builder.add_edge("classify_and_answer", "fact_check")

    # 搜索 → 重生成 → 再次事实核查
    builder.add_edge("search_web", "regenerate")
    builder.add_edge("regenerate", "fact_check")

    # ── 条件路由 ──
    # fact_check → [reflect | respond]
    builder.add_conditional_edges(
        "fact_check",
        fact_check_router,
        {"reflect": "reflect", "respond": "respond"},
    )

    # reflect → [record_error | respond]
    builder.add_conditional_edges(
        "reflect",
        reflect_router,
        {
            "record_error": "record_error",
            "respond": "respond",
        },
    )

    # record_error → search_web（强制搜索修正）
    builder.add_edge("record_error", "search_web")

    # ── 编译 ──
    compiled = builder.compile()

    logger.info(
        "Graph 构建完成：parse → rewrite_query → retrieve → classify_and_answer → "
        "fact_check → [reflect→record_error→search_web|respond]"
    )

    return compiled
