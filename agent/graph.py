import logging

from langgraph.graph import StateGraph
from agent.state import AgentState
from agent.nodes.parse import parse
from agent.nodes.retrieve import retrieve
from agent.nodes.classify_and_answer import classify_and_answer
from agent.nodes.fact_check import fact_check
from agent.nodes.search_web import search_web_node
from agent.nodes.regenerate import regenerate
from agent.nodes.store import store
from agent.nodes.respond import respond

logger = logging.getLogger(__name__)


def needs_search_router(state: dict) -> str:
    if state.get("needs_search"):
        logger.info("Router: needs_search=True → search_web branch")
        return "search_web"
    logger.info("Router: needs_search=False → store branch")
    return "store"


def build_graph() -> StateGraph:
    """
        入口: parse - 解析用户输入
        顺序执行: parse → retrieve → classify_and_answer
        条件分支: classify_and_answer 后根据 needs_search 判断：
        如果需要搜索 → search_web → regenerate
        如果不需要搜索 → 直接到 fact_check
        汇聚: 两条路径都到 fact_check → store → respond
        :return: StateGraph 状态图
    """
    builder = StateGraph(AgentState)

    builder.add_node("parse", parse)
    builder.add_node("retrieve", retrieve)
    builder.add_node("classify_and_answer", classify_and_answer)
    builder.add_node("fact_check", fact_check)
    builder.add_node("search_web", search_web_node)
    builder.add_node("regenerate", regenerate)
    builder.add_node("store", store)
    builder.add_node("respond", respond)

    builder.set_entry_point("parse")
    builder.add_edge("parse", "retrieve")
    builder.add_edge("retrieve", "classify_and_answer")

    # 条件路由：是否需要搜索
    builder.add_conditional_edges(
        "classify_and_answer",
        needs_search_router,
        {"search_web": "search_web", "store": "fact_check"},
    )

    # 搜索路径
    builder.add_edge("search_web", "regenerate")
    builder.add_edge("regenerate", "fact_check")

    # 公共下游
    builder.add_edge("fact_check", "store")
    builder.add_edge("store", "respond")

    compiled = builder.compile()

    logger.info("Graph built: parse → retrieve → classify_and_answer → [search_web→regenerate|direct] → fact_check → store → respond")

    return compiled
