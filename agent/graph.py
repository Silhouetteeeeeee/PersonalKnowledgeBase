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
from agent.nodes.correct_knowledge import correct_knowledge
from agent.nodes.record_error import record_error
from agent.nodes.store import store
from agent.nodes.respond import respond
from agent.nodes.update_profile import update_profile

logger = logging.getLogger(__name__)

MAX_CORRECTION_ATTEMPTS = 2



def fact_check_router(state: dict) -> str:
    if state.get("contradiction_found") and state.get("contradiction_details"):
        logger.info("Router: contradiction detected → reflect")
        return "reflect"
    logger.info("Router: no contradiction → store")
    return "store"


def reflect_router(state: dict) -> str:
    attempts = state.get("correction_attempts", 0)
    if attempts >= MAX_CORRECTION_ATTEMPTS:
        logger.info("Router: max correction attempts (%d) reached → respond", MAX_CORRECTION_ATTEMPTS)
        return "respond"

    result = state.get("reflection_result", "unresolved")

    if result == "stored_knowledge_wrong":
        logger.info("Router: stored knowledge wrong → correct_knowledge")
        return "correct_knowledge"
    elif result == "answer_wrong":
        logger.info("Router: answer wrong → record_error")
        return "record_error"
    else:
        logger.info("Router: unresolved → respond")
        return "respond"


def post_correction_router(state: dict) -> str:
    if state.get("force_web_search"):
        logger.info("Router: correction done, verifying via web search")
        return "search_web"
    logger.info("Router: correction done, re-running fact check")
    return "fact_check"


def post_error_router(state: dict) -> str:
    logger.info("Router: error recorded, fetching web data for correction")
    return "search_web"


def build_graph() -> StateGraph:
    """
        入口: parse - 解析用户输入
        顺序执行: parse → rewrite_query → retrieve → classify_and_answer
        classify_and_answer 后直接到 fact_check（搜索在节点内部完成）
        fact_check 后如果发现矛盾 → reflect → 修正循环
        否则 → store → respond
        修正循环最多 2 次，之后到 respond
        :return: StateGraph 状态图
    """
    builder = StateGraph(AgentState)

    # Core nodes
    builder.add_node("parse", parse)
    builder.add_node("rewrite_query", rewrite_query)
    builder.add_node("retrieve", retrieve)
    builder.add_node("classify_and_answer", classify_and_answer)
    builder.add_node("fact_check", fact_check)
    builder.add_node("search_web", search_web_node)
    builder.add_node("regenerate", regenerate)
    builder.add_node("store", store)
    builder.add_node("respond", respond)

    # Reflection nodes
    builder.add_node("reflect", reflect)
    builder.add_node("correct_knowledge", correct_knowledge)
    builder.add_node("record_error", record_error)
    builder.add_node("update_profile", update_profile)

    builder.set_entry_point("parse")

    # parse → rewrite_query → retrieve → classify_and_answer
    builder.add_edge("parse", "rewrite_query")
    builder.add_edge("rewrite_query", "retrieve")
    builder.add_edge("retrieve", "classify_and_answer")

    # classify_and_answer → [update_profile, fact_check] (parallel fan-out)
    builder.add_edge("classify_and_answer", "update_profile")
    builder.add_edge("classify_and_answer", "fact_check")
    builder.add_edge("search_web", "regenerate")
    builder.add_edge("regenerate", "fact_check")

    # fact_check → [reflect | store]
    builder.add_conditional_edges(
        "fact_check",
        fact_check_router,
        {"reflect": "reflect", "store": "store"},
    )

    # reflect → [correct_knowledge | record_error | respond]
    builder.add_conditional_edges(
        "reflect",
        reflect_router,
        {
            "correct_knowledge": "correct_knowledge",
            "record_error": "record_error",
            "respond": "respond",
        },
    )

    # correct_knowledge → [search_web | fact_check]
    builder.add_conditional_edges(
        "correct_knowledge",
        post_correction_router,
        {"search_web": "search_web", "fact_check": "fact_check"},
    )

    # record_error → search_web (always force search for correction)
    builder.add_edge("record_error", "search_web")

    # store → respond
    builder.add_edge("store", "respond")

    compiled = builder.compile()

    logger.info(
        "Graph built: parse → rewrite_query → retrieve → classify_and_answer → "
        "fact_check → "
        "[reflect→correct_knowledge/record_error|store] → respond"
    )

    return compiled
