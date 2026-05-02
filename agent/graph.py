from langgraph.graph import StateGraph
from agent.state import AgentState
from agent.nodes.parse import parse
from agent.nodes.retrieve import retrieve
from agent.nodes.classify_and_answer import classify_and_answer
from agent.nodes.search_web import search_web_node
from agent.nodes.regenerate import regenerate
from agent.nodes.store import store
from agent.nodes.respond import respond


def needs_search_router(state: dict) -> str:
    if state.get("needs_search"):
        return "search_web"
    return "store"


def build_graph() -> StateGraph:
    """
        入口: parse - 解析用户输入
        顺序执行: parse → retrieve → classify_and_answer
        条件分支: 在 classify_and_answer 后根据 needs_search 判断：
        如果需要搜索 → search_web → regenerate → store
        如果不需要搜索 → 直接到 store
        最终输出: store → respond → 结束
        :return: StateGraph 状态图
    """
    builder = StateGraph(AgentState)

    builder.add_node("parse", parse)
    builder.add_node("retrieve", retrieve)
    builder.add_node("classify_and_answer", classify_and_answer)
    builder.add_node("search_web", search_web_node)
    builder.add_node("regenerate", regenerate)
    builder.add_node("store", store)
    builder.add_node("respond", respond)

    builder.set_entry_point("parse")
    builder.add_edge("parse", "retrieve")
    builder.add_edge("retrieve", "classify_and_answer")
    builder.add_conditional_edges(
        "classify_and_answer",
        needs_search_router,
        {"search_web": "search_web", "store": "store"},
    )
    builder.add_edge("search_web", "regenerate")
    builder.add_edge("regenerate", "store")
    builder.add_edge("store", "respond")

    return builder.compile()
