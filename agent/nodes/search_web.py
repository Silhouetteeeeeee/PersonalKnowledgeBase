import logging

from agent.tools.web_search import *
from agent.models.nodes import SearchWebResult
from agent.models.value_objects import LogicChainStep

logger = logging.getLogger(__name__)


def search_web_node(state: dict) -> dict:
    query = state["user_message"]
    logger.info("Searching web for: '%s'", query[:60])
    source = "百度"
    try:
        results = search_web_from_baidu(query)
    except Exception as e:
        logger.info(f"Web search from Baidu error:{e}")
        source = "通用"
        results = search_web(query)
        if not results:
            return SearchWebResult(search_results=[], logic_chain=[LogicChainStep(
                node="search_web",
                action="网络搜索失败",
                reasoning=f"百度搜索异常（{e}），{source}搜索也未返回结果",
            )]).model_dump()
    logger.info("Web search returned %d results", len(results))
    return SearchWebResult(search_results=results, logic_chain=[LogicChainStep(
        node="search_web",
        action=f"网络搜索：{query[:40]}",
        reasoning=f"{source}搜索返回 {len(results)} 条结果",
    )]).model_dump()
