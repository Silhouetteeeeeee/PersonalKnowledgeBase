"""
网络搜索节点：当 contradiction 循环或 force_web_search 触发时执行。

搜索策略：
  1. 优先尝试百度搜索 API（配置了 BAIDU_API_KEY 时）
  2. 百度失败则回退到通用搜索（duckduckgo）
  3. 全部失败则返回空结果 → regenerate 节点会处理
"""

import logging

from agent.tools.web_search import *
from agent.models.nodes import SearchWebResult
from agent.models.value_objects import LogicChainStep

logger = logging.getLogger(__name__)


def search_web_node(state: dict) -> dict:
    """
    网络搜索节点：从 config 配置的搜索引擎获取实时信息。

    入口来源：
    1. contradiction 循环中的 record_error → search_web
    2. Graph 路由中主动触发的 force_web_search
    """
    query = state["user_message"]
    logger.info("网络搜索查询: '%s'", query[:60])
    source = "百度"
    try:
        results = search_web_from_baidu(query)
    except Exception as e:
        logger.info(f"百度搜索异常: {e}")
        source = "通用"
        results = search_web(query)
        if not results:
            return SearchWebResult(search_results=[], logic_chain=[LogicChainStep(
                node="search_web",
                action="网络搜索失败",
                reasoning=f"百度搜索异常（{e}），{source}搜索也未返回结果",
            )]).model_dump()
    logger.info("搜索返回 %d 条结果", len(results))
    return SearchWebResult(search_results=results, logic_chain=[LogicChainStep(
        node="search_web",
        action=f"网络搜索：{query[:40]}",
        reasoning=f"{source}搜索返回 {len(results)} 条结果",
    )]).model_dump()
