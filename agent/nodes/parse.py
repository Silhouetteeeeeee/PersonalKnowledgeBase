import logging
import re

from server.url_processor import fetch_urls_concurrent
from agent.utils.agent_utils import build_url_context
from agent.models.nodes import ParseResult
from agent.models.value_objects import LogicChainStep, UrlContent

logger = logging.getLogger(__name__)

_URL_PATTERN = re.compile(r'https?://[^\s]+')


def parse(state: dict) -> dict:
    user_message = state["user_message"].strip()
    logger.info("Parsed message from %s: '%s'", state.get("user_id", "unknown"), user_message[:60])

    # URL 提取 + 并发抓取
    urls = _URL_PATTERN.findall(user_message)
    url_contents: list[UrlContent] = []
    logic_chain: list[LogicChainStep] = []
    if urls:
        url_contents = fetch_urls_concurrent(urls)
        logger.info("Fetched %d URLs from message", len(url_contents))
        logic_chain = [LogicChainStep(
            node="parse",
            action=f"提取 {len(urls)} 个 URL",
            reasoning=f"从消息中提取了 {len(urls)} 个 URL，成功获取 {len(url_contents)} 个\n"
                     f"{build_url_context(url_contents, 200)}",
        )]

    result = ParseResult(
        user_message=user_message,
        user_id=state.get("user_id", "unknown"),
        timestamp=state.get("timestamp", ""),
        url_contents=url_contents,
        logic_chain=logic_chain,
    ).model_dump()
    # Keep UrlContent as objects so downstream consumers can use attribute access
    result["url_contents"] = url_contents
    return result
