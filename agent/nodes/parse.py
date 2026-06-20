import logging
import re

from server.url_processor import fetch_urls_concurrent
from agent.utils.agent_utils import build_url_context
from agent.models.nodes import ParseResult
from agent.models.value_objects import LogicChainStep, UrlContent
from server.config import IMG_READER_MODEL

logger = logging.getLogger(__name__)

_URL_PATTERN = re.compile(r'https?://[^\s]+')
# 图片扩展名检测（用于识别直接图片 URL）
_IMG_EXT_PATTERN = re.compile(r'\.(jpe?g|png|gif|webp|bmp|avif)(\?|#|$)', re.IGNORECASE)


def parse(state: dict) -> dict:
    """解析节点：提取用户消息中的 URL，并发抓取网页/理解图片。

    URL 类型自动识别：
    - 网页 URL → 提取正文 + <img> VLM 理解
    - 图片 URL → 直接 VLM 理解
    - 无 URL → 直接传递给下一节点
    """
    user_message = state["user_message"].strip()
    logger.info("解析消息 from %s: '%s'", state.get("user_id", "unknown"), user_message[:60])

    urls = _URL_PATTERN.findall(user_message)
    url_contents: list[UrlContent] = []
    logic_chain: list[LogicChainStep] = []

    if urls:
        # 区分图片 URL 和网页 URL（仅用于日志统计）
        img_urls = [u for u in urls if _IMG_EXT_PATTERN.search(u)]
        web_urls = [u for u in urls if not _IMG_EXT_PATTERN.search(u)]

        url_contents = fetch_urls_concurrent(urls)
        logger.info(
            "获取了 %d 个 URL 内容（%d 个图片, %d 个网页）",
            len(url_contents), len(img_urls), len(web_urls),
        )

        # 统计成功/失败的图片理解
        img_ok = sum(
            1 for uc in url_contents
            if uc.url in img_urls and uc.content and uc.content != "[抓取失败]"
        )
        img_fail = len(img_urls) - img_ok

        reasoning_parts = [
            f"从消息中提取了 {len(urls)} 个 URL",
            f"网页 {len(web_urls)} 个",
        ]
        if img_urls and IMG_READER_MODEL:
            reasoning_parts.append(f"图片 {len(img_urls)} 个（成功 {img_ok}，失败 {img_fail}）")
        if img_urls and not IMG_READER_MODEL:
            reasoning_parts.append("（IMG_READER_MODEL 未配置，跳过图片理解）")

        logic_chain = [LogicChainStep(
            node="parse",
            action=f"提取 {len(urls)} 个 URL（{len(web_urls)} 网页, {len(img_urls)} 图片）",
            reasoning="，".join(reasoning_parts) + "\n"
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
