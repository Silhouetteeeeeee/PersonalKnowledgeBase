import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urljoin

import requests
import trafilatura
from tavily import TavilyClient

from server.config import TAVILY_API_KEY, IMG_READER_MODEL
from agent.models.value_objects import UrlContent
from agent.utils.openaillm import OpenAILLM

logger = logging.getLogger(__name__)


# ── Content Cleaning ─────────────────────────────────────────────────────────


def clean_content(raw: str) -> str:
    """清洗网页内容：移除图片引用、空标题行、分隔线、压缩空行。"""
    # 移除 ![alt](url) 图片引用
    text = re.sub(r'!\[.*?\]\(.*?\)', '', raw)
    # 移除仅有 # ## ### 等无文字的标题行
    text = re.sub(r'^#{1,6}\s*$', '', text, flags=re.MULTILINE)
    # 移除连续分隔线 --- 或 ___
    text = re.sub(r'^[-]{3,}\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^[_]{3,}\s*$', '', text, flags=re.MULTILINE)
    # 压缩连续空行为单行
    text = re.sub(r'\n{3,}', '\n\n', text)
    # 去除首尾空白
    return text.strip()


# ── Single URL Fetch ─────────────────────────────────────────────────────────


def _extract_img_urls(html: str, base_url: str) -> list[str]:
    """从 HTML 中提取有意义的图片 URL。

    过滤规则：
    - 跳过 data:base64 内嵌图片
    - 跳过 SVG 图标
    - 跳过常见图标/Logo 文件名（favicon, logo, icon 等）
    - 相对路径转为绝对路径

    Returns:
        最多 5 张图片的 URL 列表
    """
    img_urls = []
    # 匹配 <img> 标签，兼容单引号/双引号/无引号
    for match in re.finditer(
        r'<img[^>]+src\s*=\s*["\']([^"\']+)["\']',
        html, re.IGNORECASE
    ):
        src = match.group(1).strip()
        # 跳过 base64 内嵌图片
        if src.startswith("data:"):
            continue
        # 跳过极小图标（文件名仅为 "icon"、"arrow" 等无意义内容）
        skip_keywords = ["favicon", "spacer", "pixel", "thumbnail"]
        src_lower = src.lower()
        if any(kw in src_lower for kw in skip_keywords):
            continue
        # 相对路径 → 绝对路径
        if src.startswith("//"):
            src = "https:" + src
        elif src.startswith("/"):
            from urllib.parse import urlparse
            parsed = urlparse(base_url)
            src = f"{parsed.scheme}://{parsed.netloc}{src}"
        elif not src.startswith(("http://", "https://")):
            src = urljoin(base_url, src)
        img_urls.append(src)
        if len(img_urls) >= 5:  # 最多 5 张
            break
    return img_urls


def _describe_page_images(html: str, base_url: str) -> list[str]:
    """提取网页中的图片 → 使用 VLM 逐张理解 → 返回描述列表。

    单张图片理解失败不影响其他图片，全部失败则返回空列表。
    """
    img_urls = _extract_img_urls(html, base_url)
    if not img_urls:
        return []
    if not IMG_READER_MODEL:
        logger.debug("IMG_READER_MODEL 未配置，跳过图片理解")
        return []

    logger.info("提取到 %d 张图片，开始 VLM 理解...", len(img_urls))
    descriptions = []
    for url in img_urls:
        desc = OpenAILLM.getImgContent(IMG_READER_MODEL, url)
        if desc and desc != "图像理解失败":
            descriptions.append(desc)
            logger.info("图片描述: %s", desc[:60])
    logger.info(
        "图片理解完成: %d/%d 张成功",
        len(descriptions), len(img_urls),
    )
    return descriptions


# 常见图片文件扩展名（用于直接图片 URL 检测）
_IMAGE_EXT = re.compile(r'\.(jpe?g|png|gif|webp|bmp|avif)(\?|#|$)', re.IGNORECASE)


def fetch_url_text(url: str) -> UrlContent:
    """下载单个 URL → 提取正文/理解图片 → 清洗 → 返回结构化结果。

    根据 URL 类型自动选择策略：
    - 网页 URL → trafilatura 提取 → tavily 后备 → 提取<img>并 VLM 理解
    - 图片 URL → 直接 VLM 理解，不尝试网页抓取
    单 URL 超时 30 秒。
    """
    result = UrlContent(url=url)

    # ── 直接图片 URL：跳过网页抓取，直接 VLM 理解 ──
    if _IMAGE_EXT.search(url) and IMG_READER_MODEL:
        logger.info("检测到直接图片 URL，调用 VLM 理解: %s", url[:60])
        description = OpenAILLM.getImgContent(IMG_READER_MODEL, url)
        if description and description != "图像理解失败":
            result.content = f"[图片内容] {description}"
            # 尝试从 URL 提取文件名作为标题
            filename = url.rstrip("/").split("/")[-1].split("?")[0]
            result.title = filename
            logger.info("图片理解成功: %s", description[:60])
            return result
        else:
            result.content = "[抓取失败]"
            return result

    raw_html: str | None = None  # 保存原始 HTML，用于图片提取
    try:
        # 尝试提取 title + 保存原始 HTML
        try:
            resp = requests.get(url, timeout=30, headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                )
            })
            if resp.status_code == 200:
                raw_html = resp.text  # 保存用于图片提取
                m = re.search(r'<title>(.*?)</title>', resp.text, re.IGNORECASE | re.DOTALL)
                if m:
                    title = m.group(1).strip()
                    if title:
                        result.title = title[:200]

                # trafilatura 提取正文
                downloaded = trafilatura.fetch_url(url)
                if downloaded:
                    text = trafilatura.extract(downloaded)
                    if text and len(text) >= 200:
                        result.content = clean_content(text)
                        # ── 图片理解 ──
                        if raw_html:
                            descriptions = _describe_page_images(raw_html, url)
                            if descriptions:
                                img_section = "\n\n---\n### 网页图片描述\n" + "\n".join(
                                    f"- {d}" for d in descriptions
                                )
                                result.content += img_section
                        return result
        except Exception as e:
            logger.warning("Trafilatura extraction failed for %s: %s", url, e)

        # 后备：tavily extract
        if TAVILY_API_KEY:
            try:
                client = TavilyClient(api_key=TAVILY_API_KEY)
                tavily_result = client.extract(url)
                if tavily_result and tavily_result.get("results"):
                    content = tavily_result["results"][0].get("raw_content", "")
                    if content:
                        result.title = tavily_result["results"][0].get("title", "")
                        result.content = clean_content(content)
                        # ── 图片理解（使用之前 requests 获取的原始 HTML）──
                        if raw_html:
                            descriptions = _describe_page_images(raw_html, url)
                            if descriptions:
                                img_section = "\n\n---\n### 网页图片描述\n" + "\n".join(
                                    f"- {d}" for d in descriptions
                                )
                                result.content += img_section
                        return result
            except Exception as e:
                logger.warning("Tavily extraction failed for %s: %s", url, e)

        # 全部失败
        result.content = "[抓取失败]"
        return result

    except Exception as e:
        logger.error("Unexpected error fetching %s: %s", url, e)
        result.content = "[抓取失败]"
        return result


# ── Concurrent URL Fetch ─────────────────────────────────────────────────────


def fetch_urls_concurrent(urls: list[str]) -> list[UrlContent]:
    """多线程并发抓取，max_workers=5。单条失败不影响整体。"""
    if not urls:
        return []

    # 去重，保持顺序
    seen = set()
    unique_urls = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            unique_urls.append(u)

    results = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(fetch_url_text, url): url for url in unique_urls}
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                url = futures[future]
                logger.error("Concurrent fetch failed for %s: %s", url, e)
                results.append(UrlContent(url=url, content="[抓取失败]"))

    # 按原始顺序排序
    order = {url: i for i, url in enumerate(unique_urls)}
    results.sort(key=lambda r: order.get(r.url, 999))
    return results
