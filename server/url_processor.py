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
    text = re.sub(r'!\[.*?\]\(.*?\)', '', raw)
    text = re.sub(r'^#{1,6}\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^[-]{3,}\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^[_]{3,}\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# ── 图片过滤器 ────────────────────────────────────────────────────────────────

# 常见广告 CDN 域名（URL 中包含则直接跳过）
_AD_DOMAINS = [
    "doubleclick.net", "googleadservices", "googleads", "googlesyndication",
    "amazon-adsystem", "adnxs.com", "casalemedia.com",
    "criteo.com", "outbrain", "taboola",
    "baidustatic.com", "pos.baidu.com", "cpro.baidustatic.com",
]

# 图片类名/ID 中的无用关键词
_SKIP_CLASS_KEYWORDS = [
    "avatar", "gravatar", "photo-circle", "user-photo", "profile-photo",
    "emoji", "icon", "symbol", "sprite",
    "ad", "advert", "banner", "sponsor", "promotion", "propaganda",
    "share", "social-share", "weibo-share", "wx-share",
    "qrcode", "watermark", "water-mark",
    "bg-", "background", "divider",
    "tracking", "beacon", "pixel", "spacer", "blank",
    "loading", "placeholder", "thumbnail", "thumb-",
    "btn-", "button", "nav-", "menu-", "sidebar-", "footer-",
]

# 图片 class/id 中的保留关键词（明确是正文配图）
_KEEP_CLASS_KEYWORDS = [
    "content-img", "article-img", "post-img", "aligncenter",
    "wp-image", "size-full", "size-large",
    "gallery", "carousel",
]


def _is_useless_image(tag_html: str, src_url: str) -> bool:
    """判断一张图片是否是无用图片（头像/广告/图标/装饰等）。

    Args:
        tag_html: 完整的 <img ...> 标签 HTML
        src_url: 已规范化的图片 URL

    Returns:
        True = 无用，应跳过；False = 可能是正文配图
    """
    tag_lower = tag_html.lower()

    # ── 1. URL 级别过滤 ──
    # 1a. 广告 CDN
    for domain in _AD_DOMAINS:
        if domain in src_url.lower():
            return True
    # 1b. 已知无用文件名
    skip_url_keywords = ["favicon", "pixel", "beacon", "spacer", "tracking",
                         "avatar", "gravatar", "default-avatar"]
    if any(kw in src_url.lower() for kw in skip_url_keywords):
        return True

    # ── 2. class/id 级别过滤 ──
    # 2a. 先检查是否明确是正文配图（保留类），有则直接保留
    class_match = re.search(r'''class\s*=\s*["']([^"']+)["']''', tag_lower)
    id_match = re.search(r'''id\s*=\s*["']([^"']+)["']''', tag_lower)
    combined = (class_match.group(1) if class_match else "") + " " + \
               (id_match.group(1) if id_match else "")
    for kw in _KEEP_CLASS_KEYWORDS:
        if kw in combined:
            return False
    # 2b. 检查无用关键词
    for kw in _SKIP_CLASS_KEYWORDS:
        if kw in combined:
            return True

    # ── 3. 属性级别过滤 ──
    # 3a. 明确是装饰性图片
    if 'aria-hidden="true"' in tag_lower or 'role="presentation"' in tag_lower:
        return True
    # 3b. alt 文本包含广告/装饰关键词
    alt_match = re.search(r'''alt\s*=\s*["']([^"']*)["']''', tag_lower)
    if alt_match:
        alt_text = alt_match.group(1).lower()
        alt_skip = ["广告", "ad", "sponsor", "推广", "二维码", "头像",
                    "装饰", "背景图", "分割线", "图标"]
        if any(kw in alt_text for kw in alt_skip):
            return True
    # 3c. 宽高明确很小（< 100px） → 大概率是图标/头像
    width = _parse_dimension(tag_lower, "width")
    height = _parse_dimension(tag_lower, "height")
    if (width is not None and width < 100) or (height is not None and height < 100):
        return True

    # ── 4. 默认：可能是有用图片 ──
    return False


def _parse_dimension(tag_lower: str, attr: str) -> int | None:
    """从 <img> 标签中解析 width/height 属性值（支持 px 和纯数字）。"""
    m = re.search(rf'{attr}\s*=\s*["\'](\d+)["\']', tag_lower)
    if m:
        return int(m.group(1))
    # style="width: 200px; height: 100px"
    m2 = re.search(rf'{attr}\s*:\s*(\d+)', tag_lower)
    if m2:
        return int(m2.group(1))
    return None


# 常见图片文件扩展名（用于直接图片 URL 检测）
_IMAGE_EXT = re.compile(r'\.(jpe?g|png|gif|webp|bmp|avif)(\?|#|$)', re.IGNORECASE)

# 正文容器标签/属性 — 用于优先提取图片
_CONTENT_SELECTORS = [
    r'<article[^>]*>',
    r'<main[^>]*>',
    r'<div[^>]*class=["\'][^"\']*\b(content|post|article|entry|main|body|detail)[^"\']*["\']',
    r'<div[^>]*id=["\'][^"\']*\b(content|post|article|entry|main|body|detail)[^"\']*["\']',
    r'<section[^>]*class=["\'][^"\']*\b(content|post|article|entry)[^"\']*["\']',
]

# 图片 lazy-load 属性（获取图片 URL 时的回退顺序）
_LAZY_SRC_ATTRS = ["data-src", "data-original", "data-lazy-src", "data-original-src", "src"]


def _normalize_image_url(src: str, base_url: str) -> str | None:
    """规范化图片 URL：相对路径→绝对路径，无效 URL 返回 None。"""
    src = src.strip()
    if not src or src.startswith("data:"):
        return None
    if src.startswith("//"):
        src = "https:" + src
    elif src.startswith("/"):
        from urllib.parse import urlparse
        parsed = urlparse(base_url)
        src = f"{parsed.scheme}://{parsed.netloc}{src}"
    elif not src.startswith(("http://", "https://")):
        src = urljoin(base_url, src)
    return src


def _extract_img_urls(html: str, base_url: str) -> list[str]:
    """从 HTML 中提取有意义的图片 URL，自动过滤广告/头像/图标。

    策略：
    1. 优先从正文容器（article/main/content）提取图片
    2. 不足则全文档补全
    3. 对每张图片过滤：广告域名、头像类名、小尺寸、装饰 aria-hidden 等

    Returns:
        有用的图片 URL 列表
    """
    seen = set()
    content_area_urls: list[str] = []
    full_page_urls: list[str] = []

    def _collect_and_filter(html_fragment: str, target_list: list[str]) -> None:
        """从 HTML 片段中提取并过滤图片 URL。"""
        # 匹配完整 <img ...> 标签，支持单引号/双引号
        for match in re.finditer(
            r'<img\s[^>]*?(?:src|data-src|data-original|data-lazy-src)[^>]*>',
            html_fragment, re.IGNORECASE
        ):
            tag_html = match.group(0)
            # 尝试按优先级取图片 URL（data-src > data-original > src）
            src_url = None
            for attr in _LAZY_SRC_ATTRS:
                src_m = re.search(
                    rf'{attr}\s*=\s*["\']([^"\']+)["\']',
                    tag_html, re.IGNORECASE
                )
                if src_m:
                    src_url = _normalize_image_url(src_m.group(1), base_url)
                    if src_url:
                        break
            if not src_url or src_url in seen:
                continue

            # 过滤无用图片
            if _is_useless_image(tag_html, src_url):
                continue

            seen.add(src_url)
            target_list.append(src_url)

    # Step 1: 从正文容器区域提取（优先）
    for selector in _CONTENT_SELECTORS:
        m = re.search(selector, html, re.IGNORECASE)
        if m:
            start = m.end()
            depth = 1
            i = start
            while i < len(html) and depth > 0:
                open_tag = re.search(r'<(article|main|div|section)[>\s]', html[i:], re.IGNORECASE)
                close_tag = re.search(r'</(article|main|div|section)>', html[i:], re.IGNORECASE)
                if close_tag and (not open_tag or close_tag.start() < open_tag.start()):
                    depth -= 1
                    i += close_tag.end()
                elif open_tag:
                    depth += 1
                    i += open_tag.end()
                else:
                    i += 1
            fragment = html[m.start():i]
            _collect_and_filter(fragment, content_area_urls)

    # Step 2: 全文档补全（去重）
    _collect_and_filter(html, full_page_urls)

    # 正文区排前，其余在后
    return content_area_urls + [u for u in full_page_urls if u not in seen]


def _describe_page_images(html: str, base_url: str, max_images: int | None = None) -> list[str]:
    """提取网页中的图片 → 过滤无用图片 → VLM 逐张理解 → 返回描述列表。

    Args:
        html: 原始 HTML
        base_url: 页面 URL（用于相对路径转换）
        max_images: VLM 理解上限，默认从 IMG_READER_MAX_IMAGES 读取

    单张图片理解失败不影响其他图片，全部失败则返回空列表。
    """
    if max_images is None:
        from server.config import IMG_READER_MAX_IMAGES
        max_images = IMG_READER_MAX_IMAGES

    img_urls = _extract_img_urls(html, base_url)
    if not img_urls:
        return []
    if not IMG_READER_MODEL:
        logger.debug("IMG_READER_MODEL 未配置，跳过图片理解")
        return []

    to_process = img_urls[:max_images]
    if len(img_urls) > max_images:
        logger.info(
            "提取到 %d 张有用图片，取前 %d 张进行 VLM 理解",
            len(img_urls), max_images,
        )
    else:
        logger.info("提取到 %d 张有用图片，开始 VLM 理解...", len(img_urls))

    descriptions = []
    for url in to_process:
        desc = OpenAILLM.getImgContent(IMG_READER_MODEL, url)
        if desc and desc != "图像理解失败":
            descriptions.append(desc)
            logger.info("图片描述: %s", desc[:60])

    logger.info("图片理解完成: %d/%d 张成功", len(descriptions), len(to_process))
    return descriptions


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
