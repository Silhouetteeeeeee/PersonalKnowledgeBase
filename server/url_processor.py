import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import trafilatura
from tavily import TavilyClient

from server.config import TAVILY_API_KEY

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


def fetch_url_text(url: str) -> dict:
    """下载单个 URL → 提取正文 → 清洗 → 返回结构化结果。

    使用两层策略：trafilatura 静态提取，不足 200 字则 tavily 后备。
    单 URL 超时 30 秒。
    """
    result = {"url": url, "title": None, "content": ""}
    try:
        # 尝试提取 title
        try:
            resp = requests.get(url, timeout=30, headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                )
            })
            if resp.status_code == 200:
                m = re.search(r'<title>(.*?)</title>', resp.text, re.IGNORECASE | re.DOTALL)
                if m:
                    title = m.group(1).strip()
                    if title:
                        result["title"] = title[:200]

            # trafilatura 提取正文
            downloaded = trafilatura.fetch_url(url)
            if downloaded:
                text = trafilatura.extract(downloaded)
                if text and len(text) >= 200:
                    result["content"] = clean_content(text)
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
                        result['title'] = tavily_result["results"][0].get("title", "")
                        result["content"] = clean_content(content)
                        return result
            except Exception as e:
                logger.warning("Tavily extraction failed for %s: %s", url, e)

        # 全部失败
        result["content"] = "[抓取失败]"
        return result

    except Exception as e:
        logger.error("Unexpected error fetching %s: %s", url, e)
        result["content"] = "[抓取失败]"
        return result


# ── Concurrent URL Fetch ─────────────────────────────────────────────────────


def fetch_urls_concurrent(urls: list[str]) -> list[dict]:
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
                results.append({"url": url, "title": None, "content": "[抓取失败]"})

    # 按原始顺序排序
    order = {url: i for i, url in enumerate(unique_urls)}
    results.sort(key=lambda r: order.get(r.get("url", ""), 999))
    return results
