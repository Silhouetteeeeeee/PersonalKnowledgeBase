"""
网络搜索工具集：提供多种搜索引擎后端的统一接口。

搜索策略：
  1. 百度智能搜索（首选，中文支持好，需 BAIDU_API_KEY）
  2. DuckDuckGo 搜索（备选，无需 API Key）
"""

import requests
from ddgs import DDGS


def search_web(query: str, max_results: int = 5) -> list[str]:
    """
    通用网络搜索：使用 DuckDuckGo（无需 API Key，适合英文搜索）。
    作为百度搜索的备选方案。
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
            return [r.get("body", "") for r in results if r.get("body")]
    except Exception as e:
        return [f"[Search error: {e}]"]


def search_web_from_baidu(query: str) -> list[str]:
    """
    百度智能搜索 API（首选，中文搜索效果好）。

    限制：每日 100 次免费调用
    文档：https://cloud.baidu.com/doc/qianfan-api/s/wmjqtqr7w
    """
    try:
        from server.config import BAIDU_API_KEY as api_key
        if not api_key:
            return [f"[Search error: api key is None]"]
        url = "https://qianfan.baidubce.com/v2/ai_search/web_search"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "messages": [
                {
                    "content": query,
                    "role": "user"
                }
            ],
            "resource_type_filter": [
                {
                    "type": "web",
                    "top_k": 5
                }
            ],
            "block_websites": ["https://blog.csdn.net"]  # 屏蔽 CSDN（内容质量参差不齐）
        }
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()
        data = response.json()
        return [item['content'] for item in data["references"]]
    except Exception as e:
        return [f"[Search error: {e}]"]


