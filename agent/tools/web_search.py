import requests
from ddgs import DDGS


def search_web(query: str, max_results: int = 5) -> list[str]:
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
            return [r.get("body", "") for r in results if r.get("body")]
    except Exception as e:
        return [f"[Search error: {e}]"]

def search_web_from_baidu(query: str) -> list[str]:
    """调用百度的智能搜索API 每日限用100次 更好地支持中文问题，并且回答更加精确
        api文档：https://cloud.baidu.com/doc/qianfan-api/s/wmjqtqr7w
    """
    try:
        from server.config import BAIDU_AI_SEARCH_API_KEY as api_key
        if not api_key:
            return [f"[Search error: api key is None]"]
        url = "https://qianfan.baidubce.com/v2/ai_search/web_summary"
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
            "stream": False,
            "model": "ernie-4.5-turbo-32k",
            "instruction": "##",
            "enable_corner_markers": True,
            "enable_deep_search": False
        }
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()
        data = response.json()
        return data["choice"]
    except Exception as e:
        return [f"[Search error: {e}]"]



