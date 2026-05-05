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
            "instruction": "如果是技术问题，请你以专业的视角回答这个问题，如果你不了解，请回复你不了解，不要随便回复。"
                           "如果是生活方面的问题，请你以合乎逻辑的方式回答问题。"
                           "请以人类和机器都能理解的语言回答问题。",
        }
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()
        data = response.json()
        return [d["message"]["content"] for d in data["choices"]]
    except Exception as e:
        return [f"[Search error: {e}]"]



