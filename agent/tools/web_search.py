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
        搜索生成api文档：https://cloud.baidu.com/doc/qianfan-api/s/wmjqtqr7w
        搜索API文档：https://cloud.baidu.com/doc/qianfan-api/s/Wmbq4z7e5
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
                    "type": "web", # video image aladdin
                    "top_k": 5
                }
            ],
            "block_websites": ["https://blog.csdn.net"] # 排除csdn
        }
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()
        data = response.json()
        return [item['content'] for item in data["references"]]
    except Exception as e:
        return [f"[Search error: {e}]"]


