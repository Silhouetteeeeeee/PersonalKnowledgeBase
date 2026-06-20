import sys
import os
from openai import OpenAI

client = OpenAI(base_url="https://api.xiaomimimo.com/v1", api_key="sk-cupum6tulyt38pu1z5joz2vu6s79y3uex7jxlt73lmj6mxqr")
response = client.chat.completions.create(
    model="mimo-v2.5",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "请用简洁的语言描述这张图片的内容，不需要多余的名词解释，尽可能像人类理解图片内容一样，理解图片的重点信息"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://pic1.zhimg.com/v2-6cac1c618d6c04ea5ce94ccfc2c287ca_r.jpg"
                    }
                }
            ]
        }
    ]
)
print(response)


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
