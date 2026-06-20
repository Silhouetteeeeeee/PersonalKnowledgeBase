import logging
from typing import Optional

from openai import OpenAI

logger = logging.getLogger(__name__)

class OpenAILLM:
    _client: Optional[OpenAI] = None

    def __init__(self, base_url, api_key):
        self._client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )

    @classmethod
    def getImgContent(cls, model: str, img_url: str):
        try:
            response = cls._client.chat.completions.create(
                model=model,
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
                                    "url": img_url
                                }
                            }
                        ]
                    }
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error("图像理解失败%s", e)
            return "图像理解失败"
