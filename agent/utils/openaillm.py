"""OpenAI 兼容接口的图像理解封装。

用 OpenAI 格式的 API 调用视觉语言模型（VLM），
支持 GPT-4o、GPT-4o-mini 以及任何 OpenAI 兼容的第三方视觉模型。

用法：
    description = OpenAILLM.getImgContent("gpt-4o-mini", "https://example.com/image.png")
"""

import logging
from typing import Optional

from openai import OpenAI

logger = logging.getLogger(__name__)


class OpenAILLM:
    """图像理解工具类（类级别懒加载单例）。

    无需手动初始化，首次调用 getImgContent 时自动从环境变量读取配置。
    """

    _client: Optional[OpenAI] = None

    @classmethod
    def _ensure_client(cls) -> None:
        """确保客户端已初始化（懒加载）。"""
        if cls._client is not None:
            return
        from server.config import IMG_READER_BASE_URL, IMG_READER_API_KEY
        if not IMG_READER_API_KEY:
            logger.warning("IMG_READER_API_KEY 未配置，图像理解不可用")
            return
        base_url = IMG_READER_BASE_URL or "https://api.openai.com/v1"
        cls._client = OpenAI(base_url=base_url, api_key=IMG_READER_API_KEY)
        logger.info("图像理解客户端已初始化: base_url=%s", base_url)

    @classmethod
    def getImgContent(cls, model: str, img_url: str) -> str:
        """调用视觉模型理解一张图片，返回中文文本描述。

        Args:
            model: 视觉模型名称（如 gpt-4o-mini、qwen-vl-plus 等）
            img_url: 图片的 URL 或 data URL

        Returns:
            图片的文字描述；失败时返回 "图像理解失败"
        """
        cls._ensure_client()
        if cls._client is None:
            return "图像理解失败"

        try:
            response = cls._client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "请用简洁的语言描述这张图片的内容，"
                                    "不需要多余的名词解释，"
                                    "尽可能像人类理解图片内容一样，理解图片的重点信息"
                                ),
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": img_url},
                            },
                        ],
                    }
                ],
                timeout=15,  # 单张图片超时 15s
            )
            description = response.choices[0].message.content
            logger.info("图片理解成功: %s -> %s", img_url, description[:50])
            return description
        except Exception as e:
            logger.error("图像理解失败: url=%s, error=%s", img_url, e)
            return "图像理解失败"
