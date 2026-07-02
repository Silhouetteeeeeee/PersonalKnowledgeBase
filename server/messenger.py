"""企业微信消息发送公共组件。

统一封装 WSClient 的消息发送，所有模块（bot/daily_summary/thinker）
都通过此组件发送消息，避免重复处理 msgtype/markdown 结构。

用法：
    # 初始化（在 bot 启动时调用一次）
    Messenger.init(ws_client)

    # 发送主动消息
    await Messenger.send_markdown(user_id, "**你好** 这是markdown内容")

    # 回复消息
    await Messenger.reply_markdown(frame, "**回复** 这是markdown内容")
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class Messenger:
    """企业微信消息发送器（全局单例，持有 WSClient 引用）。"""

    _client: Optional[object] = None

    @classmethod
    def init(cls, client: object) -> None:
        """初始化 Messenger，绑定 WSClient 实例。

        Args:
            client: WSClient 实例（来自 aibot 库）
        """
        cls._client = client
        logger.info("Messenger 已初始化")

    @classmethod
    def _ensure_client(cls) -> None:
        """确保客户端已初始化。"""
        if cls._client is None:
            raise RuntimeError(
                "Messenger 未初始化，请先调用 Messenger.init(client)"
            )

    @classmethod
    async def send_markdown(cls, user_id: str, content: str) -> bool:
        """向指定用户发送主动 Markdown 消息。

        Args:
            user_id: 企业微信用户 ID
            content: Markdown 格式的文本内容

        Returns:
            True 发送成功，False 发送失败
        """
        cls._ensure_client()
        try:
            await cls._client.send_message(user_id, {
                "msgtype": "markdown",
                "markdown": {"content": content},
            })
            logger.info("已向 %s 发送 markdown 消息（%d 字符）", user_id, len(content))
            return True
        except Exception as e:
            logger.error("向 %s 发送消息失败: %s", user_id, e)
            return False

    @classmethod
    async def reply_markdown(cls, frame: dict, content: str) -> bool:
        """回复指定消息帧，发送 Markdown 回复。

        Args:
            frame: 需要回复的原始消息 frame（来自 on_message 回调）
            content: Markdown 格式的回复内容

        Returns:
            True 发送成功，False 发送失败
        """
        cls._ensure_client()
        try:
            await cls._client.reply(frame, {
                "msgtype": "markdown",
                "markdown": {"content": content},
            })
            logger.info("已回复消息（%d 字符）", len(content))
            return True
        except Exception as e:
            logger.error("回复消息失败: %s", e)
            return False
