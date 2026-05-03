"""
企业微信智能机器人 — 官方 SDK 异步客户端。
使用 WSClient + BotID + Secret 直连，无需公网服务器。
"""

import asyncio
import logging

from aibot import WSClient, WSClientOptions
from server.config import WECOM_BOT_ID, WECOM_BOT_SECRET
from agent.graph import build_graph

logger = logging.getLogger(__name__)
graph = build_graph()


class KnowledgeBot:
    def __init__(self):
        self.client = WSClient(WSClientOptions(
            bot_id=WECOM_BOT_ID,
            secret=WECOM_BOT_SECRET,
            max_reconnect_attempts=-1,
        ))
        self._setup_handlers()

    def _setup_handlers(self):
        @self.client.on("connected")
        def _on_connected():
            logger.info("Connected to WeChat Work WebSocket")

        @self.client.on("authenticated")
        def _on_auth():
            logger.info("Authenticated successfully, waiting for messages...")

        @self.client.on("message.text")
        async def _on_text(frame):
            body = frame.get("body", {})
            content = body.get("text", {}).get("content", "").strip()
            user_id = body.get("from", {}).get("userid", "unknown")

            if not content:
                return

            logger.info("Received text from %s: %s", user_id, content[:60])

            try:
                result = await asyncio.to_thread(graph.invoke, {
                    "user_message": content,
                    "user_id": user_id,
                    "timestamp": "",
                })
                response = result.get("final_response", "")

                await self.client.reply(frame, {
                    "msgtype": "markdown",
                    "markdown": {
                        "content": response,
                    },
                })
                logger.info("Response sent to user_id=%s", user_id)
            except Exception:
                logger.exception("Error handling message from %s", user_id)

        @self.client.on("error")
        def _on_error(error):
            logger.error("Client error: %s", error)

    def run(self):
        self.client.run()


def run_bot():
    if not WECOM_BOT_ID or not WECOM_BOT_SECRET:
        logger.error("WECOM_BOT_ID and WECOM_BOT_SECRET must be set in .env")
        return

    logger.info("Starting Knowledge Agent Bot (BotID: %s****)", WECOM_BOT_ID[:4])
    KnowledgeBot().run()
