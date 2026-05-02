"""
企业微信智能机器人 — WebSocket 长连接客户端。
使用 BotID + Secret 直连，无需公网服务器。
"""

import json
import logging
import time

import websocket
from server.config import WECOM_BOT_ID, WECOM_BOT_SECRET
from agent.graph import build_graph

logger = logging.getLogger(__name__)
graph = build_graph()


def _on_open(ws):
    logger.info("Connected to WeChat Work WebSocket")
    ws.send(json.dumps({
        "cmd": "aibot_subscribe",
        "headers": {"req_id": "init"},
        "body": {
            "bot_id": WECOM_BOT_ID,
            "secret": WECOM_BOT_SECRET,
        },
    }))


def _on_message(ws, raw):
    if raw == "pong":
        return
    try:
        data = json.loads(raw)

        # Subscribe acknowledgment: {"errcode":0,"errmsg":"ok","headers":{...}}
        if "errcode" in data:
            if data["errcode"] == 0:
                logger.info("Subscription confirmed")
            else:
                logger.error("Subscription failed: errcode=%s errmsg=%s",
                             data["errcode"], data.get("errmsg"))

        # Incoming message
        elif data.get("cmd") == "aibot_msg_callback":
            _handle_msg(ws, data.get("body", {}))

        else:
            logger.debug("Unhandled message: %s", raw[:150])
    except json.JSONDecodeError:
        logger.warning("Invalid message received: %s", raw[:100])


def _handle_msg(ws, body):
    user_id = body.get("from", {}).get("userid", "unknown")
    msgtype = body.get("msgtype", "")
    seq = body.get("msgid", "")
    chattype = body.get("chattype", "single")

    if msgtype != "text":
        return

    content = body.get("text", {}).get("content", "").strip()
    if not content:
        return

    logger.info("Received %s from %s: %s", chattype, user_id, content[:60])

    try:
        result = graph.invoke({
            "user_message": content,
            "user_id": user_id,
            "timestamp": "",
        })
        response = result.get("final_response", "")

        ws.send(json.dumps({
            "cmd": "aibot_respond_msg",
            "headers": {"req_id": str(int(time.time()))},
            "body": {
                "seq": seq,
                "content": response,
                "content_type": 1,
            },
        }))
        logger.info("Response sent to %s", user_id)
    except Exception:
        logger.exception("Error handling message from %s", user_id)


def _on_error(ws, error):
    logger.error("WebSocket error: %s", error)


def _on_close(ws, close_status_code, close_msg):
    logger.info("WebSocket closed (code=%s). Reconnecting in 5s...", close_status_code)


def run_bot():
    if not WECOM_BOT_ID or not WECOM_BOT_SECRET:
        logger.error("WECOM_BOT_ID and WECOM_BOT_SECRET must be set in .env")
        return

    logger.info("Starting Knowledge Agent Bot (BotID: %s****)", WECOM_BOT_ID[:4])
    while True:
        try:
            ws = websocket.WebSocketApp(
                "wss://openws.work.weixin.qq.com",
                on_open=_on_open,
                on_message=_on_message,
                on_error=_on_error,
                on_close=_on_close,
            )
            ws.run_forever(ping_interval=30)
        except Exception as e:
            logger.error("Connection error: %s", e)
        time.sleep(5)
