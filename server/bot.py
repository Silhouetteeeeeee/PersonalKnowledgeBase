"""
企业微信智能机器人 — 官方 SDK 异步客户端。
使用 WSClient + BotID + Secret 直连，无需公网服务器。
"""

import asyncio
import logging
import os

from aibot import WSClient, WSClientOptions
from server.config import WECOM_BOT_ID, WECOM_BOT_SECRET, CLAUDE_CODE_BRIDGE_ENABLED
from agent.graph import build_graph
from storage.profile import load_profile
from memory.session_manager import SessionManager
from memory.context_builder import ContextBuilder
from memory.message_history import MessageHistory
from memory.episodic import EpisodicMemory

from server.claude_bridge import ClaudeCodeBridge

logger = logging.getLogger(__name__)
graph = build_graph()

session_manager = SessionManager()
context_builder = ContextBuilder()
message_history = MessageHistory()
episodic_memory = EpisodicMemory()

claude_bridge = ClaudeCodeBridge() if CLAUDE_CODE_BRIDGE_ENABLED else None


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

            # ── Claude Code bridge route ──
            if content.startswith("/code"):
                cmd = content[5:].strip()
                if CLAUDE_CODE_BRIDGE_ENABLED and claude_bridge:
                    response = claude_bridge.handle(cmd)
                else:
                    response = "⚠️ 远程编码功能未启用\n请在 .env 中设置 CLAUDE_CODE_BRIDGE_ENABLED=true"
                await self.client.reply(frame, {
                    "msgtype": "markdown",
                    "markdown": {"content": response},
                })
                logger.info("Claude Code bridge response sent to user_id=%s", user_id)
                return

            logger.info("Received text from %s: %s", user_id, content[:60])

            try:
                # ── 上下文管理 ──
                session = session_manager.lookup(user_id)
                context = context_builder.build(user_id, session["id"], content)

                session_manager.refresh(session["id"])

                result = await asyncio.to_thread(graph.invoke, {
                    "user_message": content,
                    "user_id": user_id,
                    "session_id": str(session["id"]),
                    "message_history": context.get("history_section", ""),
                    "episodic_memories": context.get("episodic_section", ""),
                    "user_profile": load_profile(user_id),
                    "timestamp": "",
                    "confidence": 0.0,
                    "needs_store": False,
                    "search_results": [],
                    "stored_knowledge": [],
                    "stored_knowledge_ids": [],
                    "answer": "",
                    "final_response": "",
                    "reasoning_log_path": "",
                    "contradiction_found": False,
                    "contradiction_details": "",
                    "search_time": 0,
                    "contradiction_severity": "",
                    "contradiction_knowledge_ids": [],
                    "contradiction_knowledge_texts": [],
                    "reflection_result": "",
                    "reflection_reasoning": "",
                    "reflection_correction": "",
                    "force_web_search": False,
                    "correction_attempts": 0,
                    "knowledge_corrected": False,
                    "error_recorded": False,
                    "logic_chain": [],
                })
                response = result.get("final_response", "")

                await self.client.reply(frame, {
                    "msgtype": "markdown",
                    "markdown": {
                        "content": response,
                    },
                })
                logger.info("Response sent to user_id=%s", user_id)

                # ── 异步持久化（不阻塞回复）──
                answer_text = result.get("final_response", "")
                asyncio.create_task(self._save_turn(
                    session["id"], user_id, content, answer_text,
                ))
            except Exception:
                logger.exception("Error handling message from %s", user_id)

        @self.client.on("message.image")
        async def _on_image(frame):
            body = frame.get("body", {})
            user_id = body.get("from", {}).get("userid", "unknown")
            image_data = body.get("image", {})

            url = image_data.get("url") or image_data.get("imageurl", "")
            aes_key = image_data.get("aeskey")

            if not url:
                logger.warning("Image message without download URL, body keys: %s", list(image_data.keys()))
                return

            logger.info("Received image from %s", user_id)
            await self._handle_file_upload(frame, url, aes_key, "image.jpg", user_id)

        @self.client.on("message.file")
        async def _on_file(frame):
            body = frame.get("body", {})
            user_id = body.get("from", {}).get("userid", "unknown")
            file_data = body.get("file", {})

            url = file_data.get("url") or file_data.get("fileurl", "")
            aes_key = file_data.get("aeskey")
            filename = file_data.get("filename", "unknown")

            if not url:
                logger.warning("File message without download URL, body keys: %s", list(file_data.keys()))
                return

            logger.info("Received file from %s: %s", user_id, filename)
            await self._handle_file_upload(frame, url, aes_key, filename, user_id)

        @self.client.on("error")
        def _on_error(error):
            logger.error("Client error: %s", error)

    async def _handle_file_upload(self, frame, url, aes_key, filename, user_id):
        """下载文件并异步处理（提取文字、LLM 蒸馏、存储）。"""
        try:
            file_bytes, sdk_filename = await self.client.download_file(url, aes_key)
            actual_filename = sdk_filename or filename

            logger.info("File downloaded: %s (%d bytes)", actual_filename, len(file_bytes))

            reply_text = await asyncio.to_thread(
                _process_and_store_file, file_bytes, actual_filename, user_id
            )

            await self.client.reply(frame, {
                "msgtype": "markdown",
                "markdown": {"content": reply_text},
            })
            logger.info("File processing reply sent to user_id=%s", user_id)
        except Exception:
            logger.exception("Error processing file upload from %s", user_id)

    async def _save_turn(self, session_id: int, user_id: str, user_msg: str, asst_msg: str):
        """Persist conversation turn asynchronously (non-blocking)."""
        try:
            await asyncio.to_thread(message_history.add_message, session_id, user_id, "user", user_msg)
            await asyncio.to_thread(message_history.add_message, session_id, user_id, "assistant", asst_msg)
            await asyncio.to_thread(episodic_memory.summarize_and_embed, session_id, user_id)
        except Exception as e:
            logger.warning("Async memory persistence failed: %s", e)

    def run(self):
        self.client.run()


def run_bot():
    if not WECOM_BOT_ID or not WECOM_BOT_SECRET:
        logger.error("WECOM_BOT_ID and WECOM_BOT_SECRET must be set in .env")
        return

    logger.info("Starting Knowledge Agent Bot (BotID: %s****)", WECOM_BOT_ID[:4])
    KnowledgeBot().run()


# ── 文件处理（同步，在独立线程中执行）──

def _process_and_store_file(file_bytes: bytes, filename: str, user_id: str) -> str:
    """保存文件、提取文字、LLM 蒸馏、存储知识。返回回复文本。"""
    from storage.database import DB_DIR
    from storage.file_processor import extract_text_from_file, compute_file_hash
    from storage.models import (
        save_knowledge_points_bulk_with_embeddings,
        ensure_category,
        save_file_record,
        get_file_record_by_hash,
    )
    from agent.utils.llm import LLM
    from agent.nodes.store import DistillOutput

    file_hash = compute_file_hash(file_bytes)
    ext = os.path.splitext(filename)[1].lower() or ".bin"

    # 去重：检查是否已处理过
    existing = get_file_record_by_hash(file_hash)
    if existing:
        logger.info("File already processed: %s (hash=%s)", filename, file_hash)
        return f"文件「{filename}」已处理过，无需重复处理。"

    # 保存到本地 data/files/
    files_dir = os.path.join(DB_DIR, "files")
    os.makedirs(files_dir, exist_ok=True)
    safe_name = f"{file_hash}{ext}"
    file_path = os.path.join(files_dir, safe_name)
    with open(file_path, "wb") as f:
        f.write(file_bytes)
    logger.info("File saved to %s", file_path)

    # 提取文字内容
    text = extract_text_from_file(file_path)
    if not text.strip():
        logger.warning("No text extracted from %s", filename)
        return f"未能从文件「{filename}」中提取到文字内容。"

    logger.info("Extracted %d chars from %s", len(text), filename)

    # LLM 蒸馏为知识点
    prompt = (
        f"请将以下从文件「{filename}」中提取的文字内容提炼为知识点。\n"
        f"如果内容涉及多个主题，请分多条知识点存储，并为每条添加合适的标签。\n\n"
        f"内容：\n{text}"
    )
    result = LLM.generate_structured(prompt, DistillOutput, use_language=False)
    ensure_category(result.category)

    # 存储知识点
    knowledge_points = [
        {
            "knowledge_text": kp.knowledge_text,
            "source_question": f"[文件] {filename}",
            "category": result.category,
            "tags": kp.tags,
        }
        for kp in result.knowledge_points
    ]
    ids = save_knowledge_points_bulk_with_embeddings(knowledge_points)

    # 记录文件处理记录
    save_file_record(filename, ext, file_hash, text, ids, user_id)

    reply = (
        f"已从文件「{filename}」中提取并存储了 {len(ids)} 条知识点\n"
        f"分类：{result.category}"
    )
    logger.info("File processed: %s → %d points in '%s'", filename, len(ids), result.category)
    return reply
