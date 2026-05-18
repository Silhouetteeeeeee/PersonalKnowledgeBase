"""
企业微信智能机器人 — 官方 SDK 异步客户端。
使用 WSClient + BotID + Secret 直连，无需公网服务器。
"""

import asyncio
import logging
import os

from aibot import WSClient, WSClientOptions
from server.config import WECOM_BOT_ID, WECOM_BOT_SECRET, CLAUDE_CODE_BRIDGE_ENABLED
from server.config import DAILY_SUMMARY_ENABLED, DAILY_SUMMARY_USER_ID
from server.config import THINKER_USER_ID, THINKER_CHECK_INTERVAL
from agent.graph import build_graph
from agent.nodes.store import run_background_store
from storage.profile import load_profile
from memory.session_manager import SessionManager
from memory.context_builder import ContextBuilder
from memory.message_history import MessageHistory
from memory.episodic import EpisodicMemory

from server.claude_bridge import ClaudeCodeBridge
from server.thinker import check_due_reviews, handle_review_response
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from server.daily_summary import send_daily_summary
from storage.models import cleanup_old_versions, record_sent_review

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
        self.scheduler = AsyncIOScheduler()
        self._setup_handlers()
        self._start_daily_summary_scheduler()
        self._start_cleanup_scheduler()
        self._start_thinker_scheduler()

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
            # 引用文字
            quote = body.get('quote')

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

            # ── Thinker review response route ──
            if quote and isinstance(quote, dict):
                quoted_text = quote.get('text', '')
                if '#review_' in quoted_text:
                    logger.info("Thinker: user %s replied to review: %s", user_id, content[:30])
                    response = handle_review_response(quoted_text, content)
                    await self.client.reply(frame, {
                        "msgtype": "markdown",
                        "markdown": {"content": response},
                    })
                    logger.info("Thinker response sent to user_id=%s", user_id)
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
                    "url_contents": [],
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
                asyncio.create_task(run_background_store(result))
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

    def _start_daily_summary_scheduler(self):
        """Schedule daily knowledge summary at 09:00 via APScheduler."""
        if not DAILY_SUMMARY_ENABLED:
            logger.info("Daily summary disabled via config")
            return
        if not DAILY_SUMMARY_USER_ID:
            logger.warning("DAILY_SUMMARY_USER_ID not set, daily summary disabled")
            return

        self.scheduler.add_job(
            send_daily_summary,
            "cron",
            hour=9,
            minute=0,
            args=[self.client, DAILY_SUMMARY_USER_ID],
            misfire_grace_time=300,
            id="daily_summary",
            replace_existing=True,
        )
        logger.info(
            "Daily summary scheduler started (09:00, user=%s)",
            DAILY_SUMMARY_USER_ID,
        )

    def _start_cleanup_scheduler(self):
        """Schedule daily wiki version cleanup at 03:00."""
        self.scheduler.add_job(
            cleanup_old_versions,
            "cron",
            hour=3,
            minute=0,
            kwargs={"days": 30},
            id="wiki_cleanup",
            replace_existing=True,
        )
        logger.info("Wiki cleanup scheduler started (03:00, keep 30 days)")

    def _start_thinker_scheduler(self):
        """Schedule periodic thinker review checks."""
        if not THINKER_USER_ID:
            logger.info("Thinker disabled: no THINKER_USER_ID configured")
            return

        self.scheduler.add_job(
            self._run_thinker_check,
            "interval",
            hours=THINKER_CHECK_INTERVAL,
            id="thinker_review_check",
            replace_existing=True,
            misfire_grace_time=600,
        )
        self.scheduler.add_job(
            self._run_weekly_integration,
            "cron",
            day_of_week="mon",
            hour=10,
            minute=0,
            id="thinker_weekly_integration",
            replace_existing=True,
            misfire_grace_time=600,
        )
        logger.info(
            "Thinker scheduler started (check every %dh, weekly Mon 10:00, user=%s)",
            THINKER_CHECK_INTERVAL, THINKER_USER_ID,
        )

    async def _run_thinker_check(self):
        """Async wrapper for thinker review check."""
        try:
            pushed = await asyncio.to_thread(check_due_reviews, THINKER_USER_ID)
            for item in pushed:
                await self.client.send_message(THINKER_USER_ID, {
                    "msgtype": "markdown",
                    "markdown": {"content": item["message"]},
                })
                await asyncio.to_thread(
                    record_sent_review,
                    item["schedule_id"], item["page_id"], item["marker_id"],
                )
                logger.info("Thinker: pushed review '%s' to %s", item["page_title"], THINKER_USER_ID)
        except Exception:
            logger.exception("Thinker check failed")

    async def _run_weekly_integration(self):
        """Async wrapper for weekly integration."""
        from server.thinker import generate_weekly_integration
        try:
            message = await asyncio.to_thread(generate_weekly_integration, THINKER_USER_ID)
            if message:
                await self.client.send_message(THINKER_USER_ID, message)
                logger.info("Thinker: pushed weekly integration")
        except Exception:
            logger.exception("Weekly integration failed")

    async def _save_turn(self, session_id: int, user_id: str, user_msg: str, asst_msg: str):
        """Persist conversation turn asynchronously (non-blocking)."""
        try:
            await asyncio.to_thread(message_history.add_message, session_id, user_id, "user", user_msg)
            await asyncio.to_thread(message_history.add_message, session_id, user_id, "assistant", asst_msg)
            await asyncio.to_thread(episodic_memory.summarize_and_embed, session_id, user_id)
        except Exception as e:
            logger.warning("Async memory persistence failed: %s", e)

    def run(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self.client.connect())
            loop.call_soon(self.scheduler.start)
            loop.run_forever()
        except KeyboardInterrupt:
            self.client.disconnect()
        finally:
            self.scheduler.shutdown(wait=False)
            loop.close()


def run_bot():
    if not WECOM_BOT_ID or not WECOM_BOT_SECRET:
        logger.error("WECOM_BOT_ID and WECOM_BOT_SECRET must be set in .env")
        return

    logger.info("Starting Knowledge Agent Bot (BotID: %s****)", WECOM_BOT_ID[:4])
    KnowledgeBot().run()


# ── 文件处理（同步，在独立线程中执行）──

MAX_FILE_CHARS = 20000


def _process_and_store_file(file_bytes: bytes, filename: str, user_id: str) -> str:
    """保存文件、提取文字、两步 CoT 提取 wiki 页面。返回回复文本。"""
    from storage.database import DB_DIR
    from storage.file_processor import extract_text_from_file, compute_file_hash
    from storage.models import (
        save_file_record,
        get_file_record_by_hash,
    )
    from agent.nodes.store import extract_to_wiki

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

    # 长度检查
    if len(text) > MAX_FILE_CHARS:
        logger.warning("File too long: %d chars (max %d)", len(text), MAX_FILE_CHARS)
        return f"文件「{filename}」内容过长（{len(text)} 字符），暂不支持处理超过 {MAX_FILE_CHARS} 字符的文档。"

    # 两步 CoT 提取为 wiki 页面
    saved = extract_to_wiki(
        source_text=text,
        source_id=f"file_{file_hash}",
        source_label=f"From file: {filename}",
    )

    if not saved.get("page_ids"):
        return f"未能从文件「{filename}」中提取到 wiki 页面。"

    # 记录文件处理记录
    save_file_record(filename, ext, file_hash, text, saved["page_ids"], user_id)

    reply = (
        f"已从文件「{filename}」中提取并创建了 {len(saved['page_ids'])} 篇 wiki 页面"
    )
    logger.info("File processed: %s -> %d wiki pages", filename, len(saved['page_ids']))
    return reply
