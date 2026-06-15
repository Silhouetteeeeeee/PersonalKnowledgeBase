import logging
import struct

from storage.database import get_connection
from storage.models import generate_embedding
from memory.message_history import MessageHistory
from agent.utils.llm import LLM

logger = logging.getLogger(__name__)

EPISODIC_SEARCH_LIMIT = 3


def _embed_text(text: str) -> bytes | None:
    """Generate embedding via fastembed (shared with wiki vector search)."""
    try:
        emb = generate_embedding(text)
        return struct.pack(f"{len(emb)}f", *emb)
    except Exception as e:
        logger.warning("Embedding failed: %s", e)
        return None


class EpisodicMemory:
    """Long-term memory via LLM summarization + vector search."""

    def __init__(self):
        self.message_history = MessageHistory()

    def summarize_and_embed(self, session_id: int, user_id: str):
        """Summarize the latest conversation turn and store embedding."""
        messages = self.message_history.get_session_messages(session_id)
        if len(messages) < 2:
            return

        last_turn = messages[-2:]
        user_msg = (
            last_turn[0]["content"][:500]
            if last_turn[0]["role"] == "user"
            else ""
        )
        asst_msg = (
            last_turn[1]["content"][:500] if len(last_turn) > 1 else ""
        )

        if not user_msg and not asst_msg:
            return

        prompt = (
            "请将以下对话浓缩为一句话摘要，保留关键信息（主题、结论、用户偏好）：\n"
            f"用户：{user_msg}\n助手：{asst_msg}"
        )
        try:
            summary = LLM.generate(prompt, use_language=False)
            summary_text = (
                summary[:200] if isinstance(summary, str) else str(summary)[:200]
            )

            embedding = _embed_text(summary_text)
            if embedding is None:
                return

            conn = get_connection()
            try:
                conn.execute(
                    "UPDATE messages SET embedding=? WHERE id=?",
                    (embedding, last_turn[1]["id"]),
                )
                conn.commit()
            finally:
                conn.close()

            logger.info(
                "Episodic memory saved for session %s: %s",
                session_id,
                summary_text[:60],
            )
        except Exception as e:
            logger.warning("Episodic summarization failed: %s", e)

    def search(
        self,
        user_id: str,
        query: str,
        limit: int = EPISODIC_SEARCH_LIMIT,
    ) -> list[dict]:
        """Vector search for similar past conversations (cross-session)."""
        query_embedding = _embed_text(query[:500])
        if query_embedding is None:
            return []

        conn = get_connection()
        try:
            rows = conn.execute(
                """
                SELECT m.id, m.content, m.created_at, m.session_id, m.role
                FROM messages m
                WHERE m.user_id=?
                  AND m.role='assistant'
                  AND m.embedding IS NOT NULL
                  AND m.session_id NOT IN (
                      SELECT id FROM sessions WHERE user_id=? AND status='active'
                  )
                ORDER BY embedding MATCH ? LIMIT ?
                """,
                (user_id, user_id, query_embedding, limit),
            ).fetchall()
            return [dict(r) for r in rows]
        except Exception as e:
            logger.warning("Episodic search failed: %s", e)
            return []
