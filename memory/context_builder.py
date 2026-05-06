import json
import logging

from storage.profile import load_profile
from memory.message_history import MessageHistory
from memory.episodic import EpisodicMemory

logger = logging.getLogger(__name__)


class ContextBuilder:
    """Assemble 3-tier memory context for prompt injection."""

    def __init__(self):
        self.message_history = MessageHistory()
        self.episodic = EpisodicMemory()

    def build(self, user_id: str, session_id: int, content: str) -> dict:
        """Build three context sections from memory tiers."""
        result = {}

        # Layer 1: Core memory (user profile)
        profile = load_profile(user_id)
        has_data = any(
            v for v in profile.values()
            if isinstance(v, dict) and v
        )
        if has_data:
            result["profile_section"] = (
                f"<user_profile>\n{json.dumps(profile, ensure_ascii=False, indent=2)}\n</user_profile>"
            )
        else:
            result["profile_section"] = ""

        # Layer 2: Working memory (recent conversation)
        recent = self.message_history.get_recent(session_id)
        if recent:
            lines = []
            for msg in recent:
                role = "user" if msg["role"] == "user" else "assistant"
                lines.append(f"{role}: {msg['content']}")
            result["history_section"] = (
                "<conversation_history>\n" + "\n".join(lines) + "\n</conversation_history>"
            )
        else:
            result["history_section"] = ""

        # Layer 3: Episodic memory (cross-session vector search)
        episodic_results = self.episodic.search(user_id, content)
        if episodic_results:
            entries = []
            for m in episodic_results:
                date = m.get("created_at", "")[:10]
                text = m["content"][:150]
                entries.append(f"[{date}] {text}")
            result["episodic_section"] = (
                "<your_long_term_memories>\n" + "\n".join(entries) + "\n</your_long_term_memories>"
            )
        else:
            result["episodic_section"] = ""

        logger.debug(
            "Context built for user=%s session=%s: profile=%s history=%d episodic=%d",
            user_id, session_id,
            "yes" if result["profile_section"] else "no",
            len(recent),
            len(episodic_results),
        )
        return result
