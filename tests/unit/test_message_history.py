import uuid

from memory.session_manager import SessionManager
from memory.message_history import MessageHistory


class TestMessageHistory:
    def test_add_and_get_recent(self):
        """Add messages and retrieve them in order."""
        uid = uuid.uuid4().hex[:8]
        manager = SessionManager()
        session = manager.lookup(f"msg_user_1_{uid}")

        history = MessageHistory()
        history.add_message(session["id"], f"msg_user_1_{uid}", "user", "hello")
        history.add_message(session["id"], f"msg_user_1_{uid}", "assistant", "hi there")
        history.add_message(session["id"], f"msg_user_1_{uid}", "user", "how are you?")

        recent = history.get_recent(session["id"])
        assert len(recent) == 3
        assert recent[0]["content"] == "hello"
        assert recent[-1]["content"] == "how are you?"

    def test_empty_session(self):
        """No messages returns empty list."""
        uid = uuid.uuid4().hex[:8]
        manager = SessionManager()
        session = manager.lookup(f"msg_user_empty_{uid}")

        history = MessageHistory()
        recent = history.get_recent(session["id"])
        assert recent == []

    def test_get_recent_limit(self):
        """Limit parameter returns at most N messages."""
        uid = uuid.uuid4().hex[:8]
        manager = SessionManager()
        session = manager.lookup(f"msg_user_limit_{uid}")

        history = MessageHistory()
        for i in range(20):
            history.add_message(session["id"], f"msg_user_limit_{uid}", "user", f"msg_{i}")

        recent = history.get_recent(session["id"], limit=5)
        assert len(recent) == 5
        assert recent[0]["content"] == "msg_15"
        assert recent[-1]["content"] == "msg_19"

    def test_get_session_messages_all(self):
        """get_session_messages returns every message in chronological order."""
        uid = uuid.uuid4().hex[:8]
        manager = SessionManager()
        session = manager.lookup(f"msg_user_all_{uid}")

        history = MessageHistory()
        for i in range(10):
            history.add_message(session["id"], f"msg_user_all_{uid}", "user", f"msg_{i}")

        all_msgs = history.get_session_messages(session["id"])
        assert len(all_msgs) == 10
        assert all_msgs[0]["content"] == "msg_0"
        assert all_msgs[-1]["content"] == "msg_9"
