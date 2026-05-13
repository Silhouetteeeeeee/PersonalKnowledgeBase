from unittest.mock import patch

from memory.session_manager import SessionManager
from memory.message_history import MessageHistory
from memory.context_builder import ContextBuilder


class TestContextBuilder:
    def test_empty_for_new_user(self):
        """New user with no profile, no history, no episodic memories."""
        from storage.profile import load_profile
        manager = SessionManager()
        session = manager.lookup("cb_new_user")

        builder = ContextBuilder()
        ctx = builder.build("cb_new_user", session["id"], "hello")

        assert ctx["profile_section"] == ""
        assert ctx["history_section"] == ""
        assert ctx["episodic_section"] == ""

    def test_history_included_after_messages(self):
        """After messages, history section contains conversation."""
        manager = SessionManager()
        session = manager.lookup("cb_user_msgs")

        history = MessageHistory()
        history.add_message(session["id"], "cb_user_msgs", "user", "hello")
        history.add_message(session["id"], "cb_user_msgs", "assistant", "world")

        builder = ContextBuilder()
        ctx = builder.build("cb_user_msgs", session["id"], "hello")

        assert "<conversation_history>" in ctx["history_section"]
        assert "hello" in ctx["history_section"]
        assert "world" in ctx["history_section"]

    def test_profile_included_when_exists(self):
        """When profile has data, profile_section is populated."""
        # Save a profile for this user
        from storage.profile import save_profile
        save_profile(
            {"identity": {"name": "TestUser"}, "preferences": {}, "habits": [], "plans": {}, "updated_at": "2026-01-01"},
            "cb_user_profile",
        )

        manager = SessionManager()
        session = manager.lookup("cb_user_profile")

        builder = ContextBuilder()
        ctx = builder.build("cb_user_profile", session["id"], "hello")

        assert "<user_profile>" in ctx["profile_section"]
        assert "TestUser" in ctx["profile_section"]

    def test_episodic_section_empty_when_no_results(self):
        """No episodic memories yields empty section."""
        # We need to make sure no embedding exists, so episodic search returns empty
        manager = SessionManager()
        session = manager.lookup("cb_user_noep")

        builder = ContextBuilder()
        ctx = builder.build("cb_user_noep", session["id"], "hello")

        assert ctx["episodic_section"] == ""
