import json
from pathlib import Path

from storage.profile import load_profile, save_profile, update_profile_field


class TestProfilePerUser:
    def test_load_profile_no_user(self):
        """Empty user_id returns default."""
        profile = load_profile("")
        assert profile["identity"] == {}

    def test_profile_isolation(self):
        """Different users get different profiles."""
        save_profile(
            {"identity": {"name": "Alice"}, "preferences": {}, "habits": [], "plans": {}, "updated_at": ""},
            "alice",
        )
        save_profile(
            {"identity": {"name": "Bob"}, "preferences": {}, "habits": [], "plans": {}, "updated_at": ""},
            "bob",
        )

        alice = load_profile("alice")
        bob = load_profile("bob")
        assert alice["identity"]["name"] == "Alice"
        assert bob["identity"]["name"] == "Bob"
        assert alice is not bob

    def test_load_nonexistent_user(self):
        """Unknown user returns default."""
        profile = load_profile("nonexistent_user_xyz")
        assert profile["identity"] == {}

    def test_update_field_dot_notation(self):
        """update_profile_field handles dot notation."""
        profile = {"identity": {}, "preferences": {}, "habits": [], "plans": {}, "updated_at": ""}
        profile = update_profile_field(profile, "identity.name", "Charlie")
        assert profile["identity"]["name"] == "Charlie"
