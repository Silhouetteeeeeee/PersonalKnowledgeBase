import json
import pytest
from pathlib import Path


def test_load_profile_empty(tmp_path):
    """Empty user_id returns default dict."""
    import storage.profile as pf

    profile = pf.load_profile()
    assert "identity" in profile
    assert "preferences" in profile
    assert "habits" in profile
    assert "plans" in profile
    assert "updated_at" in profile
    assert profile["identity"] == {}


def test_save_and_load_profile(tmp_path):
    import storage.profile as pf
    pf.PROFILE_DIR = tmp_path
    pf.BACKUP_DIR = tmp_path / "profile_backups"

    profile = pf.load_profile("test_user")
    profile["identity"]["name"] = "TestUser"
    pf.save_profile(profile, "test_user")

    loaded = pf.load_profile("test_user")
    assert loaded["identity"]["name"] == "TestUser"


def test_update_field_dot_notation():
    from storage.profile import update_profile_field

    profile = {"identity": {}, "preferences": {}, "habits": [], "plans": {}, "updated_at": ""}
    profile = update_profile_field(profile, "plans.current_study", "LangChain")
    assert profile["plans"]["current_study"] == "LangChain"
    assert profile["updated_at"] != ""

    profile = update_profile_field(profile, "identity.name", "李明")
    assert profile["identity"]["name"] == "李明"


def test_backup_rotation(tmp_path):
    import storage.profile as pf
    pf.PROFILE_DIR = tmp_path
    pf.BACKUP_DIR = tmp_path / "profile_backups"
    pf.MAX_BACKUPS = 3

    pf.save_profile({"identity": {"name": "u1"}, "preferences": {}, "habits": [], "plans": {}, "updated_at": "t1"}, "test_user")
    pf.save_profile({"identity": {"name": "u2"}, "preferences": {}, "habits": [], "plans": {}, "updated_at": "t2"}, "test_user")
    pf.save_profile({"identity": {"name": "u3"}, "preferences": {}, "habits": [], "plans": {}, "updated_at": "t3"}, "test_user")
    pf.save_profile({"identity": {"name": "u4"}, "preferences": {}, "habits": [], "plans": {}, "updated_at": "t4"}, "test_user")

    backups = list(pf.BACKUP_DIR.glob("test_user_*.json"))
    assert len(backups) == 3  # MAX_BACKUPS


def test_load_profile_corrupt(tmp_path):
    import storage.profile as pf
    pf.PROFILE_DIR = tmp_path
    pf.BACKUP_DIR = tmp_path / "profile_backups"

    path = pf.PROFILE_DIR / "test_user.json"
    path.write_text("{invalid json}", encoding="utf-8")
    profile = pf.load_profile("test_user")
    assert "identity" in profile


def test_update_field_creates_nested_dicts():
    from storage.profile import update_profile_field

    profile = {"identity": {}, "preferences": {}, "habits": [], "plans": {}, "updated_at": ""}
    profile = update_profile_field(profile, "a.b.c", "deep")
    assert profile["a"]["b"]["c"] == "deep"
