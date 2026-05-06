import json
import logging
import shutil
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

PROFILE_DIR = Path("data") / "profiles"
BACKUP_DIR = PROFILE_DIR / "backups"
MAX_BACKUPS = 10

_DEFAULT_TEMPLATE = {
    "identity": {},
    "preferences": {},
    "habits": [],
    "plans": {},
    "updated_at": "",
}


def _default_profile() -> dict:
    """Return a fresh default profile dict (no shared mutable state)."""
    return {
        "identity": {},
        "preferences": {},
        "habits": [],
        "plans": {},
        "updated_at": "",
    }


def _ensure_dirs():
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)


def _profile_path(user_id: str) -> Path:
    return PROFILE_DIR / f"{user_id}.json"


def load_profile(user_id: str = "") -> dict:
    """Load profile for a specific user. Returns default structure if not found."""
    if not user_id:
        return _default_profile()
    path = _profile_path(user_id)
    if not path.exists():
        logger.debug("No profile file for user '%s', returning default", user_id)
        return _default_profile()
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for k, v in _DEFAULT_TEMPLATE.items():
            data.setdefault(k, v)
        return data
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to load profile for '%s' (%s), returning default", user_id, e)
        return _default_profile()


def save_profile(profile: dict, user_id: str = "") -> None:
    """Save profile for a specific user with backup rotation."""
    if not user_id:
        logger.warning("save_profile called without user_id, skipping")
        return
    _ensure_dirs()
    path = _profile_path(user_id)
    if path.exists():
        timestamp = datetime.now().strftime("%Y-%m-%dT%H%M%S%f")
        backup_path = BACKUP_DIR / f"{user_id}_{timestamp}.json"
        shutil.copy2(path, backup_path)
        _clean_old_backups(user_id)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)
    logger.info("Profile saved for user '%s'", user_id)


def _clean_old_backups(user_id: str):
    backups = sorted(BACKUP_DIR.glob(f"{user_id}_*.json"))
    while len(backups) > MAX_BACKUPS:
        backups[0].unlink()
        backups = sorted(BACKUP_DIR.glob(f"{user_id}_*.json"))


def update_profile_field(profile: dict, field: str, value: object) -> dict:
    """Update a field using dot-notation. Returns updated dict, does NOT save."""
    keys = field.split(".")
    obj = profile
    for key in keys[:-1]:
        if key not in obj or not isinstance(obj[key], dict):
            obj[key] = {}
        obj = obj[key]
    obj[keys[-1]] = value
    profile["updated_at"] = datetime.now().isoformat()
    return profile
