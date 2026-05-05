import json
import logging
import shutil
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

PROFILE_DIR = Path("data")
PROFILE_PATH = PROFILE_DIR / "profile.json"
BACKUP_DIR = PROFILE_DIR / "profile_backups"
MAX_BACKUPS = 10

_DEFAULT_PROFILE = {
    "identity": {},
    "preferences": {},
    "habits": [],
    "plans": {},
    "updated_at": "",
}


def _ensure_dirs():
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)


def load_profile() -> dict:
    """Load profile from JSON file. Returns default structure if not found or corrupt."""
    if not PROFILE_PATH.exists():
        logger.info("No profile file found, returning default")
        return dict(_DEFAULT_PROFILE)
    try:
        with open(PROFILE_PATH, encoding="utf-8") as f:
            data = json.load(f)
        for k, v in _DEFAULT_PROFILE.items():
            data.setdefault(k, v)
        return data
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to load profile (%s), returning default", e)
        return dict(_DEFAULT_PROFILE)


def save_profile(profile: dict) -> None:
    """Backup current profile, write new profile, clean old backups."""
    _ensure_dirs()
    if PROFILE_PATH.exists():
        timestamp = datetime.now().strftime("%Y-%m-%dT%H%M%S%f")
        backup_path = BACKUP_DIR / f"{timestamp}.json"
        shutil.copy2(PROFILE_PATH, backup_path)
        _clean_old_backups()
    with open(PROFILE_PATH, "w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)
    logger.info("Profile saved to %s", PROFILE_PATH)


def _clean_old_backups():
    backups = sorted(BACKUP_DIR.glob("*.json"))
    while len(backups) > MAX_BACKUPS:
        backups[0].unlink()
        backups = sorted(BACKUP_DIR.glob("*.json"))


def update_profile_field(profile: dict, field: str, value: object) -> dict:
    """Update a field supporting dot-notation (e.g. 'plans.current_study').

    Returns the updated profile dict. Does NOT auto-save — caller must
    call save_profile() separately.
    """
    keys = field.split(".")
    obj = profile
    for key in keys[:-1]:
        if key not in obj or not isinstance(obj[key], dict):
            obj[key] = {}
        obj = obj[key]
    obj[keys[-1]] = value
    profile["updated_at"] = datetime.now().isoformat()
    return profile
