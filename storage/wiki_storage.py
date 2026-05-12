"""Filesystem operations for wiki pages: read/write files, parse frontmatter,
extract [[wikilinks]], title↔filename conversion, checksum."""

import hashlib
import logging
import os
import re
from typing import Optional

logger = logging.getLogger(__name__)

WIKI_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "wiki")
PAGES_DIR = os.path.join(WIKI_DIR, "pages")
INDEX_PATH = os.path.join(WIKI_DIR, "index.md")
SCHEMA_PATH = os.path.join(WIKI_DIR, "SCHEMA.md")


def ensure_dirs() -> None:
    os.makedirs(PAGES_DIR, exist_ok=True)


def title_to_filename(title: str) -> str:
    """Convert a page title to a filename.

    Examples:
        "Django ORM"     -> "django-orm.md"
        "Python 基础"    -> "python基础.md"
        "HTTP/2 协议"    -> "http2协议.md"
    """
    name = title.lower().strip()
    name = name.replace(" ", "-")
    # Remove characters unsafe for filenames
    name = re.sub(r'[<>:"/\\|?*]', "", name)
    return name + ".md"


def read_schema() -> str:
    """Read SCHEMA.md content. Returns empty string if file doesn't exist."""
    if not os.path.exists(SCHEMA_PATH):
        return ""
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        return f.read()


def parse_frontmatter(content: str) -> tuple[dict, str]:
    """Parse YAML-like frontmatter from markdown content.

    Handles the simple subset used by wiki pages (strings, lists, dates).
    Returns (metadata_dict, body_string).
    """
    match = re.match(r'^---\n(.*?)\n---\n(.*)', content, re.DOTALL)
    if not match:
        return {}, content.strip()

    raw = match.group(1)
    body = match.group(2).strip()
    metadata = _parse_simple_yaml(raw)
    return metadata, body


def _parse_simple_yaml(raw: str) -> dict:
    """Parse a simplified YAML subset: strings and lists only."""
    result = {}
    for line in raw.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        m = re.match(r'(\w+):\s*(.*)', line)
        if not m:
            continue
        key = m.group(1)
        val = m.group(2).strip()

        if val.startswith("[") and val.endswith("]"):
            # List: [a, b, c]
            items = [x.strip().strip("\"'") for x in val[1:-1].split(",") if x.strip()]
            result[key] = items
        elif val and val[0] in ("'", '"') and val[-1] == val[0]:
            result[key] = val[1:-1]
        else:
            result[key] = val
    return result


def build_frontmatter(title: str, tags: list[str], sources: list[str],
                      created: str = "", updated: str = "") -> str:
    """Build YAML frontmatter string."""
    lines = ["---"]
    lines.append(f"title: {title}")
    lines.append(f"tags: [{', '.join(tags)}]")
    lines.append(f"sources: [{', '.join(sources)}]")
    if created:
        lines.append(f"created: {created}")
    if updated:
        lines.append(f"updated: {updated}")
    lines.append("---")
    return "\n".join(lines)


def read_page(file_path: str) -> Optional[dict]:
    """Read a wiki page from disk.

    Returns dict with keys: title, body, tags, sources, checksum, created
    Returns None if file does not exist.
    """
    full_path = os.path.join(WIKI_DIR, file_path)
    if not os.path.exists(full_path):
        return None

    with open(full_path, "r", encoding="utf-8") as f:
        content = f.read()

    metadata, body = parse_frontmatter(content)
    checksum = compute_checksum(content)

    return {
        "title": metadata.get("title", ""),
        "body": body,
        "tags": metadata.get("tags", []),
        "sources": metadata.get("sources", []),
        "checksum": checksum,
        "created": metadata.get("created", ""),
    }


def write_page(file_path: str, content: str) -> str:
    """Write a wiki page to disk. Returns SHA256 checksum of written content."""
    full_path = os.path.join(WIKI_DIR, file_path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(content)
    checksum = compute_checksum(content)
    logger.info("Wrote wiki page: %s (%d bytes, checksum=%s)",
                file_path, len(content.encode("utf-8")), checksum[:12])
    return checksum


def compute_checksum(content: str) -> str:
    """Compute SHA256 checksum of page content."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def extract_wikilinks(content: str) -> list[str]:
    """Extract unique [[link]] targets from content.

    Handles both [[title]] and [[title#section]] forms.
    Returns unique target titles (without section suffix).
    """
    links = re.findall(r'\[\[(.+?)\]\]', content)
    # Strip section references: [[title#section]] -> [[title]]
    titles = [link.split("#")[0].strip() for link in links]
    seen = set()
    result = []
    for t in titles:
        if t and t not in seen:
            seen.add(t)
            result.append(t)
    return result


def read_index() -> str:
    """Read index.md content. Returns empty string if it doesn't exist."""
    if not os.path.exists(INDEX_PATH):
        return ""
    with open(INDEX_PATH, "r", encoding="utf-8") as f:
        return f.read()


def write_index(content: str) -> None:
    """Write index.md content."""
    os.makedirs(WIKI_DIR, exist_ok=True)
    with open(INDEX_PATH, "w", encoding="utf-8") as f:
        f.write(content)
    logger.info("Updated index.md (%d bytes)", len(content.encode("utf-8")))
