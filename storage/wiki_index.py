"""Manage data/wiki/index.md: build from DB, format for prompt injection."""

import logging

from storage.wiki_storage import write_index
from storage.models import get_all_pages_index

logger = logging.getLogger(__name__)


_INDEX_HEADER = """# Wiki Index

| 页面 | 标签 | 来源 | 最后更新 |
|------|------|------|----------|
"""


def rebuild_index() -> None:
    """Read all active pages from DB and rewrite index.md."""
    pages = get_all_pages_index()
    lines = [_INDEX_HEADER]
    for p in pages:
        tags = p.get("tags", [])
        tags_str = ", ".join(tags) if isinstance(tags, list) else str(tags)
        sources = p.get("sources", [])
        src_count = len(sources) if isinstance(sources, list) else 0
        updated = (p.get("updated_at") or "")[:10]
        title = p["title"]
        lines.append(f"| [[{title}]] | {tags_str} | {src_count} 条对话 | {updated} |\n")

    write_index("".join(lines))
    logger.info("Rebuilt index.md with %d pages", len(pages))


def get_index_for_prompt(max_entries: int = 50) -> str:
    """Get a condensed index text for LLM prompt injection.

    Returns a plain list format that fits within prompt limits.
    """
    pages = get_all_pages_index()
    if not pages:
        return "(当前 Wiki 为空，暂无页面)"

    lines = ["当前 Wiki 页面索引："]
    for p in pages[:max_entries]:
        title = p["title"]
        tags = p.get("tags", [])
        tags_str = ", ".join(tags[:3]) if isinstance(tags, list) else ""
        lines.append(f"  - {title} [{tags_str}]")
    if len(pages) > max_entries:
        lines.append(f"  ...及其他 {len(pages) - max_entries} 个页面")

    return "\n".join(lines)
