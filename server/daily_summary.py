"""Daily knowledge summary: query yesterday's pages, LLM summarize, push to WeChat."""

import json
import logging
import os
from datetime import datetime, timedelta

from pydantic import BaseModel, Field

from agent.utils.llm import LLM
from storage.database import get_connection
from storage import wiki_storage

logger = logging.getLogger(__name__)


class SummaryOutput(BaseModel):
    summary: str = Field(description="Full markdown daily summary content")


def _get_yesterday_range() -> tuple[datetime, datetime]:
    """Return (yesterday_00:00, today_00:00) as datetimes."""
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    yesterday = today - timedelta(days=1)
    return yesterday, today


def _get_yesterday_pages() -> list[dict]:
    """Query all pages created or updated yesterday, with full content and source questions."""
    y_bound, t_bound = _get_yesterday_range()
    y_str = y_bound.strftime("%Y-%m-%d %H:%M:%S")
    t_str = t_bound.strftime("%Y-%m-%d %H:%M:%S")

    conn = get_connection()
    rows = conn.execute(
        """SELECT id, title, file_path, tags, sources, created_at, updated_at
           FROM pages
           WHERE status = 'active'
             AND ((created_at >= ? AND created_at < ?)
               OR (updated_at >= ? AND updated_at < ?))
           ORDER BY created_at DESC""",
        (y_str, t_str, y_str, t_str),
    ).fetchall()
    conn.close()

    if not rows:
        return []

    # Build page list, querying source_question from page_versions in one connection
    vconn = get_connection()
    pages = []
    for r in rows:
        fp = r["file_path"]
        full_path = os.path.join(wiki_storage.WIKI_DIR, fp)
        content = ""
        if os.path.exists(full_path):
            with open(full_path, encoding="utf-8") as f:
                raw = f.read()
            _, content = wiki_storage.parse_frontmatter(raw)

        # Look up source question from latest page_versions entry
        questions = []
        vr = vconn.execute(
            "SELECT source_question FROM page_versions WHERE page_id = ? ORDER BY version DESC LIMIT 1",
            (r["id"],),
        ).fetchone()
        if vr and vr["source_question"]:
            questions = [vr["source_question"]]

        pages.append({
            "id": r["id"],
            "title": r["title"],
            "file_path": r["file_path"],
            "tags": r["tags"],
            "sources": r["sources"],
            "created_at": r["created_at"],
            "updated_at": r["updated_at"],
            "content": content[:500],
            "source_questions": questions,
        })

    vconn.close()
    return pages


def _split_pages(
    pages: list[dict],
    y_bound: datetime,
    t_bound: datetime,
) -> tuple[list[dict], list[dict]]:
    """Split pages into 'new' (created yesterday) and 'updated' (modified yesterday)."""
    new = []
    updated = []
    for p in pages:
        created = datetime.strptime(p["created_at"], "%Y-%m-%d %H:%M:%S") if p.get("created_at") else t_bound
        if y_bound <= created < t_bound:
            new.append(p)
        else:
            updated.append(p)
    return new, updated


def _generate_summary_text(pages: list[dict]) -> str:
    """Use LLM to generate a structured markdown daily summary."""
    y_bound, t_bound = _get_yesterday_range()
    new_pages, updated_pages = _split_pages(pages, y_bound, t_bound)

    lines = [
        f"你是一个知识库助手。请根据以下{y_bound.date()}新增和更新的知识页面，生成一篇中文知识日报。",
        "",
        "## 输出要求",
        "- 使用markdown格式，标题用emoji：📌 新增知识、📝 更新知识",
        "- 相关主题的页面归为一组",
        "- 每个页面给出标题、摘要、以及它来自什么用户问题",
        "- 如果用户问了多个相关问题，在最后加一个「💡 知识关联」部分说明这些知识之间的联系",
        "- 语言：中文，技术术语保留英文",
        "",
    ]

    if new_pages:
        lines.append(f"## 新增页面（{len(new_pages)}篇）")
        for p in new_pages:
            lines.append(f"\n### {p['title']}")
            questions = p.get("source_questions", [])
            if questions:
                for q in questions:
                    lines.append(f"- 用户问：{q}")
            if p.get("content"):
                lines.append(f"- 摘要：{p['content'][:200]}")
        lines.append("")

    if updated_pages:
        lines.append(f"## 更新页面（{len(updated_pages)}篇）")
        for p in updated_pages:
            lines.append(f"\n### {p['title']}")
            questions = p.get("source_questions", [])
            if questions:
                for q in questions:
                    lines.append(f"- 用户问：{q}")
            if p.get("content"):
                lines.append(f"- 摘要：{p['content'][:200]}")
        lines.append("")

    prompt = "\n".join(lines)

    result = LLM.generate_structured(prompt, SummaryOutput)
    if result is None:
        logger.error("LLM summary generation returned None")
        return ""
    return result.summary


async def send_daily_summary(client, user_id: str) -> bool:
    """Query yesterday's pages, generate summary, push to WeChat user.

    Args:
        client: WSClient instance for sending messages.
        user_id: WeChat user ID to push to.

    Returns:
        True if message was sent, False if skipped or failed.
    """
    pages = _get_yesterday_pages()
    if not pages:
        logger.info("No new or updated pages yesterday, skipping daily summary")
        return False

    logger.info("Generating daily summary for %d pages", len(pages))
    summary = _generate_summary_text(pages)
    if not summary:
        logger.error("Failed to generate summary text")
        return False

    try:
        client.send_message(user_id, {
            "msgtype": "markdown",
            "markdown": {"content": summary},
        })
        logger.info("Daily summary pushed to user %s (%d pages)", user_id, len(pages))
        return True
    except Exception as e:
        logger.error("Failed to send daily summary to %s: %s", user_id, e)
        return False
