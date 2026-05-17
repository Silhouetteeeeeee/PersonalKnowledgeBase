"""Spaced-repetition thinker module.

SM-2 algorithm, review push, feedback handling, weekly integration.
Completely independent of the LangGraph Q&A flow.
"""

import logging
import re
from datetime import datetime, timedelta

from pydantic import BaseModel, Field

from agent.utils.llm import LLM
from storage.models import (
    get_due_reviews,
    has_pending_review,
    get_sent_review_by_marker,
    get_reviewed_pages_since,
    process_review_feedback,
)
from storage.wiki_storage import read_page

logger = logging.getLogger(__name__)

# ── SM-2 constants ──

MIN_EF = 1.3
MAX_EF = 3.0
QUALITY_PERFECT = 5     # 记住了
QUALITY_HESITANT = 3    # 模糊
QUALITY_FORGOT = 1      # 忘了

FEEDBACK_KEYWORDS = {
    "记住了": QUALITY_PERFECT,
    "模糊": QUALITY_HESITANT,
    "忘了": QUALITY_FORGOT,
}


def _sm2_update(quality: int, easiness_factor: float, interval: int, repetitions: int) -> tuple[float, int, int]:
    """Compute new SM-2 parameters given a quality rating (0-5).

    Returns (new_ef, new_interval, new_repetitions).
    """
    if not 0 <= quality <= 5:
        raise ValueError(f"quality must be 0-5, got {quality}")

    if quality < 3:
        repetitions = 0
        interval = 1
    else:
        if repetitions == 0:
            interval = 1
        elif repetitions == 1:
            interval = 6
        else:
            interval = round(interval * easiness_factor)
        repetitions += 1

    ef = easiness_factor + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
    ef = max(MIN_EF, min(MAX_EF, ef))

    return ef, interval, repetitions


# ── LLM output models ──

class ReviewContent(BaseModel):
    summary: str = Field(description="Review summary of the wiki page, ~100-200 chars")
    key_points: list[str] = Field(description="3-5 key knowledge points from this page")
    review_question: str = Field(description="A self-test question to check recall")


class WeeklyIntegration(BaseModel):
    title: str = Field(description="Title for the weekly knowledge integration")
    content: str = Field(description="Full markdown content for the weekly review")
    new_questions: list[str] = Field(description="New questions raised by cross-linking these topics")


# ── Review generation ──

def _generate_review_content(page_title: str, page_body: str) -> ReviewContent | None:
    """LLM generates a review summary + key points + question from wiki body."""
    prompt = (
        "You are a spaced-repetition tutor. Given a wiki page, generate review material.\n\n"
        f"## Page Title\n{page_title}\n\n"
        f"## Page Content\n{page_body[:2000]}\n\n"
        "## Output Requirements\n"
        "1. summary: A concise review (~100-200 chars) covering the core concept\n"
        "2. key_points: 3-5 bullet-point knowledge items\n"
        "3. review_question: One self-test question to verify recall\n"
        "Use Chinese for explanations, English for technical terms."
    )
    return LLM.generate_structured(prompt, ReviewContent, use_language=False)


def _generate_weekly_integration(pages: list[dict]) -> WeeklyIntegration | None:
    """LLM generates a cross-page integration from reviewed pages."""
    pages_text = ""
    for p in pages:
        file_data = read_page(p["file_path"])
        body = file_data["body"][:1000] if file_data else "(content unavailable)"
        pages_text += f"\n### {p['title']}\n{body}\n---\n"

    prompt = (
        "You are a knowledge integration expert. Given multiple wiki pages reviewed this week, "
        "generate an integrated summary that connects them, identifies patterns, "
        "and raises new questions.\n\n"
        "## Pages Reviewed This Week\n"
        f"{pages_text}\n\n"
        "## Output Requirements\n"
        "1. title: A concise title for this integration\n"
        "2. content: Full markdown (~300-500 chars) connecting the topics, "
        "pointing out relationships, contrasts, and deeper insights\n"
        "3. new_questions: 2-3 new questions that arise from combining these topics\n"
        "Use Chinese for explanations, English for technical terms."
    )
    return LLM.generate_structured(prompt, WeeklyIntegration, use_language=False)


# ── Public API ──

def get_review_marker(page_id: int) -> str:
    """Generate a unique marker for a review message, e.g. #review_42_20260517."""
    return f"#review_{page_id}_{datetime.now().strftime('%Y%m%d')}"


def check_due_reviews(user_id: str = "") -> list[dict]:
    """Check for due reviews, build messages, return list of prepared reviews.

    Called by APScheduler. Returns list of dicts ready for sending by bot.py.

    Returns list of dicts: [{marker_id, page_id, page_title, message, schedule_id}, ...]
    """
    if not user_id:
        logger.warning("check_due_reviews: no user_id provided")
        return []

    due = get_due_reviews(limit=10)
    if not due:
        logger.info("Thinker: no pages due for review")
        return []

    pushed = []
    for item in due:
        page_id = item["page_id"]
        page_title = item["title"]
        file_path = item["file_path"]

        # Skip if pending review exists
        if has_pending_review(page_id):
            logger.info("Thinker: page %d already has pending review, skipping", page_id)
            continue

        # Read wiki content
        page_data = read_page(file_path)
        if page_data is None:
            logger.warning("Thinker: page %s not found on disk, skipping", file_path)
            continue

        # Generate review content
        review = _generate_review_content(page_title, page_data["body"])
        if review is None:
            logger.warning("Thinker: failed to generate review for '%s'", page_title)
            continue

        marker = get_review_marker(page_id)

        # Build message
        key_points = "\n".join(f"- {kp}" for kp in review.key_points)
        message = (
            f"🧠 **复习提醒：{page_title}**\n\n"
            f"**摘要：** {review.summary}\n\n"
            f"**关键知识点：**\n{key_points}\n\n"
            f"**自测问题：** {review.review_question}\n\n"
            f"**反馈：** 引用此消息回复「记住了」「模糊」或「忘了」\n\n"
            f"{marker}"
        )

        try:
            pushed.append({
                "marker_id": marker,
                "page_id": page_id,
                "page_title": page_title,
                "message": message,
                "schedule_id": item["id"],
            })
            logger.info("Thinker: prepared review for '%s' (%s)", page_title, marker)
        except Exception as e:
            logger.error("Thinker: failed to prepare review for '%s': %s", page_title, e)
            continue

    return pushed


def handle_review_response(quote_text: str, user_feedback: str) -> str:
    """Process a user's quote reply to a review message.

    Args:
        quote_text: The quoted message text (contains #review_ marker).
        user_feedback: User's reply text ("记住了", "模糊", or "忘了").

    Returns:
        A response string to send back to the user.
    """
    # Extract marker from quoted text
    match = re.search(r'(#review_\d+_\d+)', quote_text)
    if not match:
        return "抱歉，无法识别这条复习消息。请直接引用我发送的复习消息。😅"

    marker_id = match.group(1)

    # Look up the sent review
    sent = get_sent_review_by_marker(marker_id)
    if sent is None:
        return "这条复习消息已过期或无法找到对应记录。不过没关系，继续加油学习吧！💪"

    if sent["status"] == "reviewed":
        return "这条复习你已经回复过啦！记得保持复习节奏哦～"

    # Normalize feedback
    quality = None
    for keyword, q in FEEDBACK_KEYWORDS.items():
        if keyword in user_feedback:
            quality = q
            break

    if quality is None:
        return "请回复「记住了」「模糊」或「忘了」来告诉我你掌握得怎么样～"

    # Get SM-2 params
    from storage.database import get_connection
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM review_schedule WHERE id = ?", (sent["schedule_id"],)
    ).fetchone()
    conn.close()

    if row is None:
        return "找不到对应的复习记录。不过没关系，继续加油！💪"

    # SM-2 update (atomic via process_review_feedback)
    new_ef, new_interval, new_reps = _sm2_update(
        quality, row["easiness_factor"], row["interval_days"], row["repetitions"],
    )
    next_review = (datetime.now() + timedelta(days=new_interval)).strftime("%Y-%m-%d %H:%M:%S")
    process_review_feedback(
        schedule_id=row["id"],
        sent_id=sent["id"],
        easiness_factor=new_ef,
        interval_days=new_interval,
        repetitions=new_reps,
        next_review_at=next_review,
        quality=quality,
    )

    # Build confirmation
    quality_labels = {5: "记住了 ✅", 3: "模糊 🤔", 1: "忘了 ❌"}
    label = quality_labels.get(quality, str(quality))
    next_date = next_review[:10]

    return (
        f"收到！你的反馈：**{label}**\n\n"
        f"下次复习：**{next_date}**（{new_interval} 天后）\n"
        f"坚持复习，效果更佳！📚"
    )


def generate_weekly_integration(user_id: str = "") -> dict | None:
    """Generate a weekly knowledge integration.

    Called by APScheduler (e.g., every Monday 10:00).
    Returns a message dict with "msgtype" and "markdown" keys, or None if skipped.
    """
    if not user_id:
        logger.warning("generate_weekly_integration: no user_id provided")
        return None

    pages = get_reviewed_pages_since(days=7)
    if len(pages) < 2:
        logger.info("Thinker: only %d pages reviewed this week, skipping integration", len(pages))
        return None

    integration = _generate_weekly_integration(pages)
    if integration is None:
        logger.warning("Thinker: failed to generate weekly integration")
        return None

    new_qs = "\n".join(f"- {q}" for q in integration.new_questions)
    content = (
        f"📚 **本周知识整合：{integration.title}**\n\n"
        f"{integration.content}\n\n"
        f"**延伸思考：**\n{new_qs}\n\n"
        f"本周共复习了 {len(pages)} 个知识点，继续保持！🎯"
    )

    logger.info("Thinker: prepared weekly integration '%s'", integration.title)
    return {
        "msgtype": "markdown",
        "markdown": {"content": content},
    }
