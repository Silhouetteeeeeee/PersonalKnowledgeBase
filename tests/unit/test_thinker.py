"""Unit tests for spaced repetition thinker module."""
import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from storage.database import init_db


@pytest.fixture(autouse=True)
def temp_db(monkeypatch, tmp_path):
    db_dir = tmp_path / "data"
    db_dir.mkdir()
    monkeypatch.setattr("storage.database.DB_DIR", str(db_dir))
    monkeypatch.setattr("storage.database.DB_PATH", str(db_dir / "knowledge.db"))
    init_db()


@pytest.fixture
def wiki_dir(monkeypatch, tmp_path):
    wiki_dir = tmp_path / "wiki"
    pages_dir = wiki_dir / "pages"
    pages_dir.mkdir(parents=True)
    monkeypatch.setattr("storage.wiki_storage.WIKI_DIR", str(wiki_dir))
    return wiki_dir


@pytest.fixture
def sample_page(wiki_dir):
    """Insert a sample wiki page and return its id."""
    from storage.database import get_connection
    content = "---\ntitle: test page\ntags: [test]\n---\n\nTest content for SM-2 review."
    file_path = "pages/test-page.md"
    # Write the disk file
    full_path = wiki_dir / file_path
    full_path.parent.mkdir(parents=True, exist_ok=True)
    full_path.write_text(content, encoding="utf-8")

    conn = get_connection()
    conn.execute(
        "INSERT INTO pages (title, file_path, tags) VALUES (?, ?, ?)",
        ("test page", file_path, '["test"]'),
    )
    conn.commit()
    pid = conn.execute("SELECT id FROM pages WHERE title = ?", ("test page",)).fetchone()[0]
    conn.close()
    return pid


def test_sm2_perfect_recall():
    """Quality=5 → EF increases, interval grows."""
    from server.thinker import _sm2_update
    ef, interval, reps = _sm2_update(5, 2.5, 1, 0)
    assert reps == 1
    assert interval == 1
    assert ef > 2.5


def test_sm2_second_review():
    """Second perfect recall → interval = 6."""
    from server.thinker import _sm2_update
    ef, interval, reps = _sm2_update(5, 2.5, 1, 1)
    assert reps == 2
    assert interval == 6


def test_sm2_subsequent_reviews():
    """Third+ perfect recall → interval = round(prev * EF)."""
    from server.thinker import _sm2_update
    ef, interval, reps = _sm2_update(5, 2.5, 6, 2)
    assert reps == 3
    assert interval == 15  # round(6 * 2.5)


def test_sm2_forgot_resets():
    """Quality=1 → reps=0, interval=1."""
    from server.thinker import _sm2_update
    ef, interval, reps = _sm2_update(1, 2.5, 6, 3)
    assert reps == 0
    assert interval == 1
    assert ef < 2.5  # EF decreases


def test_sm2_ef_floor():
    """EF should not go below MIN_EF (1.3)."""
    from server.thinker import _sm2_update
    ef, interval, reps = _sm2_update(0, 1.3, 1, 0)
    assert ef >= 1.3


def test_sm2_ef_ceiling():
    """EF should not exceed MAX_EF (3.0)."""
    from server.thinker import _sm2_update
    ef, interval, reps = _sm2_update(5, 3.0, 1, 0)
    assert ef <= 3.0


def test_sm2_invalid_quality():
    """Quality outside 0-5 should raise ValueError."""
    from server.thinker import _sm2_update
    with pytest.raises(ValueError):
        _sm2_update(6, 2.5, 1, 0)
    with pytest.raises(ValueError):
        _sm2_update(-1, 2.5, 1, 0)


def test_get_review_marker():
    """Marker format: #review_{pageId}_{date}."""
    from server.thinker import get_review_marker
    marker = get_review_marker(42)
    assert marker.startswith("#review_42_")
    assert len(marker) == len("#review_42_20260517")


def test_handle_review_response_bad_quote():
    """No marker in quote → friendly error."""
    from server.thinker import handle_review_response
    result = handle_review_response("some random text", "记住了")
    assert "无法识别" in result


def test_handle_review_response_no_feedback(sample_page):
    """Unrecognized feedback → guidance message."""
    from storage.models import init_review_schedule, record_sent_review
    from server.thinker import handle_review_response
    sid = init_review_schedule(sample_page)
    record_sent_review(sid, sample_page, "#review_1_20260517")
    result = handle_review_response("#review_1_20260517 hello", "huh?")
    assert "记住了" in result


def test_init_review_schedule_creates_record(sample_page):
    """init_review_schedule inserts a row with default 1-day interval."""
    from storage.models import init_review_schedule
    sid = init_review_schedule(sample_page)
    assert sid > 0

    from storage.database import get_connection
    conn = get_connection()
    row = conn.execute("SELECT * FROM review_schedule WHERE id = ?", (sid,)).fetchone()
    conn.close()
    assert row is not None
    assert row["page_id"] == sample_page
    assert row["interval_days"] == 1
    assert row["repetitions"] == 0
    assert row["easiness_factor"] == 2.5


def test_get_due_reviews_returns_due_only(sample_page):
    """Only schedules with next_review_at <= now are returned."""
    from storage.models import init_review_schedule, get_due_reviews
    # Default next_review_at = now + 1 day → not due
    init_review_schedule(sample_page)
    due = get_due_reviews(limit=10)
    assert len(due) == 0

    # Set to past → should be due
    from storage.database import get_connection
    conn = get_connection()
    conn.execute(
        "UPDATE review_schedule SET next_review_at = ? WHERE page_id = ?",
        ((datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S"), sample_page),
    )
    conn.commit()
    conn.close()

    due = get_due_reviews(limit=10)
    assert len(due) >= 1
    assert due[0]["page_id"] == sample_page


def test_update_review_schedule_persists(sample_page):
    """After update, SM-2 params are saved."""
    from storage.models import init_review_schedule, update_review_schedule
    from storage.database import get_connection

    sid = init_review_schedule(sample_page)
    # Make it due first
    conn = get_connection()
    conn.execute(
        "UPDATE review_schedule SET next_review_at = ? WHERE id = ?",
        ((datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S"), sid),
    )
    conn.commit()
    conn.close()

    update_review_schedule(sid, 2.6, 6, 1,
                           (datetime.now() + timedelta(days=6)).strftime("%Y-%m-%d %H:%M:%S"), 5)

    conn = get_connection()
    row = conn.execute("SELECT * FROM review_schedule WHERE id = ?", (sid,)).fetchone()
    conn.close()
    assert row["easiness_factor"] == 2.6
    assert row["interval_days"] == 6
    assert row["repetitions"] == 1
    assert row["last_quality"] == 5


def test_has_pending_review(sample_page):
    """has_pending_review detects unanswered sent reviews."""
    from storage.models import init_review_schedule, has_pending_review, record_sent_review
    sid = init_review_schedule(sample_page)

    # No sent reviews yet
    assert has_pending_review(sample_page) is False

    record_sent_review(sid, sample_page, "#review_1_test")
    assert has_pending_review(sample_page) is True


def test_get_reviewed_pages_since(sample_page):
    """get_reviewed_pages_since returns pages reviewed in the window."""
    from storage.models import init_review_schedule, update_review_schedule, get_reviewed_pages_since
    from storage.database import get_connection

    sid = init_review_schedule(sample_page)

    # Simulate a recent review
    update_review_schedule(sid, 2.5, 6, 1,
                           (datetime.now() + timedelta(days=6)).strftime("%Y-%m-%d %H:%M:%S"), 5)

    pages = get_reviewed_pages_since(days=7)
    assert len(pages) >= 1
    assert pages[0]["page_id"] == sample_page


def test_get_sent_review_by_marker(sample_page):
    """Can look up sent review by marker."""
    from storage.models import init_review_schedule, record_sent_review, get_sent_review_by_marker
    sid = init_review_schedule(sample_page)
    record_sent_review(sid, sample_page, "#review_99_test")

    sent = get_sent_review_by_marker("#review_99_test")
    assert sent is not None
    assert sent["page_id"] == sample_page


def test_handle_review_response_already_reviewed(sample_page):
    """Already reviewed → message says already answered."""
    from storage.models import init_review_schedule, record_sent_review
    from server.thinker import handle_review_response
    sid = init_review_schedule(sample_page)
    record_sent_review(sid, sample_page, "#review_1_999")

    from storage.database import get_connection
    conn = get_connection()
    conn.execute("UPDATE sent_reviews SET status = 'reviewed' WHERE marker_id = ?",
                 ("#review_1_999",))
    conn.commit()
    conn.close()

    result = handle_review_response("#review_1_999", "记住了")
    assert "已经回复过" in result


def test_handle_review_response_success(sample_page):
    """Full feedback flow: marker found → SM-2 update → confirmation."""
    from storage.models import init_review_schedule, record_sent_review
    from storage.database import get_connection
    from server.thinker import handle_review_response

    sid = init_review_schedule(sample_page)
    # Make it due
    conn = get_connection()
    conn.execute(
        "UPDATE review_schedule SET next_review_at = ? WHERE id = ?",
        (("2000-01-01 00:00:00", sid)),
    )
    conn.commit()
    conn.close()

    record_sent_review(sid, sample_page, "#review_2_456")

    result = handle_review_response("#review_2_456", "记住了")
    assert "收到" in result
    assert "下次复习" in result

    # Verify SM-2 was updated atomically
    conn = get_connection()
    row = conn.execute("SELECT * FROM review_schedule WHERE id = ?", (sid,)).fetchone()
    sent = conn.execute("SELECT * FROM sent_reviews WHERE marker_id = ?", ("#review_2_456",)).fetchone()
    conn.close()
    assert row["repetitions"] == 1
    assert row["last_quality"] == 5
    assert sent["status"] == "reviewed"


def test_check_due_reviews_no_user():
    """No user_id provided → returns empty list."""
    from server.thinker import check_due_reviews
    result = check_due_reviews(user_id="")
    assert result == []


def test_process_review_feedback_atomic(sample_page):
    """process_review_feedback updates both schedule and sent_review atomically."""
    from storage.models import init_review_schedule, record_sent_review, process_review_feedback
    from storage.database import get_connection

    sid = init_review_schedule(sample_page)
    sent_id = record_sent_review(sid, sample_page, "#review_atomic_test")

    process_review_feedback(sid, sent_id, 2.7, 10, 2, "2026-06-01 10:00:00", 5)

    conn = get_connection()
    row = conn.execute("SELECT * FROM review_schedule WHERE id = ?", (sid,)).fetchone()
    sent = conn.execute("SELECT * FROM sent_reviews WHERE id = ?", (sent_id,)).fetchone()
    conn.close()

    assert row["easiness_factor"] == 2.7
    assert row["interval_days"] == 10
    assert sent["status"] == "reviewed"
