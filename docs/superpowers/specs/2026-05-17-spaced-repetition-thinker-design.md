# Spaced Repetition Thinker Module Design

**Goal:** Build a standalone "thinker" module that periodically reviews wiki pages using SM-2 spaced repetition, pushes review reminders via WeChat Work, and processes user feedback to dynamically adjust review schedules.

**Non-goal:** Modify the existing LangGraph conversation flow. The thinker runs independently and does not affect Q&A.

---

## Interaction Design

### Review Push (Bot → User)

- APScheduler periodic job checks `review_schedule` for due pages
- For each due page: LLM generates a short review message containing:
  - Wiki page link
  - Summary of key points
  - Knowledge points extracted from the page
- Message is sent as plain markdown with a `#review_<pageId>_<date>` marker tag
- Record is inserted into `sent_reviews` table

### User Response (User → Bot)

- User **quotes** the thinker's markdown message and replies with feedback
- bot.py examines `body.quote.text` — if it contains `#review_`, routes to thinker handler (bypasses normal graph)
- User feedback is one of: **记住了 / 模糊 / 忘了**
- Bot maps feedback to SM-2 quality (5/3/1), updates `review_schedule`, confirms in reply

### Weekly Integration

- Separate APScheduler cron job runs weekly
- Queries all pages reviewed in the past week
- LLM generates a comprehensive integration note: cross-linking related knowledge, raising new questions, suggesting further reading
- Pushed as a normal markdown message (no `#review_` marker, no feedback expected)

---

## Data Model

### review_schedule

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | |
| page_id | INTEGER UNIQUE FK | References pages(id) |
| easiness_factor | REAL NOT NULL DEFAULT 2.5 | SM-2 EF |
| interval_days | INTEGER NOT NULL DEFAULT 1 | Current interval in days |
| repetitions | INTEGER NOT NULL DEFAULT 0 | Consecutive correct recalls |
| next_review_at | TEXT NOT NULL | ISO datetime |
| last_reviewed_at | TEXT | ISO datetime |
| last_quality | INTEGER | 0-5 SM-2 quality |
| created_at | TEXT | Auto timestamp |

### sent_reviews

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | |
| schedule_id | INTEGER FK | References review_schedule(id) |
| page_id | INTEGER FK | References pages(id) |
| marker_id | TEXT UNIQUE | `review_{pageId}_{YYYYMMDD}` |
| sent_at | TEXT | Auto timestamp |
| status | TEXT | pending / reviewed / expired |

---

## SM-2 Algorithm

Quality mapping for 3-choice feedback:

| User says | Quality | Meaning |
|-----------|---------|---------|
| 记住了 | 5 | Perfect recall |
| 模糊 | 3 | Recalled with difficulty |
| 忘了 | 1 | Forgotten / incorrect |

On each review:
```
if quality >= 3:
    if repetitions == 0:
        interval = 1
    elif repetitions == 1:
        interval = 6
    else:
        interval = round(interval * easiness_factor)
    repetitions += 1
else:
    repetitions = 0
    interval = 1

easiness_factor = max(1.3, EF + (0.1 - (5 - q) * (0.08 + (5 - q) * 0.02)))
next_review_at = now + interval days
```

---

## File Changes

| File | Change |
|------|--------|
| `server/thinker.py` | **New** — SM-2 logic, review generation, feedback handling, weekly integration |
| `server/bot.py` | **Modify** — add thinker route before graph invoke, add APScheduler job |
| `agent/nodes/store.py` | **Modify** — after creating a page, initialize `review_schedule` |
| `storage/database.py` | **Modify** — add `review_schedule` and `sent_reviews` tables |
| `server/config.py` | **Modify** — add thinker config (interval, user_id) |

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     APScheduler                          │
│  ┌────────────────┐  ┌─────────────────────────────┐    │
│  │ thinker_check  │  │  weekly_integration         │    │
│  │ (every 4h)     │  │  (every Monday 10:00)       │    │
│  └───────┬────────┘  └──────────┬──────────────────┘    │
│          │                      │                        │
└──────────┼──────────────────────┼────────────────────────┘
           │                      │
           ▼                      ▼
┌─────────────────────────────────────────────────────────┐
│                   server/thinker.py                      │
│  ┌──────────────────┐  ┌────────────────────────────┐   │
│  │ check_due_reviews│  │ generate_weekly_integration │   │
│  │ → LLM summarize  │  │ → LLM cross-link + expand  │   │
│  │ → push message   │  │ → push message             │   │
│  └────────┬─────────┘  └────────────────────────────┘   │
│           │                                              │
│  ┌────────▼─────────┐                                    │
│  │ handle_review_   │                                    │
│  │ response         │                                    │
│  │ → SM-2 update    │                                    │
│  │ → confirm reply  │                                    │
│  └──────────────────┘                                    │
└─────────────────────────────────────────────────────────┘

┌─────────────────────┐
│      bot.py         │
│  _on_text:          │
│   if quote has      │
│   #review_ → thinker│
│   else → graph      │
└─────────────────────┘
```

---

## Edge Cases

- **No pages due**: `check_due_reviews` logs and returns, no message sent
- **User responds with unrecognized feedback**: thinker replies asking to use "记住了/模糊/忘了"
- **Page deleted between push and response**: `handle_review_response` looks up page, skips update if 404
- **Same page due before previous review is answered**: skip if `sent_reviews` has pending entry for this page
- **Empty weekly review pool**: no pages reviewed this week, skip integration
- **SM-2 EF ceiling**: cap at 3.0 to prevent absurdly long intervals
