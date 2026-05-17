"""One-time migration: initialize review_schedule for existing wiki pages.

Run: python -m scripts.migrate_review_schedule
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from storage.database import get_connection, init_db
from storage.models import init_review_schedule


def main():
    init_db()
    conn = get_connection()
    pages = conn.execute(
        "SELECT id, title FROM pages WHERE status = 'active'"
    ).fetchall()
    conn.close()

    count = 0
    for p in pages:
        if init_review_schedule(p["id"]):
            count += 1

    print(f"Initialized review schedule for {count} existing pages")


if __name__ == "__main__":
    main()
