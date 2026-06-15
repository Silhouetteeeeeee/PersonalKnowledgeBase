"""Recover wiki page records from disk files into SQLite database.

Run: python -m scripts.recover_wiki_pages

This reads all .md files from data/wiki/pages/ and re-registers
them in the database (pages table + page_vectors), then rebuilds index.md.
"""

import sys
import os
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from storage.database import get_connection, init_db
from storage.models import upsert_page, init_review_schedule
from storage.wiki_storage import read_page, PAGES_DIR
from storage.wiki_index import rebuild_index

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def main():
    init_db()

    # Fix: re-register page_vectors virtual table if needed
    # Walk the pages directory
    if not os.path.exists(PAGES_DIR):
        print(f"ERROR: pages directory not found: {PAGES_DIR}")
        return

    md_files = sorted(
        f for f in os.listdir(PAGES_DIR)
        if f.endswith(".md")
    )
    print(f"Found {len(md_files)} .md files in {PAGES_DIR}")

    restored = 0
    for filename in md_files:
        file_path = os.path.join("pages", filename)
        page_data = read_page(file_path)
        if page_data is None:
            print(f"  SKIP: {filename} (could not read)")
            continue

        title = page_data["title"]
        if not title:
            print(f"  SKIP: {filename} (no title in frontmatter)")
            continue

        # Read full content (frontmatter + body)
        full_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "data", "wiki", file_path,
        )
        try:
            with open(full_path, "r", encoding="utf-8") as f:
                full_content = f.read()
        except Exception as e:
            print(f"  SKIP: {filename} (read error: {e})")
            continue

        tags = page_data.get("tags", [])
        sources = page_data.get("sources", [])
        checksum = page_data.get("checksum", "")

        pid = upsert_page(
            title=title,
            file_path=file_path,
            tags=tags,
            sources=sources,
            checksum=checksum,
            content=full_content,
        )
        init_review_schedule(pid)
        restored += 1
        print(f"  OK: {title} -> pages id={pid}")

    print(f"\nRestored {restored} wiki page records.")

    # Rebuild index
    print("Rebuilding index.md...")
    rebuild_index()
    print("Done!")


if __name__ == "__main__":
    main()
