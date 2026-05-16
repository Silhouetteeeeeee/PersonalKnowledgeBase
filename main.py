"""
Personal Knowledge Base Agent — WeChat Work Smart Bot.
Run this to start the long-connection bot.
"""

import logging
from server.bot import run_bot

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# Suppress verbose third-party library logs (embedding model HTTP checks, etc.)
for _lib in ("httpx", "sentence_transformers", "fastembed", "httpcore"):
    logging.getLogger(_lib).setLevel(logging.WARNING)

if __name__ == "__main__":
    from storage.database import init_db
    init_db()
    print("Knowledge Agent started (WebSocket long-connection mode)...")
    run_bot()
