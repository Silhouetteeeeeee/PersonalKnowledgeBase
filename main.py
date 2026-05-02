"""
Personal Knowledge Base Agent — WeChat Work Bot.
Run this to start the Flask webhook server.
"""

import logging
from server.webhook import run_server

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

if __name__ == "__main__":
    from storage.database import init_db
    init_db()
    print("Knowledge Agent started. Listening for WeChat Work messages...")
    run_server()
