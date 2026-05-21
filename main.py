"""
Personal Knowledge Base Agent — WeChat Work Smart Bot.
Run this to start the long-connection bot.
"""

import asyncio
import logging

from server.bot import KnowledgeBot
from server.config import FUND_BOT_ENABLED
from storage.database import init_db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# Suppress verbose third-party library logs (embedding model HTTP checks, etc.)
for _lib in ("httpx", "sentence_transformers", "fastembed", "httpcore", "akshare"):
    logging.getLogger(_lib).setLevel(logging.WARNING)

if __name__ == "__main__":
    init_db()
    print("Knowledge Agent started (WebSocket long-connection mode)...")

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Start KnowledgeBot
    kb = KnowledgeBot()
    loop.run_until_complete(kb.client.connect())
    kb.scheduler.start()

    # Start FundBot if enabled
    if FUND_BOT_ENABLED:
        from fund.bot import FundBot
        fb = FundBot()
        loop.run_until_complete(fb.client.connect())
        fb.scheduler.start()
        print("FundBot started.")

    try:
        loop.run_forever()
    except KeyboardInterrupt:
        kb.client.disconnect()
        if FUND_BOT_ENABLED:
            fb.client.disconnect()
    finally:
        kb.scheduler.shutdown(wait=False)
        if FUND_BOT_ENABLED:
            fb.scheduler.shutdown(wait=False)
        loop.close()
