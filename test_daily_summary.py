"""Manual test: trigger daily knowledge summary."""
import asyncio, logging, os
os.environ['DAILY_SUMMARY_USER_ID'] = 'SunHeXuanCheng'
from aibot import WSClient, WSClientOptions
from server.config import WECOM_BOT_ID, WECOM_BOT_SECRET
from server.daily_summary import send_daily_summary
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    logger.info('Creating WSClient...')
    client = WSClient(WSClientOptions(bot_id=WECOM_BOT_ID, secret=WECOM_BOT_SECRET, max_reconnect_attempts=-1))
    connected = asyncio.Event()
    @client.on('connected')
    def _on_connected():
        logger.info('Connected!')
    @client.on('authenticated')
    def _on_auth():
        logger.info('Authenticated!')
        connected.set()
    client_task = asyncio.create_task(client.run())
    try:
        await asyncio.wait_for(connected.wait(), timeout=15)
    except asyncio.TimeoutError:
        logger.error('Failed to connect within 15s')
        client_task.cancel()
        return
    logger.info('Sending daily summary...')
    result = await send_daily_summary(client, 'SunHeXuanCheng')
    logger.info('send_daily_summary returned: %s', result)
    client_task.cancel()
    try:
        await client_task
    except (asyncio.CancelledError, KeyboardInterrupt):
        pass
    logger.info('Done.')

if __name__ == '__main__':
    asyncio.run(main())
