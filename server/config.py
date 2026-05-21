import os
from dotenv import load_dotenv

load_dotenv()

# WeChat Work Smart Bot (长连接)
WECOM_BOT_ID = os.getenv("WECOM_BOT_ID", "")
WECOM_BOT_SECRET = os.getenv("WECOM_BOT_SECRET", "")

# LLM configuration
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")
LLM_APIKEY = os.getenv("DEEPSEEK_API_KEY", "")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))

# Task-specific model overrides. Set task → model_name to use a different
# model for specific tasks. Unregistered tasks fall back to LLM_MODEL.
TASK_MODEL_MAP: dict[str, str] = {
    "default": LLM_MODEL,
    "rewrite": LLM_MODEL,
}

OUTPUT_LANGUAGE = os.getenv("OUTPUT_LANGUAGE", "English")
BAIDU_API_KEY = os.getenv("BAIDU_API_KEY", "")

# Claude Code Bridge (remote coding via WeChat Work)
CLAUDE_CODE_BRIDGE_ENABLED = os.getenv("CLAUDE_CODE_BRIDGE_ENABLED", "false").lower() == "true"

# Daily knowledge summary (APScheduler cron at 09:00)
DAILY_SUMMARY_ENABLED = os.getenv("DAILY_SUMMARY_ENABLED", "true").lower() == "true"
DAILY_SUMMARY_USER_ID = os.getenv("DAILY_SUMMARY_USER_ID", "")
DAILY_SUMMARY_TIME = os.getenv("DAILY_SUMMARY_TIME", "09:00")

# URL fetching
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

# Thinker module (spaced repetition)
THINKER_USER_ID = os.getenv("THINKER_USER_ID", "")
THINKER_CHECK_INTERVAL = int(os.getenv("THINKER_CHECK_INTERVAL", "4"))

# Fund Bot (personal fund portfolio management)
FUND_BOT_ENABLED = os.getenv("FUND_BOT_ENABLED", "false").lower() == "true"
FUND_BOT_ID = os.getenv("FUND_BOT_ID", "")
FUND_BOT_SECRET = os.getenv("FUND_BOT_SECRET", "")
FUND_CHECKPOINT_ENABLED = os.getenv("FUND_CHECKPOINT_ENABLED", "false").lower() == "true"

# Fund bot tasks
TASK_MODEL_MAP["fund_analyst"] = LLM_MODEL
TASK_MODEL_MAP["fund_researcher"] = LLM_MODEL
TASK_MODEL_MAP["fund_manager"] = LLM_MODEL
