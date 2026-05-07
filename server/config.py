import os
from dotenv import load_dotenv

load_dotenv()

# WeChat Work Smart Bot (长连接)
WECOM_BOT_ID = os.getenv("WECOM_BOT_ID", "")
WECOM_BOT_SECRET = os.getenv("WECOM_BOT_SECRET", "")

# LLM configuration
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))

# Task-specific model overrides. Set task → model_name to use a different
# model for specific tasks. Unregistered tasks fall back to LLM_MODEL.
TASK_MODEL_MAP: dict[str, str] = {
    "default": LLM_MODEL,
    "rewrite": LLM_MODEL,
}

OUTPUT_LANGUAGE = os.getenv("OUTPUT_LANGUAGE", "English")
BAIDU_AI_SEARCH_API_KEY = os.getenv("BAIDU_AI_SEARCH_API_KEY", "")

# Claude Code Bridge (remote coding via WeChat Work)
CLAUDE_CODE_BRIDGE_ENABLED = os.getenv("CLAUDE_CODE_BRIDGE_ENABLED", "false").lower() == "true"
