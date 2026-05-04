import os
from dotenv import load_dotenv

load_dotenv()

# WeChat Work Smart Bot (长连接)
WECOM_BOT_ID = os.getenv("WECOM_BOT_ID", "")
WECOM_BOT_SECRET = os.getenv("WECOM_BOT_SECRET", "")

# LLM configuration
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))

OUTPUT_LANGUAGE = os.getenv("OUTPUT_LANGUAGE", "English")
BAIDU_AI_SEARCH_API_KEY = os.getenv("BAIDU_AI_SEARCH_API_KEY", "")
