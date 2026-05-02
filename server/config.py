import os
from dotenv import load_dotenv

load_dotenv()

WEWORK_CORP_ID = os.getenv("WEWORK_CORP_ID", "")
WEWORK_AGENT_ID = os.getenv("WEWORK_AGENT_ID", "")
WEWORK_SECRET = os.getenv("WEWORK_SECRET", "")
WEWORK_TOKEN = os.getenv("WEWORK_TOKEN", "")
FLASK_HOST = os.getenv("FLASK_HOST", "0.0.0.0")
FLASK_PORT = int(os.getenv("FLASK_PORT", "8080"))

# LLM configuration
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))
