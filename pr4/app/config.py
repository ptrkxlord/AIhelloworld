import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "gemini-3.6-flash")
LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.3"))
LLM_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "500"))
LLM_TIMEOUT = float(os.environ.get("LLM_TIMEOUT", "20"))
LLM_MAX_RETRIES = int(os.environ.get("LLM_MAX_RETRIES", "2"))
HISTORY_TOKEN_BUDGET = int(os.environ.get("HISTORY_TOKEN_BUDGET", "2000"))
CONTEXT_FILE = os.environ.get("CONTEXT_FILE", str(BASE_DIR / "context.md"))

CHARS_PER_TOKEN = 2.5
