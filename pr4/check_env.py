import sys

import fastapi
import openai
import pydantic

from app import config
from app.llm_client import LLMError, chat
from app.schemas import AssistantReply

print("Python:", sys.version)
print("FastAPI:", fastapi.__version__)
print("OpenAI SDK:", openai.__version__)
print("Pydantic:", pydantic.VERSION)

if not config.LLM_API_KEY:
    print("LLM_API_KEY is not set in .env")
    sys.exit(1)

schema = AssistantReply.model_json_schema()
messages = [
    {"role": "system", "content": "Відповідай лише JSON-об'єктом за наданою схемою."},
    {"role": "user", "content": "Скільки коштує доставка?"},
]

try:
    result = chat(messages, response_schema=schema)
except LLMError as exc:
    print(f"Schema request failed ({exc.kind}):", exc)
    sys.exit(1)

print("Model reachable:", result["model"])
print("Usage:", result["usage"])
print("Raw content:", result["content"])

try:
    AssistantReply.model_validate_json(result["content"])
    print("Provider accepted response_format=json_schema and returned a valid object")
except Exception as exc:
    print("Provider returned content that failed schema validation on our side:", exc)
