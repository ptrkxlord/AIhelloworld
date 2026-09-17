import time
from typing import Optional

import openai

from app import config

_client: Optional[openai.OpenAI] = None


def get_client() -> openai.OpenAI:
    global _client
    if _client is None:
        _client = openai.OpenAI(base_url=config.LLM_BASE_URL, api_key=config.LLM_API_KEY, timeout=config.LLM_TIMEOUT)
    return _client


class LLMError(Exception):
    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


RETRYABLE = {"timeout", "rate_limit", "server_error", "unavailable"}


def _classify(exc: Exception) -> str:
    if isinstance(exc, openai.APITimeoutError):
        return "timeout"
    if isinstance(exc, openai.RateLimitError):
        return "rate_limit"
    if isinstance(exc, openai.AuthenticationError):
        return "auth_error"
    if isinstance(exc, openai.APIConnectionError):
        return "unavailable"
    if isinstance(exc, openai.APIStatusError):
        return "server_error" if exc.status_code >= 500 else "bad_request"
    return "unknown"


def chat(
    messages: list[dict],
    response_schema: Optional[dict] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> dict:
    client = get_client()
    kwargs: dict = {
        "model": config.LLM_MODEL,
        "messages": messages,
        "temperature": config.LLM_TEMPERATURE if temperature is None else temperature,
        "max_tokens": config.LLM_MAX_TOKENS if max_tokens is None else max_tokens,
    }
    if response_schema is not None:
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "assistant_reply", "schema": response_schema},
        }

    attempt = 0
    delay = 1.0
    while True:
        attempt += 1
        started = time.perf_counter()
        try:
            completion = client.chat.completions.create(**kwargs)
        except Exception as exc:
            kind = _classify(exc)
            if kind in RETRYABLE and attempt <= config.LLM_MAX_RETRIES:
                time.sleep(delay)
                delay *= 2
                continue
            raise LLMError(kind, str(exc)) from exc

        elapsed_ms = (time.perf_counter() - started) * 1000
        choice = completion.choices[0]
        usage = completion.usage
        return {
            "content": choice.message.content or "",
            "model": completion.model,
            "latency_ms": elapsed_ms,
            "usage": {
                "prompt_tokens": usage.prompt_tokens if usage else 0,
                "completion_tokens": usage.completion_tokens if usage else 0,
                "total_tokens": usage.total_tokens if usage else 0,
            },
        }
