import time

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI, RateLimitError

from app import config
from app.prompts import SYSTEM_PROMPT, user_prompt
from app.schemas import RAGAnswer


class GeminiFailure(Exception):
    def __init__(self, kind: str, status_code: int | None = None):
        self.kind = kind
        self.status_code = status_code
        super().__init__(kind)


def _failure_kind(exc: Exception) -> str:
    if isinstance(exc, APITimeoutError):
        return "timeout"
    if isinstance(exc, RateLimitError):
        return "rate_limit"
    if isinstance(exc, APIConnectionError):
        return "unavailable"
    if isinstance(exc, APIStatusError):
        if exc.status_code in {401, 403}:
            return "authentication"
        if exc.status_code == 404:
            return "model"
        if exc.status_code >= 500:
            return "provider"
        return "request"
    return "response"


def generate(context: str, question: str) -> dict:
    if not config.GEMINI_API_KEY:
        raise GeminiFailure("not_configured")
    client = OpenAI(
        api_key=config.GEMINI_API_KEY,
        base_url=config.GEMINI_BASE_URL,
        timeout=config.GEMINI_TIMEOUT,
        max_retries=config.GEMINI_MAX_RETRIES,
    )
    started = time.perf_counter()
    try:
        response = client.beta.chat.completions.parse(
            model=config.GEMINI_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt(question, context)},
            ],
            response_format=RAGAnswer,
            temperature=0.1,
            max_tokens=config.GEMINI_MAX_TOKENS,
        )
        parsed = response.choices[0].message.parsed
        if parsed is None:
            raise GeminiFailure("response")
        usage = response.usage
        return {
            "answer": parsed,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "tokens": {
                "prompt": usage.prompt_tokens if usage else None,
                "completion": usage.completion_tokens if usage else None,
                "total": usage.total_tokens if usage else None,
            },
        }
    except GeminiFailure:
        raise
    except Exception as exc:
        status = exc.status_code if isinstance(exc, APIStatusError) else None
        raise GeminiFailure(_failure_kind(exc), status) from exc
    finally:
        client.close()
