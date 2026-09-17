from dataclasses import dataclass
import time

from .assistant import build_messages
from .config import Settings


class AssistantError(Exception):
    status_code = 500
    message = "Помилка під час обробки запиту."


class MissingApiKeyError(AssistantError):
    status_code = 401
    message = "API-ключ не налаштовано."


class AuthError(AssistantError):
    status_code = 401
    message = "Помилка авторизації API-ключа."


class LimitError(AssistantError):
    status_code = 429
    message = "Перевищено ліміт запитів. Спробуйте пізніше."


class TimeoutError(AssistantError):
    status_code = 504
    message = "Сервіс не відповів вчасно."


class ServiceError(AssistantError):
    status_code = 503
    message = "Сервіс моделі тимчасово недоступний."


@dataclass(frozen=True)
class AssistantResult:
    answer: str
    model: str
    elapsed_ms: int


def ask_model(a: str, b: str, c: Settings) -> AssistantResult:
    if not c.api_key:
        raise MissingApiKeyError()

    try:
        from openai import APIConnectionError, APIStatusError, APITimeoutError, AuthenticationError, OpenAI, RateLimitError
    except ImportError as d:
        raise ServiceError("Не встановлено бібліотеку openai.") from d

    d = OpenAI(api_key=c.api_key, base_url=c.base_url, timeout=c.timeout_seconds)
    e = build_messages(a, b)
    f = time.perf_counter()
    g = 0

    while True:
        try:
            h = d.chat.completions.create(
                model=c.model,
                messages=e,
                temperature=c.temperature,
                top_p=c.top_p,
                max_tokens=c.max_tokens,
            )
            i = h.choices[0].message.content or ""
            return AssistantResult(answer=i.strip(), model=c.model, elapsed_ms=round((time.perf_counter() - f) * 1000))
        except AuthenticationError as j:
            raise AuthError() from j
        except RateLimitError as j:
            if g >= c.retries:
                raise LimitError() from j
            g += 1
            time.sleep(min(2 ** g, 8))
        except APITimeoutError as j:
            raise TimeoutError() from j
        except APIConnectionError as j:
            raise ServiceError() from j
        except APIStatusError as j:
            if j.status_code == 429 and g < c.retries:
                g += 1
                time.sleep(min(2 ** g, 8))
                continue
            if j.status_code == 429:
                raise LimitError() from j
            if j.status_code in {500, 502, 503, 504}:
                raise ServiceError() from j
            raise ServiceError() from j
