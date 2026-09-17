from dataclasses import dataclass, replace
import os
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    api_key: str
    base_url: str
    model: str
    temperature: float
    top_p: float
    max_tokens: int
    timeout_seconds: float
    retries: int
    context_path: Path

    def with_generation(self, a: float | None, b: float | None, c: int | None) -> "Settings":
        return replace(
            self,
            temperature=self.temperature if a is None else a,
            top_p=self.top_p if b is None else b,
            max_tokens=self.max_tokens if c is None else c,
        )


def _float(a: str, b: float) -> float:
    try:
        return float(a)
    except (TypeError, ValueError):
        return b


def _int(a: str, b: int) -> int:
    try:
        return int(a)
    except (TypeError, ValueError):
        return b


def load_settings() -> Settings:
    a = Path(__file__).resolve().parents[1]
    load_dotenv(a / ".env")
    b = os.getenv("CONTEXT_PATH", "context.md")
    c = Path(b)
    if not c.is_absolute():
        c = a / c
    return Settings(
        api_key=(os.getenv("LLM_API_KEY") or os.getenv("GEMINI_API_KEY") or "").strip(),
        base_url=os.getenv("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/").strip(),
        model=os.getenv("LLM_MODEL", "gemini-3.8-flash").strip(),
        temperature=_float(os.getenv("LLM_TEMPERATURE", "0.2"), 0.2),
        top_p=_float(os.getenv("LLM_TOP_P", "0.9"), 0.9),
        max_tokens=_int(os.getenv("LLM_MAX_TOKENS", "500"), 500),
        timeout_seconds=_float(os.getenv("LLM_TIMEOUT_SECONDS", "20"), 20),
        retries=max(0, _int(os.getenv("LLM_RETRIES", "2"), 2)),
        context_path=c,
    )
