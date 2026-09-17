from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import load_settings
from .llm_client import AssistantError, ask_model


class AskRequest(BaseModel):
    message: str = Field(..., min_length=1)
    temperature: float | None = Field(None, ge=0, le=2)
    top_p: float | None = Field(None, ge=0, le=1)
    max_tokens: int | None = Field(None, ge=1, le=4096)


a = FastAPI(title="PR3 Store Assistant")
b = Path(__file__).resolve().parent
a.mount("/static", StaticFiles(directory=b / "static"), name="static")


def read_context(c: Path) -> str:
    if not c.exists():
        raise HTTPException(status_code=500, detail="Файл context.md не знайдено.")
    return c.read_text(encoding="utf-8")


@a.get("/")
def index() -> FileResponse:
    return FileResponse(b / "static" / "index.html")


@a.post("/ask")
def ask(c: AskRequest) -> dict[str, str | int]:
    d = c.message.strip()
    if not d:
        raise HTTPException(status_code=400, detail="Введіть запит.")

    e = load_settings().with_generation(c.temperature, c.top_p, c.max_tokens)
    f = read_context(e.context_path)

    try:
        g = ask_model(d, f, e)
    except AssistantError as h:
        raise HTTPException(status_code=h.status_code, detail=h.message) from h

    return {"answer": g.answer, "model": g.model, "elapsed_ms": g.elapsed_ms}
