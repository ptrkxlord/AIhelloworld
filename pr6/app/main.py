from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from app import config
from app.index_store import load_index
from app.rag import RAGFailure, ask
from app.schemas import AskRequest


STATIC_FILE = Path(__file__).parent / "static" / "index.html"
PRODUCTS = {"Nova Air", "Nova Buds", "SoundBox Mini", "Orbit S"}
ERRORS = {
    "not_configured": (503, "Додайте ключ Gemini до файлу .env"),
    "authentication": (502, "Gemini відхилив ключ. Перевірте API-ключ у Google AI Studio."),
    "rate_limit": (429, "Gemini тимчасово обмежив кількість запитів. Спробуйте пізніше."),
    "timeout": (504, "Gemini не відповів вчасно. Спробуйте ще раз."),
    "model": (502, "Вказану модель Gemini не знайдено. Перевірте GEMINI_MODEL у .env."),
    "request": (502, "Gemini не прийняв запит. Перевірте модель і параметри."),
    "provider": (502, "Сервіс Gemini тимчасово недоступний."),
    "unavailable": (502, "Не вдалося з'єднатися з Gemini. Перевірте мережу."),
    "response": (502, "Gemini повернув відповідь у непідтримуваному форматі."),
}


@asynccontextmanager
async def lifespan(application: FastAPI):
    try:
        application.state.chunks, application.state.vectors, application.state.embedding_model = load_index(config.INDEX_DIR)
        application.state.index_error = None
    except (FileNotFoundError, ValueError) as exc:
        application.state.chunks, application.state.vectors, application.state.embedding_model = [], None, config.EMBEDDING_MODEL
        application.state.index_error = str(exc)
    yield


app = FastAPI(title="Сузір'я: RAG-помічник", lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return STATIC_FILE.read_text(encoding="utf-8")


@app.get("/api/status")
def status() -> dict:
    return {
        "ready": app.state.vectors is not None,
        "message": app.state.index_error or "Базу знань завантажено",
        "documents": len({chunk["source"] for chunk in app.state.chunks}),
        "chunks": len(app.state.chunks),
        "model": config.GEMINI_MODEL,
        "api_key_configured": bool(config.GEMINI_API_KEY),
        "context_chunk_limit": config.CONTEXT_CHUNK_LIMIT,
        "context_token_budget": config.CONTEXT_TOKEN_BUDGET,
        "products": sorted(PRODUCTS),
    }


@app.post("/api/ask")
def answer(payload: AskRequest) -> dict:
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Введіть запитання")
    if payload.product and payload.product not in PRODUCTS:
        raise HTTPException(status_code=422, detail="Невідомий товар")
    if app.state.vectors is None:
        raise HTTPException(status_code=503, detail=app.state.index_error)
    try:
        return ask(question, app.state.chunks, app.state.vectors, app.state.embedding_model, payload.product)
    except RAGFailure as exc:
        code, message = ERRORS.get(exc.kind, ERRORS["provider"])
        raise HTTPException(status_code=code, detail={"message": message, "diagnostics": exc.diagnostics}) from exc
