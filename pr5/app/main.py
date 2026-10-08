from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from app import config
from app.index_store import load_index
from app.schemas import SearchRequest
from app.search import search_both


app = FastAPI(title="Сузір'я: пошук у базі знань")
STATIC_FILE = Path(__file__).parent / "static" / "index.html"


@app.on_event("startup")
def startup() -> None:
    try:
        app.state.chunks, app.state.vectors, app.state.model_name = load_index(config.INDEX_DIR)
        app.state.index_error = None
    except (FileNotFoundError, ValueError) as exc:
        app.state.chunks, app.state.vectors, app.state.model_name = [], None, config.EMBEDDING_MODEL
        app.state.index_error = str(exc)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return STATIC_FILE.read_text(encoding="utf-8")


@app.get("/api/status")
def status() -> dict:
    return {"ready": app.state.vectors is not None,
            "message": app.state.index_error or "Індекс завантажено",
            "documents": len({chunk["source"] for chunk in app.state.chunks}),
            "chunks": len(app.state.chunks),
            "model": app.state.model_name,
            "default_top_k": config.DEFAULT_TOP_K,
            "max_top_k": config.MAX_TOP_K,
            "default_min_score": config.DEFAULT_MIN_SCORE}


@app.get("/api/filters")
def filters() -> dict:
    values = {key: set() for key in ("category", "product", "audience", "status")}
    for chunk in app.state.chunks:
        if not config.ALLOW_INTERNAL_DOCS and chunk["metadata"].get("audience") != "клієнти":
            continue
        for key in values:
            values[key].add(chunk["metadata"].get(key, ""))
    return {key: sorted(value for value in entries if value) for key, entries in values.items()}


@app.post("/api/search")
def search(payload: SearchRequest) -> dict:
    if not payload.query.strip():
        raise HTTPException(status_code=422, detail="Введіть текст запиту")
    if app.state.vectors is None:
        raise HTTPException(status_code=503, detail=app.state.index_error)
    if payload.audience == "персонал" and not config.ALLOW_INTERNAL_DOCS:
        raise HTTPException(status_code=403, detail="Документи для персоналу недоступні в клієнтському режимі")
    metadata = {key: getattr(payload, key) for key in ("category", "product", "audience", "status")}
    if not config.ALLOW_INTERNAL_DOCS:
        metadata["audience"] = "клієнти"
    unknown = {key: value for key, value in metadata.items() if value and value not in {c["metadata"].get(key) for c in app.state.chunks}}
    if unknown:
        raise HTTPException(status_code=422, detail=f"Невідоме значення фільтра: {', '.join(unknown)}")
    return search_both(payload.query.strip(), app.state.chunks, app.state.vectors,
                       app.state.model_name, payload.top_k, payload.min_score, metadata)
