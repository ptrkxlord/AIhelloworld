import io
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image, UnidentifiedImageError

from app.detector import detect, load_model

app = FastAPI(title="PR2 Object Detection")


@app.on_event("startup")
def _preload_model() -> None:
    load_model()


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return Path(__file__).parent.joinpath("static/index.html").read_text(encoding="utf-8")


def validate_upload(contents: bytes) -> str | None:
    if not contents:
        return "empty file"
    try:
        Image.open(io.BytesIO(contents)).verify()
    except UnidentifiedImageError:
        return "not a valid image"
    return None


@app.post("/detect")
async def detect_endpoint(
    file: UploadFile = File(...),
    conf: float = Query(0.25, ge=0.0, le=1.0),
) -> dict:
    contents = await file.read()

    error = validate_upload(contents)
    if error:
        raise HTTPException(status_code=400, detail=error)

    suffix = Path(file.filename or "upload.jpg").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        return detect(tmp_path, conf_threshold=conf)
    finally:
        Path(tmp_path).unlink(missing_ok=True)
