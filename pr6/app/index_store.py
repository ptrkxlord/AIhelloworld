import json
from pathlib import Path

import numpy as np


INDEX_VERSION = 1


def save_index(directory: Path, chunks: list[dict], vectors: np.ndarray, model_name: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "vectors.npy", vectors)
    manifest = {
        "version": INDEX_VERSION,
        "model": model_name,
        "count": len(chunks),
        "chunks": chunks,
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_index(directory: Path) -> tuple[list[dict], np.ndarray, str]:
    manifest_path = directory / "manifest.json"
    vectors_path = directory / "vectors.npy"
    if not manifest_path.is_file() or not vectors_path.is_file():
        raise FileNotFoundError("Індекс ще не створено. Запустіть: python build_index.py")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("version") != INDEX_VERSION:
        raise ValueError("Версія індексу не підтримується. Перебудуйте його командою python build_index.py")
    vectors = np.load(vectors_path, allow_pickle=False)
    chunks = manifest["chunks"]
    if len(chunks) != len(vectors) or len(chunks) != manifest["count"]:
        raise ValueError("Фрагменти та вектори індексу не збігаються. Перебудуйте індекс")
    return chunks, vectors, manifest["model"]
