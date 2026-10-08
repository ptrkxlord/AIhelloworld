from functools import lru_cache

import numpy as np

from app import config


@lru_cache(maxsize=1)
def get_model(model_name: str = config.EMBEDDING_MODEL):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def encode_passages(texts: list[str], model_name: str = config.EMBEDDING_MODEL) -> np.ndarray:
    values = get_model(model_name).encode(
        [f"passage: {text}" for text in texts],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    return np.asarray(values, dtype=np.float32)


def encode_query(text: str, model_name: str = config.EMBEDDING_MODEL) -> np.ndarray:
    value = get_model(model_name).encode(
        f"query: {text}",
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return np.asarray(value, dtype=np.float32)
