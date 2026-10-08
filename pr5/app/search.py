import re
import time
from collections import Counter

import numpy as np

from app import config
from app.embeddings import encode_query


TOKEN_RE = re.compile(r"[\w-]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return [token.casefold() for token in TOKEN_RE.findall(text)]


def _matches(chunk: dict, filters: dict) -> bool:
    metadata = chunk["metadata"]
    return all(not value or metadata.get(key) == value for key, value in filters.items())


def _format_results(chunks: list[dict], scores: np.ndarray, indexes: np.ndarray) -> list[dict]:
    results = []
    for index in indexes:
        chunk = chunks[int(index)]
        results.append({
            "score": round(float(scores[int(index)]), 4),
            "text": chunk["text"],
            "source": chunk["source"],
            "section": chunk["section"],
            "metadata": chunk["metadata"],
        })
    return results


def semantic_search(query: str, chunks: list[dict], vectors: np.ndarray, model_name: str,
                    top_k: int, min_score: float, filters: dict) -> dict:
    started = time.perf_counter()
    query_vector = encode_query(query, model_name)
    allowed = [i for i, chunk in enumerate(chunks) if _matches(chunk, filters)]
    scores = vectors @ query_vector
    ranked = [i for i in allowed if scores[i] >= min_score]
    ranked.sort(key=lambda i: float(scores[i]), reverse=True)
    return {"results": _format_results(chunks, scores, np.asarray(ranked[:top_k], dtype=int)),
            "latency_ms": round((time.perf_counter() - started) * 1000, 2)}


def keyword_search(query: str, chunks: list[dict], top_k: int, filters: dict) -> dict:
    started = time.perf_counter()
    terms = tokenize(query)
    allowed = [i for i, chunk in enumerate(chunks) if _matches(chunk, filters)]
    documents = [tokenize(chunks[i]["text"]) for i in allowed]
    frequencies = [Counter(document) for document in documents]
    average_length = sum(map(len, documents)) / max(len(documents), 1)
    document_frequency = Counter(term for frequency in frequencies for term in frequency)
    total = len(documents)
    scores = np.zeros(len(chunks), dtype=np.float32)
    k1, b = 1.5, 0.75
    for index, document, frequency in zip(allowed, documents, frequencies):
        length = len(document)
        for term in terms:
            count = frequency[term]
            if count:
                inverse = np.log(1 + (total - document_frequency[term] + 0.5) /
                                 (document_frequency[term] + 0.5))
                scores[index] += inverse * count * (k1 + 1) / (
                    count + k1 * (1 - b + b * length / max(average_length, 1))
                )
    ranked = [i for i in allowed if scores[i] > 0]
    ranked.sort(key=lambda i: float(scores[i]), reverse=True)
    return {"results": _format_results(chunks, scores, np.asarray(ranked[:top_k], dtype=int)),
            "latency_ms": round((time.perf_counter() - started) * 1000, 2)}


def search_both(query: str, chunks: list[dict], vectors: np.ndarray, model_name: str,
                top_k: int = config.DEFAULT_TOP_K, min_score: float = config.DEFAULT_MIN_SCORE,
                filters: dict | None = None) -> dict:
    active_filters = filters or {}
    return {
        "semantic": semantic_search(query, chunks, vectors, model_name, top_k, min_score, active_filters),
        "keyword": keyword_search(query, chunks, top_k, active_filters),
    }
