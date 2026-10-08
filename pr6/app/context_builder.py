import math
import re

from app import config


HISTORICAL_QUERY = re.compile(
    r"\b(?:20\d{2}|архів\w*|стар\w*\s+редакц\w*|попередн\w*|раніше\s+діял\w*)\b",
    re.I,
)


def wants_historical_context(question: str) -> bool:
    return bool(HISTORICAL_QUERY.search(question))


def estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / config.CHARS_PER_TOKEN)


def build_context(results: list[dict], question: str, product: str | None = None) -> list[dict]:
    historical = wants_historical_context(question)
    candidates = []
    for result in results:
        metadata = result["metadata"]
        if metadata.get("audience") != "клієнти":
            continue
        if product and metadata.get("product") not in {product, "усі товари"}:
            continue
        if metadata.get("status") != "чинний" and not historical:
            continue
        if metadata.get("status") == "архівний" and not historical:
            continue
        candidates.append(result)

    selected = []
    document_counts = {}
    used_tokens = 0
    for result in candidates:
        source = result["source"]
        if document_counts.get(source, 0) >= config.MAX_CHUNKS_PER_DOCUMENT:
            continue
        metadata = result["metadata"]
        details = {
            "title": metadata["title"],
            "section": result["section"],
            "revision_date": metadata["revision_date"],
            "text": result["text"],
            "score": result["score"],
            "source": source,
        }
        cost = estimate_tokens("\n".join(str(value) for value in details.values()))
        if used_tokens + cost > config.CONTEXT_TOKEN_BUDGET:
            continue
        selected.append({"id": len(selected) + 1, **details, "estimated_tokens": cost})
        used_tokens += cost
        document_counts[source] = document_counts.get(source, 0) + 1
        if len(selected) >= config.CONTEXT_CHUNK_LIMIT:
            break
    return selected


def render_context(sources: list[dict]) -> str:
    return "\n\n".join(
        f"[{item['id']}] {item['title']} | розділ: {item['section']} | редакція: {item['revision_date']}\n{item['text']}"
        for item in sources
    )
