import re
from app import config
from app.context_builder import build_context, render_context
from app.gemini_client import GeminiFailure, generate
from app.search import semantic_search


NUMBER_PATTERN = re.compile(r"(?<!\w)\d+(?:[.,]\d+)?%?(?!\w)")
UNKNOWN_ANSWER = "У базі знань не знайдено достатньо інформації для відповіді."


class RAGFailure(Exception):
    def __init__(self, kind: str, diagnostics: dict):
        self.kind = kind
        self.diagnostics = diagnostics
        super().__init__(kind)


def prepare_context(question: str, chunks: list[dict], vectors, model_name: str, product: str | None = None) -> dict:
    retrieval = semantic_search(
        question,
        chunks,
        vectors,
        model_name,
        top_k=len(chunks),
        min_score=config.DEFAULT_MIN_SCORE,
        filters={"audience": "клієнти"},
    )
    sources = build_context(retrieval["results"], question, product)
    return {
        "sources": sources,
        "context": render_context(sources),
        "retrieval_ms": retrieval["latency_ms"],
    }


def _validate_answer(answer, sources: list[dict]) -> tuple[bool, str | None]:
    allowed_ids = {source["id"] for source in sources}
    if any(source_id not in allowed_ids for source_id in answer.source_ids):
        return False, "invalid_citation"
    if len(set(answer.source_ids)) != len(answer.source_ids):
        return False, "duplicate_citation"
    if answer.found and not answer.source_ids:
        return False, "missing_citation"
    if not answer.found and answer.source_ids:
        return False, "unexpected_citation"
    cited_text = " ".join(
        f"{source['title']} {source['section']} {source['revision_date']} {source['text']}"
        for source in sources
        if source["id"] in answer.source_ids
    )
    if answer.found:
        cited_numbers = set(NUMBER_PATTERN.findall(cited_text))
        answer_numbers = set(NUMBER_PATTERN.findall(answer.answer))
        if not answer_numbers.issubset(cited_numbers):
            return False, "unsupported_number"
    return True, None


def ask(question: str, chunks: list[dict], vectors, model_name: str, product: str | None = None) -> dict:
    prepared = prepare_context(question, chunks, vectors, model_name, product)
    sources = prepared["sources"]
    if not sources:
        return {
            "answer": UNKNOWN_ANSWER,
            "found": False,
            "sources": [],
            "context_sources": [],
            "context": "",
            "model": None,
            "retrieval_ms": prepared["retrieval_ms"],
            "generation_ms": 0,
            "tokens": None,
            "validation": "no_context",
        }
    context = prepared["context"]
    try:
        generated = generate(context, question)
    except GeminiFailure as exc:
        raise RAGFailure(exc.kind, {
            "sources": sources,
            "context": context,
            "model": config.GEMINI_MODEL,
            "retrieval_ms": prepared["retrieval_ms"],
            "generation_ms": None,
            "tokens": None,
        }) from exc
    valid, validation = _validate_answer(generated["answer"], sources)
    answer = generated["answer"]
    if not valid or not answer.found:
        answer_text = UNKNOWN_ANSWER
        found = False
        cited_sources = []
        if valid:
            validation = "model_refusal"
    else:
        answer_text = answer.answer
        found = answer.found
        cited_sources = [source for source in sources if source["id"] in answer.source_ids]
    return {
        "answer": answer_text,
        "found": found,
        "sources": cited_sources,
        "context_sources": sources,
        "context": context,
        "model": config.GEMINI_MODEL,
        "retrieval_ms": prepared["retrieval_ms"],
        "generation_ms": generated["latency_ms"],
        "tokens": generated["tokens"],
        "validation": validation or "ok",
    }
