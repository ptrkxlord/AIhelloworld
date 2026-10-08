import numpy as np

from app.rag import UNKNOWN_ANSWER, RAGFailure, _validate_answer, ask
from app.schemas import RAGAnswer


def source(source_id=1, text="Гарантія діє 12 місяців."):
    return {
        "id": source_id,
        "title": "Гарантія",
        "section": "Строк",
        "revision_date": "2026-01-01",
        "text": text,
        "score": 0.9,
        "source": "warranty.md",
    }


def test_answer_cannot_cite_reference_outside_context():
    valid, reason = _validate_answer(RAGAnswer(answer="Відповідь.", source_ids=[2], found=True), [source()])
    assert (valid, reason) == (False, "invalid_citation")


def test_duplicate_citations_are_rejected():
    answer = RAGAnswer(answer="Факт.", source_ids=[1, 1], found=True)
    assert _validate_answer(answer, [source()]) == (False, "duplicate_citation")


def test_supported_numeric_fact_requires_citation_and_matching_context():
    answer = RAGAnswer(answer="Гарантія діє 12 місяців.", source_ids=[1], found=True)
    assert _validate_answer(answer, [source()]) == (True, None)
    unsupported = RAGAnswer(answer="Гарантія діє 24 місяці.", source_ids=[1], found=True)
    assert _validate_answer(unsupported, [source()]) == (False, "unsupported_number")


def test_found_answer_requires_a_source():
    answer = RAGAnswer(answer="Готово.", source_ids=[], found=True)
    assert _validate_answer(answer, [source()]) == (False, "missing_citation")


def test_refusal_text_cannot_smuggle_an_uncited_answer(monkeypatch):
    customer = {
        "source": "warranty.md", "section": "Строк", "text": "Гарантія діє 12 місяців.", "score": 0.9,
        "metadata": {"title": "Гарантія", "audience": "клієнти", "status": "чинний", "product": "усі товари", "revision_date": "2026-01-01"},
    }
    monkeypatch.setattr("app.rag.semantic_search", lambda *args, **kwargs: {"results": [customer], "latency_ms": 2})
    monkeypatch.setattr("app.rag.generate", lambda *args, **kwargs: {
        "answer": RAGAnswer(answer="Немає даних, але гарантія 99 років.", source_ids=[], found=False),
        "latency_ms": 3,
        "tokens": {"total": 20},
    })
    response = ask("гарантія", [customer], np.empty((0, 2)), "model")
    assert response["answer"] == UNKNOWN_ANSWER
    assert response["sources"] == []


def test_no_context_returns_unknown_without_calling_model(monkeypatch):
    monkeypatch.setattr("app.rag.semantic_search", lambda *args, **kwargs: {"results": [], "latency_ms": 2})
    monkeypatch.setattr("app.rag.generate", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("model called")))
    response = ask("питання", [], np.empty((0, 2)), "model")
    assert response["answer"] == UNKNOWN_ANSWER
    assert response["found"] is False


def test_model_receives_only_selected_customer_context(monkeypatch):
    customer = {
        "source": "warranty.md",
        "section": "Строк",
        "text": "Гарантія діє 12 місяців.",
        "score": 0.9,
        "metadata": {
            "title": "Гарантія",
            "audience": "клієнти",
            "status": "чинний",
            "product": "усі товари",
            "revision_date": "2026-01-01",
        },
    }
    internal = {**customer, "source": "internal.md", "metadata": {**customer["metadata"], "audience": "персонал"}}
    monkeypatch.setattr("app.rag.semantic_search", lambda *args, **kwargs: {"results": [customer, internal], "latency_ms": 2})
    calls = []

    def fake_generate(context, question):
        calls.append(context)
        return {
            "answer": RAGAnswer(answer="Гарантія діє 12 місяців.", source_ids=[1], found=True),
            "latency_ms": 3,
            "tokens": {"total": 20},
        }

    monkeypatch.setattr("app.rag.generate", fake_generate)
    response = ask("гарантія", [customer, internal], np.empty((0, 2)), "model")
    assert "internal.md" not in calls[0]
    assert response["found"] is True


def test_generation_error_keeps_context_for_evaluation(monkeypatch):
    from app.gemini_client import GeminiFailure

    customer = {
        "source": "warranty.md", "section": "Строк", "text": "12 місяців.", "score": 0.9,
        "metadata": {"title": "Гарантія", "audience": "клієнти", "status": "чинний", "product": "усі товари", "revision_date": "2026-01-01"},
    }
    monkeypatch.setattr("app.rag.semantic_search", lambda *args, **kwargs: {"results": [customer], "latency_ms": 2})
    monkeypatch.setattr("app.rag.generate", lambda *args, **kwargs: (_ for _ in ()).throw(GeminiFailure("authentication")))
    try:
        ask("гарантія", [customer], np.empty((0, 2)), "model")
    except RAGFailure as exc:
        assert exc.kind == "authentication"
        assert "12 місяців" in exc.diagnostics["context"]
    else:
        raise AssertionError("RAGFailure was not raised")
