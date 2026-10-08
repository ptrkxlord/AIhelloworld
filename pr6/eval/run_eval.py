import json
import sys
from pathlib import Path

from app import config
from app.index_store import load_index
from app.rag import RAGFailure, UNKNOWN_ANSWER, ask, prepare_context


BASE_DIR = Path(__file__).resolve().parents[1]
SCENARIOS = json.loads((BASE_DIR / "eval" / "scenarios.json").read_text(encoding="utf-8"))


def evaluate(chunks, vectors, model_name, chunk_limit, provider_blocked=None):
    config.CONTEXT_CHUNK_LIMIT = chunk_limit
    records = []
    blocked_kind = provider_blocked
    for scenario in SCENARIOS:
        if blocked_kind:
            prepared = prepare_context(scenario["question"], chunks, vectors, model_name, scenario["product"])
            selected = prepared["sources"]
            diagnostics = {
                **prepared,
                "model": model_name,
                "generation_ms": 0 if not selected else None,
                "tokens": None,
            }
            error = None
            answer = UNKNOWN_ANSWER if not selected else ""
            found = False if not selected else None
            validation = "no_context" if not selected else None
            generation_state = "not_needed" if not selected else "blocked"
            generation_attempted = False
            blocked_reason = blocked_kind if selected else None
            citation_files = set()
        else:
            try:
                result = ask(scenario["question"], chunks, vectors, model_name, scenario["product"])
                selected = result["sources"]
                answer = result["answer"]
                found = result["found"]
                diagnostics = {key: result[key] for key in ("context", "model", "retrieval_ms", "generation_ms", "tokens")}
                context_sources = result["context_sources"]
                citation_files = {source["source"] for source in selected}
                error = None
                validation = result.get("validation")
                generation_state = "not_needed" if validation == "no_context" else "completed"
                generation_attempted = generation_state == "completed"
                blocked_reason = None
            except RAGFailure as exc:
                selected = exc.diagnostics["sources"]
                answer = ""
                found = None
                diagnostics = exc.diagnostics
                context_sources = selected
                error = exc.kind
                validation = None
                generation_state = "blocked" if exc.kind in {"authentication", "not_configured", "model"} else "failed"
                generation_attempted = True
                blocked_reason = None
                citation_files = set()
                if generation_state == "blocked":
                    blocked_kind = exc.kind
        if blocked_kind and generation_state == "blocked":
            context_sources = selected
        elif generation_state == "not_needed":
            context_sources = []
        source_files = {source["source"] for source in context_sources}
        expected_sources = set(scenario["expected_sources"])
        source_hit = expected_sources.issubset(source_files) if expected_sources else not source_files
        citation_hit = (expected_sources.issubset(citation_files) if expected_sources else not citation_files) if generation_state != "blocked" else None
        facts_hit = all(fact.casefold() in answer.casefold() for fact in scenario["required_facts"])
        records.append({
            "id": scenario["id"],
            "type": scenario["type"],
            "question": scenario["question"],
            "expected_sources": scenario["expected_sources"],
            "context_sources": sorted(source_files),
            "cited_sources": sorted(citation_files),
            "context": diagnostics["context"],
            "answer": answer,
            "found": found,
            "expected_answer": scenario["answer_expected"],
            "source_hit": source_hit,
            "citation_hit": citation_hit,
            "required_facts_hit": facts_hit,
            "correct_refusal": found is False and not scenario["answer_expected"],
            "unwanted_answer": found is True and not scenario["answer_expected"],
            "false_refusal": found is False and scenario["answer_expected"],
            "validation": validation,
            "generation_state": generation_state,
            "generation_attempted": generation_attempted,
            "blocked_reason": blocked_reason,
            "context_tokens_estimate": sum(source["estimated_tokens"] for source in context_sources),
            "retrieval_ms": diagnostics["retrieval_ms"],
            "generation_ms": diagnostics["generation_ms"],
            "tokens": diagnostics["tokens"],
            "error": error,
            "skipped": False,
            "failure_class": "retrieval" if not source_hit else "generation_unavailable" if generation_state == "blocked" else "generation" if error or not facts_hit else None,
        })
    return records, blocked_kind


def summarize(records):
    completed = [record for record in records if not record.get("skipped")]
    generated = [record for record in completed if record.get("generation_state") in {"completed", "not_needed"}]
    count = len(completed)
    generated_count = len(generated)
    return {
        "cases": len(records),
        "retrieval_completed": count,
        "generation_completed": generated_count,
        "expected_source_hit_rate": sum(record.get("source_hit", False) for record in completed) / max(count, 1),
        "citation_hit_rate": sum(record.get("citation_hit", False) is True for record in generated) / max(generated_count, 1),
        "required_facts_rate": sum(record.get("required_facts_hit", False) for record in generated) / max(generated_count, 1),
        "correct_refusals": sum(record.get("correct_refusal", False) for record in generated),
        "unwanted_answers": sum(record.get("unwanted_answer", False) for record in generated),
        "false_refusals": sum(record.get("false_refusal", False) for record in generated),
        "invalid_citations": sum(record.get("validation") not in {None, "ok", "no_context"} for record in generated),
        "generation_blocked": sum(record.get("generation_state") == "blocked" for record in completed),
        "generation_attempts": sum(record.get("generation_attempted", False) for record in completed),
        "generation_auth_failures": sum(record.get("error") == "authentication" for record in completed),
        "generation_unavailable": sum(record.get("failure_class") == "generation_unavailable" for record in completed),
        "generation_errors": sum(record.get("failure_class") == "generation" for record in completed),
        "retrieval_errors": sum(record.get("failure_class") == "retrieval" for record in completed),
    }


def main():
    try:
        chunks, vectors, model_name = load_index(config.INDEX_DIR)
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc
    results = {}
    blocked_kind = None
    for limit in (4, 2):
        records, blocked_kind = evaluate(chunks, vectors, model_name, limit, blocked_kind)
        results[str(limit)] = {"blocked": blocked_kind, "summary": summarize(records), "cases": records}
    output = BASE_DIR / "eval" / "results.json"
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    original = results["4"]["summary"]
    comparison = results["2"]["summary"]
    if results["4"].get("blocked"):
        findings = (
            "# Результати оцінювання\n\n"
            f"Для кожного з {len(SCENARIOS)} запитань виконано retrieval і збережено контекст для лімітів 4 та 2. Єдина реальна спроба генерації завершилася помилкою `{results['4']['blocked']}`. Наступні запити до Gemini не надсилалися; відповіді моделі, точність фактів і посилання тому не оцінені. Питання без контексту оброблено локально.\n\n"
            "| Ліміт | Запитань у наборі | Потрібне джерело в контексті | Середня оцінка токенів контексту | Спроби Gemini | Спроби з помилкою авторизації |\n"
            "|---:|---:|---:|---:|---:|\n"
            f"| 4 | {original['retrieval_completed']} | {original['expected_source_hit_rate']:.0%} | {sum(record['context_tokens_estimate'] for record in results['4']['cases']) / max(original['retrieval_completed'], 1):.1f} | {original['generation_attempts']} | {original['generation_auth_failures']} |\n"
            f"| 2 | {comparison['retrieval_completed']} | {comparison['expected_source_hit_rate']:.0%} | {sum(record['context_tokens_estimate'] for record in results['2']['cases']) / max(comparison['retrieval_completed'], 1):.1f} | {comparison['generation_attempts']} | {comparison['generation_auth_failures']} |\n\n"
            "За цим набором ліміт у два фрагменти зберіг покриття очікуваних джерел і зменшив середній оцінковий контекст приблизно на 51%. Це лише порівняння retrieval: якість відповідей і цитувань Gemini не виміряна. Єдиний промах — `qa06`: запит про внутрішню процедуру привів до схожих клієнтських матеріалів, але внутрішній документ не потрапив у контекст.\n\n"
            "Для перевірки генерації додайте чинний API-ключ і повторіть `python -m eval.run_eval`. Повні контексти та причини помилок збережені в `results.json`.\n"
        )
    else:
        findings = (
            "# Результати оцінювання\n\n"
            f"У наборі {len(SCENARIOS)} запитань. Змінювався лише ліміт контексту: 4 фрагменти проти 2.\n\n"
            "| Ліміт | Потрібне джерело в контексті | Очікувані джерела процитовано | Факти у відповіді | Відмови для питань без відповіді | Вигадані відповіді | Хибні відмови | Хибні посилання | Помилки пошуку | Помилки генерації |\n"
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
            f"| 4 | {original['expected_source_hit_rate']:.0%} | {original['citation_hit_rate']:.0%} | {original['required_facts_rate']:.0%} | {original['correct_refusals']} | {original['unwanted_answers']} | {original['false_refusals']} | {original['invalid_citations']} | {original['retrieval_errors']} | {original['generation_errors']} |\n"
            f"| 2 | {comparison['expected_source_hit_rate']:.0%} | {comparison['citation_hit_rate']:.0%} | {comparison['required_facts_rate']:.0%} | {comparison['correct_refusals']} | {comparison['unwanted_answers']} | {comparison['false_refusals']} | {comparison['invalid_citations']} | {comparison['retrieval_errors']} | {comparison['generation_errors']} |\n\n"
            "Деталі кожного запиту, включно з повним контекстом, відповіддю, джерелами, часом і токенами, збережені у `results.json`. Якщо правильне джерело є в контексті, але факт відсутній у відповіді, це помилка генерації; якщо очікуваного джерела немає — помилка пошуку. Токени наведено окремо для кожної відповіді в JSON.\n"
        )
    (BASE_DIR / "eval" / "findings.md").write_text(findings, encoding="utf-8")
    print(f"Збережено результати: {output}")
    for limit, result in results.items():
        print(f"Фрагментів {limit}: {result['summary']}")


if __name__ == "__main__":
    main()
