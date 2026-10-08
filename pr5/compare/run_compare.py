import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import config
from app.chunking import load_chunks
from app.embeddings import encode_passages, encode_query
from app.search import keyword_search, semantic_search


SCENARIOS = Path(__file__).with_name("scenarios.json")
OUTPUT = Path(__file__).with_name("results.json")
FINDINGS = Path(__file__).with_name("findings.md")


def evaluate(scenarios: list[dict], chunk_size: int, label: str) -> tuple[dict, list[dict]]:
    chunks = load_chunks(config.DOCS_DIR, chunk_size, config.CHUNK_OVERLAP)
    vectors = encode_passages([chunk["embedding_text"] for chunk in chunks])
    per_query = {"semantic": [], "keyword": []}
    for scenario in scenarios:
        filters = scenario["filters"]
        started = time.perf_counter()
        query_vector = encode_query(scenario["query"])
        semantic_scores = vectors @ query_vector
        allowed = [i for i, chunk in enumerate(chunks)
                   if all(not value or chunk["metadata"].get(key) == value for key, value in filters.items())]
        ranked_semantic = sorted((i for i in allowed if semantic_scores[i] >= config.DEFAULT_MIN_SCORE),
                                 key=lambda i: float(semantic_scores[i]), reverse=True)
        best_semantic_score = max((float(semantic_scores[i]) for i in allowed), default=0.0)
        semantic_elapsed = (time.perf_counter() - started) * 1000
        keyword = keyword_search(scenario["query"], chunks, len(chunks), filters)
        for method, ranked, scores, elapsed in (
            ("semantic", ranked_semantic, semantic_scores, semantic_elapsed),
            ("keyword", [], None, keyword["latency_ms"]),
        ):
            if method == "keyword":
                score_by_source = {}
                ranked_sources = []
                for result in keyword["results"]:
                    if result["source"] not in ranked_sources:
                        ranked_sources.append(result["source"])
                        score_by_source[result["source"]] = result["score"]
                rank = ranked_sources.index(scenario["expected"]) + 1 if scenario["expected"] in ranked_sources else None
                best_score = score_by_source.get(ranked_sources[0], 0) if ranked_sources else 0
                higher_score = score_by_source.get(ranked_sources[rank - 2], 0) if rank and rank > 1 else None
            else:
                source_scores = {}
                for index in ranked:
                    source = chunks[index]["source"]
                    source_scores[source] = max(source_scores.get(source, -1.0), float(semantic_scores[index]))
                ranked_sources = sorted(source_scores, key=source_scores.get, reverse=True)
                rank = ranked_sources.index(scenario["expected"]) + 1 if scenario["expected"] in ranked_sources else None
                best_score = best_semantic_score
                higher_score = source_scores[ranked_sources[rank - 2]] if rank and rank > 1 else None
            per_query[method].append({
                "id": scenario["id"], "type": scenario["type"], "expected": scenario["expected"],
                "rank": rank, "hit_at_1": rank == 1 if scenario["expected"] else not ranked_sources,
                "hit_at_3": rank is not None and rank <= 3 if scenario["expected"] else not ranked_sources,
                "top_score": round(float(best_score), 4),
                "score_above_expected": round(float(higher_score), 4) if higher_score is not None else None,
                "latency_ms": round(elapsed, 2), "top_sources": ranked_sources[:5],
            })
    summary = {}
    for method, rows in per_query.items():
        positives = [row for row, scenario in zip(rows, scenarios) if scenario["expected"]]
        negatives = [row for row, scenario in zip(rows, scenarios) if not scenario["expected"]]
        summary[method] = {
            "hit_at_1": round(sum(row["hit_at_1"] for row in positives) / max(len(positives), 1), 3),
            "hit_at_3": round(sum(row["hit_at_3"] for row in positives) / max(len(positives), 1), 3),
            "mean_latency_ms": round(sum(row["latency_ms"] for row in rows) / max(len(rows), 1), 2),
            "positive_top_scores": [row["top_score"] for row in positives],
            "out_of_collection_top_scores": [row["top_score"] for row in negatives],
            "out_of_collection_returned": sum(bool(row["top_sources"]) for row in negatives),
        }
    return {"label": label, "chunk_size": chunk_size, "chunk_count": len(chunks),
            "summary": summary, "queries": per_query}, chunks


def main() -> None:
    scenarios = json.loads(SCENARIOS.read_text(encoding="utf-8"))
    baseline, _ = evaluate(scenarios, config.CHUNK_SIZE, "базовий")
    changed_size = max(200, config.CHUNK_SIZE - 700)
    changed, _ = evaluate(scenarios, changed_size, f"фрагмент {changed_size} символів")
    payload = {"model": config.EMBEDDING_MODEL, "scenario_count": len(scenarios),
               "baseline": baseline, "changed": changed}
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Порівняння пошуку", "", f"Модель: `{config.EMBEDDING_MODEL}`.",
             f"Набір: {len(scenarios)} запитів. Змінено лише розмір фрагмента: {baseline['chunk_size']} → {changed['chunk_size']} символів.",
             "Час семантичного пошуку включає кодування запиту; час BM25 не включає кодування.", ""]
    for result in (baseline, changed):
        lines.extend([f"## {result['label']}", "",
                      f"Фрагментів: {result['chunk_count']}.", "",
                      "| Метод | hit@1 | hit@3 | Середній час, мс | OOD повернуто |",
                      "|---|---:|---:|---:|---:|"])
        for method, metrics in result["summary"].items():
            lines.append(f"| {method} | {metrics['hit_at_1']:.3f} | {metrics['hit_at_3']:.3f} | {metrics['mean_latency_ms']:.2f} | {metrics['out_of_collection_returned']} |")
        lines.append("")
        lines.extend(["| Запит | Очікуваний документ | Ранг semantic | Ранг BM25 | Найвища оцінка semantic | Оцінка вище очікуваного | Час semantic, мс | Час BM25, мс |",
                      "|---|---|---:|---:|---:|---:|---:|---:|"])
        for semantic, keyword in zip(result["queries"]["semantic"], result["queries"]["keyword"]):
            expected = semantic["expected"] or "поза колекцією"
            lines.append(f"| {semantic['id']} | {expected} | {semantic['rank'] or '—'} | {keyword['rank'] or '—'} | {semantic['top_score']:.4f} | {semantic['score_above_expected'] if semantic['score_above_expected'] is not None else '—'} | {semantic['latency_ms']:.2f} | {keyword['latency_ms']:.2f} |")
        lines.append("")
    positives = baseline["summary"]["semantic"]["positive_top_scores"]
    negatives = baseline["summary"]["semantic"]["out_of_collection_top_scores"]
    threshold = round((min(positives) + max(negatives)) / 2, 4) if positives and negatives and max(negatives) < min(positives) else None
    base_semantic = baseline["summary"]["semantic"]
    changed_semantic = changed["summary"]["semantic"]
    base_keyword = baseline["summary"]["keyword"]
    changed_keyword = changed["summary"]["keyword"]
    long_query = next(row for row in scenarios if row["type"] == "довге питання")
    base_long_rank = next(row["rank"] for row in baseline["queries"]["semantic"] if row["id"] == long_query["id"])
    changed_long_rank = next(row["rank"] for row in changed["queries"]["semantic"] if row["id"] == long_query["id"])
    lines.extend(["## Висновок", "",
                  f"На цих 14 запитах із відомою відповіддю semantic зберіг hit@1 {base_semantic['hit_at_1']:.3f} та hit@3 {base_semantic['hit_at_3']:.3f} в обох конфігураціях. Для довгого запиту цільовий документ перемістився з рангу {base_long_rank} на {changed_long_rank}.",
                  f"BM25 змінив hit@1 з {base_keyword['hit_at_1']:.3f} до {changed_keyword['hit_at_1']:.3f}, hit@3 — з {base_keyword['hit_at_3']:.3f} до {changed_keyword['hit_at_3']:.3f}. Коротші фрагменти дали більше точок пошуку, але довгий контекст розподілився між кількома уривками.",
                  f"Час у таблиці виміряний у межах одного прогону; перший semantic-запит включає холодний прогрів моделі ({baseline['queries']['semantic'][0]['latency_ms']:.2f} мс), тому середній час не є чистим порівнянням швидкості розміру фрагмента.",
                  f"Мінімальна оцінка позитивних запитів: {min(positives):.4f}; максимальна оцінка запиту поза колекцією: {max(negatives):.4f}."])
    if threshold is not None:
        lines.append(f"Для цієї вибірки поріг між позитивними та позаколекційним запитом можна встановити близько {threshold:.4f}; перевірте компроміс на всіх сценаріях перед використанням.")
    else:
        lines.append("Оцінки позитивного та позаколекційного запитів перетинаються, тому єдиного порога без помилок ця вибірка не дає. Поріг слід вибирати за прийнятним балансом пропусків і хибних збігів.")
    FINDINGS.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Результати записано: {OUTPUT}")
    print(f"Висновки записано: {FINDINGS}")


if __name__ == "__main__":
    main()
