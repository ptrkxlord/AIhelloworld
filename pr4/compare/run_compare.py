import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import prompts, store
from app.llm_client import LLMError, chat
from app.schemas import AssistantReply

SCENARIOS_PATH = Path(__file__).parent / "scenarios.json"
RESULTS_PATH = Path(__file__).parent / "results.json"

NAIVE_INSTRUCTION = "Ти помічник підтримки інтернет-магазину. Відповідай на запитання клієнта."

FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def strip_fences(text: str) -> str:
    match = FENCE_RE.search(text)
    return match.group(1) if match else text


def run_naive(user_text: str) -> dict:
    messages = [{"role": "system", "content": NAIVE_INSTRUCTION}, {"role": "user", "content": user_text}]
    started = time.perf_counter()
    try:
        result = chat(messages, response_schema=None)
        return {"raw": result, "error": None, "wall_ms": (time.perf_counter() - started) * 1000}
    except LLMError as exc:
        return {"raw": None, "error": f"{exc.kind}: {exc}", "wall_ms": (time.perf_counter() - started) * 1000}


def _run_with_schema(messages: list[dict]) -> dict:
    schema = AssistantReply.model_json_schema()
    started = time.perf_counter()
    try:
        result = chat(messages, response_schema=schema)
    except LLMError as exc:
        return {
            "parsed": None,
            "raw": None,
            "error": f"{exc.kind}: {exc}",
            "passed_schema": False,
            "wall_ms": (time.perf_counter() - started) * 1000,
        }
    try:
        parsed = AssistantReply.model_validate_json(strip_fences(result["content"])).model_dump()
        return {
            "parsed": parsed,
            "raw": result,
            "error": None,
            "passed_schema": True,
            "wall_ms": (time.perf_counter() - started) * 1000,
        }
    except Exception as exc:
        return {
            "parsed": None,
            "raw": result,
            "error": str(exc),
            "passed_schema": False,
            "wall_ms": (time.perf_counter() - started) * 1000,
        }


def run_structured(user_text: str) -> dict:
    messages = [prompts.build_system_message(), {"role": "user", "content": user_text}]
    return _run_with_schema(messages)


def run_context_engineering(conversation: store.Conversation, user_text: str) -> dict:
    messages = [
        prompts.build_system_message(),
        prompts.build_context_message(),
        *conversation.budgeted_history(2000),
        {"role": "user", "content": user_text},
    ]
    outcome = _run_with_schema(messages)
    if outcome["parsed"]:
        conversation.add_turn(user_text, outcome["parsed"]["reply"])
        conversation.update_ticket(outcome["parsed"])
    return outcome


def main() -> None:
    scenarios = json.loads(SCENARIOS_PATH.read_text(encoding="utf-8"))
    all_results = []

    for scenario in scenarios:
        entry = {"id": scenario["id"], "type": scenario["type"], "naive": [], "structured": [], "context": []}

        for turn in scenario["turns"]:
            entry["naive"].append(run_naive(turn))

        context_conversation = store.Conversation()
        for turn in scenario["turns"]:
            entry["structured"].append(run_structured(turn))
            entry["context"].append(run_context_engineering(context_conversation, turn))

        all_results.append(entry)
        print(f"scenario {scenario['id']} done")

    RESULTS_PATH.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
