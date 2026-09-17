from pathlib import Path
import json
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.config import load_settings
from app.llm_client import ask_model


def main() -> int:
    a = Path(__file__).resolve().parent
    b = json.loads((a / "prompts.json").read_text(encoding="utf-8"))
    c = load_settings()
    d = c.context_path.read_text(encoding="utf-8")
    e = [
        ("temperature_0_2", c.with_generation(0.2, 0.9, c.max_tokens)),
        ("temperature_0_8", c.with_generation(0.8, 0.9, c.max_tokens)),
    ]
    f = []

    for g, h in e:
        for i in b:
            j = ask_model(i, d, h)
            f.append({
                "config": g,
                "prompt": i,
                "answer": j.answer,
                "model": j.model,
                "elapsed_ms": j.elapsed_ms,
            })

    (a / "results.json").write_text(json.dumps(f, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Результати збережено в eval/results.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
