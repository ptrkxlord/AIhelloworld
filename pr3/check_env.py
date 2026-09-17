from pathlib import Path
import os
import sys

from dotenv import load_dotenv


def main() -> int:
    a = Path(__file__).resolve().parent
    load_dotenv(a / ".env")
    b = [
        a / "app" / "main.py",
        a / "app" / "llm_client.py",
        a / "app" / "assistant.py",
        a / "context.md",
        a / "app" / "static" / "index.html",
    ]
    c = [str(d) for d in b if not d.exists()]
    if c:
        print("Не знайдено файли:")
        print("\n".join(c))
        return 1
    if not os.getenv("LLM_API_KEY"):
        print("LLM_API_KEY не налаштовано у pr3/.env або змінних середовища.")
        return 1
    print("Середовище налаштовано.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
