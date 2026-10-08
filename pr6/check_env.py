from app import config
from app.gemini_client import GeminiFailure, generate


def main() -> None:
    if not config.GEMINI_API_KEY:
        raise SystemExit("У .env відсутній GEMINI_API_KEY")
    try:
        response = generate("[1] Перевірка: Nova Air підключається через Bluetooth.", "Як підключити Nova Air?")
    except GeminiFailure as exc:
        messages = {
            "authentication": "Google AI Studio відхилив ключ. Створіть або перевірте ключ у потрібному проєкті.",
            "rate_limit": "Перевищено ліміт запитів Gemini.",
            "timeout": "Перевірка перевищила час очікування.",
            "model": "Модель Gemini недоступна для цього ключа.",
        }
        raise SystemExit(messages.get(exc.kind, f"Перевірка Gemini не вдалася: {exc.kind}")) from exc
    print(f"Gemini доступний: {config.GEMINI_MODEL}; відповідь отримана за {response['latency_ms']} мс")


if __name__ == "__main__":
    main()
