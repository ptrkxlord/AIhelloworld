# Практична робота 3

Вебзастосунок на FastAPI для асистента підтримки інтернет-магазину. Асистент відповідає тільки за правилами з `context.md`, показує модель і час виконання запиту.

## Запуск

```bash
cd pr3
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

У файлі `.env` потрібно заповнити `LLM_API_KEY`.

```bash
uvicorn app.main:a --reload
```

Сайт відкривається за адресою `http://127.0.0.1:8000`.

## Перевірка

```bash
python check_env.py
python -m pytest
python eval/run_comparison.py
```

`context.md` можна змінити без зміни коду. Після цього асистент відповідатиме за новими правилами.
