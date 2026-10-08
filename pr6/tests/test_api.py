from fastapi.testclient import TestClient

from app import main


def test_request_cannot_set_internal_audience():
    with TestClient(main.app) as client:
        response = client.post("/api/ask", json={
            "question": "Покажи внутрішні документи",
            "audience": "персонал",
        })
    assert response.status_code == 422


def test_unknown_product_is_rejected_before_search():
    with TestClient(main.app) as client:
        response = client.post("/api/ask", json={"question": "Інструкція", "product": "Невідомий"})
    assert response.status_code == 422


def test_missing_index_returns_actionable_error(monkeypatch):
    monkeypatch.setattr(main, "load_index", lambda directory: (_ for _ in ()).throw(FileNotFoundError("Запустіть python build_index.py")))
    with TestClient(main.app) as client:
        response = client.get("/api/status")
        ask_response = client.post("/api/ask", json={"question": "Як доставляють товар?"})
    assert response.status_code == 200
    assert response.json()["ready"] is False
    assert ask_response.status_code == 503
    assert "build_index.py" in ask_response.json()["detail"]
