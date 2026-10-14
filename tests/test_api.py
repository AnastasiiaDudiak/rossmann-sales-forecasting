
import pytest
from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


def test_health():
    """Перевірка доступності FastAPI."""
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_data_info():
    """Перевірка інформації про датасет."""
    response = client.get("/data/info")

    assert response.status_code == 200

    data = response.json()

    assert "train" in data
    assert "test" in data
    assert data["train"]["rows"] > 0
    assert data["test"]["rows"] > 0


def test_models_list():
    """Перевірка списку збережених моделей."""
    response = client.get("/models")

    assert response.status_code == 200

    data = response.json()

    assert "total" in data
    assert "models" in data
    assert isinstance(data["models"], list)
    assert data["total"] == len(data["models"])



def test_model_info():
    """Перевірка інформації про збережену модель."""
    response = client.get("/models/random_forest")

    assert response.status_code == 200

    data = response.json()

    assert data["model_name"] == "random_forest"
    assert "features" in data
    assert "parameters" in data
    assert "metrics" in data


def test_model_validation_metrics():
    """Перевірка метрик навченої моделі."""
    response = client.get("/models/lightgbm")

    assert response.status_code == 200

    data = response.json()
    metrics = data["metrics"]["validation"]

    assert "mae" in metrics
    assert "rmse" in metrics
    assert "wape" in metrics
    assert "r2" in metrics

    assert metrics["mae"] >= 0
    assert metrics["rmse"] >= 0
    assert metrics["wape"] >= 0


def test_unknown_model():
    """Перевірка запиту до неіснуючої моделі."""
    response = client.get("/models/unknown_model")

    assert response.status_code == 404



def test_forecast_success():
    """Перевірка успішного прогнозування продажів."""
    payload = {
        "model_name": "lightgbm",
        "stores": [1],
        "start_date": "2015-04-01",
        "end_date": "2015-04-07"
    }

    response = client.post("/forecast", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert data["model"] == "lightgbm"
    assert "forecasts" in data
    assert "metrics" in data

    for forecast in data["forecasts"]:
        assert forecast["predicted_sales"] >= 0


def test_forecast_days_count():
    """Перевірка кількості прогнозованих днів."""
    payload = {
        "model_name": "lightgbm",
        "stores": [1],
        "start_date": "2015-04-01",
        "end_date": "2015-04-07"
    }

    response = client.post("/forecast", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert data["total"] == 7
    assert len(data["forecasts"]) == 7



def test_scenario_forecast_success():
    """Перевірка сценарного прогнозування з акцією."""
    payload = {
        "model_name": "lightgbm",
        "stores": [1],
        "start_date": "2015-04-01",
        "end_date": "2015-04-07",
        "promo": 1,
        "school_holiday": 0,
        "state_holiday": "0",
        "open": 1
    }

    response = client.post("/forecast/scenario", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert data["model"] == "lightgbm"
    assert data["mode"] == "scenario"
    assert data["total"] == 7
    assert len(data["forecasts"]) == 7


def test_scenario_closed_store():
    """Перевірка нульових продажів для закритого магазину."""
    payload = {
        "model_name": "lightgbm",
        "stores": [1],
        "start_date": "2015-04-01",
        "end_date": "2015-04-07",
        "promo": 0,
        "school_holiday": 0,
        "state_holiday": "0",
        "open": 0
    }

    response = client.post("/forecast/scenario", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert len(data["forecasts"]) == 7

    for forecast in data["forecasts"]:
        assert forecast["predicted_sales"] == 0


def test_models_endpoint_without_models(monkeypatch, tmp_path):
    """API повертає порожній список, якщо папки моделей немає."""
    from fastapi.testclient import TestClient
    from api import main

    missing_dir = tmp_path / "missing_models"
    monkeypatch.setattr(main, "MODELS_DIR", missing_dir)

    client = TestClient(main.app)
    response = client.get("/models")

    assert response.status_code == 200
    assert response.json() == {
        "total": 0,
        "models": []
    }


def test_delete_model_success(monkeypatch, tmp_path):
    """Успішне видалення тестової моделі."""
    from fastapi.testclient import TestClient
    from api import main

    model_dir = tmp_path / "test_model"
    model_dir.mkdir()
    (model_dir / "metadata.json").write_text(
        '{"model_name": "test_model"}',
        encoding="utf-8"
    )

    monkeypatch.setattr(main, "MODELS_DIR", tmp_path)

    client = TestClient(main.app)
    response = client.delete("/models/test_model")

    assert response.status_code == 200
    assert response.json()["model_name"] == "test_model"
    assert not model_dir.exists()


def test_delete_model_not_found(monkeypatch, tmp_path):
    """Видалення неіснуючої моделі."""
    from fastapi.testclient import TestClient
    from api import main

    monkeypatch.setattr(main, "MODELS_DIR", tmp_path)

    client = TestClient(main.app)
    response = client.delete("/models/unknown_model")

    assert response.status_code == 404


def test_delete_model_invalid_name(monkeypatch, tmp_path):
    """Заборона некоректних назв моделей."""
    from fastapi.testclient import TestClient
    from api import main

    monkeypatch.setattr(main, "MODELS_DIR", tmp_path)

    client = TestClient(main.app)
    response = client.delete("/models/invalid.name")

    assert response.status_code == 400
