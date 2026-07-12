"""Tests de la API FastAPI con TestClient."""

import pytest
from fastapi.testclient import TestClient

from bikeshare import config
from bikeshare.api.main import app

client = TestClient(app)

_VALID = {
    "hour": 8, "month": 6, "weekday": 2, "season": 3, "yr": 1,
    "workingday": 1, "is_holiday": 0, "weathersit": 1,
    "temperature_2m": 24.0, "precipitation": 0.0,
    "relative_humidity_2m": 55.0, "wind_speed_10m": 12.0, "recent_demand": 200.0,
}


def test_health_ok():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_predict_validation_error():
    bad = dict(_VALID, hour=30)  # fuera de rango (0-23)
    resp = client.post("/predict", json=bad)
    assert resp.status_code == 422


@pytest.mark.skipif(not config.MODEL_FILE.exists(), reason="requiere models/model.pkl")
def test_predict_happy_path():
    resp = client.post("/predict", json=_VALID)
    assert resp.status_code == 200
    demand = resp.json()["predicted_demand"]
    assert isinstance(demand, (int, float))
    assert demand >= 0
