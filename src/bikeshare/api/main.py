"""API FastAPI para servir el modelo de demanda de bicicletas.

Endpoints:
    GET  /health      → estado del servicio
    GET  /model/info  → modelo activo, features y métricas
    POST /predict     → demanda estimada a partir de entradas amigables

Ejecutar:
    uv run uvicorn bikeshare.api.main:app --port 8000
"""

from __future__ import annotations

import json

from fastapi import FastAPI
from pydantic import BaseModel, Field

from bikeshare import config
from bikeshare.features import single_feature_row
from bikeshare.models.predict import predict as model_predict

app = FastAPI(
    title="API — Demanda de bicicletas",
    description="Predice la demanda horaria de un sistema de bikesharing.",
    version="0.1.0",
)


class PredictRequest(BaseModel):
    """Entradas amigables para una predicción *what-if*."""

    hour: int = Field(..., ge=0, le=23, description="Hora del día (0-23)")
    month: int = Field(..., ge=1, le=12, description="Mes (1-12)")
    weekday: int = Field(..., ge=0, le=6, description="Día de la semana (0=lunes)")
    season: int = Field(2, ge=1, le=4, description="Estación (1=inv,2=prim,3=ver,4=oto)")
    yr: int = Field(1, ge=0, le=1, description="Año (0=2011, 1=2012)")
    workingday: int = Field(1, ge=0, le=1, description="¿Día laboral? (0/1)")
    is_holiday: int = Field(0, ge=0, le=1, description="¿Feriado? (0/1)")
    weathersit: int = Field(1, ge=1, le=4, description="Clima UCI (1=despejado…4=tormenta)")
    temperature_2m: float = Field(20.0, description="Temperatura (°C)")
    apparent_temperature: float | None = Field(None, description="Sensación térmica (°C)")
    precipitation: float = Field(0.0, ge=0, description="Precipitación (mm)")
    relative_humidity_2m: float = Field(60.0, ge=0, le=100, description="Humedad relativa (%)")
    wind_speed_10m: float = Field(10.0, ge=0, description="Velocidad del viento (km/h)")
    recent_demand: float = Field(190.0, ge=0, description="Demanda reciente típica (para lags)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "hour": 8,
                "month": 6,
                "weekday": 2,
                "season": 3,
                "yr": 1,
                "workingday": 1,
                "is_holiday": 0,
                "weathersit": 1,
                "temperature_2m": 24.0,
                "precipitation": 0.0,
                "relative_humidity_2m": 55.0,
                "wind_speed_10m": 12.0,
                "recent_demand": 200.0,
            }
        }
    }


class PredictResponse(BaseModel):
    predicted_demand: float = Field(..., description="Demanda horaria estimada (bicicletas)")


@app.get("/health")
def health() -> dict:
    """Estado del servicio y si el modelo está disponible."""
    return {"status": "ok", "model_available": config.MODEL_FILE.exists()}


@app.get("/model/info")
def model_info() -> dict:
    """Devuelve el modelo activo, sus features y las métricas de evaluación."""
    info: dict = {"model_file": config.MODEL_FILE.name}
    if config.METRICS_FILE.exists():
        info["metrics"] = json.loads(config.METRICS_FILE.read_text(encoding="utf-8"))
    if config.MODEL_FILE.exists():
        from bikeshare.models.predict import load_model

        info["features"] = load_model()["features"]
    return info


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    """Estima la demanda horaria a partir de las condiciones dadas."""
    row = single_feature_row(
        hour=req.hour,
        month=req.month,
        weekday=req.weekday,
        season=req.season,
        yr=req.yr,
        workingday=req.workingday,
        is_holiday=req.is_holiday,
        weathersit=req.weathersit,
        temperature_2m=req.temperature_2m,
        apparent_temperature=req.apparent_temperature,
        precipitation=req.precipitation,
        relative_humidity_2m=req.relative_humidity_2m,
        wind_speed_10m=req.wind_speed_10m,
        recent_demand=req.recent_demand,
    )
    pred = float(model_predict(row)[0])
    return PredictResponse(predicted_demand=round(pred, 1))
