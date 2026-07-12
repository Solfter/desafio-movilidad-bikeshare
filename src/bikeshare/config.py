"""Configuración central: rutas, coordenadas de la ciudad y parámetros de las fuentes."""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Rutas del proyecto
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

PROCESSED_FILE = PROCESSED_DIR / "bikeshare_features.parquet"
MODEL_FILE = MODELS_DIR / "model.pkl"
METRICS_FILE = MODELS_DIR / "metrics.json"
IMPORTANCES_FILE = MODELS_DIR / "feature_importances.csv"

# ---------------------------------------------------------------------------
# Fuente 1: UCI Bike Sharing Dataset (Washington DC, 2011-2012, horario)
# ---------------------------------------------------------------------------
UCI_DATASET_ID = 275

# ---------------------------------------------------------------------------
# Ciudad de referencia: Washington DC (Capital Bikeshare)
# ---------------------------------------------------------------------------
CITY_LAT = 38.90
CITY_LON = -77.04
TIMEZONE = "America/New_York"
DATE_START = "2011-01-01"
DATE_END = "2012-12-31"

# ---------------------------------------------------------------------------
# Fuente 2: Open-Meteo Archive API (clima histórico real, gratis, sin clave)
# ---------------------------------------------------------------------------
OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_HOURLY = [
    "temperature_2m",
    "apparent_temperature",
    "precipitation",
    "relative_humidity_2m",
    "wind_speed_10m",
    "weather_code",
]

# ---------------------------------------------------------------------------
# Fuente 3: feriados federales de EE.UU. (librería `holidays`)
# ---------------------------------------------------------------------------
HOLIDAYS_COUNTRY = "US"
HOLIDAYS_YEARS = [2011, 2012]

# ---------------------------------------------------------------------------
# Objetivo del modelo
# ---------------------------------------------------------------------------
TARGET = "cnt"
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Servicios (usado por el dashboard para consumir la API)
# ---------------------------------------------------------------------------
API_URL = os.environ.get("API_URL", "http://localhost:8000")


def ensure_dirs() -> None:
    """Crea los directorios de datos y modelos si no existen."""
    for d in (RAW_DIR, INTERIM_DIR, PROCESSED_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)
