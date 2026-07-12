"""Ingeniería de características para el modelo de demanda.

Incluye variables de calendario, codificación **cíclica** (sin/cos) y variables
**autoregresivas** (lags y medias móviles) calculadas respetando el orden temporal
real (usando una rejilla horaria completa para no confundir huecos con horas contiguas).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bikeshare.config import TARGET

# Columnas que consume el modelo (orden estable).
FEATURE_COLUMNS: list[str] = [
    # Clima real (Open-Meteo)
    "temperature_2m",
    "apparent_temperature",
    "precipitation",
    "relative_humidity_2m",
    "wind_speed_10m",
    # Contexto (UCI)
    "season",
    "weathersit",
    "workingday",
    "is_holiday",
    "is_weekend",
    "yr",
    # Temporales cíclicas
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "weekday_sin",
    "weekday_cos",
    # Autoregresivas
    "lag_1h",
    "lag_24h",
    "roll_mean_3h",
    "roll_mean_24h",
]


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    dt = pd.to_datetime(out["datetime"])
    out["hour"] = dt.dt.hour
    out["dayofweek"] = dt.dt.dayofweek  # 0 = lunes
    out["month"] = dt.dt.month
    out["year"] = dt.dt.year
    out["is_weekend"] = (out["dayofweek"] >= 5).astype(int)
    return out


def add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)
    out["weekday_sin"] = np.sin(2 * np.pi * out["dayofweek"] / 7)
    out["weekday_cos"] = np.cos(2 * np.pi * out["dayofweek"] / 7)
    return out


def add_lag_features(df: pd.DataFrame, target: str = TARGET) -> pd.DataFrame:
    """Agrega lags y medias móviles usando una rejilla horaria completa.

    Así, ``lag_24h`` es realmente el valor de 24 horas antes (no la 24ª observación
    previa), evitando que los huecos del dataset distorsionen las variables.
    """
    out = df.sort_values("datetime").reset_index(drop=True).copy()
    full_idx = pd.date_range(out["datetime"].min(), out["datetime"].max(), freq="1h")
    s_full = out.set_index("datetime")[target].reindex(full_idx)

    lag_1h = s_full.shift(1)
    lag_24h = s_full.shift(24)
    roll_3h = s_full.shift(1).rolling(3, min_periods=1).mean()
    roll_24h = s_full.shift(1).rolling(24, min_periods=1).mean()

    out["lag_1h"] = out["datetime"].map(lag_1h)
    out["lag_24h"] = out["datetime"].map(lag_24h)
    out["roll_mean_3h"] = out["datetime"].map(roll_3h)
    out["roll_mean_24h"] = out["datetime"].map(roll_24h)
    return out


def single_feature_row(
    *,
    hour: int,
    month: int,
    weekday: int,
    season: int,
    yr: int,
    workingday: int,
    is_holiday: int,
    weathersit: int,
    temperature_2m: float,
    relative_humidity_2m: float,
    wind_speed_10m: float,
    precipitation: float = 0.0,
    apparent_temperature: float | None = None,
    recent_demand: float = 190.0,
) -> dict[str, float]:
    """Construye una fila de features (para la API / dashboard) desde entradas amigables.

    Las variables autoregresivas (lags y medias móviles) se aproximan con ``recent_demand``,
    la demanda típica reciente, ya que en un escenario *what-if* no hay historia real.
    """
    if apparent_temperature is None:
        apparent_temperature = temperature_2m
    row = {
        "temperature_2m": float(temperature_2m),
        "apparent_temperature": float(apparent_temperature),
        "precipitation": float(precipitation),
        "relative_humidity_2m": float(relative_humidity_2m),
        "wind_speed_10m": float(wind_speed_10m),
        "season": int(season),
        "weathersit": int(weathersit),
        "workingday": int(workingday),
        "is_holiday": int(is_holiday),
        "is_weekend": int(weekday >= 5),
        "yr": int(yr),
        "hour_sin": float(np.sin(2 * np.pi * hour / 24)),
        "hour_cos": float(np.cos(2 * np.pi * hour / 24)),
        "month_sin": float(np.sin(2 * np.pi * month / 12)),
        "month_cos": float(np.cos(2 * np.pi * month / 12)),
        "weekday_sin": float(np.sin(2 * np.pi * weekday / 7)),
        "weekday_cos": float(np.cos(2 * np.pi * weekday / 7)),
        "lag_1h": float(recent_demand),
        "lag_24h": float(recent_demand),
        "roll_mean_3h": float(recent_demand),
        "roll_mean_24h": float(recent_demand),
    }
    return {c: row[c] for c in FEATURE_COLUMNS}


def build_features(
    df: pd.DataFrame, target: str = TARGET, dropna: bool = True
) -> pd.DataFrame:
    """Pipeline de features: calendario → cíclicas → autoregresivas."""
    out = add_calendar_features(df)
    out = add_cyclical_features(out)
    out = add_lag_features(out, target=target)
    if dropna:
        subset = [c for c in FEATURE_COLUMNS + [target] if c in out.columns]
        out = out.dropna(subset=subset).reset_index(drop=True)
    return out
