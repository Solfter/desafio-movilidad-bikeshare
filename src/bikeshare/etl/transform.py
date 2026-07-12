"""Transformación: construye la clave temporal, une las 3 fuentes y limpia.

Las funciones son puras (reciben DataFrames) para poder testearlas sin red.
``transform()`` orquesta y, si no se le pasan datos, los extrae.
"""

from __future__ import annotations

import pandas as pd

# Constantes de desnormalización del dataset UCI (según su README)
_TEMP_MAX = 41  # °C
_ATEMP_MAX = 50  # °C (sensación térmica)
_HUM_MAX = 100  # %
_WIND_MAX = 67  # km/h

_NUMERIC_WEATHER = [
    "temperature_2m",
    "apparent_temperature",
    "precipitation",
    "relative_humidity_2m",
    "wind_speed_10m",
]


def build_datetime(uci: pd.DataFrame) -> pd.DataFrame:
    """Crea la columna ``datetime`` = fecha (``dteday``) + hora (``hr``)."""
    df = uci.copy()
    df["dteday"] = pd.to_datetime(df["dteday"])
    df["datetime"] = df["dteday"] + pd.to_timedelta(df["hr"], unit="h")
    return df


def merge_weather(df: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Une el clima real de Open-Meteo por ``datetime`` (left join)."""
    weather = weather.copy()
    weather["datetime"] = pd.to_datetime(weather["datetime"])
    return df.merge(weather, on="datetime", how="left")


def merge_holidays(df: pd.DataFrame, holidays_df: pd.DataFrame) -> pd.DataFrame:
    """Une los feriados por fecha y agrega la bandera ``is_holiday``."""
    h = holidays_df.copy()
    h["date"] = pd.to_datetime(h["date"]).dt.normalize()
    out = df.copy()
    out["date"] = pd.to_datetime(out["dteday"]).dt.normalize()
    out = out.merge(h, on="date", how="left")
    out["is_holiday"] = out["holiday_name"].notna().astype(int)
    return out.drop(columns=["date"])


def denormalize_uci(df: pd.DataFrame) -> pd.DataFrame:
    """Recupera unidades reales de las variables normalizadas de UCI (para el dashboard)."""
    out = df.copy()
    out["temp_c"] = out["temp"] * _TEMP_MAX
    out["atemp_c"] = out["atemp"] * _ATEMP_MAX
    out["humidity_pct"] = out["hum"] * _HUM_MAX
    out["windspeed_kmh"] = out["windspeed"] * _WIND_MAX
    return out


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Ordena por tiempo e interpola huecos del clima (por si la API dejó nulos)."""
    out = df.sort_values("datetime").reset_index(drop=True)
    present = [c for c in _NUMERIC_WEATHER if c in out.columns]
    out[present] = out[present].interpolate(method="linear", limit_direction="both")
    if "weather_code" in out.columns:
        out["weather_code"] = out["weather_code"].ffill().bfill().round().astype("Int64")
    return out


def transform(
    uci: pd.DataFrame | None = None,
    weather: pd.DataFrame | None = None,
    holidays_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Orquesta la unión de las 3 fuentes en una tabla horaria limpia."""
    from bikeshare.etl.extract import extract_holidays, extract_uci, extract_weather

    uci = extract_uci() if uci is None else uci
    weather = extract_weather() if weather is None else weather
    holidays_df = extract_holidays() if holidays_df is None else holidays_df

    df = build_datetime(uci)
    df = merge_weather(df, weather)
    df = merge_holidays(df, holidays_df)
    df = denormalize_uci(df)
    df = clean(df)
    return df


def weather_match_rate(df: pd.DataFrame) -> float:
    """% de filas que encontraron clima tras el join (evidencia de calidad del ETL)."""
    if "temperature_2m" not in df.columns or len(df) == 0:
        return 0.0
    return float(df["temperature_2m"].notna().mean() * 100)


if __name__ == "__main__":
    merged = transform()
    print("Merged:", merged.shape)
    print(f"Weather match: {weather_match_rate(merged):.2f}%")
    print(merged[["datetime", "cnt", "temperature_2m", "precipitation", "is_holiday"]].head())
