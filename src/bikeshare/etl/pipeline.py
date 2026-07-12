"""Orquestador del ETL: Extract → Transform → Features → Load.

Uso:
    uv run python -m bikeshare.etl.pipeline
"""

from __future__ import annotations

import pandas as pd

from bikeshare import config
from bikeshare.etl.extract import extract_holidays, extract_uci, extract_weather
from bikeshare.etl.load import save_processed
from bikeshare.etl.transform import transform, weather_match_rate
from bikeshare.features import FEATURE_COLUMNS, build_features


def run() -> pd.DataFrame:
    print("== ETL: extracción de las 3 fuentes ==")
    uci = extract_uci()
    weather = extract_weather()
    holidays_df = extract_holidays()
    print(f"  UCI:      {uci.shape[0]:>6} filas")
    print(f"  Clima:    {weather.shape[0]:>6} filas")
    print(f"  Feriados: {holidays_df.shape[0]:>6} fechas")

    print("== ETL: transformación y unión ==")
    merged = transform(uci, weather, holidays_df)
    print(f"  Unido:    {merged.shape[0]:>6} filas x {merged.shape[1]} columnas")
    print(f"  Match clima: {weather_match_rate(merged):.2f}%")

    print("== ETL: ingeniería de características ==")
    featured = build_features(merged)
    print(f"  Final:    {featured.shape[0]:>6} filas x {featured.shape[1]} columnas")
    nulls = int(featured[FEATURE_COLUMNS + [config.TARGET]].isna().sum().sum())
    print(f"  Nulos en features+target: {nulls}")

    print("== ETL: carga ==")
    path = save_processed(featured)
    print(f"  Guardado en: {path}")
    return featured


if __name__ == "__main__":
    run()
