"""Tests de ingeniería de características."""

import numpy as np
import pandas as pd

from bikeshare.features import (
    FEATURE_COLUMNS,
    build_features,
    single_feature_row,
)


def test_single_feature_row_has_exact_feature_columns():
    row = single_feature_row(
        hour=8, month=6, weekday=2, season=3, yr=1, workingday=1,
        is_holiday=0, weathersit=1, temperature_2m=24.0,
        relative_humidity_2m=55.0, wind_speed_10m=12.0,
    )
    assert list(row.keys()) == FEATURE_COLUMNS


def test_cyclical_encoding_values():
    row = single_feature_row(
        hour=6, month=3, weekday=0, season=2, yr=0, workingday=1,
        is_holiday=0, weathersit=1, temperature_2m=15.0,
        relative_humidity_2m=60.0, wind_speed_10m=10.0,
    )
    # hora 6 → sin(2π·6/24)=sin(π/2)=1
    assert np.isclose(row["hour_sin"], 1.0)
    assert np.isclose(row["hour_cos"], 0.0, atol=1e-9)


def test_weekend_flag_from_weekday():
    weekday_row = single_feature_row(
        hour=8, month=6, weekday=5, season=3, yr=1, workingday=0,
        is_holiday=0, weathersit=1, temperature_2m=24.0,
        relative_humidity_2m=55.0, wind_speed_10m=12.0,
    )
    assert weekday_row["is_weekend"] == 1


def test_build_features_no_nulls_and_ordered():
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame(
        {
            "datetime": pd.date_range("2011-01-01", periods=n, freq="1h"),
            "cnt": rng.integers(1, 500, n),
            "season": 1, "weathersit": 1, "workingday": 1, "yr": 0,
        }
    )
    out = build_features(df)
    engineered = [
        "hour_sin", "hour_cos", "month_sin", "month_cos", "weekday_sin", "weekday_cos",
        "lag_1h", "lag_24h", "roll_mean_3h", "roll_mean_24h",
    ]
    assert out[engineered].isna().sum().sum() == 0
    assert out["datetime"].is_monotonic_increasing
    # se descartan las primeras horas sin lag_24h disponible
    assert len(out) < n
