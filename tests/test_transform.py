"""Tests del ETL de transformación (sin red, con DataFrames sintéticos)."""

import pandas as pd

from bikeshare.etl import transform as T


def _uci_sample() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dteday": ["2011-01-01", "2011-01-01"],
            "hr": [0, 1],
            "temp": [0.5, 0.6],
            "atemp": [0.5, 0.5],
            "hum": [0.8, 0.7],
            "windspeed": [0.2, 0.3],
            "cnt": [16, 40],
        }
    )


def test_build_datetime_combines_date_and_hour():
    out = T.build_datetime(_uci_sample())
    assert out["datetime"].iloc[0] == pd.Timestamp("2011-01-01 00:00:00")
    assert out["datetime"].iloc[1] == pd.Timestamp("2011-01-01 01:00:00")


def test_merge_weather_joins_on_datetime():
    base = T.build_datetime(_uci_sample())
    weather = pd.DataFrame(
        {
            "datetime": ["2011-01-01 00:00:00", "2011-01-01 01:00:00"],
            "temperature_2m": [5.0, 6.0],
            "precipitation": [0.0, 0.1],
        }
    )
    out = T.merge_weather(base, weather)
    assert out["temperature_2m"].tolist() == [5.0, 6.0]
    assert out["precipitation"].tolist() == [0.0, 0.1]


def test_merge_holidays_sets_flag():
    base = T.build_datetime(_uci_sample())
    hol = pd.DataFrame({"date": ["2011-01-01"], "holiday_name": ["New Year's Day"]})
    out = T.merge_holidays(base, hol)
    assert out["is_holiday"].tolist() == [1, 1]


def test_denormalize_uci_units():
    out = T.denormalize_uci(_uci_sample())
    assert out["temp_c"].iloc[0] == 0.5 * 41
    assert out["humidity_pct"].iloc[0] == 80.0
