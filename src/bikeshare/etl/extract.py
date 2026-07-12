"""Extracción de las 3 fuentes de datos.

1. UCI Bike Sharing (horario) vía `ucimlrepo`.
2. Clima histórico real vía Open-Meteo Archive API (sin clave).
3. Feriados federales de EE.UU. vía la librería `holidays`.

Cada función devuelve un ``pandas.DataFrame`` y (opcionalmente) cachea el crudo en ``data/raw``.
"""

from __future__ import annotations

import pandas as pd
import requests

from bikeshare import config


# ---------------------------------------------------------------------------
# Fuente 1 — UCI Bike Sharing (horario)
# ---------------------------------------------------------------------------
def extract_uci(use_cache: bool = True) -> pd.DataFrame:
    """Descarga el dataset horario de UCI (id=275) y lo devuelve como DataFrame.

    Combina *features* + *targets* para conservar todas las columnas originales.
    """
    config.ensure_dirs()
    cache = config.RAW_DIR / "uci_bike_hour.csv"
    if use_cache and cache.exists():
        return pd.read_csv(cache)

    from ucimlrepo import fetch_ucirepo

    dataset = fetch_ucirepo(id=config.UCI_DATASET_ID)
    features = dataset.data.features
    targets = dataset.data.targets
    df = pd.concat([features, targets], axis=1)

    # El dataset horario tiene la columna `hr`; si no está, estamos ante el diario.
    if "hr" not in df.columns:
        raise ValueError(
            "El dataset descargado no es el horario (falta la columna 'hr'). "
            f"Columnas: {list(df.columns)}"
        )

    df.to_csv(cache, index=False)
    return df


# ---------------------------------------------------------------------------
# Fuente 2 — Open-Meteo Archive (clima histórico real, sin clave)
# ---------------------------------------------------------------------------
def extract_weather(use_cache: bool = True, timeout: int = 60) -> pd.DataFrame:
    """Descarga clima horario histórico de Open-Meteo para la ciudad configurada.

    Devuelve un DataFrame con columna ``datetime`` (naive, hora local) y las variables
    meteorológicas. Es una llamada real a una API pública **sin autenticación**.
    """
    config.ensure_dirs()
    cache = config.RAW_DIR / "open_meteo_weather.csv"
    if use_cache and cache.exists():
        return pd.read_csv(cache, parse_dates=["datetime"])

    params = {
        "latitude": config.CITY_LAT,
        "longitude": config.CITY_LON,
        "start_date": config.DATE_START,
        "end_date": config.DATE_END,
        "hourly": ",".join(config.OPEN_METEO_HOURLY),
        "timezone": config.TIMEZONE,
    }
    resp = requests.get(config.OPEN_METEO_URL, params=params, timeout=timeout)
    resp.raise_for_status()
    hourly = resp.json()["hourly"]

    df = pd.DataFrame(hourly)
    df = df.rename(columns={"time": "datetime"})
    df["datetime"] = pd.to_datetime(df["datetime"])
    df.to_csv(cache, index=False)
    return df


# ---------------------------------------------------------------------------
# Fuente 3 — Feriados federales de EE.UU.
# ---------------------------------------------------------------------------
def extract_holidays() -> pd.DataFrame:
    """Devuelve un DataFrame con las fechas feriadas y su nombre para los años configurados."""
    import holidays as holidays_lib

    us_holidays = holidays_lib.country_holidays(
        config.HOLIDAYS_COUNTRY, years=config.HOLIDAYS_YEARS
    )
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(list(us_holidays.keys())),
            "holiday_name": list(us_holidays.values()),
        }
    ).sort_values("date", ignore_index=True)
    return df


if __name__ == "__main__":
    uci = extract_uci()
    weather = extract_weather()
    hol = extract_holidays()
    print("UCI    ", uci.shape, "->", list(uci.columns))
    print("Weather", weather.shape, "->", list(weather.columns))
    print("Holidays", hol.shape)
    print(hol.to_string(index=False))
