"""Tests de la segmentación no supervisada de tipos de día."""

import numpy as np
import pandas as pd

from bikeshare.models.clustering import (
    LABORAL,
    NO_LABORAL,
    cluster_summary,
    compute_clusters,
    daily_profiles,
)


def _synthetic_hourly(n_weeks: int = 8) -> pd.DataFrame:
    """Semanas sintéticas: laborales bimodales (peaks 8 y 18), findes unimodales."""
    rng = np.random.default_rng(0)
    rows = []
    start = pd.Timestamp("2011-01-03")  # lunes
    for day in range(n_weeks * 7):
        date = start + pd.Timedelta(days=day)
        working = int(date.dayofweek < 5)
        for hour in range(24):
            if working:
                base = 40 + 200 * (hour in (8, 9, 17, 18, 19))
            else:
                base = 40 + 150 * (10 <= hour <= 16)
            rows.append(
                {
                    "datetime": date + pd.Timedelta(hours=hour),
                    "hour": hour,
                    "cnt": base + rng.normal(scale=3.0),
                    "workingday": working,
                    "is_holiday": 0,
                    "temperature_2m": 15.0 + rng.normal(scale=2.0),
                    "precipitation": 0.0,
                }
            )
    return pd.DataFrame(rows)


def test_daily_profiles_shape_and_normalization():
    profiles = daily_profiles(_synthetic_hourly())
    assert profiles.shape == (8 * 7, 24)
    assert np.allclose(profiles.sum(axis=1), 1.0)


def test_daily_profiles_drops_incomplete_days():
    df = _synthetic_hourly().iloc[:-1]  # el último día queda con 23 horas
    profiles = daily_profiles(df)
    assert len(profiles) == 8 * 7 - 1


def test_k2_recovers_working_vs_weekend_without_calendar():
    res = compute_clusters(_synthetic_hourly())
    pct_lab = res.meta.groupby("tipo2")["workingday"].mean()
    assert pct_lab[LABORAL] > 0.9
    assert pct_lab[NO_LABORAL] < 0.1


def test_k4_names_every_day_and_summary_is_consistent():
    res = compute_clusters(_synthetic_hourly())
    assert res.meta["tipo4"].notna().all()
    resumen = cluster_summary(res.meta, "tipo4")
    assert len(resumen) == res.meta["tipo4"].nunique()
    assert resumen["dias"].sum() == len(res.meta)


def test_pca_projection_matches_days():
    res = compute_clusters(_synthetic_hourly())
    assert len(res.pca) == len(res.profiles)
    assert 0 < res.var_pc1 <= 100
    assert 0 <= res.var_pc2 <= res.var_pc1
