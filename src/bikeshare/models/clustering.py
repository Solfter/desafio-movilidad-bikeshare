"""Segmentación no supervisada de tipos de día (K-Means sobre perfiles horarios).

Cada día se representa por su distribución horaria de demanda (24 valores que
suman 1) y se agrupa con K-Means:

- k=2 redescubre la división laboral / no laboral sin mirar el calendario.
- k=4 separa además los días laborales según el clima (cálido, frío, lluvioso).

El cálculo es liviano (~730 días), por lo que no se persisten artefactos: el
dashboard lo ejecuta una vez al arrancar.

Uso:
    from bikeshare.models.clustering import compute_clusters
    res = compute_clusters()
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from bikeshare import config
from bikeshare.etl.load import load_processed

# Nombres de los tipos de día (compartidos con el dashboard)
LABORAL = "Día laboral"
NO_LABORAL = "Fin de semana / feriado"
LABORAL_CALIDO = "Laboral cálido"
LABORAL_FRIO = "Laboral frío"
LABORAL_LLUVIOSO = "Laboral lluvioso"


@dataclass(frozen=True)
class DayClusters:
    """Resultado de la segmentación, listo para graficar."""

    profiles: pd.DataFrame  # día × 24 horas; cada fila suma 1
    meta: pd.DataFrame  # metadatos por día + columnas tipo2 / tipo4
    pca: pd.DataFrame  # PC1, PC2, fecha, tipo2, tipo4
    var_pc1: float  # % de varianza explicada por PC1
    var_pc2: float  # % de varianza explicada por PC2


def daily_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """Matriz día × 24 horas normalizada (distribución horaria de cada día).

    Solo se conservan los días con las 24 horas presentes; cada fila se
    divide por su total para comparar la *forma* del día, no su volumen.
    """
    tmp = df.assign(date=df["datetime"].dt.date)
    pivot = tmp.pivot_table(index="date", columns="hour", values="cnt", aggfunc="mean")
    complete = pivot.dropna()
    return complete.div(complete.sum(axis=1), axis=0)


def day_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Metadatos diarios usados para interpretar y nombrar los clusters."""
    tmp = df.assign(date=df["datetime"].dt.date)
    return tmp.groupby("date").agg(
        workingday=("workingday", "max"),
        is_holiday=("is_holiday", "max"),
        temp_media=("temperature_2m", "mean"),
        precip_total=("precipitation", "sum"),
        demanda_total=("cnt", "sum"),
    )


def fit_labels(profiles: pd.DataFrame, k: int) -> pd.Series:
    """Ajusta K-Means y devuelve la etiqueta de cluster de cada día."""
    km = KMeans(n_clusters=k, random_state=config.RANDOM_STATE, n_init=10)
    return pd.Series(km.fit_predict(profiles.values), index=profiles.index, name=f"cluster{k}")


def _name_k2(meta: pd.DataFrame) -> dict[int, str]:
    """k=2: el cluster con mayoría de días laborales es "Día laboral"."""
    pct_lab = meta.groupby("cluster2")["workingday"].mean()
    return {c: (LABORAL if p > 0.5 else NO_LABORAL) for c, p in pct_lab.items()}


def _name_k4(meta: pd.DataFrame) -> dict[int, str]:
    """k=4: entre los clusters laborales, el más lluvioso y luego cálido vs frío."""
    stats = meta.groupby("cluster4").agg(
        pct_lab=("workingday", "mean"),
        precip=("precip_total", "mean"),
        temp=("temp_media", "mean"),
    )
    nombres: dict[int, str] = {}
    laborales = stats[stats["pct_lab"] > 0.5]
    if not laborales.empty:
        lluvioso = laborales["precip"].idxmax()
        nombres[lluvioso] = LABORAL_LLUVIOSO
        resto = laborales.drop(lluvioso)
        if not resto.empty:
            nombres[resto["temp"].idxmax()] = LABORAL_CALIDO
            nombres[resto["temp"].idxmin()] = LABORAL_FRIO
    for c in stats.index:
        nombres.setdefault(c, NO_LABORAL)
    return nombres


def cluster_summary(meta: pd.DataFrame, tipo_col: str) -> pd.DataFrame:
    """Tabla resumen por tipo de día: tamaño, composición, clima y demanda."""
    return (
        meta.groupby(tipo_col)
        .agg(
            dias=(tipo_col, "size"),
            pct_dia_laboral=("workingday", "mean"),
            pct_feriado=("is_holiday", "mean"),
            temp_media=("temp_media", "mean"),
            precip_total_media=("precip_total", "mean"),
            demanda_diaria_media=("demanda_total", "mean"),
        )
        .round(2)
        .reset_index()
    )


def compute_clusters(df: pd.DataFrame | None = None) -> DayClusters:
    """Corre la segmentación completa (k=2 y k=4) más la proyección PCA 2D."""
    if df is None:
        df = load_processed()
    profiles = daily_profiles(df)
    meta = day_metadata(df).loc[profiles.index].copy()
    meta["cluster2"] = fit_labels(profiles, 2)
    meta["cluster4"] = fit_labels(profiles, 4)
    meta["tipo2"] = meta["cluster2"].map(_name_k2(meta))
    meta["tipo4"] = meta["cluster4"].map(_name_k4(meta))

    pca = PCA(n_components=2, random_state=config.RANDOM_STATE)
    coords = pca.fit_transform(profiles.values)
    var1, var2 = (pca.explained_variance_ratio_ * 100).tolist()
    pca_df = pd.DataFrame(
        {
            "PC1": coords[:, 0],
            "PC2": coords[:, 1],
            "fecha": [str(d) for d in profiles.index],
            "tipo2": meta["tipo2"].to_numpy(),
            "tipo4": meta["tipo4"].to_numpy(),
        }
    )
    return DayClusters(profiles=profiles, meta=meta, pca=pca_df, var_pc1=var1, var_pc2=var2)
