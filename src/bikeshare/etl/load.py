"""Carga: persiste el dataset procesado en formato Parquet."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from bikeshare import config


def save_processed(df: pd.DataFrame, path: Path | None = None) -> Path:
    """Guarda el DataFrame final en ``data/processed/bikeshare_features.parquet``."""
    config.ensure_dirs()
    target = path or config.PROCESSED_FILE
    df.to_parquet(target, index=False)
    return target


def load_processed(path: Path | None = None) -> pd.DataFrame:
    """Lee el dataset procesado."""
    source = path or config.PROCESSED_FILE
    if not source.exists():
        raise FileNotFoundError(
            f"No existe {source}. Ejecuta primero: uv run python -m bikeshare.etl.pipeline"
        )
    return pd.read_parquet(source)
