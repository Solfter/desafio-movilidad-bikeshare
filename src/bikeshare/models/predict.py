"""Carga del modelo entrenado y predicción (usado por la API y el dashboard)."""

from __future__ import annotations

import functools

import joblib
import numpy as np
import pandas as pd

from bikeshare import config


@functools.lru_cache(maxsize=1)
def load_model() -> dict:
    """Carga el artefacto {pipeline, features, target} desde ``models/model.pkl``."""
    if not config.MODEL_FILE.exists():
        raise FileNotFoundError(
            f"No existe {config.MODEL_FILE}. Ejecuta: uv run python -m bikeshare.models.train"
        )
    return joblib.load(config.MODEL_FILE)


def predict(features: pd.DataFrame | dict) -> np.ndarray:
    """Predice la demanda. Acepta un DataFrame o un dict con las columnas del modelo."""
    artifact = load_model()
    cols = artifact["features"]
    if isinstance(features, dict):
        features = pd.DataFrame([features])
    x = features.reindex(columns=cols)
    pred = artifact["pipeline"].predict(x)
    return np.clip(pred, 0, None)
