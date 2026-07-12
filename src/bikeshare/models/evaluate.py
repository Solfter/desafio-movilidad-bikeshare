"""Métricas de evaluación para regresión de demanda."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Error porcentual absoluto medio (%), ignorando valores reales igual a 0."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Devuelve MAE, RMSE, R² y MAPE en un diccionario."""
    y_pred = np.clip(y_pred, 0, None)  # la demanda no puede ser negativa
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "mape": mape(y_true, y_pred),
    }
