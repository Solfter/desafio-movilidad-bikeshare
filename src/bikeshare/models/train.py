"""Entrenamiento y comparación de modelos de demanda.

- Split **temporal** (sin barajar): el test es el último tramo cronológico.
- Compara: baseline estacional (lag 24h) → Regresión Lineal → Random Forest → XGBoost.
- Validación cruzada temporal (`TimeSeriesSplit`) sobre el set de entrenamiento.
- Guarda el mejor modelo (`model.pkl`), las métricas (`metrics.json`) y las importancias.

Uso:
    uv run python -m bikeshare.models.train
"""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from bikeshare import config
from bikeshare.etl.load import load_processed
from bikeshare.features import FEATURE_COLUMNS
from bikeshare.models.evaluate import regression_metrics

TEST_FRACTION = 0.2


def make_models() -> dict[str, Pipeline]:
    """Define los modelos a comparar (los lineales van con escalado)."""
    return {
        "linear_regression": Pipeline(
            [("scaler", StandardScaler()), ("model", LinearRegression())]
        ),
        "random_forest": Pipeline(
            [
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=300,
                        max_depth=None,
                        n_jobs=-1,
                        random_state=config.RANDOM_STATE,
                    ),
                )
            ]
        ),
        "xgboost": Pipeline(
            [
                (
                    "model",
                    XGBRegressor(
                        n_estimators=500,
                        max_depth=6,
                        learning_rate=0.05,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        random_state=config.RANDOM_STATE,
                        n_jobs=-1,
                    ),
                )
            ]
        ),
    }


def temporal_split(df: pd.DataFrame):
    """Divide cronológicamente en train/test."""
    df = df.sort_values("datetime").reset_index(drop=True)
    split = int(len(df) * (1 - TEST_FRACTION))
    train, test = df.iloc[:split], df.iloc[split:]
    x_train, y_train = train[FEATURE_COLUMNS], train[config.TARGET]
    x_test, y_test = test[FEATURE_COLUMNS], test[config.TARGET]
    return x_train, x_test, y_train, y_test, train, test


def feature_importances(pipeline: Pipeline) -> pd.DataFrame | None:
    """Extrae importancias si el modelo final las expone."""
    model = pipeline.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    elif hasattr(model, "coef_"):
        imp = np.abs(model.coef_)
    else:
        return None
    return (
        pd.DataFrame({"feature": FEATURE_COLUMNS, "importance": imp})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def run() -> dict:
    config.ensure_dirs()
    df = load_processed()
    x_train, x_test, y_train, y_test, _train, test = temporal_split(df)
    print(f"Train: {len(x_train)}  |  Test: {len(x_test)}  (split temporal)")

    tscv = TimeSeriesSplit(n_splits=5)
    results: dict[str, dict] = {}

    # Baseline estacional: predecir la demanda de hace 24 h (lag_24h ya es una feature).
    baseline_pred = test["lag_24h"].to_numpy()
    results["baseline_lag24h"] = regression_metrics(y_test.to_numpy(), baseline_pred)

    fitted: dict[str, Pipeline] = {}
    for name, pipe in make_models().items():
        print(f"Entrenando {name} ...")
        pipe.fit(x_train, y_train)
        pred = pipe.predict(x_test)
        metrics = regression_metrics(y_test.to_numpy(), pred)
        # CV temporal sobre train (RMSE medio)
        cv_rmse = -cross_val_score(
            pipe, x_train, y_train, cv=tscv, scoring="neg_root_mean_squared_error"
        ).mean()
        metrics["cv_rmse"] = float(cv_rmse)
        results[name] = metrics
        fitted[name] = pipe

    # Elegir el mejor por RMSE de test (entre los modelos entrenados, no el baseline).
    best_name = min(fitted, key=lambda n: results[n]["rmse"])
    best_pipe = fitted[best_name]
    print(f"\nMejor modelo: {best_name}  (RMSE test = {results[best_name]['rmse']:.2f})")

    # Persistir artefactos
    joblib.dump(
        {"pipeline": best_pipe, "features": FEATURE_COLUMNS, "target": config.TARGET},
        config.MODEL_FILE,
    )
    payload = {
        "best_model": best_name,
        "test_fraction": TEST_FRACTION,
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
        "metrics": results,
    }
    config.METRICS_FILE.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    imp = feature_importances(best_pipe)
    if imp is not None:
        imp.to_csv(config.IMPORTANCES_FILE, index=False)

    print(f"Guardado: {config.MODEL_FILE.name}, {config.METRICS_FILE.name}")
    print(json.dumps(results, indent=2, ensure_ascii=False))
    return payload


if __name__ == "__main__":
    run()
