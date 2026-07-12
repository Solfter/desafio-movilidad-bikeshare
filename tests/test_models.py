"""Tests de métricas y de un entrenamiento mínimo."""

import numpy as np
import pandas as pd

from bikeshare.features import FEATURE_COLUMNS as _FEATURES
from bikeshare.models.evaluate import mape, regression_metrics
from bikeshare.models.train import make_models


def test_regression_metrics_perfect_prediction():
    y = np.array([10.0, 20.0, 30.0])
    m = regression_metrics(y, y)
    assert m["mae"] == 0.0
    assert m["rmse"] == 0.0
    assert m["r2"] == 1.0
    assert m["mape"] == 0.0


def test_mape_ignores_zero_truth():
    y_true = np.array([0.0, 100.0])
    y_pred = np.array([50.0, 90.0])
    # solo cuenta el segundo punto: |100-90|/100 = 10%
    assert np.isclose(mape(y_true, y_pred), 10.0)


def test_linear_model_fits_and_predicts_shape():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(50, len(_FEATURES)))
    x_df = pd.DataFrame(x, columns=_FEATURES)
    y = x_df.sum(axis=1)
    model = make_models()["linear_regression"]
    model.fit(x_df, y)
    pred = model.predict(x_df)
    assert pred.shape == (50,)
