# tests/test_model_tools.py
import math
import pytest
from tools.model_tools import (
    ALGORITHM_REGISTRY,
    get_model,
    compute_classification_metrics,
    compute_regression_metrics,
)


def test_registry_contains_required_algorithms():
    required = {
        "LogisticRegression", "RandomForestClassifier",
        "LinearRegression", "Ridge", "RandomForestRegressor",
    }
    assert required.issubset(set(ALGORITHM_REGISTRY.keys()))


def test_registry_contains_xgb_lgbm():
    assert "XGBClassifier" in ALGORITHM_REGISTRY
    assert "XGBRegressor" in ALGORITHM_REGISTRY
    assert "LGBMClassifier" in ALGORITHM_REGISTRY
    assert "LGBMRegressor" in ALGORITHM_REGISTRY


def test_get_model_returns_fresh_instance():
    m1 = get_model("LogisticRegression")
    m2 = get_model("LogisticRegression")
    assert m1 is not m2


def test_get_model_raises_for_unknown():
    with pytest.raises(KeyError, match="not in registry"):
        get_model("NoSuchModel")


def test_classification_metrics_keys():
    y_true = [0, 1, 0, 1]
    y_pred = [0, 0, 0, 1]
    metrics = compute_classification_metrics(y_true, y_pred)
    assert set(metrics.keys()) == {"accuracy", "precision", "recall", "f1"}


def test_classification_metrics_perfect():
    y = [0, 1, 0, 1, 0]
    metrics = compute_classification_metrics(y, y)
    assert metrics["accuracy"] == 1.0
    assert metrics["f1"] == 1.0


def test_regression_metrics_keys():
    y_true = [1.0, 2.0, 3.0, 4.0]
    y_pred = [1.1, 1.9, 3.1, 3.9]
    metrics = compute_regression_metrics(y_true, y_pred)
    assert set(metrics.keys()) == {"mae", "rmse", "r2"}


def test_regression_metrics_perfect():
    y = [1.0, 2.0, 3.0]
    metrics = compute_regression_metrics(y, y)
    assert metrics["mae"] == 0.0
    assert metrics["r2"] == 1.0


def test_regression_metrics_known_values():
    y_true = [1.0, 2.0, 3.0, 4.0]
    y_pred = [1.5, 2.5, 3.5, 4.5]  # all off by +0.5
    metrics = compute_regression_metrics(y_true, y_pred)
    assert metrics["mae"] == 0.5
    assert abs(metrics["rmse"] - 0.5) < 1e-4   # sqrt(0.25) = 0.5
    assert metrics["r2"] < 1.0                  # not perfect
    assert metrics["r2"] > 0.0                  # still positive correlation


def test_regression_metrics_constant_target():
    y_true = [3.0, 3.0, 3.0, 3.0]
    y_pred = [3.1, 2.9, 3.0, 3.2]
    metrics = compute_regression_metrics(y_true, y_pred)
    assert math.isfinite(metrics["r2"])
    assert metrics["r2"] == -1.0   # clamped to sentinel


def test_classification_metrics_known_values():
    # 4 correct, 1 wrong out of 5
    y_true = [0, 1, 0, 1, 1]
    y_pred = [0, 1, 0, 0, 1]   # index 3 is wrong
    metrics = compute_classification_metrics(y_true, y_pred)
    assert metrics["accuracy"] == 0.8
    assert metrics["recall"] < 1.0       # not perfect recall
    assert metrics["precision"] > 0.0    # some precision
    assert metrics["f1"] > 0.0 and metrics["f1"] <= 1.0
