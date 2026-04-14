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


def test_registry_contains_clustering_algorithms():
    required = {"KMeans", "DBSCAN", "AgglomerativeClustering"}
    assert required.issubset(set(ALGORITHM_REGISTRY.keys()))


def test_registry_contains_anomaly_algorithms():
    required = {"IsolationForest", "LocalOutlierFactor", "OneClassSVM"}
    assert required.issubset(set(ALGORITHM_REGISTRY.keys()))


def test_registry_contains_svc_and_lasso():
    assert "SVC" in ALGORITHM_REGISTRY
    assert "Lasso" in ALGORITHM_REGISTRY


def test_clustering_metrics_keys():
    from tools.model_tools import compute_clustering_metrics
    import numpy as np
    X = np.random.randn(50, 2)
    labels = [0] * 25 + [1] * 25
    metrics = compute_clustering_metrics(X, labels)
    assert "silhouette" in metrics
    assert "davies_bouldin" in metrics


def test_clustering_metrics_values_are_finite():
    from tools.model_tools import compute_clustering_metrics
    import numpy as np
    import math
    X = np.random.randn(50, 2)
    labels = [0] * 25 + [1] * 25
    metrics = compute_clustering_metrics(X, labels)
    assert math.isfinite(metrics["silhouette"])
    assert math.isfinite(metrics["davies_bouldin"])


def test_clustering_metrics_single_cluster_returns_sentinel():
    from tools.model_tools import compute_clustering_metrics
    import numpy as np
    X = np.random.randn(20, 2)
    labels = [0] * 20  # all same cluster
    metrics = compute_clustering_metrics(X, labels)
    assert metrics["silhouette"] == -1.0
    assert metrics["davies_bouldin"] == 999.0


def test_anomaly_metrics_keys():
    from tools.model_tools import compute_anomaly_metrics
    preds = [-1, 1, -1, 1, 1] * 10
    metrics = compute_anomaly_metrics(preds, contamination=0.1)
    assert "anomaly_ratio" in metrics
    assert "anomaly_count" in metrics
    assert "contamination" in metrics


def test_anomaly_metrics_ratio_correct():
    from tools.model_tools import compute_anomaly_metrics
    preds = [1] * 18 + [-1, -1]
    metrics = compute_anomaly_metrics(preds, contamination=0.05)
    assert metrics["anomaly_count"] == 2
    assert abs(metrics["anomaly_ratio"] - 0.1) < 0.01


def test_tune_model_returns_fitted_model(clf_arrays):
    from tools.model_tools import tune_model
    X_train, X_test, y_train, _ = clf_arrays
    y_fit = y_train.values.ravel()
    model = tune_model("RandomForestClassifier", X_train, y_fit, "classification")
    assert hasattr(model, "predict")
    preds = model.predict(X_test)
    assert len(preds) == len(X_test)


def test_tune_model_raises_for_unsupervised():
    from tools.model_tools import tune_model
    import pandas as pd, numpy as np
    X = pd.DataFrame(np.random.randn(50, 2))
    with pytest.raises(ValueError, match="tune_model"):
        tune_model("KMeans", X, None, "clustering")
