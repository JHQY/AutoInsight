# tests/test_evaluation.py
import pandas as pd
import pytest
from nodes.evaluation import evaluation_node


def _state(model_results, y_test, task_type):
    # y_test as DataFrame — same as data_processing output
    if not isinstance(y_test, pd.DataFrame):
        y_test = pd.DataFrame({"target": y_test})
    return {
        "model_results": model_results,
        "y_test": y_test,
        "task_type": task_type,
        "logs": [],
    }


def test_evaluation_classification_metrics_keys():
    y_test = pd.DataFrame({"t": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]})
    model_results = {
        "ModelA": {"y_pred": [0, 1, 0, 1, 0, 1, 0, 1, 0, 0], "model": None},
        "ModelB": {"y_pred": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1], "model": None},
    }
    result = evaluation_node(_state(model_results, y_test, "classification"))
    for model in result["metrics"]:
        assert set(result["metrics"][model].keys()) == {"accuracy", "precision", "recall", "f1"}


def test_evaluation_selects_best_by_f1():
    y_test = pd.DataFrame({"t": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]})
    model_results = {
        "BadModel":  {"y_pred": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], "model": None},
        "GoodModel": {"y_pred": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1], "model": None},
    }
    result = evaluation_node(_state(model_results, y_test, "classification"))
    assert result["best_model"] == "GoodModel"


def test_evaluation_regression_metrics_keys():
    y_test = pd.DataFrame({"p": [1.0, 2.0, 3.0, 4.0, 5.0]})
    model_results = {
        "ModelA": {"y_pred": [1.1, 1.9, 3.1, 3.9, 5.1], "model": None},
    }
    result = evaluation_node(_state(model_results, y_test, "regression"))
    assert set(result["metrics"]["ModelA"].keys()) == {"mae", "rmse", "r2"}


def test_evaluation_selects_best_by_r2():
    y_test = pd.DataFrame({"p": [1.0, 2.0, 3.0, 4.0, 5.0]})
    model_results = {
        "WeakModel":   {"y_pred": [3.0, 3.0, 3.0, 3.0, 3.0], "model": None},
        "StrongModel": {"y_pred": [1.1, 1.9, 3.1, 3.9, 5.1], "model": None},
    }
    result = evaluation_node(_state(model_results, y_test, "regression"))
    assert result["best_model"] == "StrongModel"


def test_evaluation_skips_error_models():
    y_test = pd.DataFrame({"t": [0, 1, 0, 1, 0]})
    model_results = {
        "BrokenModel":  {"error": "something went wrong"},
        "WorkingModel": {"y_pred": [0, 1, 0, 1, 0], "model": None},
    }
    result = evaluation_node(_state(model_results, y_test, "classification"))
    assert "BrokenModel" not in result["metrics"]
    assert result["best_model"] == "WorkingModel"


def test_evaluation_appends_log():
    y_test = pd.DataFrame({"t": [0, 1, 0, 1]})
    model_results = {"M": {"y_pred": [0, 1, 0, 1], "model": None}}
    result = evaluation_node(_state(model_results, y_test, "classification"))
    assert any("[evaluation]" in log for log in result["logs"])


def test_evaluation_raises_if_all_models_failed():
    y_test = pd.DataFrame({"t": [0, 1]})
    model_results = {"Broken": {"error": "failed"}}
    with pytest.raises(RuntimeError, match="All models failed"):
        evaluation_node(_state(model_results, y_test, "classification"))


def test_evaluation_clustering_metrics_keys():
    import numpy as np
    X_test = pd.DataFrame(np.random.randn(20, 2), columns=["a", "b"])
    model_results = {
        "KMeans": {"labels": [0]*10 + [1]*10, "model": None}
    }
    state = {
        "model_results": model_results,
        "X_test": X_test,
        "y_test": None,
        "task_type": "clustering",
        "logs": [],
    }
    result = evaluation_node(state)
    assert "KMeans" in result["metrics"]
    assert "silhouette" in result["metrics"]["KMeans"]
    assert "davies_bouldin" in result["metrics"]["KMeans"]


def test_evaluation_clustering_best_model_by_silhouette():
    import numpy as np
    X_test = pd.DataFrame(np.random.randn(40, 2), columns=["a", "b"])
    # KMeans gets good silhouette, DBSCAN gets sentinel -1
    model_results = {
        "KMeans": {"labels": [0]*20 + [1]*20, "model": None},
        "DBSCAN": {"labels": [-1]*40, "model": None},  # all noise → sentinel
    }
    state = {
        "model_results": model_results,
        "X_test": X_test,
        "y_test": None,
        "task_type": "clustering",
        "logs": [],
    }
    result = evaluation_node(state)
    assert result["best_model"] == "KMeans"


def test_evaluation_anomaly_metrics_keys():
    import numpy as np
    X_test = pd.DataFrame(np.random.randn(20, 2), columns=["a", "b"])
    preds = [1] * 18 + [-1, -1]
    model_results = {
        "IsolationForest": {"labels": preds, "model": None}
    }
    state = {
        "model_results": model_results,
        "X_test": X_test,
        "y_test": None,
        "task_type": "anomaly_detection",
        "logs": [],
    }
    result = evaluation_node(state)
    assert "IsolationForest" in result["metrics"]
    assert "anomaly_ratio" in result["metrics"]["IsolationForest"]
    assert abs(result["metrics"]["IsolationForest"]["anomaly_ratio"] - 0.1) < 0.01


def test_evaluation_correlation_analysis_returns_empty():
    state = {
        "model_results": {},
        "X_test": pd.DataFrame(),
        "y_test": None,
        "task_type": "correlation_analysis",
        "logs": [],
    }
    result = evaluation_node(state)
    assert result["metrics"] == {}
    assert result["best_model"] == ""


def test_evaluation_anomaly_appends_log():
    import numpy as np
    X_test = pd.DataFrame(np.random.randn(10, 2), columns=["a", "b"])
    model_results = {"IsolationForest": {"labels": [1]*9 + [-1], "model": None}}
    state = {
        "model_results": model_results,
        "X_test": X_test,
        "y_test": None,
        "task_type": "anomaly_detection",
        "logs": [],
    }
    result = evaluation_node(state)
    assert any("[evaluation]" in log for log in result["logs"])
