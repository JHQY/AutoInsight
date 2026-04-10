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
