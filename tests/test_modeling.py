# tests/test_modeling.py
import pytest
import pandas as pd
from nodes.modeling import modeling_node


def _state(X_train, X_test, y_train, algorithms):
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,   # pd.DataFrame shape (n, 1) — same as data_processing output
        "selected_algorithms": algorithms,
        "logs": [],
    }


def test_modeling_runs_each_algorithm(clf_arrays):
    X_train, X_test, y_train, _ = clf_arrays
    result = modeling_node(_state(X_train, X_test, y_train,
                                  ["LogisticRegression", "RandomForestClassifier"]))
    assert "LogisticRegression" in result["model_results"]
    assert "RandomForestClassifier" in result["model_results"]


def test_modeling_predictions_are_list(clf_arrays):
    X_train, X_test, y_train, _ = clf_arrays
    result = modeling_node(_state(X_train, X_test, y_train, ["LogisticRegression"]))
    y_pred = result["model_results"]["LogisticRegression"]["y_pred"]
    assert isinstance(y_pred, list)
    assert len(y_pred) == 20


def test_modeling_stores_fitted_model(clf_arrays):
    X_train, X_test, y_train, _ = clf_arrays
    result = modeling_node(_state(X_train, X_test, y_train, ["LogisticRegression"]))
    model = result["model_results"]["LogisticRegression"]["model"]
    assert hasattr(model, "predict")


def test_modeling_works_with_dataframe_y(clf_arrays):
    """Explicitly test that DataFrame y_train does not crash any algorithm."""
    X_train, X_test, y_train, _ = clf_arrays
    assert isinstance(y_train, pd.DataFrame), "fixture must return DataFrame"
    result = modeling_node(_state(X_train, X_test, y_train,
                                  ["LogisticRegression", "XGBClassifier"]))
    assert "LogisticRegression" in result["model_results"]
    assert "XGBClassifier" in result["model_results"]
    # Neither should be an error entry
    assert "error" not in result["model_results"]["LogisticRegression"]
    assert "error" not in result["model_results"]["XGBClassifier"]


def test_modeling_handles_unknown_algorithm_gracefully(clf_arrays):
    X_train, X_test, y_train, _ = clf_arrays
    result = modeling_node(_state(X_train, X_test, y_train, ["NoSuchModel"]))
    assert "error" in result["model_results"]["NoSuchModel"]


def test_modeling_regression(reg_arrays):
    X_train, X_test, y_train, _ = reg_arrays
    result = modeling_node(_state(X_train, X_test, y_train,
                                  ["Ridge", "RandomForestRegressor"]))
    assert "Ridge" in result["model_results"]
    assert "error" not in result["model_results"]["Ridge"]
    assert len(result["model_results"]["Ridge"]["y_pred"]) == 20


def test_modeling_appends_log(clf_arrays):
    X_train, X_test, y_train, _ = clf_arrays
    result = modeling_node(_state(X_train, X_test, y_train, ["LogisticRegression"]))
    assert any("[modeling]" in log for log in result["logs"])
