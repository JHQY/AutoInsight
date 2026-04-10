import pytest
import numpy as np
import pandas as pd


def _make_state(n=100, task_category="supervised"):
    np.random.seed(42)
    X = pd.DataFrame({
        "a": np.random.randn(n),
        "b": np.random.randn(n),
    })
    y_vals = X["a"] * 2 + np.random.randn(n) * 0.1
    y = pd.DataFrame({"target": y_vals})
    return {
        "X_train": X,
        "y_train": y,
        "target_column": "target",
        "task_category": task_category,
        "charts": [],
        "quality_issues": [],
        "logs": [],
    }


def test_modeling_hints_present():
    from nodes.eda import run_eda
    result = run_eda(_make_state())
    assert "modeling_hints" in result
    hints = result["modeling_hints"]
    assert "linearity_score" in hints
    assert "outlier_ratio" in hints
    assert "sample_size" in hints
    assert "feature_count" in hints
    assert "high_corr_pairs" in hints


def test_linearity_score_range():
    from nodes.eda import run_eda
    result = run_eda(_make_state())
    score = result["modeling_hints"]["linearity_score"]
    assert 0.0 <= score <= 1.0


def test_sample_size_correct():
    from nodes.eda import run_eda
    result = run_eda(_make_state(n=100))
    assert result["modeling_hints"]["sample_size"] == 100


def test_feature_count_correct():
    from nodes.eda import run_eda
    result = run_eda(_make_state(n=100))
    # X has 2 columns (a, b), target is separate
    assert result["modeling_hints"]["feature_count"] == 2


def test_imbalance_ratio_for_classification():
    from nodes.eda import run_eda
    np.random.seed(0)
    n = 100
    X = pd.DataFrame({"a": np.random.randn(n), "b": np.random.randn(n)})
    y_vals = [1] * 10 + [0] * 90
    y = pd.DataFrame({"label": y_vals})
    state = {
        "X_train": X, "y_train": y,
        "target_column": "label",
        "task_category": "supervised",
        "charts": [], "quality_issues": [], "logs": [],
    }
    result = run_eda(state)
    ratio = result["modeling_hints"].get("imbalance_ratio")
    assert ratio is not None
    assert abs(ratio - 0.1) < 0.02


def test_unsupervised_has_no_imbalance_ratio():
    from nodes.eda import run_eda
    state = _make_state(task_category="unsupervised")
    state["y_train"] = None
    result = run_eda(state)
    hints = result.get("modeling_hints", {})
    assert hints.get("imbalance_ratio") is None or "imbalance_ratio" not in hints


def test_high_linearity_detected():
    """When feature 'a' strongly correlates with target, linearity_score should be > 0.5."""
    from nodes.eda import run_eda
    np.random.seed(42)
    n = 100
    a = np.random.randn(n)
    X = pd.DataFrame({"a": a, "b": np.random.randn(n)})
    y = pd.DataFrame({"target": a * 3.0})  # perfect linear
    state = {
        "X_train": X, "y_train": y,
        "target_column": "target",
        "task_category": "supervised",
        "charts": [], "quality_issues": [], "logs": [],
    }
    result = run_eda(state)
    assert result["modeling_hints"]["linearity_score"] > 0.5
