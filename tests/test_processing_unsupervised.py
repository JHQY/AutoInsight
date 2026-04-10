import pytest
import pandas as pd
import numpy as np
from nodes.processing import data_processing


def _schema(cols):
    result = {}
    for col in cols:
        result[col] = {"type": "numeric", "null_rate": 0.0}
    result["gender"] = {"type": "categorical", "null_rate": 0.0}
    result["_meta"] = {
        "row_count": 100, "col_count": len(cols) + 1,
        "data_scope": "", "unit": "", "mean_target": 0.0,
        "core_features": ",".join(cols), "file_name": "t.csv",
    }
    return result


@pytest.fixture
def csv_path(tmp_path):
    np.random.seed(0)
    df = pd.DataFrame({
        "age":    np.random.randint(20, 60, 100).astype(float),
        "income": np.random.randint(3000, 20000, 100).astype(float),
        "gender": np.random.choice(["M", "F"], 100),
    })
    p = str(tmp_path / "data.csv")
    df.to_csv(p, index=False)
    return p


def test_unsupervised_y_is_none(csv_path):
    state = {
        "file_path": csv_path,
        "target_column": "",
        "task_category": "unsupervised",
        "schema": _schema(["age", "income"]),
        "quality_issues": [],
        "logs": [],
    }
    result = data_processing(state)
    assert result["y_train"] is None
    assert result["y_test"] is None


def test_unsupervised_x_is_set(csv_path):
    state = {
        "file_path": csv_path,
        "target_column": "",
        "task_category": "unsupervised",
        "schema": _schema(["age", "income"]),
        "quality_issues": [],
        "logs": [],
    }
    result = data_processing(state)
    assert isinstance(result["X_train"], pd.DataFrame)
    assert len(result["X_train"]) > 0


def test_analytical_y_is_none(csv_path):
    state = {
        "file_path": csv_path,
        "target_column": "",
        "task_category": "analytical",
        "schema": _schema(["age", "income"]),
        "quality_issues": [],
        "logs": [],
    }
    result = data_processing(state)
    assert result["y_train"] is None


def test_unsupervised_log_appended(csv_path):
    state = {
        "file_path": csv_path,
        "target_column": "",
        "task_category": "unsupervised",
        "schema": _schema(["age", "income"]),
        "quality_issues": [],
        "logs": [],
    }
    result = data_processing(state)
    assert any("[processing]" in log for log in result["logs"])
