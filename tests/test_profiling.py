import os
import pytest
import pandas as pd
from nodes.profiling import profiling_node


@pytest.fixture
def sample_csv(tmp_path):
    df = pd.DataFrame({
        "age":    [25, 30, None, 40, 25],
        "gender": ["M", "F", "M", "F", "M"],
        "price":  [100.0, 200.0, 150.0, 300.0, 100.0],
    })
    path = str(tmp_path / "sample.csv")
    df.to_csv(path, index=False)
    return path


def _state(csv_path, target="price"):
    return {"file_path": csv_path, "target_column": target, "logs": []}


def test_schema_has_all_columns(sample_csv):
    result = profiling_node(_state(sample_csv))
    assert "age" in result["schema"]
    assert "gender" in result["schema"]
    assert "price" in result["schema"]


def test_column_types_detected(sample_csv):
    result = profiling_node(_state(sample_csv))
    assert result["schema"]["age"]["type"] == "numeric"
    assert result["schema"]["gender"]["type"] == "categorical"


def test_null_rate_detected(sample_csv):
    result = profiling_node(_state(sample_csv))
    assert result["schema"]["age"]["null_rate"] == pytest.approx(0.2, abs=0.01)
    assert result["schema"]["gender"]["null_rate"] == 0.0


def test_quality_issues_includes_nulls(sample_csv):
    result = profiling_node(_state(sample_csv))
    assert any("age" in issue for issue in result["quality_issues"])


def test_quality_issues_includes_duplicates(sample_csv):
    # age=25,gender=M,price=100 appears twice
    result = profiling_node(_state(sample_csv))
    assert any("重复" in issue for issue in result["quality_issues"])


def test_meta_row_col_count(sample_csv):
    result = profiling_node(_state(sample_csv))
    meta = result["schema"]["_meta"]
    assert meta["row_count"] == 5
    assert meta["col_count"] == 3


def test_appends_log(sample_csv):
    result = profiling_node(_state(sample_csv))
    assert any("[profiling]" in log for log in result["logs"])


def test_empty_target_column(sample_csv):
    """无监督任务可能不传 target_column。"""
    result = profiling_node({"file_path": sample_csv, "target_column": "", "logs": []})
    assert "schema" in result
