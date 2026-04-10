# tests/test_routing.py
from unittest.mock import MagicMock, patch
import pytest
from nodes.routing import routing_node, RoutingOutput


def _schema():
    return {
        "age":      {"type": "numeric",     "null_rate": 0.0},
        "income":   {"type": "numeric",     "null_rate": 0.0},
        "category": {"type": "categorical", "null_rate": 0.0},
        "_meta": {
            "file_name": "test.csv",
            "row_count": 100,
            "col_count": 4,
            "data_scope": "100 rows",
            "unit": "",
            "mean_target": 0.5,
            "core_features": "age,income,category",
        },
    }


def _mock_llm(task_type, algorithms, reasoning="test reason"):
    mock_output = RoutingOutput(
        task_type=task_type,
        selected_algorithms=algorithms,
        reasoning=reasoning,
    )
    mock_structured = MagicMock()
    mock_structured.invoke.return_value = mock_output
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured
    return mock_llm


@patch("nodes.routing.ChatAnthropic")
def test_routing_classification_task_type(mock_chat):
    mock_chat.return_value = _mock_llm(
        "classification", ["LogisticRegression", "RandomForestClassifier"]
    )
    result = routing_node({
        "target_column": "target",
        "user_query": "Predict churn",
        "schema": _schema(),
        "logs": [],
    })
    assert result["task_type"] == "classification"


@patch("nodes.routing.ChatAnthropic")
def test_routing_returns_algorithms_list(mock_chat):
    mock_chat.return_value = _mock_llm(
        "regression", ["Ridge", "XGBRegressor"]
    )
    result = routing_node({
        "target_column": "price",
        "user_query": "",
        "schema": _schema(),
        "logs": [],
    })
    assert isinstance(result["selected_algorithms"], list)
    assert len(result["selected_algorithms"]) >= 1
    assert "Ridge" in result["selected_algorithms"]


@patch("nodes.routing.ChatAnthropic")
def test_routing_returns_reasoning_string(mock_chat):
    mock_chat.return_value = _mock_llm(
        "classification", ["LogisticRegression"], "Continuous target with many unique values."
    )
    result = routing_node({
        "target_column": "target",
        "user_query": "",
        "schema": _schema(),
        "logs": [],
    })
    assert isinstance(result["reasoning"], str)
    assert len(result["reasoning"]) > 0


@patch("nodes.routing.ChatAnthropic")
def test_routing_appends_to_logs(mock_chat):
    mock_chat.return_value = _mock_llm("classification", ["LogisticRegression"])
    result = routing_node({
        "target_column": "target",
        "user_query": "",
        "schema": _schema(),
        "logs": ["[profiling] done"],
    })
    assert len(result["logs"]) == 2
    assert any("[routing]" in log for log in result["logs"])
