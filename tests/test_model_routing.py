from unittest.mock import MagicMock, patch
import pytest
from nodes.model_routing import model_routing_node, ModelRoutingOutput


def _state(task_category, task_type, hints, algorithms, reasoning="test"):
    return {
        "task_category":       task_category,
        "task_type":           task_type,
        "user_intent_summary": "test intent",
        "schema": {"_meta": {"row_count": 500, "col_count": 5,
                              "mean_target": 0.5, "core_features": "a,b,c"}},
        "modeling_hints": hints,
        "logs": [],
    }


def _mock_llm(task_type, algorithms, reasoning="selected based on hints"):
    out = ModelRoutingOutput(
        task_type=task_type,
        selected_algorithms=algorithms,
        reasoning=reasoning,
    )
    mock_s = MagicMock()
    mock_s.invoke.return_value = out
    mock_l = MagicMock()
    mock_l.with_structured_output.return_value = mock_s
    return mock_l


@patch("nodes.model_routing.ChatOpenAI")
def test_classification_routing(mock_chat):
    mock_chat.return_value = _mock_llm("classification", ["XGBClassifier", "RandomForestClassifier"])
    result = model_routing_node(_state(
        "supervised", "classification",
        {"linearity_score": 0.3, "imbalance_ratio": 0.45, "sample_size": 500, "feature_count": 4},
        [], ""
    ))
    assert result["task_type"] == "classification"
    assert "XGBClassifier" in result["selected_algorithms"]


@patch("nodes.model_routing.ChatOpenAI")
def test_clustering_routing(mock_chat):
    mock_chat.return_value = _mock_llm("clustering", ["KMeans", "AgglomerativeClustering"])
    result = model_routing_node(_state(
        "unsupervised", "clustering",
        {"sample_size": 300, "feature_count": 3},
        [], ""
    ))
    assert result["task_type"] == "clustering"
    assert "KMeans" in result["selected_algorithms"]


@patch("nodes.model_routing.ChatOpenAI")
def test_correlation_analysis_returns_empty_algorithms(mock_chat):
    mock_chat.return_value = _mock_llm("correlation_analysis", [])
    result = model_routing_node(_state(
        "analytical", "correlation_analysis",
        {"sample_size": 200, "feature_count": 5},
        [], ""
    ))
    assert result["task_type"] == "correlation_analysis"
    assert result["selected_algorithms"] == []


@patch("nodes.model_routing.ChatOpenAI")
def test_returns_reasoning(mock_chat):
    mock_chat.return_value = _mock_llm("regression", ["Ridge"], "High linearity detected")
    result = model_routing_node(_state(
        "supervised", "regression",
        {"linearity_score": 0.85, "sample_size": 1000, "feature_count": 8},
        [], ""
    ))
    assert isinstance(result["reasoning"], str)
    assert len(result["reasoning"]) > 0


@patch("nodes.model_routing.ChatOpenAI")
def test_appends_log(mock_chat):
    mock_chat.return_value = _mock_llm("classification", ["LogisticRegression"])
    result = model_routing_node(_state(
        "supervised", "classification",
        {"linearity_score": 0.6, "sample_size": 200, "feature_count": 3},
        [], ""
    ))
    assert any("[model_routing]" in log for log in result["logs"])
