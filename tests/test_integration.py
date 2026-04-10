"""
End-to-end smoke test: synthetic CSV, mocked LLM nodes, full graph run.
Tests that graph executes start-to-finish and key state fields are written.
"""
import os
import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch


@pytest.fixture
def iris_csv(tmp_path):
    """Simple 3-class synthetic dataset."""
    np.random.seed(42)
    n = 150
    df = pd.DataFrame({
        "sepal_length": np.random.randn(n) + 5.0,
        "sepal_width":  np.random.randn(n) + 3.0,
        "petal_length": np.random.randn(n) + 4.0,
        "species":      np.random.choice(["setosa", "versicolor", "virginica"], n),
    })
    p = str(tmp_path / "iris.csv")
    df.to_csv(p, index=False)
    return p


def _mock_intent(target, category, task_type, summary):
    from nodes.intent_routing import IntentOutput
    out = IntentOutput(target_column=target, task_category=category,
                       task_type=task_type, user_intent_summary=summary)
    ms = MagicMock(); ms.invoke.return_value = out
    ml = MagicMock(); ml.with_structured_output.return_value = ms
    return ml


def _mock_model_routing(task_type, algorithms):
    from nodes.model_routing import ModelRoutingOutput
    out = ModelRoutingOutput(task_type=task_type,
                             selected_algorithms=algorithms,
                             reasoning="mock reasoning")
    ms = MagicMock(); ms.invoke.return_value = out
    ml = MagicMock(); ml.with_structured_output.return_value = ms
    return ml


@patch("nodes.reporting.Anthropic")
@patch("nodes.model_routing.ChatOpenAI")
@patch("nodes.intent_routing.ChatAnthropic")
def test_classification_pipeline_end_to_end(
    mock_intent_chat, mock_model_chat, mock_anthropic, iris_csv
):
    mock_intent_chat.return_value = _mock_intent(
        "species", "supervised", "classification", "预测鸢尾花种类"
    )
    mock_model_chat.return_value = _mock_model_routing(
        "classification", ["LogisticRegression", "RandomForestClassifier"]
    )
    mock_resp = MagicMock()
    mock_resp.content = [MagicMock(text="# Mock Report\nTest report content.")]
    mock_anthropic.return_value.messages.create.return_value = mock_resp

    from app.graph import build_graph

    initial_state = {
        "user_query":   "预测鸢尾花种类",
        "file_path":    iris_csv,
        "target_column": "",
        "user_level":   "general",
        "tune":         False,
        "logs":         [],
        "schema": {}, "quality_issues": [], "task_category": "",
        "task_type": "", "user_intent_summary": "",
        "selected_algorithms": [], "reasoning": "",
        "model_results": {}, "metrics": {}, "best_model": "",
        "charts": [], "eda_summary": {}, "modeling_hints": {},
        "feature_names": [], "X_train": None, "X_test": None,
        "y_train": None, "y_test": None, "report_path": "",
    }

    graph = build_graph()
    final = graph.invoke(initial_state)

    assert final["task_type"] == "classification"
    assert final["best_model"] in ("LogisticRegression", "RandomForestClassifier")
    assert len(final["metrics"]) >= 1
    assert final["report_path"] != ""
    assert os.path.exists(final["report_path"])
    assert len(final["logs"]) >= 6
