from unittest.mock import MagicMock, patch
import pytest
from nodes.intent_routing import intent_routing_node, IntentOutput


def _schema():
    return {
        "age":    {"type": "numeric",     "null_rate": 0.0},
        "income": {"type": "numeric",     "null_rate": 0.0},
        "churn":  {"type": "categorical", "null_rate": 0.0},
        "_meta":  {"row_count": 500, "col_count": 3,
                   "data_scope": "500 rows x 3 columns",
                   "unit": "", "mean_target": 0.4,
                   "core_features": "age,income", "file_name": "test.csv"},
    }


def _mock_llm(target_column, task_category, task_type, summary="test summary"):
    out = IntentOutput(
        target_column=target_column,
        task_category=task_category,
        task_type=task_type,
        user_intent_summary=summary,
    )
    mock_structured = MagicMock()
    mock_structured.invoke.return_value = out
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured
    return mock_llm


@patch("nodes.intent_routing.ChatAnthropic")
def test_infers_target_when_not_provided(mock_chat):
    mock_chat.return_value = _mock_llm("churn", "supervised", "classification")
    result = intent_routing_node({
        "user_query": "预测客户是否流失",
        "target_column": "",
        "schema": _schema(),
        "logs": [],
    })
    assert result["target_column"] == "churn"
    assert result["task_category"] == "supervised"
    assert result["task_type"] == "classification"


@patch("nodes.intent_routing.ChatAnthropic")
def test_preserves_user_provided_target(mock_chat):
    mock_chat.return_value = _mock_llm("income", "supervised", "regression")
    result = intent_routing_node({
        "user_query": "预测收入",
        "target_column": "income",
        "schema": _schema(),
        "logs": [],
    })
    assert result["target_column"] == "income"


@patch("nodes.intent_routing.ChatAnthropic")
def test_unsupervised_does_not_set_target(mock_chat):
    mock_chat.return_value = _mock_llm("", "unsupervised", "clustering")
    result = intent_routing_node({
        "user_query": "把客户分成几类",
        "target_column": "",
        "schema": _schema(),
        "logs": [],
    })
    assert result["task_category"] == "unsupervised"
    # When LLM returns empty string, target_column should NOT be added to updates
    assert result.get("target_column", "") == ""


@patch("nodes.intent_routing.ChatAnthropic")
def test_returns_user_intent_summary(mock_chat):
    mock_chat.return_value = _mock_llm("churn", "supervised", "classification",
                                       "判断客户是否会流失以支持运营决策")
    result = intent_routing_node({
        "user_query": "预测流失",
        "target_column": "",
        "schema": _schema(),
        "logs": [],
    })
    assert isinstance(result["user_intent_summary"], str)
    assert len(result["user_intent_summary"]) > 0


@patch("nodes.intent_routing.ChatAnthropic")
def test_appends_log(mock_chat):
    mock_chat.return_value = _mock_llm("churn", "supervised", "classification")
    result = intent_routing_node({
        "user_query": "",
        "target_column": "",
        "schema": _schema(),
        "logs": ["[profiling] done"],
    })
    assert len(result["logs"]) == 2
    assert any("[intent_routing]" in log for log in result["logs"])


@patch("nodes.intent_routing.ChatAnthropic")
def test_analytical_task_type(mock_chat):
    mock_chat.return_value = _mock_llm("", "analytical", "correlation_analysis")
    result = intent_routing_node({
        "user_query": "分析哪些因素影响房价",
        "target_column": "",
        "schema": _schema(),
        "logs": [],
    })
    assert result["task_category"] == "analytical"
    assert result["task_type"] == "correlation_analysis"
