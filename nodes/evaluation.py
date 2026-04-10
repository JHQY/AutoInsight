# nodes/evaluation.py
from app.state import AgentState
from tools.model_tools import compute_classification_metrics, compute_regression_metrics


def evaluation_node(state: AgentState) -> dict:
    model_results = state["model_results"]
    y_test = state["y_test"]
    task_type = state["task_type"]

    # y_test from data_processing is a DataFrame with shape (n, 1).
    y_true = y_test.values.ravel() if hasattr(y_test, "values") else y_test

    metrics = {}
    for algo_name, result in model_results.items():
        if "error" in result:
            continue
        y_pred = result["y_pred"]
        if task_type == "classification":
            metrics[algo_name] = compute_classification_metrics(y_true, y_pred)
        else:
            metrics[algo_name] = compute_regression_metrics(y_true, y_pred)

    if not metrics:
        raise RuntimeError("All models failed — cannot select best_model.")

    if task_type == "classification":
        best_model = max(metrics, key=lambda k: metrics[k]["f1"])
    else:
        best_model = max(metrics, key=lambda k: metrics[k]["r2"])

    return {
        "metrics": metrics,
        "best_model": best_model,
        "logs": list(state.get("logs", [])) + [
            f"[evaluation] 最优模型: {best_model}, 指标: {metrics[best_model]}"
        ],
    }
