import numpy as np
from app.state import AgentState
from tools.model_tools import (
    compute_classification_metrics,
    compute_regression_metrics,
    compute_clustering_metrics,
    compute_anomaly_metrics,
)


def evaluation_node(state: AgentState) -> dict:
    model_results = state["model_results"]
    y_test        = state.get("y_test")
    X_test        = state.get("X_test")
    task_type     = state.get("task_type", "")

    # correlation_analysis: no model trained — return empty
    if task_type == "correlation_analysis":
        return {
            "metrics":    {},
            "best_model": "",
            "logs": list(state.get("logs", [])) + [
                "[evaluation] correlation_analysis — no model evaluation needed"
            ],
        }

    y_true = None
    if y_test is not None:
        y_true = y_test.values.ravel() if hasattr(y_test, "values") else y_test

    X_arr = X_test.values if hasattr(X_test, "values") else np.asarray(X_test)

    metrics: dict = {}
    for algo_name, result in model_results.items():
        if "error" in result:
            continue
        if task_type == "classification":
            metrics[algo_name] = compute_classification_metrics(y_true, result["y_pred"])
        elif task_type == "regression":
            metrics[algo_name] = compute_regression_metrics(y_true, result["y_pred"])
        elif task_type == "clustering":
            # DBSCAN labels come from X_train (stored as X_fit); others use X_test
            X_eval = np.asarray(result["X_fit"]) if "X_fit" in result else X_arr
            metrics[algo_name] = compute_clustering_metrics(X_eval, result["labels"])
        elif task_type == "anomaly_detection":
            contamination = 0.05
            model_obj = result.get("model")
            if model_obj is not None and hasattr(model_obj, "contamination"):
                contamination = float(model_obj.contamination) if model_obj.contamination != "auto" else 0.05
            metrics[algo_name] = compute_anomaly_metrics(result["labels"], contamination)

    if not metrics:
        raise RuntimeError("All models failed — cannot select best_model.")

    best_model = _select_best(metrics, task_type)

    return {
        "metrics":    metrics,
        "best_model": best_model,
        "logs": list(state.get("logs", [])) + [
            f"[evaluation] 最优模型: {best_model}"
        ],
    }


def _select_best(metrics: dict, task_type: str) -> str:
    if not metrics:
        return ""
    if task_type == "classification":
        return max(metrics, key=lambda k: metrics[k].get("f1", 0))
    if task_type == "regression":
        return max(metrics, key=lambda k: metrics[k].get("r2", -999))
    if task_type == "clustering":
        return max(metrics, key=lambda k: metrics[k].get("silhouette", -1))
    if task_type == "anomaly_detection":
        return next(iter(metrics))   # no "best" concept — return first
    return next(iter(metrics))
