from app.state import AgentState
from tools.model_tools import get_model, tune_model

_UNSUPERVISED_TASKS = {"clustering", "anomaly_detection"}
_DBSCAN_LIKE = {"DBSCAN"}   # no predict() — use fit_predict on X_train


def modeling_node(state: AgentState) -> dict:
    X_train = state["X_train"]
    X_test  = state["X_test"]
    y_train = state.get("y_train")
    selected_algorithms = state["selected_algorithms"]
    task_type = state.get("task_type", "")
    do_tune   = state.get("tune", False)

    y_fit = None
    if y_train is not None:
        y_fit = y_train.values.ravel() if hasattr(y_train, "values") else y_train

    model_results: dict = {}

    for algo_name in selected_algorithms:
        try:
            if task_type in _UNSUPERVISED_TASKS:
                model_results[algo_name] = _fit_unsupervised(
                    algo_name, X_train, X_test, task_type
                )
            else:
                model_results[algo_name] = _fit_supervised(
                    algo_name, X_train, X_test, y_fit, task_type, do_tune
                )
        except Exception as exc:
            model_results[algo_name] = {"error": str(exc)}

    return {
        "model_results": model_results,
        "logs": list(state.get("logs", [])) + [
            f"[modeling] 训练完成: {list(model_results.keys())}"
        ],
    }


def _fit_supervised(algo_name, X_train, X_test, y_fit, task_type, do_tune):
    if do_tune:
        model = tune_model(algo_name, X_train, y_fit, task_type)
    else:
        model = get_model(algo_name)
        model.fit(X_train, y_fit)
    y_pred = model.predict(X_test)
    return {
        "y_pred": y_pred.tolist() if hasattr(y_pred, "tolist") else list(y_pred),
        "model":  model,
    }


def _fit_unsupervised(algo_name, X_train, X_test, task_type):
    model = get_model(algo_name)

    if algo_name in _DBSCAN_LIKE:
        # DBSCAN has no predict() — fit_predict on X_train directly
        labels = model.fit_predict(X_train).tolist()
        return {"labels": labels, "model": model}

    model.fit(X_train)
    preds = model.predict(X_test).tolist()  # clustering: cluster id; anomaly: 1/-1
    return {"labels": preds, "model": model}
