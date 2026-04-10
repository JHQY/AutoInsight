# nodes/modeling.py
from app.state import AgentState
from tools.model_tools import get_model


def modeling_node(state: AgentState) -> dict:
    X_train = state["X_train"]
    y_train = state["y_train"]
    X_test  = state["X_test"]
    selected_algorithms = state["selected_algorithms"]

    # y_train from data_processing is a DataFrame with shape (n, 1).
    # All sklearn/XGB/LGB models expect a 1-D array — ravel() converts safely.
    y_fit = y_train.values.ravel() if hasattr(y_train, "values") else y_train

    model_results = {}
    for algo_name in selected_algorithms:
        try:
            model = get_model(algo_name)
            model.fit(X_train, y_fit)
            y_pred = model.predict(X_test)
            model_results[algo_name] = {
                "y_pred": y_pred.tolist() if hasattr(y_pred, "tolist") else list(y_pred),
                "model": model,
            }
        except Exception as exc:
            model_results[algo_name] = {"error": str(exc)}

    return {
        "model_results": model_results,
        "logs": list(state.get("logs", [])) + [
            f"[modeling] 训练完成: {list(model_results.keys())}"
        ],
    }
