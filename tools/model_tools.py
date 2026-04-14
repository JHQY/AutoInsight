# tools/model_tools.py
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score,
    silhouette_score, davies_bouldin_score,
)
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.svm import OneClassSVM, SVC
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor


ALGORITHM_REGISTRY: dict = {
    "LogisticRegression":     lambda: LogisticRegression(max_iter=1000, random_state=42),
    "RandomForestClassifier": lambda: RandomForestClassifier(n_estimators=100, random_state=42),
    "LinearRegression":       lambda: LinearRegression(),
    "Ridge":                  lambda: Ridge(),
    "RandomForestRegressor":  lambda: RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBClassifier":          lambda: XGBClassifier(
        n_estimators=100, random_state=42, eval_metric="logloss", verbosity=0
    ),
    "XGBRegressor":           lambda: XGBRegressor(
        n_estimators=100, random_state=42, verbosity=0
    ),
    "LGBMClassifier":         lambda: LGBMClassifier(
        n_estimators=100, random_state=42, verbosity=-1
    ),
    "LGBMRegressor":          lambda: LGBMRegressor(
        n_estimators=100, random_state=42, verbosity=-1
    ),
    # Supervised extras
    "SVC":                     lambda: SVC(probability=True, random_state=42),
    "Lasso":                   lambda: Lasso(),

    # Clustering
    "KMeans":                  lambda: KMeans(n_clusters=3, random_state=42, n_init="auto"),
    "DBSCAN":                  lambda: DBSCAN(eps=0.5, min_samples=5),
    "AgglomerativeClustering": lambda: AgglomerativeClustering(n_clusters=3),

    # Anomaly Detection
    "IsolationForest":         lambda: IsolationForest(contamination=0.05, random_state=42),
    "LocalOutlierFactor":      lambda: LocalOutlierFactor(contamination=0.05, novelty=True),
    "OneClassSVM":             lambda: OneClassSVM(nu=0.05),
}


def get_model(name: str):
    """Return a fresh (unfitted) model instance by registry name."""
    if name not in ALGORITHM_REGISTRY:
        raise KeyError(
            f"Algorithm '{name}' not in registry. "
            f"Available: {sorted(ALGORITHM_REGISTRY.keys())}"
        )
    return ALGORITHM_REGISTRY[name]()


def compute_classification_metrics(y_true, y_pred) -> dict:
    return {
        "accuracy":  round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "recall":    round(float(recall_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "f1":        round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
    }


def compute_regression_metrics(y_true, y_pred) -> dict:
    import math
    r2_raw = float(r2_score(y_true, y_pred))
    # sklearn >= 1.8 returns 0.0 (not nan/inf) for zero-variance target;
    # detect zero-variance explicitly and clamp to sentinel -1.0.
    y_arr = np.asarray(y_true, dtype=float)
    zero_variance = float(np.var(y_arr)) == 0.0
    r2_val = round(r2_raw if (math.isfinite(r2_raw) and not zero_variance) else -1.0, 4)
    return {
        "mae":  round(float(mean_absolute_error(y_true, y_pred)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "r2":   r2_val,
    }


def compute_clustering_metrics(X, labels) -> dict:
    """Silhouette + Davies-Bouldin. Returns sentinels if only one cluster."""
    labels_arr = np.asarray(labels)
    valid_mask = labels_arr != -1   # DBSCAN marks noise as -1
    unique = np.unique(labels_arr[valid_mask])
    if len(unique) < 2:
        return {"silhouette": -1.0, "davies_bouldin": 999.0}
    X_arr = X.values if hasattr(X, "values") else np.asarray(X)
    X_valid = X_arr[valid_mask]
    l_valid = labels_arr[valid_mask]
    return {
        "silhouette":     round(float(silhouette_score(X_valid, l_valid)), 4),
        "davies_bouldin": round(float(davies_bouldin_score(X_valid, l_valid)), 4),
    }


def compute_anomaly_metrics(preds, contamination: float) -> dict:
    """preds: array of 1 (normal) / -1 (anomaly)."""
    preds_arr = np.asarray(preds)
    anomaly_count = int((preds_arr == -1).sum())
    total = len(preds_arr)
    return {
        "anomaly_count": anomaly_count,
        "anomaly_ratio": round(anomaly_count / max(total, 1), 4),
        "contamination": round(float(contamination), 4),
    }


_TUNE_PARAM_GRIDS: dict = {
    "LogisticRegression":     {"C": [0.01, 0.1, 1, 10, 100]},
    "RandomForestClassifier": {"n_estimators": [50, 100, 200], "max_depth": [None, 5, 10]},
    "XGBClassifier":          {"n_estimators": [50, 100], "max_depth": [3, 5, 7], "learning_rate": [0.05, 0.1, 0.2]},
    "LGBMClassifier":         {"n_estimators": [50, 100], "num_leaves": [31, 63], "learning_rate": [0.05, 0.1]},
    "SVC":                    {"C": [0.1, 1, 10], "gamma": ["scale", "auto"]},
    "LinearRegression":       {},
    "Ridge":                  {"alpha": [0.1, 1.0, 10.0, 100.0]},
    "Lasso":                  {"alpha": [0.01, 0.1, 1.0, 10.0]},
    "RandomForestRegressor":  {"n_estimators": [50, 100, 200], "max_depth": [None, 5, 10]},
    "XGBRegressor":           {"n_estimators": [50, 100], "max_depth": [3, 5, 7], "learning_rate": [0.05, 0.1, 0.2]},
    "LGBMRegressor":          {"n_estimators": [50, 100], "num_leaves": [31, 63], "learning_rate": [0.05, 0.1]},
}


def tune_model(name: str, X_train, y_train, task_type: str):
    """RandomizedSearchCV (20 iter) on supervised models only.
    Raises ValueError for unsupervised/analytical task types."""
    if task_type in ("clustering", "anomaly_detection", "correlation_analysis"):
        raise ValueError(
            f"tune_model does not support unsupervised/analytical tasks (task_type={task_type})"
        )
    if name not in _TUNE_PARAM_GRIDS:
        return get_model(name)
    param_grid = _TUNE_PARAM_GRIDS[name]
    if not param_grid:
        model = get_model(name)
        model.fit(X_train, y_train)
        return model
    scoring = "f1_weighted" if task_type == "classification" else "r2"
    search = RandomizedSearchCV(
        get_model(name),
        param_distributions=param_grid,
        n_iter=20, cv=3, scoring=scoring,
        random_state=42, n_jobs=-1, error_score="raise",
    )
    search.fit(X_train, y_train)
    return search.best_estimator_
