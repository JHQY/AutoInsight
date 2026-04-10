# tools/model_tools.py
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score,
)
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
