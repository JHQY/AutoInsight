"""
nodes/insight.py — Insight Extraction Node

Purpose: extract task-specific insights from the trained model and data.
The model is a tool for finding signal; this node translates model internals
into human-readable findings for the reporting LLM.

Per-task behaviour:
  regression        — feature importance + direction + test-set prediction range
  classification    — feature importance + per-class profiles + test distribution
  clustering        — full-dataset cluster assignment + cluster profiles (original scale)
  anomaly_detection — full-dataset anomaly flagging + anomaly vs normal profiles
  correlation_analysis — skipped (no model)
"""

import os
import numpy as np
import pandas as pd
from collections import Counter
from datetime import datetime

from app.state import AgentState

_SUPERVISED    = {"regression", "classification"}
_UNSUPERVISED  = {"clustering", "anomaly_detection"}
# Models that have no predict() on new data
_NO_PREDICT    = {"DBSCAN", "LocalOutlierFactor"}


# ── Entry point ───────────────────────────────────────────────────────────────

def insight_node(state: AgentState) -> dict:
    task_type = state.get("task_type", "")
    logs      = list(state.get("logs", []))

    if task_type == "correlation_analysis":
        logs.append("[insight] correlation_analysis — 跳过洞察提取")
        return {"logs": logs, "insight_data": {}, "prediction_path": ""}

    if task_type in _SUPERVISED:
        insight_data, path = _supervised_insight(state, logs)
    elif task_type in _UNSUPERVISED:
        insight_data, path = _unsupervised_insight(state, logs)
    else:
        logs.append(f"[insight] 未知任务类型: {task_type}，跳过")
        return {"logs": logs, "insight_data": {}, "prediction_path": ""}

    return {"insight_data": insight_data, "prediction_path": path, "logs": logs}


# ── Supervised: regression & classification ───────────────────────────────────

def _supervised_insight(state: AgentState, logs: list) -> tuple[dict, str]:
    best_model   = state.get("best_model", "")
    model_results = state.get("model_results") or {}
    feature_names = state.get("feature_names") or []
    task_type    = state.get("task_type", "")

    best  = model_results.get(best_model, {})
    model = best.get("model")
    y_pred = best.get("y_pred")        # test-set predictions (already computed)

    insight: dict = {"task_type": task_type}

    # ── Feature importance ────────────────────────────────────────────────
    importance_list = _extract_feature_importance(model, feature_names)
    if importance_list:
        # Attach direction using correlation between feature and y_train
        directions = _compute_feature_directions(state, feature_names)
        for item in importance_list:
            item["direction"] = directions.get(item["feature"], "unknown")
        insight["feature_importance"] = importance_list
        logs.append(f"[insight] 特征重要性提取完成（top {len(importance_list)} 个特征）")
    else:
        logs.append("[insight] 该模型不支持特征重要性提取")

    # ── Test-set prediction stats (regression) ────────────────────────────
    if task_type == "regression" and y_pred is not None:
        arr = np.array(y_pred, dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr):
            insight["test_prediction_stats"] = {
                "count":  int(len(arr)),
                "mean":   round(float(np.mean(arr)),   2),
                "median": round(float(np.median(arr)), 2),
                "std":    round(float(np.std(arr)),    2),
                "min":    round(float(np.min(arr)),    2),
                "max":    round(float(np.max(arr)),    2),
                "p25":    round(float(np.percentile(arr, 25)), 2),
                "p75":    round(float(np.percentile(arr, 75)), 2),
                "p90":    round(float(np.percentile(arr, 90)), 2),
            }
            logs.append("[insight] 测试集预测分布统计完成")

    # ── Per-class feature profiles (classification) ───────────────────────
    if task_type == "classification":
        class_profiles = _compute_class_profiles(state, y_pred, feature_names)
        if class_profiles:
            insight["class_profiles"] = class_profiles
            logs.append(f"[insight] 分类画像提取完成（{len(class_profiles)} 个类别）")

        # Low-confidence ratio
        if model is not None and hasattr(model, "predict_proba"):
            X_test = state.get("X_test")
            if X_test is not None:
                try:
                    proba = model.predict_proba(
                        X_test.values if hasattr(X_test, "values") else X_test
                    )
                    max_conf = proba.max(axis=1)
                    low_conf = float((max_conf < 0.6).mean())
                    insight["low_confidence_ratio"] = round(low_conf, 4)
                    logs.append(f"[insight] 低置信度样本比例: {low_conf:.1%}")
                except Exception:
                    pass

    return insight, ""          # supervised tasks do not output a CSV


# ── Unsupervised: clustering & anomaly_detection ──────────────────────────────

def _unsupervised_insight(state: AgentState, logs: list) -> tuple[dict, str]:
    best_model   = state.get("best_model", "")
    model_results = state.get("model_results") or {}
    X_full       = state.get("X_full")
    task_type    = state.get("task_type", "")
    file_path    = state.get("file_path", "")
    feature_names = state.get("feature_names") or []

    best  = model_results.get(best_model, {})
    model = best.get("model")
    insight: dict = {"task_type": task_type}

    if model is None or X_full is None:
        logs.append("[insight] 跳过：模型或全量数据不可用")
        return insight, ""

    X_arr = X_full.values if hasattr(X_full, "values") else np.asarray(X_full)

    # ── Run full-dataset prediction ───────────────────────────────────────
    labels = _predict_full(model, best_model, X_arr, best, logs)
    if labels is None:
        return insight, ""

    # ── Load original CSV for human-readable profiles ─────────────────────
    original_df = None
    try:
        original_df = pd.read_csv(file_path)
    except Exception as exc:
        logs.append(f"[insight] 读取原始 CSV 失败，仅输出统计: {exc}")

    # ── Task-specific profiling ───────────────────────────────────────────
    if task_type == "clustering":
        profiles = _cluster_profiles(labels, original_df, feature_names, logs)
        insight.update(profiles)

    elif task_type == "anomaly_detection":
        profiles = _anomaly_profiles(labels, original_df, model, X_arr, feature_names, logs)
        insight.update(profiles)

    # ── Save labeled CSV ──────────────────────────────────────────────────
    pred_path = _save_labeled_csv(
        labels, original_df, task_type, file_path, logs
    )

    return insight, pred_path


# ── Helpers: feature importance ───────────────────────────────────────────────

def _extract_feature_importance(model, feature_names: list) -> list:
    """Return list of {feature, importance} sorted desc, top 15."""
    importances = None

    if model is None:
        return []

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        coef = np.array(model.coef_)
        if coef.ndim == 2:          # multi-class
            importances = np.abs(coef).mean(axis=0)
        else:
            importances = np.abs(coef)

    if importances is None or len(importances) == 0:
        return []

    names = feature_names if feature_names else [f"feature_{i}" for i in range(len(importances))]
    total = importances.sum()
    if total == 0:
        return []

    idx = np.argsort(importances)[::-1][:15]
    return [
        {
            "feature":    names[i],
            "importance": round(float(importances[i]), 6),
            "importance_pct": round(float(importances[i] / total * 100), 1),
        }
        for i in idx
        if importances[i] > 0
    ]


def _compute_feature_directions(state: AgentState, feature_names: list) -> dict:
    """Compute sign of correlation between each feature and y_train."""
    X_train = state.get("X_train")
    y_train = state.get("y_train")
    if X_train is None or y_train is None or not feature_names:
        return {}
    try:
        y = y_train.values.ravel() if hasattr(y_train, "values") else np.array(y_train).ravel()
        X = X_train.values if hasattr(X_train, "values") else np.array(X_train)
        directions = {}
        for i, name in enumerate(feature_names):
            if i >= X.shape[1]:
                break
            corr = np.corrcoef(X[:, i], y)[0, 1]
            if np.isfinite(corr):
                directions[name] = "positive" if corr >= 0 else "negative"
        return directions
    except Exception:
        return {}


# ── Helpers: classification profiles ─────────────────────────────────────────

def _compute_class_profiles(state: AgentState, y_pred, feature_names: list) -> dict:
    """Compute per-class mean of top features from original (unscaled) data."""
    file_path  = state.get("file_path", "")
    target_col = state.get("target_column", "")
    y_test     = state.get("y_test")

    if y_pred is None or y_test is None or not file_path:
        return {}
    try:
        original_df = pd.read_csv(file_path)
        feat_cols = [c for c in feature_names if c in original_df.columns][:8]
        if not feat_cols:
            return {}

        y_true = y_test.values.ravel() if hasattr(y_test, "values") else np.array(y_test).ravel()
        y_pred_arr = np.array(y_pred).ravel()

        # Use test-set rows from original_df (last len(y_test) rows after 80/20 split)
        n_test = len(y_true)
        test_df = original_df.iloc[-n_test:].copy().reset_index(drop=True)
        test_df["__pred__"] = y_pred_arr
        test_df["__true__"] = y_true

        profiles = {}
        for cls in sorted(test_df["__pred__"].unique()):
            subset = test_df[test_df["__pred__"] == cls]
            count  = len(subset)
            total  = len(test_df)
            means  = {col: round(float(subset[col].mean()), 4)
                      for col in feat_cols
                      if pd.api.types.is_numeric_dtype(subset[col])}
            profiles[str(cls)] = {"count": count,
                                   "ratio": round(count / total, 4),
                                   **means}
        return profiles
    except Exception:
        return {}


# ── Helpers: clustering profiles ─────────────────────────────────────────────

def _cluster_profiles(labels, original_df, feature_names: list, logs: list) -> dict:
    counts = Counter(int(l) for l in labels)
    n_clusters = len(counts)
    total = len(labels)
    logs.append(f"[insight] 聚类完成：{n_clusters} 个 cluster，共 {total} 条")

    profiles = {}
    if original_df is not None and len(original_df) == total:
        feat_cols = [c for c in feature_names if c in original_df.columns][:8]
        tmp = original_df.copy()
        tmp["__cluster__"] = labels
        for cid in sorted(counts):
            subset = tmp[tmp["__cluster__"] == cid]
            means  = {col: round(float(subset[col].mean()), 4)
                      for col in feat_cols
                      if pd.api.types.is_numeric_dtype(subset[col])}
            profiles[str(cid)] = {
                "size":  counts[cid],
                "ratio": round(counts[cid] / total, 4),
                **means,
            }

    return {
        "n_clusters":      n_clusters,
        "cluster_profiles": profiles,
    }


# ── Helpers: anomaly profiles ─────────────────────────────────────────────────

def _anomaly_profiles(labels, original_df, model, X_arr, feature_names: list, logs: list) -> dict:
    arr    = np.array(labels)
    n_anom = int((arr == -1).sum())
    n_norm = int((arr ==  1).sum())
    total  = len(arr)
    ratio  = round(n_anom / total, 4) if total else 0
    logs.append(f"[insight] 异常检测完成：{n_anom} 异常 / {total} 总计 ({ratio:.1%})")

    result: dict = {
        "n_anomaly":     n_anom,
        "n_normal":      n_norm,
        "anomaly_ratio": ratio,
    }

    # Anomaly score (if available)
    try:
        if hasattr(model, "decision_function"):
            scores = model.decision_function(X_arr)
            result["has_scores"] = True
            result["score_stats"] = {
                "anomaly_mean": round(float(np.mean(scores[arr == -1])), 4) if n_anom else None,
                "normal_mean":  round(float(np.mean(scores[arr ==  1])), 4) if n_norm else None,
            }
    except Exception:
        pass

    # Per-group feature profiles (original scale)
    if original_df is not None and len(original_df) == total:
        feat_cols = [c for c in feature_names if c in original_df.columns][:8]
        tmp = original_df.copy()
        tmp["__anomaly__"] = arr
        for label, key in [(-1, "anomaly_profile"), (1, "normal_profile")]:
            subset = tmp[tmp["__anomaly__"] == label]
            if len(subset):
                result[key] = {
                    col: round(float(subset[col].mean()), 4)
                    for col in feat_cols
                    if pd.api.types.is_numeric_dtype(subset[col])
                }

    return result


# ── Helpers: full-dataset prediction ─────────────────────────────────────────

def _predict_full(model, model_name: str, X_arr, best: dict, logs: list):
    """Run predict on full dataset. Returns labels array or None on failure."""
    # Models that can only fit_predict (not predict on new data)
    if model_name in _NO_PREDICT:
        stored = best.get("labels")
        if stored is not None:
            logs.append(f"[insight] {model_name} 无 predict()，使用已存储标签")
            return np.array(stored)
        # Re-run fit_predict on full data as best effort
        try:
            labels = model.fit_predict(X_arr)
            logs.append(f"[insight] {model_name} fit_predict 全量数据完成")
            return labels
        except Exception as exc:
            logs.append(f"[insight] {model_name} fit_predict 失败: {exc}")
            return None

    try:
        labels = model.predict(X_arr)
        logs.append(f"[insight] 全量预测完成，共 {len(labels)} 条")
        return labels
    except Exception as exc:
        logs.append(f"[insight] predict 失败: {exc}")
        return None


# ── Helpers: save labeled CSV ─────────────────────────────────────────────────

def _save_labeled_csv(labels, original_df, task_type: str, file_path: str, logs: list) -> str:
    if original_df is None:
        return ""
    try:
        df = original_df.copy()
        if len(labels) == len(df):
            if task_type == "clustering":
                df["cluster_id"] = labels
            elif task_type == "anomaly_detection":
                df["is_anomaly"] = (np.array(labels) == -1).astype(int)

        out_dir = os.path.join(os.path.dirname(__file__), "..", "outputs", "predictions")
        os.makedirs(out_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(file_path))[0] if file_path else "data"
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(out_dir, f"predictions_{stem}_{ts}.csv")
        df.to_csv(path, index=False)
        logs.append(f"[insight] 标注结果已保存: {os.path.basename(path)}")
        return path
    except Exception as exc:
        logs.append(f"[insight] 保存 CSV 失败: {exc}")
        return ""
