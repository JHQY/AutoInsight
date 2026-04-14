import os
import numpy as np
import pandas as pd
from collections import Counter
from datetime import datetime

from app.state import AgentState

_UNSUPERVISED_TASKS = {"clustering", "anomaly_detection"}
_NO_PREDICT = {"DBSCAN"}   # no predict() method


def inference_node(state: AgentState) -> dict:
    """Run best model on full dataset; save predictions CSV and compute stats for reporting."""
    best_model   = state.get("best_model", "")
    model_results = state.get("model_results") or {}
    X_full       = state.get("X_full")
    task_type    = state.get("task_type", "")
    target_col   = state.get("target_column", "")
    file_path    = state.get("file_path", "")
    logs         = list(state.get("logs", []))

    if not best_model or X_full is None:
        logs.append("[inference] 跳过：无最优模型或全量特征矩阵")
        return {"logs": logs, "prediction_path": "", "prediction_stats": {}}

    best = model_results.get(best_model, {})
    model = best.get("model")

    if model is None:
        logs.append("[inference] 跳过：模型对象不可用（可能为无监督结果）")
        return {"logs": logs, "prediction_path": "", "prediction_stats": {}}

    X_arr = X_full.values if hasattr(X_full, "values") else np.asarray(X_full)

    try:
        # DBSCAN has no predict() — fall back to stored labels from training
        if best_model in _NO_PREDICT:
            all_preds = np.array(best.get("labels", []))
            logs.append("[inference] DBSCAN 无 predict()，使用训练集标签统计")
        else:
            all_preds = model.predict(X_arr)
            logs.append(f"[inference] 全量推理完成，共 {len(all_preds)} 条预测")
    except Exception as exc:
        logs.append(f"[inference] 推理失败: {exc}")
        return {"logs": logs, "prediction_path": "", "prediction_stats": {}}

    # ── Save predictions CSV ──────────────────────────────────────────────
    pred_path = ""
    try:
        original_df = pd.read_csv(file_path)
        pred_col = f"predicted_{target_col}" if target_col else "prediction"
        if len(all_preds) == len(original_df):
            original_df[pred_col] = all_preds
        else:
            # Lengths differ (e.g. DBSCAN fallback used train labels only)
            # Attach as a separate series truncated/padded to match
            padded = list(all_preds) + [None] * max(0, len(original_df) - len(all_preds))
            original_df[pred_col] = padded[:len(original_df)]

        out_dir = os.path.join(os.path.dirname(__file__), "..", "outputs", "predictions")
        os.makedirs(out_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(file_path))[0] if file_path else "data"
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        pred_path = os.path.join(out_dir, f"predictions_{stem}_{ts}.csv")
        original_df.to_csv(pred_path, index=False)
        logs.append(f"[inference] 预测结果已保存: {os.path.basename(pred_path)}")
    except Exception as exc:
        logs.append(f"[inference] 保存预测 CSV 失败: {exc}")

    # ── Compute prediction statistics for reporting LLM ───────────────────
    prediction_stats = _compute_stats(all_preds, task_type)

    return {
        "prediction_path":  pred_path,
        "prediction_stats": prediction_stats,
        "logs":             logs,
    }


def _compute_stats(preds, task_type: str) -> dict:
    if task_type == "regression":
        arr = np.array(preds, dtype=float)
        arr = arr[np.isfinite(arr)]
        if not len(arr):
            return {}
        return {
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

    if task_type == "classification":
        counts = Counter(str(p) for p in preds)
        total  = len(preds)
        return {
            "total": total,
            "class_distribution": dict(counts),
            "class_ratios": {k: round(v / total, 4) for k, v in counts.items()},
        }

    if task_type == "clustering":
        counts = Counter(int(p) for p in preds)
        return {
            "n_clusters":   len(counts),
            "cluster_sizes": {str(k): v for k, v in sorted(counts.items())},
        }

    if task_type == "anomaly_detection":
        arr = np.array(preds)
        n_anom  = int((arr == -1).sum())
        n_norm  = int((arr ==  1).sum())
        total   = len(arr)
        return {
            "total":         total,
            "n_anomaly":     n_anom,
            "n_normal":      n_norm,
            "anomaly_ratio": round(n_anom / total, 4) if total else 0,
        }

    return {}
