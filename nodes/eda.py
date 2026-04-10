import os
from typing import Tuple, List, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from app.state import AgentState


def run_eda(state: AgentState) -> AgentState:
    """
    Node 3: EDA
    输入: 处理后的特征/目标数据（优先 X_train + y_train）
    输出: charts + eda_summary + quality_issues(追加)
    """
    updated_state = state.copy()
    updated_state.setdefault("charts", [])
    updated_state.setdefault("quality_issues", [])
    updated_state.setdefault("logs", [])

    # ── Step 1: 数据加载与 modeling_hints（必须成功，否则提前返回） ──────
    try:
        data = _build_eda_dataframe(updated_state)
        target_column = updated_state.get("target_column", "")
        if not target_column or target_column not in data.columns:
            raise ValueError("target_column 缺失或不在 EDA 数据中")
        updated_state["modeling_hints"] = _compute_modeling_hints(
            data,
            target_column,
            updated_state.get("task_category", "supervised"),
            updated_state.get("y_train"),
        )
    except Exception as exc:
        updated_state["logs"].append(f"[EDA] 数据加载失败: {exc}")
        updated_state["modeling_hints"] = {}
        return updated_state

    # ── Step 2: 图表生成（各步独立隔离，失败只记日志不中断） ────────────
    chart_dir = _ensure_chart_dir()
    chart_paths: List[str] = []
    top3_features: List[str] = []
    top3_corr: List[float] = []

    try:
        dist_path = _plot_target_distribution(data, target_column, chart_dir)
        if dist_path:
            chart_paths.append(dist_path)
    except Exception as exc:
        updated_state["logs"].append(f"[EDA] 目标分布图生成失败: {exc}")

    try:
        corr_path = _plot_correlation_heatmap(data, target_column, chart_dir)
        if corr_path:
            chart_paths.append(corr_path)
    except Exception as exc:
        updated_state["logs"].append(f"[EDA] 相关性热力图生成失败: {exc}")

    try:
        top3_features, top3_corr = _top3_correlations(data, target_column)
        rel_paths = _plot_feature_target_relations(data, target_column, top3_features, chart_dir)
        chart_paths.extend(rel_paths)
    except Exception as exc:
        updated_state["logs"].append(f"[EDA] Top3特征图生成失败: {exc}")

    updated_state["charts"].extend(chart_paths)

    # ── Step 3: 文字摘要（各步独立隔离） ───────────────────────────────
    layer_desc = ""
    abnormal_desc = ""
    distribution_desc = ""

    try:
        layer_desc = _build_layer_desc(data, target_column)
    except Exception as exc:
        updated_state["logs"].append(f"[EDA] 分层描述生成失败: {exc}")

    try:
        abnormal_desc, abnormal_issue = _detect_target_outliers(data, target_column)
        if abnormal_issue:
            updated_state["quality_issues"].append(abnormal_issue)
    except Exception as exc:
        updated_state["logs"].append(f"[EDA] 异常值检测失败: {exc}")

    try:
        distribution_desc = _build_distribution_desc(data[target_column], target_column)
    except Exception as exc:
        updated_state["logs"].append(f"[EDA] 分布描述生成失败: {exc}")

    f1, f2, f3 = (top3_features + ["", "", ""])[:3]
    c1, c2, c3 = (top3_corr + [0.0, 0.0, 0.0])[:3]
    updated_state["eda_summary"] = {
        "top3_features": ",".join(top3_features),
        "distribution_desc": distribution_desc,
        "layer_desc": layer_desc,
        "abnormal_desc": abnormal_desc,
        "feature_1": f1,
        "feature_2": f2,
        "feature_3": f3,
        "feature_1_corr": float(c1),
        "feature_2_corr": float(c2),
        "feature_3_corr": float(c3),
    }

    updated_state["logs"].append("[EDA] EDA 分析完成，已生成图表与洞察摘要")
    return updated_state


def _build_eda_dataframe(state: AgentState) -> pd.DataFrame:
    x_train = state.get("X_train")
    y_train = state.get("y_train")
    target_column = state.get("target_column", "")

    if x_train is None or y_train is None:
        raise ValueError("X_train / y_train 不存在，无法执行 EDA")

    x_df = x_train if isinstance(x_train, pd.DataFrame) else pd.DataFrame(x_train)

    if isinstance(y_train, pd.DataFrame):
        if target_column in y_train.columns:
            y_series = y_train[target_column]
        else:
            y_series = y_train.iloc[:, 0]
    elif isinstance(y_train, pd.Series):
        y_series = y_train
    else:
        y_series = pd.Series(y_train)

    y_series = y_series.rename(target_column or "target")
    data = x_df.copy()
    data[y_series.name] = y_series.values
    return data


def _ensure_chart_dir() -> str:
    chart_dir = os.path.join(os.path.dirname(__file__), "..", "outputs", "charts")
    os.makedirs(chart_dir, exist_ok=True)
    return chart_dir


def _plot_target_distribution(data: pd.DataFrame, target_column: str, chart_dir: str) -> str:
    series = pd.to_numeric(data[target_column], errors="coerce").dropna()
    if series.empty:
        return ""

    path = os.path.join(chart_dir, f"target_distribution_{target_column}.png")
    plt.figure(figsize=(8, 5))
    try:
        sns.histplot(series, kde=True, bins=30)
        plt.title(f"Target Distribution: {target_column}")
        plt.xlabel(target_column)
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(path, dpi=150)
    finally:
        plt.close()
    return path


def _plot_correlation_heatmap(data: pd.DataFrame, target_column: str, chart_dir: str) -> str:
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2 or target_column not in numeric_cols:
        return ""

    corr = data[numeric_cols].corr()
    path = os.path.join(chart_dir, "correlation_heatmap.png")

    plt.figure(figsize=(10, 8))
    try:
        sns.heatmap(corr, cmap="coolwarm", center=0)
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        plt.savefig(path, dpi=150)
    finally:
        plt.close()
    return path


def _top3_correlations(data: pd.DataFrame, target_column: str) -> Tuple[List[str], List[float]]:
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    if target_column not in numeric_cols:
        return [], []

    corr = data[numeric_cols].corr()[target_column].drop(labels=[target_column], errors="ignore")
    corr = corr.dropna()
    if corr.empty:
        return [], []

    top = corr.abs().sort_values(ascending=False).head(3)
    features = top.index.tolist()
    values = [float(corr.loc[f]) for f in features]
    return features, values


def _plot_feature_target_relations(
    data: pd.DataFrame, target_column: str, features: List[str], chart_dir: str
) -> List[str]:
    paths: List[str] = []
    for feature in features:
        if feature not in data.columns:
            continue
        series = pd.to_numeric(data[feature], errors="coerce")
        target = pd.to_numeric(data[target_column], errors="coerce")
        plot_data = pd.DataFrame({feature: series, target_column: target}).dropna()
        if plot_data.empty:
            continue

        path = os.path.join(chart_dir, f"feature_target_{feature}.png")
        plt.figure(figsize=(7, 5))
        try:
            sns.scatterplot(data=plot_data, x=feature, y=target_column, s=20, alpha=0.7)
            plt.title(f"{feature} vs {target_column}")
            plt.tight_layout()
            plt.savefig(path, dpi=150)
        finally:
            plt.close()
        paths.append(path)
    return paths


def _build_distribution_desc(target: pd.Series, target_column: str) -> str:
    numeric_target = pd.to_numeric(target, errors="coerce").dropna()
    if numeric_target.empty:
        return f"{target_column} 无法进行数值分布分析。"

    q25 = numeric_target.quantile(0.25)
    q75 = numeric_target.quantile(0.75)
    median = numeric_target.median()
    mean = numeric_target.mean()
    return (
        f"{target_column}均值{mean:.2f}，中位数{median:.2f}，"
        f"主要分布区间约为[{q25:.2f}, {q75:.2f}]。"
    )


def _build_layer_desc(data: pd.DataFrame, target_column: str) -> str:
    cat_cols = data.select_dtypes(include=["object", "category"]).columns.tolist()
    low_card_cols = [c for c in cat_cols if 1 < data[c].nunique(dropna=True) <= 12]
    if not low_card_cols:
        return "当前训练数据中未找到低基数类别特征，未执行分层均值分析。"

    layer_col = low_card_cols[0]
    grouped = data.groupby(layer_col)[target_column].mean(numeric_only=True).dropna()
    if grouped.empty:
        return f"按 {layer_col} 分层后未得到有效均值结果。"

    top = grouped.sort_values(ascending=False).head(3)
    detail = "；".join([f"{idx}: {val:.2f}" for idx, val in top.items()])
    return f"按 {layer_col} 分组后，目标均值 Top 分层为 {detail}。"


def _detect_target_outliers(data: pd.DataFrame, target_column: str) -> Tuple[str, str]:
    target = pd.to_numeric(data[target_column], errors="coerce").dropna()
    if target.empty:
        return "目标列无法执行异常值检测。", ""

    q1 = target.quantile(0.25)
    q3 = target.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return "目标列分布较集中，未识别到明显异常值。", ""

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outlier_mask = (target < lower) | (target > upper)
    outlier_count = int(outlier_mask.sum())
    ratio = outlier_count / max(len(target), 1)

    if outlier_count == 0:
        return "未识别到明显异常值。", ""

    desc = f"检测到{outlier_count}个异常值，占比{ratio:.2%}。"
    issue = f"{target_column} 存在异常值 {outlier_count} 个（占比 {ratio:.2%}）"
    return desc, issue


def _compute_modeling_hints(data: pd.DataFrame, target_column: str,
                             task_category: str, y_train) -> dict:
    """Compute structured modeling hints for model_routing_node."""
    hints: dict = {
        "sample_size":   len(data),
        "feature_count": len([c for c in data.columns if c != target_column]),
    }

    numeric_data = data.select_dtypes(include=[np.number])

    # linearity_score: mean |pearson| of features vs target (supervised only)
    if task_category == "supervised" and target_column in numeric_data.columns:
        corr = (
            numeric_data.corr()[target_column]
            .drop(labels=[target_column], errors="ignore")
            .dropna()
        )
        hints["linearity_score"] = round(float(corr.abs().mean()), 4) if not corr.empty else 0.0
    else:
        hints["linearity_score"] = 0.0

    # outlier_ratio: IQR method on target column
    if target_column in data.columns:
        target = pd.to_numeric(data[target_column], errors="coerce").dropna()
        if len(target) > 0:
            q1, q3 = target.quantile(0.25), target.quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                mask = (target < q1 - 1.5 * iqr) | (target > q3 + 1.5 * iqr)
                hints["outlier_ratio"] = round(float(mask.sum()) / len(target), 4)
            else:
                hints["outlier_ratio"] = 0.0
        else:
            hints["outlier_ratio"] = 0.0
    else:
        hints["outlier_ratio"] = 0.0

    # imbalance_ratio: minority class fraction (supervised + y_train not None)
    if task_category == "supervised" and y_train is not None:
        y_series = (
            y_train.iloc[:, 0] if isinstance(y_train, pd.DataFrame)
            else pd.Series(y_train)
        )
        counts = y_series.value_counts(normalize=True)
        if len(counts) >= 2:
            hints["imbalance_ratio"] = round(float(counts.min()), 4)

    # high_corr_pairs: feature pairs with |correlation| > 0.9
    num_cols = [c for c in numeric_data.columns if c != target_column]
    pairs: list[str] = []
    if len(num_cols) >= 2:
        corr_matrix = numeric_data[num_cols].corr().abs()
        for i in range(len(num_cols)):
            for j in range(i + 1, len(num_cols)):
                if corr_matrix.iloc[i, j] > 0.9:
                    pairs.append(f"{num_cols[i]}-{num_cols[j]}")
    hints["high_corr_pairs"] = pairs

    return hints
