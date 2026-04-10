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

    try:
        data = _build_eda_dataframe(updated_state)
        target_column = updated_state.get("target_column", "")
        if not target_column or target_column not in data.columns:
            raise ValueError("target_column 缺失或不在 EDA 数据中")

        chart_dir = _ensure_chart_dir()
        chart_paths: List[str] = []

        # 1) 目标分布图
        dist_path = _plot_target_distribution(data, target_column, chart_dir)
        if dist_path:
            chart_paths.append(dist_path)

        # 2) 相关性热力图（仅数值列）
        corr_path = _plot_correlation_heatmap(data, target_column, chart_dir)
        if corr_path:
            chart_paths.append(corr_path)

        # 3) Top3 特征与目标关系图
        top3_features, top3_corr = _top3_correlations(data, target_column)
        rel_paths = _plot_feature_target_relations(data, target_column, top3_features, chart_dir)
        chart_paths.extend(rel_paths)

        # 4) 目标的分层均值（选一个低基数类别列）
        layer_desc = _build_layer_desc(data, target_column)

        # 5) 目标异常值
        abnormal_desc, abnormal_issue = _detect_target_outliers(data, target_column)
        if abnormal_issue:
            updated_state["quality_issues"].append(abnormal_issue)

        # 6) 目标分布描述
        distribution_desc = _build_distribution_desc(data[target_column], target_column)

        # 回填 eda_summary
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

        updated_state["charts"].extend(chart_paths)
        updated_state["logs"].append("[EDA] EDA 分析完成，已生成图表与洞察摘要")
        return updated_state
    except Exception as exc:
        updated_state["logs"].append(f"[EDA] EDA 分析失败: {exc}")
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
    sns.histplot(series, kde=True, bins=30)
    plt.title(f"Target Distribution: {target_column}")
    plt.xlabel(target_column)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def _plot_correlation_heatmap(data: pd.DataFrame, target_column: str, chart_dir: str) -> str:
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2 or target_column not in numeric_cols:
        return ""

    corr = data[numeric_cols].corr()
    path = os.path.join(chart_dir, "correlation_heatmap.png")

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
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
        sns.scatterplot(data=plot_data, x=feature, y=target_column, s=20, alpha=0.7)
        plt.title(f"{feature} vs {target_column}")
        plt.tight_layout()
        plt.savefig(path, dpi=150)
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
