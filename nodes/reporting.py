import os
import json
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from openai import OpenAI
from app.state import AgentState, ReportData


_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "..", "prompts", "report_prompt.txt")
_CHART_DIR   = os.path.join(os.path.dirname(__file__), "..", "outputs", "charts")


# ── Entry point ──────────────────────────────────────────────────────────────

def generate_report(state: AgentState) -> dict:
    try:
        # 1. 生成模型专属图表
        model_charts = _generate_model_charts(state)
        all_charts   = list(state.get("charts", [])) + model_charts

        # 2. LLM 生成叙事段落（JSON 格式）
        narrative = _call_llm_narrative(state)

        # 3. 程序化组装最终报告
        content     = _assemble_report(state, all_charts, narrative)
        report_path = _save_report(content, state.get("file_path", ""))

        return {
            "report_path": report_path,
            "charts":      all_charts,
            "logs": list(state.get("logs", [])) + ["[reporting] 报告生成完成"],
        }
    except Exception as exc:
        return {
            "logs": list(state.get("logs", [])) + [f"[reporting] 报告生成失败: {exc}"],
        }


# ── Model chart generation ────────────────────────────────────────────────────

def _generate_model_charts(state: AgentState) -> list:
    """Generate model-specific visualizations; return list of saved paths."""
    os.makedirs(_CHART_DIR, exist_ok=True)
    task_type    = state.get("task_type", "")
    model_results = state.get("model_results", {})
    best_model   = state.get("best_model", "")
    paths        = []

    if not model_results or not best_model:
        return paths

    best = model_results.get(best_model, {})
    fitted_model = best.get("model")
    predictions  = best.get("predictions")
    feature_names = state.get("feature_names", [])
    y_test = state.get("y_test")

    # ── Feature importance (tree-based models) ────────────────────────────
    if fitted_model is not None and hasattr(fitted_model, "feature_importances_"):
        try:
            importances = fitted_model.feature_importances_
            names = feature_names if feature_names else [f"f{i}" for i in range(len(importances))]
            idx   = np.argsort(importances)[::-1][:15]  # top 15

            fig, ax = plt.subplots(figsize=(8, max(4, len(idx) * 0.4)))
            try:
                ax.barh([names[i] for i in reversed(idx)],
                        [importances[i] for i in reversed(idx)],
                        color="steelblue")
                ax.set_xlabel("Feature Importance")
                ax.set_title(f"Feature Importance — {best_model}")
                plt.tight_layout()
                path = os.path.join(_CHART_DIR, "model_feature_importance.png")
                plt.savefig(path, dpi=150)
                paths.append(path)
            finally:
                plt.close(fig)
        except Exception:
            pass

    # ── Regression: prediction vs actual + residuals ──────────────────────
    if task_type == "regression" and predictions is not None and y_test is not None:
        try:
            import pandas as pd
            y_true = y_test.values.ravel() if hasattr(y_test, "values") else np.array(y_test).ravel()
            y_pred = np.array(predictions).ravel()
            mask   = np.isfinite(y_true) & np.isfinite(y_pred)
            y_true, y_pred = y_true[mask], y_pred[mask]

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            try:
                # Pred vs Actual
                axes[0].scatter(y_true, y_pred, alpha=0.3, s=10, color="steelblue")
                mn, mx = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
                axes[0].plot([mn, mx], [mn, mx], "r--", linewidth=1.5, label="Perfect fit")
                axes[0].set_xlabel("Actual")
                axes[0].set_ylabel("Predicted")
                axes[0].set_title(f"Predicted vs Actual — {best_model}")
                axes[0].legend()

                # Residuals
                residuals = y_pred - y_true
                axes[1].scatter(y_pred, residuals, alpha=0.3, s=10, color="coral")
                axes[1].axhline(0, color="black", linewidth=1)
                axes[1].set_xlabel("Predicted")
                axes[1].set_ylabel("Residual")
                axes[1].set_title("Residual Plot")

                plt.tight_layout()
                path = os.path.join(_CHART_DIR, "model_regression_diagnosis.png")
                plt.savefig(path, dpi=150)
                paths.append(path)
            finally:
                plt.close(fig)
        except Exception:
            pass

    # ── Classification: confusion matrix ─────────────────────────────────
    if task_type == "classification" and predictions is not None and y_test is not None:
        try:
            from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
            y_true = y_test.values.ravel() if hasattr(y_test, "values") else np.array(y_test).ravel()
            y_pred = np.array(predictions).ravel()
            cm     = confusion_matrix(y_true, y_pred)
            labels = sorted(set(y_true))

            fig, ax = plt.subplots(figsize=(max(5, len(labels)), max(4, len(labels))))
            try:
                disp = ConfusionMatrixDisplay(cm, display_labels=labels)
                disp.plot(ax=ax, colorbar=False, cmap="Blues")
                ax.set_title(f"Confusion Matrix — {best_model}")
                plt.tight_layout()
                path = os.path.join(_CHART_DIR, "model_confusion_matrix.png")
                plt.savefig(path, dpi=150)
                paths.append(path)
            finally:
                plt.close(fig)
        except Exception:
            pass

    # ── Clustering: PCA 2D visualization ─────────────────────────────────
    if task_type == "clustering" and predictions is not None:
        try:
            from sklearn.decomposition import PCA
            X_test = state.get("X_test")
            if X_test is not None:
                X_arr  = X_test.values if hasattr(X_test, "values") else np.array(X_test)
                labels = np.array(predictions).ravel()
                pca    = PCA(n_components=2)
                X_2d   = pca.fit_transform(X_arr)

                fig, ax = plt.subplots(figsize=(7, 6))
                try:
                    scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=labels,
                                         cmap="tab10", alpha=0.6, s=20)
                    plt.colorbar(scatter, ax=ax, label="Cluster")
                    ax.set_xlabel("PC1")
                    ax.set_ylabel("PC2")
                    ax.set_title(f"Cluster Visualization (PCA 2D) — {best_model}")
                    plt.tight_layout()
                    path = os.path.join(_CHART_DIR, "model_cluster_pca.png")
                    plt.savefig(path, dpi=150)
                    paths.append(path)
                finally:
                    plt.close(fig)
        except Exception:
            pass

    # ── Anomaly: score distribution ───────────────────────────────────────
    if task_type == "anomaly_detection" and predictions is not None:
        try:
            preds  = np.array(predictions).ravel()
            n_anom = int((preds == -1).sum())
            n_norm = int((preds == 1).sum())

            fig, ax = plt.subplots(figsize=(6, 4))
            try:
                ax.bar(["Normal", "Anomaly"], [n_norm, n_anom],
                       color=["steelblue", "tomato"])
                ax.set_title(f"Anomaly Detection Results — {best_model}")
                ax.set_ylabel("Count")
                for i, v in enumerate([n_norm, n_anom]):
                    ax.text(i, v + max(n_norm, n_anom) * 0.01, str(v), ha="center")
                plt.tight_layout()
                path = os.path.join(_CHART_DIR, "model_anomaly_distribution.png")
                plt.savefig(path, dpi=150)
                paths.append(path)
            finally:
                plt.close(fig)
        except Exception:
            pass

    # ── All models: metric comparison bar chart ───────────────────────────
    metrics   = state.get("metrics", {})
    metric_key = {
        "regression":       "r2",
        "classification":   "f1",
        "clustering":       "silhouette",
        "anomaly_detection": "anomaly_ratio",
    }.get(task_type)

    if metric_key and len(metrics) > 1:
        try:
            names  = [m for m in metrics if metric_key in metrics[m]]
            values = [metrics[m][metric_key] for m in names]
            colors = ["tomato" if n == best_model else "steelblue" for n in names]

            fig, ax = plt.subplots(figsize=(7, 4))
            try:
                bars = ax.bar(names, values, color=colors)
                ax.set_title(f"Model Comparison — {metric_key.upper()}")
                ax.set_ylabel(metric_key.upper())
                for bar, val in zip(bars, values):
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + max(values) * 0.01,
                            f"{val:.3f}", ha="center", fontsize=9)
                plt.xticks(rotation=15, ha="right")
                plt.tight_layout()
                path = os.path.join(_CHART_DIR, "model_comparison.png")
                plt.savefig(path, dpi=150)
                paths.append(path)
            finally:
                plt.close(fig)
        except Exception:
            pass

    return paths


# ── LLM narrative (JSON sections only) ───────────────────────────────────────

def _call_llm_narrative(state: AgentState) -> dict:
    """Ask LLM to return narrative sections as JSON. Falls back to empty strings."""
    user_level   = state.get("user_level", "general")
    task_type    = state.get("task_type", "")
    eda_summary  = state.get("eda_summary", {})
    metrics      = state.get("metrics", {})
    best_model   = state.get("best_model", "")
    reasoning    = state.get("reasoning", "")

    best_metrics = metrics.get(best_model, {}) if best_model else {}
    metrics_str  = json.dumps(
        {m: {k: round(v, 4) for k, v in mv.items() if isinstance(v, float)}
         for m, mv in metrics.items() if isinstance(mv, dict) and "error" not in mv},
        ensure_ascii=False, indent=2
    )

    # Build prediction statistics so the LLM has real numbers to cite
    prediction_stats_str = ""
    best_result = (state.get("model_results") or {}).get(best_model, {})
    predictions = best_result.get("predictions")
    y_test = state.get("y_test")
    if predictions is not None and task_type == "regression":
        try:
            preds = np.array(predictions).ravel()
            preds = preds[np.isfinite(preds)]
            if len(preds):
                prediction_stats_str = (
                    f"Predicted value distribution — "
                    f"mean: {np.mean(preds):.2f}, median: {np.median(preds):.2f}, "
                    f"std: {np.std(preds):.2f}, "
                    f"range: [{np.min(preds):.2f}, {np.max(preds):.2f}], "
                    f"25th pct: {np.percentile(preds, 25):.2f}, "
                    f"75th pct: {np.percentile(preds, 75):.2f}"
                )
        except Exception:
            pass

    conclusions_spec_expert = (
        "- \"conclusions\": 2-3 conclusions written as a DIRECT ANSWER to the user's specific question above.\n"
        "  • If the question is about prediction/forecasting: state what the model says about the target's level, range, and direction, using the prediction statistics provided.\n"
        "  • If the question is about key drivers: state which factors push the target up or down and by how much.\n"
        "  • If the question is about segmentation/clustering: describe the groups found and what differentiates them.\n"
        "  Do NOT list generic feature importances. Do NOT mention model names or accuracy scores. (bullet points, Chinese)"
    )
    conclusions_spec_general = (
        "- \"conclusions\": 2-3 plain-language sentences that are a DIRECT ANSWER to the user's specific question above.\n"
        "  • If the question is about prediction/forecasting: state the predicted price level/range/trend using the prediction statistics provided — give concrete numbers where available.\n"
        "  • If the question is about key drivers: explain in plain language which factors matter most and how they affect the outcome.\n"
        "  • If the question is about segmentation/clustering: describe the groups in terms the user cares about.\n"
        "  Do NOT just list feature importances as if that were the answer. Do NOT mention model names or accuracy scores. (bullet points, Chinese)"
    )

    if user_level == "expert":
        section_spec = f"""Return a JSON object with these keys:
- "eda_narrative": technical EDA findings (2-3 paragraphs, Chinese)
- "model_rationale": explain the algorithm selection reasoning (1-2 paragraphs, Chinese)
- "model_analysis": interpret the best model's metrics technically (1-2 paragraphs, Chinese)
{conclusions_spec_expert}
- "recommendations": 3-5 concrete actions the user should take based on what the data revealed about their specific question — short-term (1-3 months) and long-term (3-12 months). Ground each action in the findings, not in model improvement. (bullet points, Chinese)
- "risks": 2-3 risks or caveats relevant to acting on these findings (bullet points, Chinese)"""
    else:
        section_spec = f"""Return a JSON object with these keys:
- "eda_narrative": explain EDA findings in plain language without jargon (2-3 paragraphs, Chinese)
- "model_rationale": explain in simple terms why these models were chosen (1 paragraph, Chinese)
- "model_analysis": translate model performance into business meaning, no metric numbers (1-2 paragraphs, Chinese)
{conclusions_spec_general}
- "recommendations": 3-5 specific actions the user should take to act on the answer to their question — short-term (1-3 months) and long-term (3-12 months). These should follow naturally from the conclusions, not from model improvement ideas. (bullet points, Chinese)
- "risks": 2-3 plain-language warnings about limitations of these findings for the user's decision (bullet points, Chinese)"""

    prediction_block = f"\nPrediction statistics:\n{prediction_stats_str}\n" if prediction_stats_str else ""

    prompt = f"""You are a business analyst answering a specific question the user asked about their data.
Your ONE job: use the data findings to directly answer the user's question below.
Do NOT write a generic data science report. Do NOT default to "key factors are X, Y, Z" unless that IS the answer to the question.

User's question: {state.get("user_query", "")}
Restated intent: {state.get("user_intent_summary", "")}
Task type: {task_type}

EDA summary:
- Top 3 features: {eda_summary.get("top3_features", "")}
- Distribution: {eda_summary.get("distribution_desc", "")}
- Layer analysis: {eda_summary.get("layer_desc", "")}
- Outliers: {eda_summary.get("abnormal_desc", "")}
{prediction_block}
Model selection reasoning: {reasoning}
Model results (all candidates):
{metrics_str}

Best model: {best_model}
Best model metrics: {json.dumps({k: round(v, 4) for k, v in best_metrics.items() if isinstance(v, float)}, ensure_ascii=False)}

{section_spec}

IMPORTANT: Return ONLY valid JSON. No markdown code fences, no extra text."""

    client = OpenAI(
        api_key=os.environ.get("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
    )
    response = client.chat.completions.create(
        model="deepseek-chat",
        max_tokens=3000,
        temperature=0.3,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.choices[0].message.content.strip()

    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        raw = raw.rsplit("```", 1)[0].strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {k: "" for k in ["eda_narrative", "model_rationale",
                                  "model_analysis", "conclusions",
                                  "recommendations", "risks"]}


# ── Report assembly ───────────────────────────────────────────────────────────

def _rel_path(abs_path: str, report_dir: str) -> str:
    """Return path relative to report file for markdown image embedding."""
    try:
        return os.path.relpath(abs_path, report_dir).replace("\\", "/")
    except ValueError:
        return abs_path.replace("\\", "/")


def _embed_charts(paths: list, report_dir: str, captions: dict = None) -> str:
    captions = captions or {}
    lines = []
    for p in paths:
        if p and os.path.exists(p):
            name    = os.path.basename(p).replace(".png", "").replace("_", " ").title()
            caption = captions.get(os.path.basename(p), name)
            rel     = _rel_path(p, report_dir)
            lines.append(f"![{caption}]({rel})\n*{caption}*\n")
    return "\n".join(lines)


def _metrics_table(metrics: dict, task_type: str, best_model: str) -> str:
    if not metrics:
        return ""

    metric_cols = {
        "classification":   ["accuracy", "precision", "recall", "f1"],
        "regression":       ["mae", "rmse", "r2"],
        "clustering":       ["silhouette", "davies_bouldin"],
        "anomaly_detection": ["anomaly_ratio"],
    }.get(task_type, [])

    valid = {m: v for m, v in metrics.items()
             if isinstance(v, dict) and "error" not in v}
    if not valid or not metric_cols:
        return ""

    actual_cols = [c for c in metric_cols
                   if any(c in v for v in valid.values())]
    if not actual_cols:
        return ""

    header = "| 模型 | " + " | ".join(c.upper() for c in actual_cols) + " |"
    sep    = "|---|" + "---|" * len(actual_cols)
    rows   = []
    for model, mv in valid.items():
        mark = " ✓" if model == best_model else ""
        vals = " | ".join(
            f"{mv.get(c, '-'):.4f}" if isinstance(mv.get(c), float) else str(mv.get(c, "-"))
            for c in actual_cols
        )
        rows.append(f"| **{model}{mark}** | {vals} |")

    return "\n".join([header, sep] + rows)


def _narr(narrative: dict, key: str) -> str:
    """Safely extract narrative section as string regardless of LLM return type."""
    val = narrative.get(key, "")
    if isinstance(val, list):
        return "\n".join(str(v) for v in val)
    return str(val) if val else ""


def _assemble_report(state: AgentState, all_charts: list, narrative: dict) -> str:
    schema       = state.get("schema", {})
    meta         = schema.get("_meta", {})
    task_type    = state.get("task_type", "")
    best_model   = state.get("best_model", "")
    metrics      = state.get("metrics", {})
    eda_summary  = state.get("eda_summary", {})
    quality      = state.get("quality_issues", [])
    features     = state.get("feature_names", [])
    user_level   = state.get("user_level", "general")
    reasoning    = state.get("reasoning", "")
    selected_alg = state.get("selected_algorithms", [])
    hints        = state.get("modeling_hints", {})

    report_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "outputs", "reports")
    )

    # Categorise charts
    eda_charts   = [p for p in all_charts if p and "feature_importance" not in p
                    and "model_" not in os.path.basename(p)]
    model_charts = [p for p in all_charts if p and "model_" in os.path.basename(p)]

    chart_captions = {
        "target_distribution_median_house_value.png": "目标变量分布",
        "correlation_heatmap.png": "特征相关性热力图",
        "feature_target_a.png": "特征与目标关系",
        "model_feature_importance.png": "特征重要性排名",
        "model_regression_diagnosis.png": "预测 vs 实际 & 残差分析",
        "model_confusion_matrix.png": "混淆矩阵",
        "model_cluster_pca.png": "聚类可视化（PCA降维）",
        "model_anomaly_distribution.png": "异常检测结果分布",
        "model_comparison.png": "候选模型性能对比",
    }

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    level_tag  = "通用报告" if user_level == "general" else "专家报告"

    lines = []

    # ── Header ────────────────────────────────────────────────────────────
    lines += [
        f"# AutoInsight 分析报告",
        f"",
        f"> **生成时间:** {timestamp} | "
        f"**数据集:** {os.path.basename(state.get('file_path', ''))} | "
        f"**任务类型:** {task_type} | "
        f"**报告等级:** {level_tag}",
        f"",
        f"**用户诉求:** {state.get('user_intent_summary', '')}",
        f"",
        "---",
    ]

    # ── Section 1: Dataset Overview ───────────────────────────────────────
    lines += [
        "", "## 一、数据概览", "",
        "| 指标 | 值 |",
        "|---|---|",
        f"| 数据集 | {os.path.basename(state.get('file_path', ''))} |",
        f"| 行数 | {meta.get('row_count', '-'):,} |",
        f"| 列数 | {meta.get('col_count', '-')} |",
        f"| 目标列 | `{state.get('target_column', '-')}` |",
        f"| 任务类型 | {task_type} |",
        f"| 特征列数 | {len(features)} |",
        "",
    ]
    if quality:
        lines += ["**数据质量问题：**", ""]
        for q in quality:
            lines.append(f"- {q}")
        lines.append("")

    if user_level == "expert" and hints:
        lines += ["**EDA Modeling Hints：**", ""]
        for k, v in hints.items():
            lines.append(f"- {k}: `{v}`")
        lines.append("")

    # ── Section 2: EDA ────────────────────────────────────────────────────
    lines += ["---", "", "## 二、探索性数据分析（EDA）", ""]

    if eda_charts:
        lines += ["### 关键图表", ""]
        lines.append(_embed_charts(eda_charts, report_dir, chart_captions))

    if eda_summary.get("top3_features"):
        lines += [
            "### 关键特征",
            "",
            "| 排名 | 特征 | 与目标相关系数 |",
            "|---|---|---|",
        ]
        for i, feat_key in enumerate(["feature_1", "feature_2", "feature_3"], 1):
            feat = eda_summary.get(feat_key, "")
            corr = eda_summary.get(f"feature_{i}_corr", 0.0)
            if feat:
                lines.append(f"| {i} | `{feat}` | {corr:.4f} |")
        lines.append("")

    eda_narr = _narr(narrative, "eda_narrative")
    if eda_narr:
        lines += ["### 数据洞察", "", eda_narr, ""]

    # ── Section 3: Model Selection & Training ─────────────────────────────
    lines += ["---", "", "## 三、模型选择与训练", ""]

    if selected_alg:
        lines += [
            f"**候选算法：** {', '.join(selected_alg)}",
            "",
            f"**选择依据：** {reasoning}",
            "",
        ]

    model_rationale = _narr(narrative, "model_rationale")
    if model_rationale:
        lines += [model_rationale, ""]

    # Model comparison table
    table = _metrics_table(metrics, task_type, best_model)
    if table:
        lines += ["### 候选模型对比", "", table, ""]

    # Model comparison chart
    comp_chart = [p for p in model_charts if "comparison" in os.path.basename(p)]
    if comp_chart:
        lines.append(_embed_charts(comp_chart, report_dir, chart_captions))

    # ── Section 4: Best Model Analysis ────────────────────────────────────
    lines += ["---", "", f"## 四、最优模型分析（{best_model}）", ""]

    # Model visualisation charts (excluding comparison)
    viz_charts = [p for p in model_charts if "comparison" not in os.path.basename(p)]
    if viz_charts:
        lines += ["### 模型可视化", ""]
        lines.append(_embed_charts(viz_charts, report_dir, chart_captions))

    model_analysis = _narr(narrative, "model_analysis")
    if model_analysis:
        lines += ["### 模型解读", "", model_analysis, ""]

    # ── Section 5: Conclusions ────────────────────────────────────────────
    lines += ["---", "", "## 五、结论", ""]
    conclusions = _narr(narrative, "conclusions")
    if conclusions:
        lines += [conclusions, ""]

    # ── Section 6: Recommendations ────────────────────────────────────────
    lines += ["---", "", "## 六、建议", ""]
    recommendations = _narr(narrative, "recommendations")
    if recommendations:
        lines += [recommendations, ""]

    # ── Section 7: Risks ──────────────────────────────────────────────────
    lines += ["---", "", "## 七、风险提示", ""]
    risks = _narr(narrative, "risks")
    if risks:
        lines += [risks, ""]

    lines += ["---", f"*本报告由 AutoInsight 自动生成 · {timestamp}*"]

    return "\n".join(lines)


# ── Save ─────────────────────────────────────────────────────────────────────

def _save_report(content: str, file_path: str = "") -> str:
    out_dir = os.path.join(os.path.dirname(__file__), "..", "outputs", "reports")
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(file_path))[0] if file_path else "report"
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(out_dir, f"report_{stem}_{ts}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path


# ── Legacy helpers (kept for test compatibility) ──────────────────────────────

def build_report_data(state: AgentState) -> ReportData:
    schema = state.get("schema", {})
    meta   = schema.get("_meta", {})
    eda    = state.get("eda_summary", {})
    return {
        "timestamp":     datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "file_name":     os.path.basename(state.get("file_path", "")),
        "target_column": state.get("target_column", ""),
        "task_type":     state.get("task_type", ""),
        "user_level":    state.get("user_level", "general"),
        "row_count":     meta.get("row_count", 0),
        "col_count":     meta.get("col_count", 0),
        "quality_issues": ";".join(state.get("quality_issues", [])),
        "top3_features": eda.get("top3_features", ""),
        "best_model":    state.get("best_model", ""),
        "all_metrics":   state.get("metrics", {}),
    }
