import os
import json
from datetime import datetime
from anthropic import Anthropic
from app.state import AgentState, ReportData


_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "..", "prompts", "report_prompt.txt")


def _load_template(user_level: str) -> str:
    with open(_PROMPT_PATH, "r", encoding="utf-8") as f:
        content = f.read()
    if user_level == "expert":
        marker = "=== EXPERT TEMPLATE ==="
    else:
        marker = "=== GENERAL TEMPLATE ==="
    idx = content.find(marker)
    if idx == -1:
        return content
    next_marker = content.find("===", idx + len(marker))
    if next_marker == -1 or user_level == "expert":
        return content[idx + len(marker):].strip()
    return content[idx + len(marker):next_marker].strip()


def generate_report(state: AgentState) -> AgentState:
    try:
        user_level  = state.get("user_level", "general")
        report_data = build_report_data(state)
        template    = _load_template(user_level)
        content     = _call_llm(template, report_data)
        report_path = _save_report(content)

        updated = state.copy()
        updated["report_path"] = report_path
        updated["logs"] = list(state.get("logs", [])) + ["[reporting] 报告生成完成"]
        return updated
    except Exception as e:
        updated = state.copy()
        updated["logs"] = list(state.get("logs", [])) + [f"[reporting] 报告生成失败: {e}"]
        return updated


def build_report_data(state: AgentState) -> ReportData:
    schema = state.get("schema", {})
    meta = schema.get("_meta", {})
    eda_summary = state.get("eda_summary", {})
    metrics = state.get("metrics", {})
    best_model = state.get("best_model", "")
    task_type = state.get("task_type", "")

    report_data: ReportData = {
        "timestamp":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "file_name":    os.path.basename(state.get("file_path", "")),
        "target_column": state.get("target_column", ""),
        "task_type":    task_type,
        "user_level":   state.get("user_level", "general"),
        "user_intent_summary": state.get("user_intent_summary", ""),
        "reasoning":    state.get("reasoning", ""),

        # Dataset overview
        "data_scope":   meta.get("data_scope", ""),
        "row_count":    meta.get("row_count", 0),
        "col_count":    meta.get("col_count", 0),
        "core_features": ",".join((state.get("feature_names", []) or [])[:5]),
        "unit":         meta.get("unit", ""),
        "mean_target":  meta.get("mean_target", 0.0),

        # Quality
        "quality_issues": ";".join(state.get("quality_issues", [])),

        # EDA
        "top3_features":    eda_summary.get("top3_features", ""),
        "distribution_desc": eda_summary.get("distribution_desc", ""),
        "layer_desc":       eda_summary.get("layer_desc", ""),
        "abnormal_desc":    eda_summary.get("abnormal_desc", ""),
        "key_charts":       ";".join(state.get("charts", [])),
        "feature_1":        eda_summary.get("feature_1", ""),
        "feature_2":        eda_summary.get("feature_2", ""),
        "feature_3":        eda_summary.get("feature_3", ""),
        "feature_1_corr":   float(eda_summary.get("feature_1_corr", 0.0)),
        "feature_2_corr":   float(eda_summary.get("feature_2_corr", 0.0)),
        "feature_3_corr":   float(eda_summary.get("feature_3_corr", 0.0)),

        # Modeling hints (for expert report)
        "modeling_hints":   state.get("modeling_hints", {}),

        # All model metrics (for expert report comparison table)
        "all_metrics":      metrics,

        # Best model metrics
        "best_model":       best_model,
        "accuracy":  0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0,
        "mae":       0.0, "rmse":      0.0,  "r2":    0.0,
        "silhouette":        0.0,
        "anomaly_ratio":     0.0,

        # Defaults
        "business_context":     "本次分析基于业务数据，旨在挖掘影响核心指标的关键因素",
        "feature_relation_desc": "",
        "model_value":          "",
        "conclusion_1": "", "conclusion_2": "", "conclusion_3": "",
        "short_suggest_1": "", "short_suggest_2": "",
        "long_suggest_1": "",  "long_suggest_2": "",
        "data_risk":  "数据可能存在局限性，结论需结合业务经验验证。",
        "exec_risk":  "模型预测结果仅作参考，实际执行需综合考量。",
    }

    # Fill best model metrics
    if best_model and best_model in metrics:
        m = metrics[best_model]
        if task_type == "classification":
            report_data["accuracy"]  = m.get("accuracy", 0.0)
            report_data["precision"] = m.get("precision", 0.0)
            report_data["recall"]    = m.get("recall", 0.0)
            report_data["f1"]        = m.get("f1", 0.0)
        elif task_type == "regression":
            report_data["mae"]  = m.get("mae", 0.0)
            report_data["rmse"] = m.get("rmse", 0.0)
            report_data["r2"]   = m.get("r2", 0.0)
        elif task_type == "clustering":
            report_data["silhouette"] = m.get("silhouette", 0.0)
        elif task_type == "anomaly_detection":
            report_data["anomaly_ratio"] = m.get("anomaly_ratio", 0.0)

    return report_data


def _call_llm(template: str, report_data: ReportData) -> str:
    client = Anthropic()
    # Exclude non-serializable or very large objects
    data_json = {
        k: v for k, v in report_data.items()
        if v not in (None, "", 0, 0.0, []) and k not in ("X_train", "X_test", "y_train", "y_test")
    }
    user_msg = (
        f"{template}\n\n---\n"
        f"以下是分析数据：\n```json\n{json.dumps(data_json, ensure_ascii=False, indent=2)}\n```"
    )
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        messages=[{"role": "user", "content": user_msg}],
    )
    return response.content[0].text


def _save_report(content: str) -> str:
    out_dir = os.path.join(os.path.dirname(__file__), "..", "outputs", "reports")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "analysis_report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path
