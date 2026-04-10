from typing import TypedDict, Optional
import pandas as pd


class EDASummary(TypedDict, total=False):
    top3_features: str
    distribution_desc: str
    layer_desc: str
    abnormal_desc: str
    feature_1: str
    feature_2: str
    feature_3: str
    feature_1_corr: float
    feature_2_corr: float
    feature_3_corr: float


class ModelingHints(TypedDict, total=False):
    """EDA 节点输出的结构化模型选择信号，供 model_routing 消费。"""
    linearity_score: float   # 0~1，特征与目标的整体线性相关度
    imbalance_ratio: float   # 少数类占比（仅分类任务有意义）
    outlier_ratio: float     # 目标列异常值占比
    high_corr_pairs: list    # 高共线性特征对列表，如 ["age-income"]
    sample_size: int         # 训练集样本数
    feature_count: int       # 特征列数


class AgentState(TypedDict):

    # ── 入口字段（由 main.py 初始化）────────────────────────────────
    user_query:           str
    file_path:            str
    target_column:        str        # 可为空字符串；intent_routing 会推断并回写
    user_level:           str        # "general" | "expert"，默认 "general"

    # ── Profiling ────────────────────────────────────────────────────
    schema:               dict
    quality_issues:       list

    # ── Intent Routing ───────────────────────────────────────────────
    task_category:        str        # "supervised" | "unsupervised" | "analytical"
    task_type:            str        # "classification"|"regression"|"clustering"|"anomaly_detection"|"correlation_analysis"
    user_intent_summary:  str        # 白话描述用户意图，供报告用

    # ── Processing ───────────────────────────────────────────────────
    X_train:              pd.DataFrame
    X_test:               pd.DataFrame
    y_train:              Optional[pd.DataFrame]   # 无监督任务为 None
    y_test:               Optional[pd.DataFrame]   # 无监督任务为 None
    feature_names:        list

    # ── EDA ──────────────────────────────────────────────────────────
    charts:               list
    eda_summary:          EDASummary
    modeling_hints:       ModelingHints

    # ── Model Routing ────────────────────────────────────────────────
    selected_algorithms:  list
    reasoning:            str

    # ── Modeling ─────────────────────────────────────────────────────
    model_results:        dict

    # ── Evaluation ───────────────────────────────────────────────────
    metrics:              dict
    best_model:           str

    # ── Reporting ────────────────────────────────────────────────────
    report_path:          str

    # ── 全局 ─────────────────────────────────────────────────────────
    logs:                 list


class ReportData(TypedDict, total=False):
    timestamp:            str
    file_name:            str
    target_column:        str
    task_type:            str
    user_level:           str
    data_scope:           str
    row_count:            int
    col_count:            int
    core_features:        str
    unit:                 str
    mean_target:          float
    quality_issues:       str
    top3_features:        str
    distribution_desc:    str
    layer_desc:           str
    abnormal_desc:        str
    key_charts:           str
    feature_1:            str
    feature_2:            str
    feature_3:            str
    feature_1_corr:       float
    feature_2_corr:       float
    feature_3_corr:       float
    accuracy:             float
    precision:            float
    recall:               float
    f1:                   float
    mae:                  float
    rmse:                 float
    r2:                   float
    silhouette:           float
    davies_bouldin:       float
    anomaly_ratio:        float
    best_model:           str
    user_intent_summary:  str
    reasoning:            str
    conclusion_1:         str
    conclusion_2:         str
    conclusion_3:         str
    short_suggest_1:      str
    short_suggest_2:      str
    long_suggest_1:       str
    long_suggest_2:       str
    data_risk:            str
    exec_risk:            str
