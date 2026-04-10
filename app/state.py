from typing import TypedDict, Any
import pandas as pd

class EDASummary(TypedDict, total=False):
    """
    EDA 节点的结构化输出，供 Reporting 直接消费。
    """
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

class AgentState(TypedDict):

    # ── 入口字段（由 main.py 初始化）────────────────────────────────
    user_query:           str        # 用户自然语言任务描述，如"预测房价走势"
    file_path:            str        # 输入 CSV 的路径
    target_column:        str        # 预测目标列名

    # ── Profiling Skill ──────────────────────────────────────────────
    schema:               dict
    # 示例：
    # {
    #   "age":    {"type": "numeric",     "null_rate": 0.02},
    #   "gender": {"type": "categorical", "null_rate": 0.00},
    #   "price":  {"type": "numeric",     "null_rate": 0.01}
    # }

    quality_issues:       list[str]
    # 示例：["age 列缺失率 2%", "发现 12 行重复数据"]

    # ── Routing + Modeling + Evaluation Skill ────────────────────────
    task_type:            str        # "classification" | "regression"
    selected_algorithms:  list[str]  # 示例：["RandomForestRegressor", "XGBRegressor"]
    reasoning:            str        # LLM 选择算法的理由，供 Reporting 引用

    model_results:        dict
    # 示例：
    # {
    #   "RandomForestRegressor": {
    #       "y_pred": [3.2, 4.1, ...],   # list 或 ndarray，X_test 上的预测值
    #       "model":  <fitted model>      # 模型对象，用于 feature importance 等
    #   }
    # }

    metrics:              dict
    # classification 示例：
    # {
    #   "LogisticRegression":     {"accuracy": 0.88, "precision": 0.87, "recall": 0.86, "f1": 0.87},
    #   "RandomForestClassifier": {"accuracy": 0.93, "precision": 0.92, "recall": 0.91, "f1": 0.91}
    # }
    # regression 示例：
    # {
    #   "Ridge":                  {"mae": 0.21, "rmse": 0.34, "r2": 0.89},
    #   "RandomForestRegressor":  {"mae": 0.18, "rmse": 0.29, "r2": 0.93}
    # }

    best_model:           str        # 示例："RandomForestRegressor"

    # ── Processing Skill ─────────────────────────────────────────────
    # ⚠️ 待确认：类型是 pd.DataFrame 还是 np.ndarray？
    # 建议保留 pd.DataFrame（含列名），EDA 和 Modeling 均可直接使用
    X_train:              pd.DataFrame
    X_test:               pd.DataFrame
    y_train:              pd.DataFrame
    y_test:               pd.DataFrame

    # ⚠️ 待确认：是否需要单独存列名，还是直接从 X_train.columns 取？
    feature_names:        list[str]  # 处理后的特征列名，供 EDA / Reporting 使用

    # ── EDA Skill（含所有出图）───────────────────────────────────────
    charts:               list[str]
    # 所有图表的文件路径，包括：
    #   - 分布图 / 相关性热力图 / 特征-目标关系图（EDA 阶段）
    #   - confusion matrix / residual plot（评估阶段，由 EDA skill 负责出图）
    # 示例：["outputs/charts/hist_age.png", "outputs/charts/confusion_matrix.png"]
    eda_summary:          EDASummary
    # 结构化 EDA 文本+指标结果，供 reporting.py 填充报告字段

    # ── Reporting Skill ───────────────────────────────────────────────
    report_path:          str        # 示例："outputs/reports/analysis_report.md"
    ## report 输入需求：

    # ── 全局 ──────────────────────────────────────────────────────────
    logs:                 list[str]  # 每个 skill 追加，格式："[skill名] 描述"


class ReportData(TypedDict, total=False):
    """
    Node 7 报告生成所需的完整数据结构，共 43 个变量。

    完整文档见：docs/DATA_MODEL.md
    """

    # ── 第一部分：基础信息（4 个变量）──────────────────────────────
    timestamp:            str        # 报告生成时间，格式：YYYY-MM-DD HH:MM:SS
    file_name:            str        # 输入 CSV 文件名（不含路径）
    target_column:        str        # 目标列名称
    task_type:            str        # "classification" | "regression"

    # ── 第二部分：数据概览（6 个变量）──────────────────────────────
    data_scope:           str        # 数据覆盖范围（时间/空间/业务维度）
    row_count:            int        # 样本总行数
    col_count:            int        # 特征总列数
    core_features:        str        # 逗号分隔的特征列名
    unit:                 str        # 目标列单位（无则为空）
    mean_target:          float      # 目标列均值

    # ── 第三部分：数据质量指标（1 个变量）────────────────────────
    quality_issues:       str        # 分号分隔的数据质量问题（无则为空）

    # ── 第四部分：EDA 结果（6 个变量）──────────────────────────────
    top3_features:        str        # 逗号分隔的 TOP3 相关特征名
    distribution_desc:    str        # 目标列统计指标（中位数、均值等）
    layer_desc:           str        # 按核心特征分组的均值
    abnormal_desc:        str        # 异常值统计（无则为空）
    key_charts:           str        # 分号分隔的关键图表路径

    # ── 第五部分：特征相关性（6 个变量）────────────────────────────
    feature_1:            str        # TOP1 特征名
    feature_2:            str        # TOP2 特征名
    feature_3:            str        # TOP3 特征名
    feature_1_corr:       float      # 特征 1 与目标列的相关系数
    feature_2_corr:       float      # 特征 2 与目标列的相关系数
    feature_3_corr:       float      # 特征 3 与目标列的相关系数

    # ── 第六部分：模型评估指标（7 个变量）─────────────────────────
    # 分类任务
    accuracy:             float      # 准确率（task_type="classification"）
    precision:            float      # 精确率
    recall:               float      # 召回率
    f1:                   float      # F1 分数
    # 回归任务
    mae:                  float      # 平均绝对误差（task_type="regression"）
    rmse:                 float      # 均方根误差
    r2:                   float      # 决定系数

    # ── 第七部分：模型信息（4 个变量）──────────────────────────────
    best_model:           str        # 最优模型名称
    business_context:     str        # 业务背景描述
    feature_relation_desc: str       # 特征-目标关联描述
    model_value:          str        # 模型价值描述

    # ── 第八部分：核心结论（3 个变量）──────────────────────────────
    conclusion_1:         str        # 核心结论 1
    conclusion_2:         str        # 核心结论 2
    conclusion_3:         str        # 核心结论 3

    # ── 第九部分：行动建议（4 个变量）──────────────────────────────
    short_suggest_1:      str        # 短期建议 1（1-3 个月）
    short_suggest_2:      str        # 短期建议 2（1-3 个月）
    long_suggest_1:       str        # 长期建议 1（3-12 个月）
    long_suggest_2:       str        # 长期建议 2（3-12 个月）

    # ── 第十部分：风险提示（2 个变量）──────────────────────────────
    data_risk:            str        # 数据层面风险
    exec_risk:            str        # 执行层面风险
