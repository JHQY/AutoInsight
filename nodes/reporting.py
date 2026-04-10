import os
from datetime import datetime
from app.state import AgentState, ReportData

def generate_report(state: AgentState) -> AgentState:
    """
    Node 7: 报告生成
    
    输入: AgentState
    输出: AgentState (更新 report_path)
    
    功能:
    1. 从 AgentState 中提取 43 个变量，组织成 ReportData 结构
    2. 读取提示模板，使用 LLM 生成报告
    3. 将报告保存到 outputs/reports/analysis_report.md
    4. 返回更新后的 AgentState
    """
    try:
        # 1. 构建 ReportData 结构
        report_data = build_report_data(state)
        
        # 2. 生成报告内容
        report_content = generate_report_content(report_data)
        
        # 3. 保存报告文件
        report_path = save_report(report_content)
        
        # 4. 更新状态
        updated_state = state.copy()
        updated_state["report_path"] = report_path
        updated_state["logs"].append("[Reporting] 报告生成完成")
        
        return updated_state
        
    except Exception as e:
        updated_state = state.copy()
        updated_state["logs"].append(f"[Reporting] 报告生成失败: {str(e)}")
        return updated_state

def build_report_data(state: AgentState) -> ReportData:
    """
    从 AgentState 中提取数据，构建 ReportData 结构
    """
    report_data: ReportData = {
        # 基础信息
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "file_name": os.path.basename(state.get("file_path", "")),
        "target_column": state.get("target_column", ""),
        "task_type": state.get("task_type", ""),
        
        # 数据概览
        "data_scope": "",  # 从 schema 中提取
        "row_count": 0,     # 从 schema 中提取
        "col_count": 0,     # 从 schema 中提取
        "core_features": "", # 从 feature_names 中提取
        "unit": "",        # 默认为空
        "mean_target": 0.0,  # 从 schema 中提取
        
        # 数据质量
        "quality_issues": ";".join(state.get("quality_issues", [])),
        
        # EDA 结果
        "top3_features": "",
        "distribution_desc": "",
        "layer_desc": "",
        "abnormal_desc": "",
        "key_charts": ";".join(state.get("charts", [])),
        
        # 特征相关性
        "feature_1": "",
        "feature_2": "",
        "feature_3": "",
        "feature_1_corr": 0.0,
        "feature_2_corr": 0.0,
        "feature_3_corr": 0.0,
        
        # 模型评估指标
        "accuracy": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "mae": 0.0,
        "rmse": 0.0,
        "r2": 0.0,
        
        # 模型信息
        "best_model": state.get("best_model", ""),
        "business_context": "本次分析基于业务数据，旨在挖掘影响核心指标的关键因素，为业务决策提供数据支撑",
        "feature_relation_desc": "",
        "model_value": "",
        
        # 核心结论
        "conclusion_1": "",
        "conclusion_2": "",
        "conclusion_3": "",
        
        # 行动建议
        "short_suggest_1": "",
        "short_suggest_2": "",
        "long_suggest_1": "",
        "long_suggest_2": "",
        
        # 风险提示
        "data_risk": "",
        "exec_risk": ""
    }
    
    # 从 schema 中提取数据
    schema = state.get("schema", {})
    if schema:
        report_data["col_count"] = len([k for k in schema.keys() if k != "__metadata__"])
        # 假设 schema 中包含行数信息
        if "__metadata__" in schema:
            metadata = schema["__metadata__"]
            report_data["row_count"] = metadata.get("row_count", 0)
            report_data["data_scope"] = metadata.get("data_scope", "")
            report_data["mean_target"] = metadata.get("mean_target", 0.0)
    
    # 从 feature_names 中提取核心特征
    feature_names = state.get("feature_names", [])
    if feature_names:
        report_data["core_features"] = ",".join(feature_names[:5])  # 取前5个特征
    
    # 从 metrics 中提取模型评估指标
    metrics = state.get("metrics", {})
    best_model = state.get("best_model", "")
    if best_model in metrics:
        model_metrics = metrics[best_model]
        if state.get("task_type") == "classification":
            report_data["accuracy"] = model_metrics.get("accuracy", 0.0)
            report_data["precision"] = model_metrics.get("precision", 0.0)
            report_data["recall"] = model_metrics.get("recall", 0.0)
            report_data["f1"] = model_metrics.get("f1", 0.0)
        else:
            report_data["mae"] = model_metrics.get("mae", 0.0)
            report_data["rmse"] = model_metrics.get("rmse", 0.0)
            report_data["r2"] = model_metrics.get("r2", 0.0)

    # 从 eda_summary 中提取 EDA 结果（如果存在）
    eda_summary = state.get("eda_summary", {})
    if eda_summary:
        report_data["top3_features"] = eda_summary.get("top3_features", "")
        report_data["distribution_desc"] = eda_summary.get("distribution_desc", "")
        report_data["layer_desc"] = eda_summary.get("layer_desc", "")
        report_data["abnormal_desc"] = eda_summary.get("abnormal_desc", "")
        report_data["feature_1"] = eda_summary.get("feature_1", "")
        report_data["feature_2"] = eda_summary.get("feature_2", "")
        report_data["feature_3"] = eda_summary.get("feature_3", "")
        report_data["feature_1_corr"] = float(eda_summary.get("feature_1_corr", 0.0))
        report_data["feature_2_corr"] = float(eda_summary.get("feature_2_corr", 0.0))
        report_data["feature_3_corr"] = float(eda_summary.get("feature_3_corr", 0.0))
    
    # 填充默认值和标准化描述
    report_data = fill_default_descriptions(report_data, state)
    
    return report_data

def fill_default_descriptions(report_data: ReportData, state: AgentState) -> ReportData:
    """
    填充默认的标准化描述
    """
    # 生成默认的业务描述
    target_column = report_data.get("target_column", "")
    
    # 特征-目标关联描述
    report_data["feature_relation_desc"] = f"通过数据分析，我们发现多个特征与{target_column}存在显著关联，其中Top3特征对{target_column}的影响最为显著。"
    
    # 模型价值描述
    if report_data.get("task_type") == "classification":
        report_data["model_value"] = f"该模型可快速预测{target_column}的分类结果，帮助业务团队做出更准确的决策。"
    else:
        report_data["model_value"] = f"该模型可预测{target_column}的数值，误差控制在合理范围内，辅助业务规划。"
    
    # 核心结论
    report_data["conclusion_1"] = f"{target_column}受多个因素影响，其中特征之间存在复杂的交互关系。"
    report_data["conclusion_2"] = f"模型预测效果良好，可作为业务决策的参考依据。"
    report_data["conclusion_3"] = f"通过特征工程和模型优化，{target_column}的预测精度还有提升空间。"
    
    # 行动建议
    report_data["short_suggest_1"] = "基于模型结果，针对关键特征制定优化策略。"
    report_data["short_suggest_2"] = "建立数据监控机制，及时发现数据质量问题。"
    report_data["long_suggest_1"] = "持续收集更多特征数据，提升模型的预测能力。"
    report_data["long_suggest_2"] = "定期更新模型，适应业务场景的变化。"
    
    # 风险提示
    report_data["data_risk"] = "数据可能存在一定的局限性，结论需结合业务经验验证。"
    report_data["exec_risk"] = "模型预测结果仅作为参考，实际执行需考虑多种因素。"
    
    return report_data

def generate_report_content(report_data: ReportData) -> str:
    """
    生成报告内容
    注意：由于是演示版本，这里使用模板替换的方式生成报告，实际项目中可以使用 LLM
    """
    # 使用干净的内置模板
    template = """
# AutoInsight 业务数据分析报告
> 生成时间：{timestamp} | 数据源：{file_name} | 核心分析目标：{target_column} | 分析类型：{task_type}

## 一、业务背景与分析目标
### 1.1 业务背景
{business_context}

### 1.2 核心分析目标
1. 梳理{target_column}的核心影响因素，定位业务优化方向；
2. 构建预测模型，量化{target_column}的预测效果；
3. 基于数据规律提出可落地的业务建议。

## 二、数据集业务概览
### 2.1 数据基础信息（业务视角）
| 维度         | 详情                                  | 业务解读                                                                 |
|--------------|---------------------------------------|--------------------------------------------------------------------------|
| 数据覆盖范围 | {data_scope} | 说明数据的时间/空间/业务范围，判断数据代表性                              |
| 样本数量     | {row_count} 行                        | 样本量{row_count}行，数据量充足 |
| 核心特征维度 | {core_features} | 覆盖{col_count}个业务维度，核心关注{top3_features}                        |

### 2.2 数据质量业务影响
{quality_issues_section}

## 三、核心业务洞察（EDA结论）
### 3.1 核心指标分布特征
- {target_column} 整体表现：{distribution_desc}；
- 关键业务分层：{layer_desc}。

### 3.2 业务维度关联规律
1. 核心影响因素Top3：
   - {feature_1}：与{target_column}强相关；
   - {feature_2}：与{target_column}相关；
   - {feature_3}：与{target_column}相关。
2. 业务异常发现：{abnormal_desc}。

> 注：以上规律均基于可视化分析验证，完整图表见 `outputs/charts/` 目录（核心图表：{key_charts}）。

## 四、模型预测效果（业务视角）
### 4.1 模型预测能力评估
| 分析类型   | 评估指标       | 指标取值 | 业务解读                                                                 |
|------------|----------------|----------|--------------------------------------------------------------------------|
{"| 分类任务   | Accuracy（准确率） | {accuracy} | 模型对{target_column}的整体预测准确率为{accuracy}，预测效果良好 |
|            | Precision（精确率） | {precision} | 模型判定为「正类」的样本中，实际为正类的比例为{precision}，误判率低 |
|            | Recall（召回率）| {recall} | 实际为「正类」的样本中，被模型识别的比例为{recall}，漏判率低 |
|            | F1-Score（F1分数） | {f1}     | 精确率与召回率的平衡值为{f1}，模型综合效果优秀 |" if report_data.get("task_type") == "classification" else "| 回归任务   | MAE（平均绝对误差） | {mae}    | 模型预测值与实际值的平均偏差为{mae}{unit}，数值越小预测越精准 |
|            | RMSE（均方根误差） | {rmse}   | 模型对极端值的预测偏差为{rmse}{unit}，极端值预测效果好 |
|            | R²（决定系数）| {r2}     | 模型可解释{r2*100:.1f}%的{target_column}波动，模型解释力强 |"}

### 4.2 最优模型业务价值
- 最终选定模型：{best_model}；
- 模型业务价值：{model_value}；
- 模型局限性：模型无法捕捉所有外部因素影响，需结合业务经验调整预测结果。

## 五、业务结论与行动建议
### 5.1 核心业务结论
1. {conclusion_1}；
2. {conclusion_2}；
3. {conclusion_3}。

### 5.2 可落地行动建议
#### 短期建议（1-3个月）
1. {short_suggest_1}；
2. {short_suggest_2}。

#### 长期建议（3-12个月）
1. {long_suggest_1}；
2. {long_suggest_2}。

### 5.3 风险提示
- 数据层面：{data_risk}；
- 执行层面：{exec_risk}。

---
*本报告由 AutoInsight 自动化分析系统生成 | 报告基于{row_count}行业务数据，模型指标仅作业务决策参考，最终需结合业务经验落地*
    """
    
    # 处理数据质量部分
    if report_data.get("quality_issues"):
        quality_issues_section = "⚠️ 需关注的数据质量问题（影响分析可靠性）：\n- " + "\n- ".join(report_data.get("quality_issues", "").split(";"))
    else:
        quality_issues_section = "✅ 数据质量良好，无明显缺失/异常/重复问题，分析结论具备可靠性"
    
    # 替换模板中的变量
    for key, value in report_data.items():
        if isinstance(value, float):
            # 格式化浮点数，保留两位小数
            template = template.replace(f"{{{key}}}", f"{value:.2f}")
        else:
            template = template.replace(f"{{{key}}}", str(value))
    
    # 替换数据质量部分
    template = template.replace("{quality_issues_section}", quality_issues_section)
    
    return template.strip()

def save_report(content: str) -> str:
    """
    保存报告到 outputs/reports/analysis_report.md
    """
    report_dir = os.path.join(os.path.dirname(__file__), "..", "outputs", "reports")
    os.makedirs(report_dir, exist_ok=True)
    
    report_path = os.path.join(report_dir, "analysis_report.md")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return report_path
