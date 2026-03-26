# AutoInsight — Technical Documentation

## Overview

AutoInsight is a **LangGraph-based automated data mining agent system** that accepts any tabular CSV dataset and autonomously executes a full analytics pipeline: profiling → preprocessing → EDA → modeling → evaluation → report generation.

---

## Architecture

```
┌─────────────────────────────────────┐
│           Interface Layer           │
│  (CLI / main.py — CSV file input)   │
└────────────────┬────────────────────┘
                 │
┌────────────────▼────────────────────┐
│    Agent Orchestration Layer        │
│    LangGraph (graph.py + state.py)  │
│  - Node orchestration               │
│  - Shared state management          │
│  - Conditional routing              │
└────────────────┬────────────────────┘
                 │
┌────────────────▼────────────────────┐
│       Skill Execution Layer         │
│  nodes/: 7 LangGraph node modules   │
│  tools/: reusable helper utilities  │
└─────────────────────────────────────┘
```

---

## Workflow

```
START
  │
  ▼
Load CSV
  │
  ▼
[Node 1] Dataset Profiling    — detect schema, types, nulls, duplicates
  │
  ▼
[Node 4] Task Routing         — classify task type (classification / regression)
  │
  ▼
[Node 2] Data Processing      — clean, encode, scale, train/test split
  │
  ▼
[Node 3] EDA                  — generate distribution, correlation, feature plots
  │
  ▼
[Node 5] Modeling             — train candidate models
  │
  ▼
[Node 6] Evaluation           — compute metrics, select best model
  │
  ▼
[Node 7] Report Generation    — LLM-generated markdown report
  │
  ▼
END → outputs/charts/ + outputs/reports/analysis_report.md
```

---

## Shared State Schema

Defined in `app/state.py` as a TypedDict passed between every LangGraph node.

| Field | Type | Description |
|---|---|---|
| `user_query` | str | Optional user instruction |
| `file_path` | str | Path to input CSV |
| `schema` | dict | Column names, inferred types, stats |
| `target_column` | str | Column to predict |
| `task_type` | str | `"classification"` or `"regression"` |
| `quality_issues` | list | Detected data quality problems |
| `charts` | list[str] | Paths to saved chart files |
| `model_results` | dict | Per-model train/predict outputs |
| `best_model` | str | Name of selected best model |
| `metrics` | dict | Evaluation metrics table |
| `report_path` | str | Path to final report |
| `logs` | list[str] | Step-level audit log |

---

## Node Modules

### Node 1 — `nodes/profiling.py` (Dataset Profiling)

**Input:** `file_path`
**Output:** `schema`, `quality_issues`

Responsibilities:
- Detect column types: numeric, categorical, datetime
- Compute missing value rates per column
- Count duplicate rows
- Produce schema summary dict

### Node 2 — `nodes/processing.py` (Data Processing)

**Input:** `file_path`, `schema`, `target_column`
**Output:** processed train/test DataFrames, preprocessing log

Responsibilities:
- Fill missing values (median for numeric, mode for categorical)
- Encode categorical columns (LabelEncoder / OneHotEncoder)
- Optional numeric scaling (StandardScaler)
- Stratified train/test split (80/20)

### Node 3 — `nodes/eda.py` (EDA)

**Input:** processed DataFrame, `target_column`
**Output:** `charts` (list of saved file paths), `quality_issues` append

Chart types generated:
- Distribution plots (numeric histograms)
- Correlation heatmap
- Feature–target relationship plots
- Categorical column distribution bar charts

Saved to: `outputs/charts/`

### Node 4 — `nodes/routing.py` (Task Routing)

**Input:** `schema`, `target_column`
**Output:** `task_type`

Logic:
```python
if target_column.nunique() < threshold:   # default threshold = 20
    task_type = "classification"
else:
    task_type = "regression"
```

### Node 5 — `nodes/modeling.py` (Modeling)

**Input:** training data, `task_type`
**Output:** `model_results`

| task_type | Models trained |
|---|---|
| classification | Logistic Regression, Random Forest Classifier |
| regression | Linear Regression, Random Forest Regressor |

### Node 6 — `nodes/evaluation.py` (Evaluation)

**Input:** `model_results`, ground truth labels
**Output:** `metrics`, `best_model`

| task_type | Metrics |
|---|---|
| classification | Accuracy, Precision, Recall, F1 |
| regression | MAE, RMSE, R² |

Best model selection: highest F1 (classification) or highest R² (regression).

### Node 7 — `nodes/reporting.py` (Report Generation)

**Input:** `schema`, EDA findings, `metrics`, `best_model`
**Output:** `report_path` → `outputs/reports/analysis_report.md`

Uses an LLM (Claude via Anthropic SDK) with a structured prompt from `prompts/report_prompt.txt` to generate a natural-language Markdown report covering:
- Dataset overview
- Data quality issues
- EDA findings
- Model comparison table
- Final conclusions and recommendations

#### Report Output Data Model

Node 7 生成报告时需要从 AgentState 中提取 43 个变量，按以下结构组织：

| 变量分类 | 变量名 | 数据类型 | 来源节点 | 说明 |
|---|---|---|---|---|
| **基础信息** | timestamp | str | 系统 | 报告生成时间，格式：YYYY-MM-DD HH:MM:SS |
| | file_name | str | Node1 | 输入 CSV 文件名（不含路径） |
| | target_column | str | 用户输入 | 目标列名称 |
| | task_type | str | Node4 | 任务类型：`classification` \| `regression` |
| **数据概览** | data_scope | str | Node1 | 数据基础范围（从 CSV 元数据提取） |
| | row_count | int | Node1 | 数据集总行数 |
| | col_count | int | Node1 | 数据集总列数（含目标列） |
| | core_features | str | Node1+3 | 竖线分隔的特征列名称 |
| | unit | str | Node1 | 目标列单位（无则为空） |
| | mean_target | float | Node1 | 目标列均值 |
| **质量指标** | quality_issues | str | Node1+3 | 分号分隔的数据质量问题描述 |
| **EDA结果** | top3_features | str | Node3 | TOP3相关特征名（按相关系数排序，逗号分隔） |
| | distribution_desc | str | Node3 | 目标列统计指标（中位数、均值、分位数等） |
| | layer_desc | str | Node3 | 目标列按核心特征分组的均值 |
| | abnormal_desc | str | Node3 | 数据异常值统计（无则为空） |
| | key_charts | str | str | 生成的图表文件路径（分号分隔） |
| **特征相关性** | feature_1 | str | Node3 | TOP1 特征名 |
| | feature_2 | str | Node3 | TOP2 特征名 |
| | feature_3 | str | Node3 | TOP3 特征名 |
| | feature_1_corr | float | Node3 | 特征1与目标列的相关系数 |
| | feature_2_corr | float | Node3 | 特征2与目标列的相关系数 |
| | feature_3_corr | float | Node3 | 特征3与目标列的相关系数 |
| **分类指标**（task_type=classification） | accuracy | float | Node6 | 准确率 |
| | precision | float | Node6 | 精确率 |
| | recall | float | Node6 | 召回率 |
| | f1 | float | F1 分数 |
| **回归指标**（task_type=regression） | mae | float | Node6 | 平均绝对误差 |
| | rmse | float | Node6 | 均方根误差 |
| | r2 | float | Node6 | 决定系数 |
| **模型信息** | best_model | str | Node6 | 最优模型名称 |
| | business_context | str | Node6 | 标准化业务背景描述 |
| | feature_relation_desc | str | Node6 | 特征-目标关联标准化描述 |
| | model_value | str | Node6 | 模型价值标准化描述 |
| **结论** | conclusion_1 | str | Node6 | 核心结论1（标准化） |
| | conclusion_2 | str | Node6 | 核心结论2（标准化） |
| | conclusion_3 | str | Node6 | 核心结论3（标准化） |
| **短期建议** | short_suggest_1 | str | Node6 | 短期建议1（1-3个月） |
| | short_suggest_2 | str | Node6 | 短期建议2（1-3个月） |
| **长期建议** | long_suggest_1 | str | Node6 | 长期建议1（3-12个月） |
| | long_suggest_2 | str | Node6 | 长期建议2（3-12个月） |
| **风险提示** | data_risk | str | Node6 | 数据层面风险描述 |
| | exec_risk | str | Node6 | 执行层面风险描述 |

完整详细说明见 `docs/DATA_MODEL.md`

---

## Tool Modules

| File | Purpose |
|---|---|
| `tools/data_tools.py` | DataFrame I/O, type inference, split utilities |
| `tools/plot_tools.py` | Chart generation helpers (matplotlib / seaborn) |
| `tools/model_tools.py` | Model training wrappers, metric calculators |

---

## Project Directory Structure

```
AutoInsight/
│
├── app/
│   ├── graph.py          # LangGraph graph construction & compilation
│   ├── state.py          # AgentState TypedDict definition
│   └── main.py           # Entry point — CLI argument parsing, graph.invoke()
│
├── nodes/
│   ├── profiling.py      # Skill 1: Dataset Profiling
│   ├── processing.py     # Skill 2: Data Processing
│   ├── eda.py            # Skill 3: EDA
│   ├── routing.py        # Skill 4: Task Routing
│   ├── modeling.py       # Skill 5: Modeling
│   ├── evaluation.py     # Skill 6: Evaluation
│   └── reporting.py      # Skill 7: Report Generation
│
├── tools/
│   ├── data_tools.py
│   ├── plot_tools.py
│   └── model_tools.py
│
├── prompts/
│   └── report_prompt.txt
│
├── outputs/
│   ├── charts/
│   └── reports/
│
├── pyproject.toml        # uv-managed project config
├── TECHNICAL.md          # This document
└── README.md
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Agent orchestration | [LangGraph](https://github.com/langchain-ai/langgraph) |
| LLM (report generation) | Anthropic Claude (claude-sonnet-4-6) |
| Data processing | pandas, scikit-learn |
| Visualization | matplotlib, seaborn |
| Environment management | uv |
| Language | Python ≥ 3.11 |

---

## Data Constraints

| Constraint | Value |
|---|---|
| Supported format | CSV only |
| Data shape | Tabular (rows = samples, cols = features) |
| Recommended max size | < 200,000 rows |
| Unsupported | Images, audio, nested JSON, graph data, streaming |

---

## Usage

```bash
# Activate the uv environment
uv run python app/main.py --file path/to/data.csv --target target_column_name
```

Outputs written to:
- `outputs/charts/*.png`
- `outputs/reports/analysis_report.md`
