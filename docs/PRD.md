# AutoInsight — 产品需求文档（PRD）

> 版本：v0.2  
> 更新日期：2026-04-10  
> 状态：已对齐，待实现

---

## 1. 项目目标

构建一个**基于 LangGraph 的自动化数据挖掘 Agent 系统**。

用户只需提供：
1. 一个 CSV 数据文件
2. 一段自然语言描述其业务问题的 prompt

系统自动完成数据识别、清洗、探索分析、任务定性、建模评估、报告生成的完整流程。

**核心价值主张：用户只需要懂自己的业务问题，不需要懂数据分析。**

---

## 2. 目标受众

系统面向两类用户，两类用户使用同一套系统，报告输出语言风格根据 `user_level` 自适应。

| 用户类型 | 描述 | 示例 |
|---|---|---|
| **General 用户** | 有行业背景但无数据分析基础 | 购房者想通过房价数据判断是否适合投资 |
| **Expert 用户** | 有数据分析或机器学习背景 | 数据分析师使用系统加速分析流程 |

---

## 3. 系统输入 / 输出

### 输入

| 参数 | 类型 | 是否必填 | 说明 |
|---|---|---|---|
| `file` | CSV 文件路径 | **必填** | 待分析的表格数据集 |
| `prompt` | 自然语言字符串 | **必填** | 用户对业务问题的描述 |
| `target_column` | 列名字符串 | 可选 | 不填则由系统从 prompt + schema 自动推断 |
| `user_level` | `"general"` \| `"expert"` | 可选 | 默认 `"general"`，控制报告语言风格 |
| `--tune` | flag | 可选 | 开启轻量超参调优（RandomizedSearchCV，20 轮），默认 off |

### 数据约束

| 约束项 | 值 |
|---|---|
| 支持格式 | CSV only |
| 数据形态 | 表格数据（行=样本，列=特征） |
| 建议最大规模 | < 200,000 行 |
| 不支持 | 图像、音频、嵌套 JSON、图数据、实时流数据 |

### 输出

| 输出项 | 路径 | 说明 |
|---|---|---|
| EDA 图表 | `outputs/charts/*.png` | 分布图、热力图、特征关系图等 |
| 分析报告 | `outputs/reports/analysis_report.md` | LLM 生成的 Markdown 报告 |

---

## 4. 支持的任务类型

系统支持 5 种任务类型，由 `model_routing` 节点综合三个角度后自动判断。

| 任务类型 | 学习方式 | 是否需要 target | 典型用户 prompt |
|---|---|---|---|
| `classification` | 监督学习 | 是 | "判断这个客户是否会流失" |
| `regression` | 监督学习 | 是 | "预测这套房子的价格" |
| `clustering` | 无监督学习 | 否 | "把这些客户分成几类" |
| `anomaly_detection` | 无监督学习 | 否 | "找出这批订单里的异常交易" |
| `correlation_analysis` | 统计分析 | 否（可选） | "哪些因素最影响房价" |

> `correlation_analysis` 不走 Modeling 节点，直接进入 Reporting。

---

## 5. 任务定性与模型选择：三个角度

`model_routing` 节点综合以下三个角度做最终决策：

### 角度 1 — 用户 Prompt 意图

从用户自然语言中解读：
- 期望的分析目标（预测 / 分类 / 聚类 / 异常检测 / 关联分析）
- 业务背景
- 是否隐含目标列

### 角度 2 — EDA 结构化建议信号（`modeling_hints`）

EDA 节点除图表外，额外输出 `modeling_hints` 字段：

| 字段 | 类型 | 含义 | 对模型的影响 |
|---|---|---|---|
| `linearity_score` | float 0~1 | 特征与目标的线性相关程度 | 高 → 优先 Linear/Ridge；低 → 优先 tree 系列 |
| `imbalance_ratio` | float | 少数类占比（分类任务） | < 0.1 → 启用 class_weight 或提示 |
| `outlier_ratio` | float | 异常值占比 | 高 → 提示 anomaly_detection 或 robust 模型 |
| `high_corr_pairs` | list[str] | 高共线性特征对 | 建议 Ridge/Lasso 而非 LinearRegression |
| `sample_size` | int | 训练样本数 | < 500 → 简单模型；> 50k → tree/boost 更稳定 |
| `feature_count` | int | 特征数量 | 高维 → 正则化模型优先 |

### 角度 3 — 数据集性质（来自 Profiling）

| 信息 | 来源 | 对决策的影响 |
|---|---|---|
| 样本量 | `schema._meta.row_count` | 决定模型复杂度上限 |
| 缺失率 | `schema[col].null_rate` | 高缺失 → 优先 XGB（原生支持缺失） |
| 特征类型分布 | `schema[col].type` | 全类别特征需重点 encoding |
| 目标列性质 | unique 值数量 | 辅助判断 classification vs regression |

---

## 6. 候选模型清单

### 监督学习

| 任务 | 算法 | 备注 |
|---|---|---|
| `classification` | LogisticRegression | 线性基线 |
| | RandomForestClassifier | 非线性、抗噪声 |
| | XGBClassifier | 高性能 boost |
| | LGBMClassifier | 高性能 boost，大数据集首选 |
| | SVC | 小数据集表现好 |
| `regression` | LinearRegression | 线性基线 |
| | Ridge | 含正则化，共线性场景优先 |
| | Lasso | 高维稀疏场景 |
| | RandomForestRegressor | 非线性 |
| | XGBRegressor | 高性能 boost |
| | LGBMRegressor | 大数据集首选 |

### 无监督学习

| 任务 | 算法 | 关键参数 |
|---|---|---|
| `clustering` | KMeans | k 由肘部法则自动确定 |
| | DBSCAN | eps/min_samples 自适应 |
| | AgglomerativeClustering | 层次聚类 |
| `anomaly_detection` | IsolationForest | contamination 由 EDA outlier_ratio 推断，默认 0.05 |
| | LocalOutlierFactor | |
| | OneClassSVM | |

### 超参调优（`--tune` 开启时）

- 策略：RandomizedSearchCV，20 次迭代
- 范围：仅对最终选定的 best model 调优，不对全部候选模型调优
- 目的：在合理时间内（5~15x 基线耗时）获得 3~10% 的指标提升

---

## 7. 评估指标

| 任务类型 | 指标 | 最优模型选择依据 |
|---|---|---|
| `classification` | Accuracy, Precision, Recall, F1, AUC | 最高 F1 |
| `regression` | MAE, RMSE, R² | 最高 R² |
| `clustering` | Silhouette Score, Davies-Bouldin | 最高 Silhouette |
| `anomaly_detection` | 异常比例, 异常分数分布 | 无"最优"概念，输出所有结果 |
| `correlation_analysis` | 相关系数矩阵, VIF, Mutual Information | 无模型，纯统计输出 |

---

## 8. 系统架构与节点流程

### 节点顺序

```
用户输入（CSV + prompt）
        │
        ▼
[Node 1] Profiling
        识别 schema（列类型、缺失率）、数据集性质、质量问题
        │
        ▼
[Node 2] Intent Routing
        输入：prompt + schema
        输出：
          - task_type（粗判断，supervised / unsupervised / analytical）
          - target_column（推断或透传用户输入）
          - user_intent_summary（白话描述"用户想做什么"，供报告用）
        │
        ▼
[Node 3] Processing
        根据 task 类型走不同分支：
          - 监督学习 → X_train / X_test / y_train / y_test
          - 无监督学习 → X_only（不拆分 y）
        操作：缺失值填充、类别编码、数值标准化
        │
        ▼
[Node 4] EDA
        输出：
          - charts（图表文件路径列表）
          - eda_summary（文本摘要）
          - modeling_hints（结构化建议信号，见第5节）
        │
        ▼
[Node 5] Model Routing
        综合三个角度（prompt意图 + modeling_hints + 数据集性质）
        输出：
          - task_type（精确任务类型）
          - selected_algorithms（2~3个候选算法）
          - reasoning（选择理由，供报告引用）
        │
        ▼
[Node 6] Modeling
        训练所有 selected_algorithms
        若 --tune 开启，对候选模型执行 RandomizedSearchCV
        │
        ▼
[Node 7] Evaluation
        按 task_type 计算对应指标
        选出 best_model
        （correlation_analysis 跳过此节点）
        │
        ▼
[Node 8] Reporting
        综合所有节点输出，调用 LLM 生成 Markdown 报告
        报告语言风格由 user_level 控制：
          - general：业务语言，避免技术术语，给出行动建议
          - expert：包含完整指标表、模型对比、技术细节
        │
        ▼
outputs/charts/ + outputs/reports/analysis_report.md
```

### `correlation_analysis` 特殊路径

```
Profiling → Intent Routing → Processing → EDA → Model Routing
        → [跳过 Modeling / Evaluation]
        → Reporting（基于 EDA 统计结果直接生成报告）
```

---

## 9. 共享状态（AgentState）关键字段

| 字段 | 类型 | 写入节点 | 读取节点 |
|---|---|---|---|
| `user_query` | str | main.py | Intent Routing, Reporting |
| `file_path` | str | main.py | Profiling, Processing |
| `target_column` | str | main.py / Intent Routing | 全部节点 |
| `user_level` | str | main.py | Reporting |
| `schema` | dict | Profiling | Intent Routing, Processing, Model Routing |
| `quality_issues` | list[str] | Profiling, EDA | Reporting |
| `task_type` | str | Model Routing | Processing, Modeling, Evaluation, Reporting |
| `user_intent_summary` | str | Intent Routing | Reporting |
| `modeling_hints` | dict | EDA | Model Routing |
| `eda_summary` | EDASummary | EDA | Reporting |
| `charts` | list[str] | EDA | Reporting |
| `X_train/test` | pd.DataFrame | Processing | Modeling, EDA |
| `y_train/test` | pd.DataFrame | Processing | Modeling, Evaluation |
| `selected_algorithms` | list[str] | Model Routing | Modeling |
| `reasoning` | str | Model Routing | Reporting |
| `model_results` | dict | Modeling | Evaluation |
| `metrics` | dict | Evaluation | Reporting |
| `best_model` | str | Evaluation | Reporting |
| `report_path` | str | Reporting | main.py（打印） |
| `logs` | list[str] | 全部节点 | main.py（打印） |

---

## 10. 报告内容结构

### General 用户报告（业务语言）

1. 你的问题是什么（用户意图复述）
2. 数据概况（几行几列，数据质量）
3. 关键发现（EDA 洞察，通俗描述）
4. 分析结论（最优模型表现，翻译为业务含义）
5. 行动建议（短期 1~3 月 / 长期 3~12 月）
6. 风险提示

### Expert 用户报告（技术语言）

1. 数据集概览（schema、质量问题列表）
2. EDA 发现（图表引用、统计指标）
3. 模型选择理由（三角度 reasoning）
4. 模型对比表（全部候选模型的完整指标）
5. 最优模型分析（特征重要性、残差分布等）
6. 结论与建议

---

## 11. 本期范围（In Scope）

- [x] 五类任务类型：classification / regression / clustering / anomaly_detection / correlation_analysis
- [x] target_column 自动推断
- [x] EDA 结构化 modeling_hints 输出
- [x] 两阶段 Routing（Intent Routing + Model Routing）
- [x] 无监督任务的 Processing 分支（X_only）
- [x] user_level 控制报告语言风格
- [x] 可选超参调优（--tune）

## 12. 本期不做（Out of Scope）

- [ ] 时间序列分析
- [ ] 图像 / 音频 / 非结构化数据
- [ ] 实时流数据
- [ ] Web UI / 可视化界面
- [ ] 模型持久化 / 在线预测服务
- [ ] 多文件 / 多数据集 Join 分析
