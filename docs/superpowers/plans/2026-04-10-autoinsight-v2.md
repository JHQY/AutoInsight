# AutoInsight v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在现有代码基础上完成 AutoInsight v2，支持五类任务（classification / regression / clustering / anomaly_detection / correlation_analysis），实现三角度自动模型选择、target 自动推断、user_level 报告风格自适应，并串联完整 LangGraph pipeline。

**Architecture:** 将原单一 routing 节点拆分为 IntentRouting（早期意图理解）和 ModelRouting（三角度精选）；EDA 新增结构化 modeling_hints 输出；Processing 按 task_category 分为有监督/无监督两条分支；LangGraph graph 对 correlation_analysis 设条件边跳过 Modeling+Evaluation。

**Tech Stack:** Python 3.11+, LangGraph, langchain-anthropic (claude-haiku-4-5 用于 intent_routing), langchain-openai + DeepSeek API (用于 model_routing), anthropic SDK (用于 reporting), pandas, scikit-learn, xgboost, lightgbm, pytest, uv

---

## 文件变更总表

### 新建
| 文件 | 职责 |
|---|---|
| `nodes/profiling.py` | 读取 CSV，输出 schema + quality_issues |
| `nodes/intent_routing.py` | LLM 推断 target、task_category、task_type（粗）、user_intent_summary |
| `nodes/model_routing.py` | LLM 三角度综合，输出精确 task_type + selected_algorithms + reasoning |
| `app/graph.py` | LangGraph StateGraph 编排，含 correlation_analysis 条件边 |
| `app/main.py` | CLI 入口，argparse，初始化 state，调用 graph.invoke() |
| `tests/test_profiling.py` | profiling_node 单元测试 |
| `tests/test_intent_routing.py` | intent_routing_node 单元测试（mock LLM） |
| `tests/test_model_routing.py` | model_routing_node 单元测试（mock LLM） |
| `tests/test_processing_unsupervised.py` | data_processing 无监督分支测试 |
| `tests/test_eda_hints.py` | EDA modeling_hints 输出测试 |
| `tests/test_integration.py` | 端到端冒烟测试 |

### 修改
| 文件 | 改动 |
|---|---|
| `app/state.py` | 新增 ModelingHints TypedDict；AgentState 新增 user_level / task_category / user_intent_summary / modeling_hints；y_train/y_test 改为 Optional |
| `nodes/processing.py` | 加无监督分支：task_category=="unsupervised"/"analytical" 时不拆 y |
| `nodes/eda.py` | 新增 `_compute_modeling_hints()` 并写入 state["modeling_hints"] |
| `tools/model_tools.py` | 注册 clustering + anomaly 算法；新增 compute_clustering_metrics / compute_anomaly_metrics / tune_model |
| `nodes/modeling.py` | 支持无监督算法（无 y_train）；支持 --tune |
| `nodes/evaluation.py` | 加 clustering / anomaly_detection 指标分支；correlation_analysis 直接返回空 metrics |
| `nodes/reporting.py` | 读取 user_level，选择报告模板 |
| `prompts/report_prompt.txt` | 拆为 general / expert 两段模板 |
| `nodes/__init__.py` | 更新 export 列表 |

### 废弃（保留文件，不再被 graph 引用）
| 文件 | 说明 |
|---|---|
| `nodes/routing.py` | 被 intent_routing + model_routing 取代，不删除以保留 git 历史 |

---

## Task 1 — AgentState Schema 更新

**Files:**
- Modify: `app/state.py`

- [ ] **Step 1: 更新 state.py**

用以下内容完整替换 `app/state.py`：

```python
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
```

- [ ] **Step 2: 验证导入正常**

```bash
cd P:/M-DATA/AutoInsight
uv run python -c "from app.state import AgentState, ModelingHints, EDASummary, ReportData; print('OK')"
```

期望输出：`OK`

- [ ] **Step 3: Commit**

```bash
git add app/state.py
git commit -m "feat: expand AgentState with ModelingHints, task_category, user_level, Optional y fields"
```

---

## Task 2 — Profiling Node

**Files:**
- Create: `nodes/profiling.py`
- Create: `tests/test_profiling.py`

- [ ] **Step 1: 写失败测试**

创建 `tests/test_profiling.py`：

```python
import os
import pytest
import pandas as pd
from nodes.profiling import profiling_node


@pytest.fixture
def sample_csv(tmp_path):
    df = pd.DataFrame({
        "age":    [25, 30, None, 40, 25],
        "gender": ["M", "F", "M", "F", "M"],
        "price":  [100.0, 200.0, 150.0, 300.0, 100.0],
    })
    path = str(tmp_path / "sample.csv")
    df.to_csv(path, index=False)
    return path


def _state(csv_path, target="price"):
    return {"file_path": csv_path, "target_column": target, "logs": []}


def test_schema_has_all_columns(sample_csv):
    result = profiling_node(_state(sample_csv))
    assert "age" in result["schema"]
    assert "gender" in result["schema"]
    assert "price" in result["schema"]


def test_column_types_detected(sample_csv):
    result = profiling_node(_state(sample_csv))
    assert result["schema"]["age"]["type"] == "numeric"
    assert result["schema"]["gender"]["type"] == "categorical"


def test_null_rate_detected(sample_csv):
    result = profiling_node(_state(sample_csv))
    assert result["schema"]["age"]["null_rate"] == pytest.approx(0.2, abs=0.01)
    assert result["schema"]["gender"]["null_rate"] == 0.0


def test_quality_issues_includes_nulls(sample_csv):
    result = profiling_node(_state(sample_csv))
    assert any("age" in issue for issue in result["quality_issues"])


def test_quality_issues_includes_duplicates(sample_csv):
    # age=25,gender=M,price=100 appears twice
    result = profiling_node(_state(sample_csv))
    assert any("重复" in issue for issue in result["quality_issues"])


def test_meta_row_col_count(sample_csv):
    result = profiling_node(_state(sample_csv))
    meta = result["schema"]["_meta"]
    assert meta["row_count"] == 5
    assert meta["col_count"] == 3


def test_appends_log(sample_csv):
    result = profiling_node(_state(sample_csv))
    assert any("[profiling]" in log for log in result["logs"])


def test_empty_target_column(sample_csv):
    """无监督任务可能不传 target_column。"""
    result = profiling_node({"file_path": sample_csv, "target_column": "", "logs": []})
    assert "schema" in result
```

- [ ] **Step 2: 运行确认失败**

```bash
cd P:/M-DATA/AutoInsight
uv run pytest tests/test_profiling.py -v 2>&1 | head -20
```

期望：`ImportError: cannot import name 'profiling_node'`

- [ ] **Step 3: 实现 profiling_node**

创建 `nodes/profiling.py`：

```python
import pandas as pd
from app.state import AgentState


def profiling_node(state: AgentState) -> dict:
    file_path = state["file_path"]
    target_col = state.get("target_column", "")

    df = pd.read_csv(file_path)

    schema: dict = {}
    for col in df.columns:
        null_rate = round(float(df[col].isnull().mean()), 4)
        if pd.api.types.is_numeric_dtype(df[col]):
            col_type = "numeric"
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            col_type = "datetime"
        else:
            col_type = "categorical"
        schema[col] = {"type": col_type, "null_rate": null_rate}

    feature_cols = [c for c in df.columns if c != target_col]
    target_mean = 0.0
    if target_col and target_col in df.columns:
        try:
            target_mean = round(float(pd.to_numeric(df[target_col], errors="coerce").mean()), 4)
        except Exception:
            pass

    schema["_meta"] = {
        "file_name": file_path.replace("\\", "/").split("/")[-1],
        "row_count": len(df),
        "col_count": len(df.columns),
        "data_scope": f"{len(df)} rows × {len(df.columns)} columns",
        "unit": "",
        "mean_target": target_mean,
        "core_features": ",".join(feature_cols),
    }

    quality_issues: list[str] = []
    dup_count = int(df.duplicated().sum())
    if dup_count > 0:
        quality_issues.append(f"发现 {dup_count} 行重复数据")
    for col in df.columns:
        nr = schema[col]["null_rate"]
        if nr > 0:
            quality_issues.append(f"{col} 列缺失率 {nr * 100:.1f}%")

    return {
        "schema": schema,
        "quality_issues": quality_issues,
        "logs": list(state.get("logs", [])) + [
            f"[profiling] {len(df)} 行 × {len(df.columns)} 列，质量问题 {len(quality_issues)} 条"
        ],
    }
```

- [ ] **Step 4: 运行测试确认通过**

```bash
uv run pytest tests/test_profiling.py -v
```

期望：所有测试 PASS

- [ ] **Step 5: Commit**

```bash
git add nodes/profiling.py tests/test_profiling.py
git commit -m "feat: add profiling_node — schema detection, null rates, duplicate check"
```

---

## Task 3 — Intent Routing Node

**Files:**
- Create: `nodes/intent_routing.py`
- Create: `tests/test_intent_routing.py`

- [ ] **Step 1: 写失败测试**

创建 `tests/test_intent_routing.py`：

```python
from unittest.mock import MagicMock, patch
import pytest
from nodes.intent_routing import intent_routing_node, IntentOutput


def _schema():
    return {
        "age":    {"type": "numeric",     "null_rate": 0.0},
        "income": {"type": "numeric",     "null_rate": 0.0},
        "churn":  {"type": "categorical", "null_rate": 0.0},
        "_meta":  {"row_count": 500, "col_count": 3,
                   "data_scope": "500 rows × 3 columns",
                   "unit": "", "mean_target": 0.4,
                   "core_features": "age,income", "file_name": "test.csv"},
    }


def _mock_llm(target_column, task_category, task_type, summary="test summary"):
    out = IntentOutput(
        target_column=target_column,
        task_category=task_category,
        task_type=task_type,
        user_intent_summary=summary,
    )
    mock_structured = MagicMock()
    mock_structured.invoke.return_value = out
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured
    return mock_llm


@patch("nodes.intent_routing.ChatAnthropic")
def test_infers_target_when_not_provided(mock_chat):
    mock_chat.return_value = _mock_llm("churn", "supervised", "classification")
    result = intent_routing_node({
        "user_query": "预测客户是否流失",
        "target_column": "",
        "schema": _schema(),
        "logs": [],
    })
    assert result["target_column"] == "churn"
    assert result["task_category"] == "supervised"
    assert result["task_type"] == "classification"


@patch("nodes.intent_routing.ChatAnthropic")
def test_preserves_user_provided_target(mock_chat):
    mock_chat.return_value = _mock_llm("income", "supervised", "regression")
    result = intent_routing_node({
        "user_query": "预测收入",
        "target_column": "income",
        "schema": _schema(),
        "logs": [],
    })
    assert result["target_column"] == "income"


@patch("nodes.intent_routing.ChatAnthropic")
def test_unsupervised_returns_empty_target(mock_chat):
    mock_chat.return_value = _mock_llm("", "unsupervised", "clustering")
    result = intent_routing_node({
        "user_query": "把客户分成几类",
        "target_column": "",
        "schema": _schema(),
        "logs": [],
    })
    assert result["task_category"] == "unsupervised"
    # target_column 不在 updates 里（不覆盖原空值）
    assert "target_column" not in result or result.get("target_column") == ""


@patch("nodes.intent_routing.ChatAnthropic")
def test_returns_user_intent_summary(mock_chat):
    mock_chat.return_value = _mock_llm("churn", "supervised", "classification",
                                       "判断客户是否会流失以支持运营决策")
    result = intent_routing_node({
        "user_query": "预测流失",
        "target_column": "",
        "schema": _schema(),
        "logs": [],
    })
    assert isinstance(result["user_intent_summary"], str)
    assert len(result["user_intent_summary"]) > 0


@patch("nodes.intent_routing.ChatAnthropic")
def test_appends_log(mock_chat):
    mock_chat.return_value = _mock_llm("churn", "supervised", "classification")
    result = intent_routing_node({
        "user_query": "",
        "target_column": "",
        "schema": _schema(),
        "logs": ["[profiling] done"],
    })
    assert len(result["logs"]) == 2
    assert any("[intent_routing]" in log for log in result["logs"])
```

- [ ] **Step 2: 运行确认失败**

```bash
uv run pytest tests/test_intent_routing.py -v 2>&1 | head -10
```

- [ ] **Step 3: 实现 intent_routing_node**

创建 `nodes/intent_routing.py`：

```python
from pydantic import BaseModel
from langchain_anthropic import ChatAnthropic
from app.state import AgentState


class IntentOutput(BaseModel):
    target_column:       str   # 推断结果；无监督返回空字符串
    task_category:       str   # "supervised" | "unsupervised" | "analytical"
    task_type:           str   # 五类之一（粗判断）
    user_intent_summary: str   # 白话一句话，供报告使用


_TASK_TYPES = """
task_category options:
  "supervised"   — user wants to predict/classify a specific column (needs target)
  "unsupervised" — user wants clustering or anomaly detection (no target needed)
  "analytical"   — user wants correlation or association analysis (no model trained)

task_type options (pick exactly one):
  "classification"      — predicting a categorical/binary outcome
  "regression"          — predicting a continuous numeric outcome
  "clustering"          — grouping data into natural clusters
  "anomaly_detection"   — finding unusual or suspicious data points
  "correlation_analysis"— analyzing which variables are related
"""


def intent_routing_node(state: AgentState) -> dict:
    schema   = state["schema"]
    query    = state.get("user_query", "")
    provided = state.get("target_column", "")
    meta     = schema.get("_meta", {})

    col_lines = [
        f"  - {col}: type={info['type']}, null_rate={info['null_rate']}"
        for col, info in schema.items()
        if col != "_meta"
    ]

    prompt = f"""You are a data analysis assistant helping a non-technical user.

Dataset overview:
  Rows: {meta.get('row_count', '?')}
  Columns: {meta.get('col_count', '?')}
Column details:
{chr(10).join(col_lines)}

User's goal (in their own words): {query or "(not provided)"}
User-specified target column: {provided or "(not specified — infer if supervised)"}

{_TASK_TYPES}

Instructions:
1. Determine task_category and task_type based on the user's goal.
2. If task_category is "supervised" and target_column was not specified,
   infer the most likely target column from schema + user goal.
   If task_category is "unsupervised" or "analytical", set target_column to "".
3. Write user_intent_summary as ONE plain-language sentence (no ML jargon)
   describing what the user wants to achieve. Write in the same language
   as the user's goal (Chinese if user wrote Chinese, English otherwise).
"""

    llm = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0)
    structured_llm = llm.with_structured_output(IntentOutput)
    result: IntentOutput = structured_llm.invoke(prompt)

    updates: dict = {
        "task_category":       result.task_category,
        "task_type":           result.task_type,
        "user_intent_summary": result.user_intent_summary,
        "logs": list(state.get("logs", [])) + [
            f"[intent_routing] category={result.task_category}, "
            f"type={result.task_type}, target={result.target_column or 'none'}"
        ],
    }
    if result.target_column:
        updates["target_column"] = result.target_column

    return updates
```

- [ ] **Step 4: 运行测试确认通过**

```bash
uv run pytest tests/test_intent_routing.py -v
```

- [ ] **Step 5: Commit**

```bash
git add nodes/intent_routing.py tests/test_intent_routing.py
git commit -m "feat: add intent_routing_node — LLM-based target inference and task categorization"
```

---

## Task 4 — Processing Node 无监督分支

**Files:**
- Modify: `nodes/processing.py`
- Create: `tests/test_processing_unsupervised.py`

- [ ] **Step 1: 写失败测试**

创建 `tests/test_processing_unsupervised.py`：

```python
import pytest
import pandas as pd
from nodes.processing import data_processing


def _schema():
    return {
        "age":    {"type": "numeric",     "null_rate": 0.0},
        "income": {"type": "numeric",     "null_rate": 0.0},
        "gender": {"type": "categorical", "null_rate": 0.0},
        "_meta":  {"row_count": 100, "col_count": 3,
                   "data_scope": "", "unit": "", "mean_target": 0.0,
                   "core_features": "age,income,gender", "file_name": "t.csv"},
    }


@pytest.fixture
def csv_path(tmp_path):
    import numpy as np
    np.random.seed(0)
    df = pd.DataFrame({
        "age":    np.random.randint(20, 60, 100).astype(float),
        "income": np.random.randint(3000, 20000, 100).astype(float),
        "gender": np.random.choice(["M", "F"], 100),
    })
    p = str(tmp_path / "data.csv")
    df.to_csv(p, index=False)
    return p


def test_unsupervised_y_is_none(csv_path):
    state = {
        "file_path": csv_path,
        "target_column": "",
        "task_category": "unsupervised",
        "schema": _schema(),
        "quality_issues": [],
        "logs": [],
    }
    result = data_processing(state)
    assert result["y_train"] is None
    assert result["y_test"] is None


def test_unsupervised_x_is_set(csv_path):
    state = {
        "file_path": csv_path,
        "target_column": "",
        "task_category": "unsupervised",
        "schema": _schema(),
        "quality_issues": [],
        "logs": [],
    }
    result = data_processing(state)
    assert isinstance(result["X_train"], pd.DataFrame)
    assert len(result["X_train"]) > 0


def test_analytical_y_is_none(csv_path):
    state = {
        "file_path": csv_path,
        "target_column": "",
        "task_category": "analytical",
        "schema": _schema(),
        "quality_issues": [],
        "logs": [],
    }
    result = data_processing(state)
    assert result["y_train"] is None
```

- [ ] **Step 2: 运行确认失败**

```bash
uv run pytest tests/test_processing_unsupervised.py -v 2>&1 | head -15
```

- [ ] **Step 3: 修改 processing.py**

在 `data_processing` 函数中，将数据集拆分部分替换为以下逻辑（在标准化之后）：

```python
    task_category = state.get("task_category", "supervised")

    if task_category in ("unsupervised", "analytical"):
        # 无监督/分析任务：全量 X，不拆分 y
        from sklearn.model_selection import train_test_split as _split
        X_train, X_test = _split(X, test_size=0.2, random_state=42)
        state["X_train"]      = X_train
        state["X_test"]       = X_test
        state["y_train"]      = None
        state["y_test"]       = None
        state["feature_names"] = feature_names
        state["logs"]         = logs + ["[processing] 无监督分支，跳过 y 拆分"]
        return state

    # 有监督任务：保持原有 stratify 逻辑
    y = df[[target_col]]
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        logs.append("[processing] 完成 80/20 分层拆分")
    except Exception:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        logs.append("[processing] 回归任务，使用普通 80/20 拆分")

    state["X_train"]       = X_train
    state["X_test"]        = X_test
    state["y_train"]       = y_train
    state["y_test"]        = y_test
    state["feature_names"] = feature_names
    state["logs"]          = logs
    return state
```

同时在函数顶部将 `y = df[[target_col]]` 这行删除（移动到有监督分支内），并确保 `target_col` 在无监督时可为空字符串：

```python
    # 读取数据
    df = pd.read_csv(file_path)
    logs.append("[processing] 成功读取 CSV 文件")

    target_col = state.get("target_column", "")
    task_category = state.get("task_category", "supervised")

    # 无监督任务：X 为全部列；有监督任务：X 去掉 target 列
    if task_category in ("unsupervised", "analytical") or not target_col:
        X = df.copy()
    else:
        X = df.drop(columns=[target_col])

    feature_names = list(X.columns)
```

- [ ] **Step 4: 运行全部 processing 相关测试**

```bash
uv run pytest tests/test_processing_unsupervised.py -v
```

期望：全部 PASS（现有监督路径不受影响）

- [ ] **Step 5: Commit**

```bash
git add nodes/processing.py tests/test_processing_unsupervised.py
git commit -m "feat: add unsupervised branch to data_processing — y_train/y_test=None for clustering/anomaly"
```

---

## Task 5 — EDA 新增 modeling_hints

**Files:**
- Modify: `nodes/eda.py`
- Create: `tests/test_eda_hints.py`

- [ ] **Step 1: 写失败测试**

创建 `tests/test_eda_hints.py`：

```python
import pytest
import numpy as np
import pandas as pd


def _make_state(n=100, task_category="supervised"):
    np.random.seed(42)
    X = pd.DataFrame({
        "a": np.random.randn(n),
        "b": np.random.randn(n),
    })
    y_vals = X["a"] * 2 + np.random.randn(n) * 0.1
    y = pd.DataFrame({"target": y_vals})
    return {
        "X_train": X,
        "y_train": y,
        "target_column": "target",
        "task_category": task_category,
        "charts": [],
        "quality_issues": [],
        "logs": [],
    }


def test_modeling_hints_present():
    from nodes.eda import run_eda
    result = run_eda(_make_state())
    assert "modeling_hints" in result
    hints = result["modeling_hints"]
    assert "linearity_score" in hints
    assert "outlier_ratio" in hints
    assert "sample_size" in hints
    assert "feature_count" in hints


def test_linearity_score_range():
    from nodes.eda import run_eda
    result = run_eda(_make_state())
    score = result["modeling_hints"]["linearity_score"]
    assert 0.0 <= score <= 1.0


def test_sample_size_correct():
    from nodes.eda import run_eda
    result = run_eda(_make_state(n=100))
    assert result["modeling_hints"]["sample_size"] == 100


def test_imbalance_ratio_for_classification():
    from nodes.eda import run_eda
    np.random.seed(0)
    n = 100
    X = pd.DataFrame({"a": np.random.randn(n), "b": np.random.randn(n)})
    # 10% positive class
    y_vals = [1] * 10 + [0] * 90
    y = pd.DataFrame({"label": y_vals})
    state = {
        "X_train": X, "y_train": y,
        "target_column": "label",
        "task_category": "supervised",
        "charts": [], "quality_issues": [], "logs": [],
    }
    result = run_eda(state)
    ratio = result["modeling_hints"].get("imbalance_ratio", None)
    assert ratio is not None
    assert ratio == pytest.approx(0.1, abs=0.01)


def test_unsupervised_hints_no_imbalance():
    from nodes.eda import run_eda
    state = _make_state(task_category="unsupervised")
    state["y_train"] = None
    result = run_eda(state)
    # 无监督无 imbalance_ratio，或为 None
    hints = result.get("modeling_hints", {})
    assert hints.get("imbalance_ratio") is None or "imbalance_ratio" not in hints
```

- [ ] **Step 2: 运行确认失败**

```bash
uv run pytest tests/test_eda_hints.py -v 2>&1 | head -15
```

- [ ] **Step 3: 在 eda.py 中添加 _compute_modeling_hints 并调用**

在 `nodes/eda.py` 末尾添加：

```python
def _compute_modeling_hints(data: pd.DataFrame, target_column: str,
                             task_category: str, y_train) -> dict:
    hints: dict = {
        "sample_size":   len(data),
        "feature_count": len(data.columns) - (1 if target_column in data.columns else 0),
    }

    numeric_data = data.select_dtypes(include=[np.number])

    # linearity_score: 平均 |pearson| 与 target 的相关度（仅有监督）
    if task_category == "supervised" and target_column in numeric_data.columns:
        corr = numeric_data.corr()[target_column].drop(labels=[target_column], errors="ignore")
        corr = corr.dropna()
        hints["linearity_score"] = round(float(corr.abs().mean()), 4) if not corr.empty else 0.0
    else:
        hints["linearity_score"] = 0.0

    # outlier_ratio: IQR 方法检测 target 异常比例
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

    # imbalance_ratio: 少数类占比（仅分类任务，y_train 非 None）
    if task_category == "supervised" and y_train is not None:
        y_series = y_train.iloc[:, 0] if isinstance(y_train, pd.DataFrame) else pd.Series(y_train)
        counts = y_series.value_counts(normalize=True)
        if len(counts) >= 2:
            hints["imbalance_ratio"] = round(float(counts.min()), 4)

    # high_corr_pairs: 特征间相关系数 > 0.9 的列对
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
```

在 `run_eda` 函数的 try 块中，回填 `eda_summary` 之后加入：

```python
        # 计算 modeling_hints
        modeling_hints = _compute_modeling_hints(
            data, target_column, 
            updated_state.get("task_category", "supervised"),
            updated_state.get("y_train")
        )
        updated_state["modeling_hints"] = modeling_hints
```

- [ ] **Step 4: 运行测试**

```bash
uv run pytest tests/test_eda_hints.py -v
```

- [ ] **Step 5: Commit**

```bash
git add nodes/eda.py tests/test_eda_hints.py
git commit -m "feat: EDA node outputs structured modeling_hints for model routing"
```

---

## Task 6 — Model Tools 扩展（无监督算法 + 调参）

**Files:**
- Modify: `tools/model_tools.py`
- Modify: `tests/test_model_tools.py`

- [ ] **Step 1: 写新测试（追加到 test_model_tools.py 末尾）**

```python
# 追加到 tests/test_model_tools.py

def test_registry_contains_clustering_algorithms():
    required = {"KMeans", "DBSCAN", "AgglomerativeClustering"}
    assert required.issubset(set(ALGORITHM_REGISTRY.keys()))


def test_registry_contains_anomaly_algorithms():
    required = {"IsolationForest", "LocalOutlierFactor", "OneClassSVM"}
    assert required.issubset(set(ALGORITHM_REGISTRY.keys()))


def test_clustering_metrics_keys():
    from tools.model_tools import compute_clustering_metrics
    import numpy as np
    X = np.random.randn(50, 2)
    labels = [0] * 25 + [1] * 25
    metrics = compute_clustering_metrics(X, labels)
    assert "silhouette" in metrics
    assert "davies_bouldin" in metrics


def test_anomaly_metrics_keys():
    from tools.model_tools import compute_anomaly_metrics
    scores = [-0.1, -0.2, 0.5, 0.1, -0.3] * 10
    preds  = [-1, 1, -1, 1, 1] * 10
    metrics = compute_anomaly_metrics(preds, contamination=0.1)
    assert "anomaly_ratio" in metrics
    assert "anomaly_count" in metrics


def test_tune_model_returns_fitted_model(clf_arrays):
    from tools.model_tools import tune_model
    X_train, X_test, y_train, _ = clf_arrays
    y_fit = y_train.values.ravel()
    model = tune_model("RandomForestClassifier", X_train, y_fit, "classification")
    assert hasattr(model, "predict")
    preds = model.predict(X_test)
    assert len(preds) == len(X_test)


def test_tune_model_raises_for_unsupervised():
    from tools.model_tools import tune_model
    import pandas as pd, numpy as np
    X = pd.DataFrame(np.random.randn(50, 2))
    with pytest.raises(ValueError, match="tune_model"):
        tune_model("KMeans", X, None, "clustering")
```

- [ ] **Step 2: 运行确认失败**

```bash
uv run pytest tests/test_model_tools.py -v -k "clustering or anomaly or tune" 2>&1 | head -20
```

- [ ] **Step 3: 扩展 model_tools.py**

在 `tools/model_tools.py` 中追加以下内容（在现有 imports 后添加缺失的 imports，再追加函数）：

```python
# 在文件顶部 imports 区域添加
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.svm import OneClassSVM, SVC
from sklearn.linear_model import Lasso
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.model_selection import RandomizedSearchCV
```

在 `ALGORITHM_REGISTRY` 中添加：

```python
    # Supervised extras
    "SVC":                    lambda: SVC(probability=True, random_state=42),
    "Lasso":                  lambda: Lasso(),

    # Clustering
    "KMeans":                 lambda: KMeans(n_clusters=3, random_state=42, n_init="auto"),
    "DBSCAN":                 lambda: DBSCAN(eps=0.5, min_samples=5),
    "AgglomerativeClustering":lambda: AgglomerativeClustering(n_clusters=3),

    # Anomaly Detection
    "IsolationForest":        lambda: IsolationForest(contamination=0.05, random_state=42),
    "LocalOutlierFactor":     lambda: LocalOutlierFactor(contamination=0.05, novelty=True),
    "OneClassSVM":            lambda: OneClassSVM(nu=0.05),
```

在文件末尾追加：

```python
def compute_clustering_metrics(X, labels) -> dict:
    """Silhouette + Davies-Bouldin。labels 必须包含至少 2 个不同值。"""
    import numpy as np
    labels_arr = np.asarray(labels)
    unique = np.unique(labels_arr[labels_arr != -1])   # DBSCAN 的噪声点标记为 -1
    if len(unique) < 2:
        return {"silhouette": -1.0, "davies_bouldin": 999.0}
    X_valid = X[labels_arr != -1] if hasattr(X, "__len__") else X
    l_valid  = labels_arr[labels_arr != -1]
    return {
        "silhouette":     round(float(silhouette_score(X_valid, l_valid)), 4),
        "davies_bouldin": round(float(davies_bouldin_score(X_valid, l_valid)), 4),
    }


def compute_anomaly_metrics(preds, contamination: float) -> dict:
    """preds: array of 1 (normal) / -1 (anomaly)。"""
    import numpy as np
    preds_arr = np.asarray(preds)
    anomaly_count = int((preds_arr == -1).sum())
    total         = len(preds_arr)
    return {
        "anomaly_count": anomaly_count,
        "anomaly_ratio": round(anomaly_count / max(total, 1), 4),
        "contamination": round(contamination, 4),
    }


_TUNE_PARAM_GRIDS = {
    "LogisticRegression":     {"C": [0.01, 0.1, 1, 10, 100]},
    "RandomForestClassifier": {"n_estimators": [50, 100, 200], "max_depth": [None, 5, 10]},
    "XGBClassifier":          {"n_estimators": [50, 100], "max_depth": [3, 5, 7], "learning_rate": [0.05, 0.1, 0.2]},
    "LGBMClassifier":         {"n_estimators": [50, 100], "num_leaves": [31, 63], "learning_rate": [0.05, 0.1]},
    "SVC":                    {"C": [0.1, 1, 10], "gamma": ["scale", "auto"]},
    "LinearRegression":       {},
    "Ridge":                  {"alpha": [0.1, 1.0, 10.0, 100.0]},
    "Lasso":                  {"alpha": [0.01, 0.1, 1.0, 10.0]},
    "RandomForestRegressor":  {"n_estimators": [50, 100, 200], "max_depth": [None, 5, 10]},
    "XGBRegressor":           {"n_estimators": [50, 100], "max_depth": [3, 5, 7], "learning_rate": [0.05, 0.1, 0.2]},
    "LGBMRegressor":          {"n_estimators": [50, 100], "num_leaves": [31, 63], "learning_rate": [0.05, 0.1]},
}


def tune_model(name: str, X_train, y_train, task_type: str):
    """RandomizedSearchCV (20 iter) on supervised models. Raises ValueError for unsupervised."""
    if task_type in ("clustering", "anomaly_detection", "correlation_analysis"):
        raise ValueError(f"tune_model 不支持无监督/分析任务，收到 task_type={task_type}")
    if name not in _TUNE_PARAM_GRIDS:
        return get_model(name)   # 无调参网格，直接返回默认模型
    param_grid = _TUNE_PARAM_GRIDS[name]
    if not param_grid:
        model = get_model(name)
        model.fit(X_train, y_train)
        return model
    scoring = "f1_weighted" if task_type == "classification" else "r2"
    search = RandomizedSearchCV(
        get_model(name), param_distributions=param_grid,
        n_iter=20, cv=3, scoring=scoring,
        random_state=42, n_jobs=-1, error_score="raise"
    )
    search.fit(X_train, y_train)
    return search.best_estimator_
```

- [ ] **Step 4: 运行测试**

```bash
uv run pytest tests/test_model_tools.py -v
```

- [ ] **Step 5: Commit**

```bash
git add tools/model_tools.py tests/test_model_tools.py
git commit -m "feat: add clustering/anomaly algorithms, clustering/anomaly metrics, tune_model to model_tools"
```

---

## Task 7 — Model Routing Node（三角度精选）

**Files:**
- Create: `nodes/model_routing.py`
- Create: `tests/test_model_routing.py`

- [ ] **Step 1: 写失败测试**

创建 `tests/test_model_routing.py`：

```python
from unittest.mock import MagicMock, patch
import pytest
from nodes.model_routing import model_routing_node, ModelRoutingOutput


def _state(task_category, task_type, hints, algorithms, reasoning="test"):
    return {
        "task_category":       task_category,
        "task_type":           task_type,
        "user_intent_summary": "test intent",
        "schema": {"_meta": {"row_count": 500, "col_count": 5,
                              "mean_target": 0.5, "core_features": "a,b,c"}},
        "modeling_hints": hints,
        "logs": [],
    }


def _mock_llm(task_type, algorithms, reasoning="selected based on hints"):
    out = ModelRoutingOutput(
        task_type=task_type,
        selected_algorithms=algorithms,
        reasoning=reasoning,
    )
    mock_s = MagicMock()
    mock_s.invoke.return_value = out
    mock_l = MagicMock()
    mock_l.with_structured_output.return_value = mock_s
    return mock_l


@patch("nodes.model_routing.ChatOpenAI")
def test_classification_routing(mock_chat):
    mock_chat.return_value = _mock_llm("classification", ["XGBClassifier", "RandomForestClassifier"])
    result = model_routing_node(_state(
        "supervised", "classification",
        {"linearity_score": 0.3, "imbalance_ratio": 0.45, "sample_size": 500, "feature_count": 4},
        [], ""
    ))
    assert result["task_type"] == "classification"
    assert "XGBClassifier" in result["selected_algorithms"]


@patch("nodes.model_routing.ChatOpenAI")
def test_clustering_routing(mock_chat):
    mock_chat.return_value = _mock_llm("clustering", ["KMeans", "AgglomerativeClustering"])
    result = model_routing_node(_state(
        "unsupervised", "clustering",
        {"sample_size": 300, "feature_count": 3},
        [], ""
    ))
    assert result["task_type"] == "clustering"
    assert "KMeans" in result["selected_algorithms"]


@patch("nodes.model_routing.ChatOpenAI")
def test_correlation_analysis_returns_empty_algorithms(mock_chat):
    mock_chat.return_value = _mock_llm("correlation_analysis", [])
    result = model_routing_node(_state(
        "analytical", "correlation_analysis",
        {"sample_size": 200, "feature_count": 5},
        [], ""
    ))
    assert result["task_type"] == "correlation_analysis"
    assert result["selected_algorithms"] == []


@patch("nodes.model_routing.ChatOpenAI")
def test_returns_reasoning(mock_chat):
    mock_chat.return_value = _mock_llm("regression", ["Ridge"], "High linearity detected")
    result = model_routing_node(_state(
        "supervised", "regression",
        {"linearity_score": 0.85, "sample_size": 1000, "feature_count": 8},
        [], ""
    ))
    assert isinstance(result["reasoning"], str)
    assert len(result["reasoning"]) > 0


@patch("nodes.model_routing.ChatOpenAI")
def test_appends_log(mock_chat):
    mock_chat.return_value = _mock_llm("classification", ["LogisticRegression"])
    result = model_routing_node(_state(
        "supervised", "classification",
        {"linearity_score": 0.6, "sample_size": 200, "feature_count": 3},
        [], ""
    ))
    assert any("[model_routing]" in log for log in result["logs"])
```

- [ ] **Step 2: 运行确认失败**

```bash
uv run pytest tests/test_model_routing.py -v 2>&1 | head -10
```

- [ ] **Step 3: 实现 model_routing_node**

创建 `nodes/model_routing.py`：

> **依赖说明**：此节点使用 DeepSeek API（OpenAI 兼容接口）。
> 运行前需设置环境变量：`DEEPSEEK_API_KEY=<your_key>`

```python
import os
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from app.state import AgentState


class ModelRoutingOutput(BaseModel):
    task_type:           str        # 精确任务类型（五选一）
    selected_algorithms: list[str]  # 2~3 个候选算法；correlation_analysis 为空列表
    reasoning:           str        # 选择理由（一句话）


_ALGORITHM_MENU = """
Supervised — Classification:   LogisticRegression, RandomForestClassifier, XGBClassifier, LGBMClassifier, SVC
Supervised — Regression:       LinearRegression, Ridge, Lasso, RandomForestRegressor, XGBRegressor, LGBMRegressor
Unsupervised — Clustering:     KMeans, DBSCAN, AgglomerativeClustering
Unsupervised — Anomaly:        IsolationForest, LocalOutlierFactor, OneClassSVM
Analytical — Correlation:      (no model — return empty list)
"""

_SELECTION_RULES = """
Selection guidelines (use modeling_hints to decide):
- linearity_score > 0.7  → prefer LinearRegression / Ridge / LogisticRegression
- linearity_score < 0.4  → prefer tree-based (RandomForest, XGB, LGBM)
- imbalance_ratio < 0.15 → note class imbalance; prefer F1-optimized models
- high_corr_pairs not empty → prefer Ridge / Lasso over plain LinearRegression
- sample_size < 500       → avoid XGB/LGBM; prefer simpler models
- sample_size > 50000     → prefer LGBM (fastest)
- feature_count > 50      → prefer Lasso (sparse) or tree-based
- For clustering: KMeans if data is likely spherical; DBSCAN if density-based clusters expected
- For anomaly_detection: IsolationForest is default; use LOF for local density anomalies
- Select 2-3 algorithms (or 0 for correlation_analysis)
"""


def model_routing_node(state: AgentState) -> dict:
    task_category  = state.get("task_category", "supervised")
    task_type_hint = state.get("task_type", "")
    intent_summary = state.get("user_intent_summary", "")
    hints          = state.get("modeling_hints", {})
    meta           = state.get("schema", {}).get("_meta", {})

    prompt = f"""You are a senior data scientist selecting ML algorithms.

User intent: {intent_summary}
Initial task assessment: category={task_category}, type={task_type_hint}

Dataset properties:
  - Rows: {meta.get('row_count', '?')}
  - Features: {meta.get('col_count', '?')}

EDA modeling hints:
  - linearity_score:   {hints.get('linearity_score', 'N/A')}   (0=nonlinear, 1=linear)
  - imbalance_ratio:   {hints.get('imbalance_ratio', 'N/A')}   (minority class fraction)
  - outlier_ratio:     {hints.get('outlier_ratio', 'N/A')}
  - high_corr_pairs:   {hints.get('high_corr_pairs', [])}
  - sample_size:       {hints.get('sample_size', '?')}
  - feature_count:     {hints.get('feature_count', '?')}

Available algorithms:
{_ALGORITHM_MENU}

{_SELECTION_RULES}

Output:
- task_type: confirm or refine (pick exactly one of the 5 task types)
- selected_algorithms: list of 2-3 algorithm names from the menu above; [] for correlation_analysis
- reasoning: one concise English sentence explaining your algorithm selection
"""

    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=os.environ.get("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
        temperature=0,
    )
    structured_llm = llm.with_structured_output(ModelRoutingOutput)
    result: ModelRoutingOutput = structured_llm.invoke(prompt)

    return {
        "task_type":           result.task_type,
        "selected_algorithms": result.selected_algorithms,
        "reasoning":           result.reasoning,
        "logs": list(state.get("logs", [])) + [
            f"[model_routing] task_type={result.task_type}, "
            f"algorithms={result.selected_algorithms}"
        ],
    }
```

- [ ] **Step 4: 运行测试**

```bash
uv run pytest tests/test_model_routing.py -v
```

- [ ] **Step 5: Commit**

```bash
git add nodes/model_routing.py tests/test_model_routing.py
git commit -m "feat: add model_routing_node — 3-angle LLM-based algorithm selection"
```

---

## Task 8 — Modeling Node 无监督扩展 + tune 支持

**Files:**
- Modify: `nodes/modeling.py`
- Modify: `tests/test_modeling.py`

- [ ] **Step 1: 追加测试到 test_modeling.py**

```python
# 追加到 tests/test_modeling.py

def _unsupervised_state(X_train, X_test, algorithms):
    return {
        "X_train": X_train,
        "X_test":  X_test,
        "y_train": None,
        "y_test":  None,
        "selected_algorithms": algorithms,
        "task_type": "clustering",
        "tune": False,
        "logs": [],
    }


def test_modeling_clustering_kmeans(clf_arrays):
    X_train, X_test, _, _ = clf_arrays
    result = modeling_node(_unsupervised_state(X_train, X_test, ["KMeans"]))
    assert "KMeans" in result["model_results"]
    assert "error" not in result["model_results"]["KMeans"]
    assert "labels" in result["model_results"]["KMeans"]


def test_modeling_isolation_forest(clf_arrays):
    X_train, X_test, _, _ = clf_arrays
    state = _unsupervised_state(X_train, X_test, ["IsolationForest"])
    state["task_type"] = "anomaly_detection"
    result = modeling_node(state)
    assert "IsolationForest" in result["model_results"]
    preds = result["model_results"]["IsolationForest"]["labels"]
    assert set(preds).issubset({1, -1})


def test_modeling_tune_flag(clf_arrays):
    X_train, X_test, y_train, _ = clf_arrays
    state = {
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": None,
        "selected_algorithms": ["RandomForestClassifier"],
        "task_type": "classification",
        "tune": True,
        "logs": [],
    }
    result = modeling_node(state)
    assert "RandomForestClassifier" in result["model_results"]
    assert "error" not in result["model_results"]["RandomForestClassifier"]
```

- [ ] **Step 2: 修改 modeling.py**

用以下内容替换 `nodes/modeling.py`：

```python
import numpy as np
from app.state import AgentState
from tools.model_tools import get_model, tune_model

_UNSUPERVISED_TASKS = {"clustering", "anomaly_detection"}
_DBSCAN_LIKE = {"DBSCAN"}   # 无 predict()，用 fit_predict(X_train)


def modeling_node(state: AgentState) -> dict:
    X_train = state["X_train"]
    X_test  = state["X_test"]
    y_train = state.get("y_train")
    selected_algorithms = state["selected_algorithms"]
    task_type = state.get("task_type", "")
    do_tune   = state.get("tune", False)

    y_fit = None
    if y_train is not None:
        y_fit = y_train.values.ravel() if hasattr(y_train, "values") else y_train

    model_results = {}

    for algo_name in selected_algorithms:
        try:
            if task_type in _UNSUPERVISED_TASKS:
                model_results[algo_name] = _fit_unsupervised(algo_name, X_train, X_test, task_type)
            else:
                model_results[algo_name] = _fit_supervised(
                    algo_name, X_train, X_test, y_fit, task_type, do_tune
                )
        except Exception as exc:
            model_results[algo_name] = {"error": str(exc)}

    return {
        "model_results": model_results,
        "logs": list(state.get("logs", [])) + [
            f"[modeling] 训练完成: {list(model_results.keys())}"
        ],
    }


def _fit_supervised(algo_name, X_train, X_test, y_fit, task_type, do_tune):
    if do_tune:
        model = tune_model(algo_name, X_train, y_fit, task_type)
    else:
        model = get_model(algo_name)
        model.fit(X_train, y_fit)
    y_pred = model.predict(X_test)
    return {
        "y_pred": y_pred.tolist() if hasattr(y_pred, "tolist") else list(y_pred),
        "model":  model,
    }


def _fit_unsupervised(algo_name, X_train, X_test, task_type):
    model = get_model(algo_name)

    if algo_name in _DBSCAN_LIKE:
        # DBSCAN 只有 fit_predict，在 X_train 上训练并得到标签
        labels = model.fit_predict(X_train).tolist()
        return {"labels": labels, "model": model}

    if task_type == "clustering":
        model.fit(X_train)
        labels = model.predict(X_test).tolist()
        return {"labels": labels, "model": model}

    # anomaly_detection: fit on X_train, predict on X_test
    model.fit(X_train)
    preds = model.predict(X_test).tolist()   # 1=normal, -1=anomaly
    return {"labels": preds, "model": model}
```

- [ ] **Step 3: 运行全部 modeling 测试**

```bash
uv run pytest tests/test_modeling.py -v
```

期望：所有 PASS（含新增无监督测试）

- [ ] **Step 4: Commit**

```bash
git add nodes/modeling.py tests/test_modeling.py
git commit -m "feat: modeling_node supports clustering/anomaly_detection and optional tune flag"
```

---

## Task 9 — Evaluation Node 无监督指标

**Files:**
- Modify: `nodes/evaluation.py`
- Modify: `tests/test_evaluation.py`

- [ ] **Step 1: 追加测试**

```python
# 追加到 tests/test_evaluation.py
import numpy as np
import pandas as pd

def test_evaluation_clustering_metrics():
    import numpy as np
    X_test = pd.DataFrame(np.random.randn(20, 2), columns=["a", "b"])
    model_results = {
        "KMeans": {"labels": [0]*10 + [1]*10, "model": None}
    }
    state = {
        "model_results": model_results,
        "X_test": X_test,
        "y_test": None,
        "task_type": "clustering",
        "logs": [],
    }
    from nodes.evaluation import evaluation_node
    result = evaluation_node(state)
    assert "KMeans" in result["metrics"]
    assert "silhouette" in result["metrics"]["KMeans"]
    assert result["best_model"] == "KMeans"


def test_evaluation_anomaly_metrics():
    X_test = pd.DataFrame(np.random.randn(20, 2), columns=["a", "b"])
    preds = [1] * 18 + [-1, -1]
    model_results = {"IsolationForest": {"labels": preds, "model": None}}
    state = {
        "model_results": model_results,
        "X_test": X_test,
        "y_test": None,
        "task_type": "anomaly_detection",
        "logs": [],
    }
    from nodes.evaluation import evaluation_node
    result = evaluation_node(state)
    assert "anomaly_ratio" in result["metrics"]["IsolationForest"]
    assert result["metrics"]["IsolationForest"]["anomaly_ratio"] == pytest.approx(0.1, abs=0.01)


def test_evaluation_correlation_analysis_returns_empty():
    state = {
        "model_results": {},
        "X_test": pd.DataFrame(),
        "y_test": None,
        "task_type": "correlation_analysis",
        "logs": [],
    }
    from nodes.evaluation import evaluation_node
    result = evaluation_node(state)
    assert result["metrics"] == {}
    assert result["best_model"] == ""
```

- [ ] **Step 2: 修改 evaluation.py**

用以下内容替换 `nodes/evaluation.py`：

```python
import numpy as np
from app.state import AgentState
from tools.model_tools import (
    compute_classification_metrics,
    compute_regression_metrics,
    compute_clustering_metrics,
    compute_anomaly_metrics,
)


def evaluation_node(state: AgentState) -> dict:
    model_results = state["model_results"]
    y_test        = state.get("y_test")
    X_test        = state.get("X_test")
    task_type     = state.get("task_type", "")

    # correlation_analysis 不训练模型，直接返回空
    if task_type == "correlation_analysis":
        return {
            "metrics":    {},
            "best_model": "",
            "logs": list(state.get("logs", [])) + [
                "[evaluation] correlation_analysis 无需模型评估"
            ],
        }

    y_true = None
    if y_test is not None:
        y_true = y_test.values.ravel() if hasattr(y_test, "values") else y_test

    X_arr = X_test.values if hasattr(X_test, "values") else np.asarray(X_test)

    metrics: dict = {}
    for algo_name, result in model_results.items():
        if "error" in result:
            continue
        if task_type == "classification":
            metrics[algo_name] = compute_classification_metrics(y_true, result["y_pred"])
        elif task_type == "regression":
            metrics[algo_name] = compute_regression_metrics(y_true, result["y_pred"])
        elif task_type == "clustering":
            metrics[algo_name] = compute_clustering_metrics(X_arr, result["labels"])
        elif task_type == "anomaly_detection":
            contamination = getattr(result.get("model"), "contamination", 0.05) or 0.05
            metrics[algo_name] = compute_anomaly_metrics(result["labels"], contamination)

    if not metrics and task_type not in ("correlation_analysis",):
        raise RuntimeError("All models failed — cannot select best_model.")

    best_model = _select_best(metrics, task_type)

    return {
        "metrics":    metrics,
        "best_model": best_model,
        "logs": list(state.get("logs", [])) + [
            f"[evaluation] 最优模型: {best_model}"
        ],
    }


def _select_best(metrics: dict, task_type: str) -> str:
    if not metrics:
        return ""
    if task_type == "classification":
        return max(metrics, key=lambda k: metrics[k].get("f1", 0))
    if task_type == "regression":
        return max(metrics, key=lambda k: metrics[k].get("r2", -999))
    if task_type == "clustering":
        return max(metrics, key=lambda k: metrics[k].get("silhouette", -1))
    if task_type == "anomaly_detection":
        # 异常检测没有"最优"模型概念，取第一个
        return next(iter(metrics))
    return next(iter(metrics))
```

- [ ] **Step 3: 运行全部 evaluation 测试**

```bash
uv run pytest tests/test_evaluation.py -v
```

- [ ] **Step 4: Commit**

```bash
git add nodes/evaluation.py tests/test_evaluation.py
git commit -m "feat: evaluation_node supports clustering/anomaly metrics and correlation_analysis skip"
```

---

## Task 10 — Reporting 双模板（general / expert）

**Files:**
- Modify: `nodes/reporting.py`
- Modify: `prompts/report_prompt.txt`

- [ ] **Step 1: 更新 prompts/report_prompt.txt**

用以下内容替换 `prompts/report_prompt.txt`：

```
=== GENERAL TEMPLATE ===
你是一位亲切的数据分析顾问，面向没有数据分析背景的用户。
请用简单的中文撰写报告，避免统计学和机器学习术语。
聚焦于"这对我意味着什么"，给出清晰的业务结论和行动建议。

报告结构：
1. 你的问题是什么（用一句话复述用户意图）
2. 数据说了什么（数据概况，通俗描述）
3. 最重要的发现（EDA 洞察，用类比或日常语言描述）
4. 分析结论（模型表现翻译为业务含义，不要出现指标数字，用"预测准确率较高/较低"等描述）
5. 建议你做什么（短期 1~3 个月的具体行动，长期 3~12 个月的方向）
6. 需要注意的风险

=== EXPERT TEMPLATE ===
You are a senior data analyst writing a technical analysis report.
Write in Chinese. Include all quantitative metrics and technical details.

Report structure:
1. Dataset Overview (rows, columns, schema, quality issues)
2. EDA Findings (key charts referenced, statistical observations, modeling hints)
3. Model Selection Rationale (3-angle reasoning)
4. Model Comparison Table (all candidate models with full metrics)
5. Best Model Analysis (feature importance if available, residual/confusion matrix notes)
6. Conclusions & Recommendations (data-driven, specific)
7. Risks & Limitations
```

- [ ] **Step 2: 修改 reporting.py 读取 user_level 并选择模板**

在 `generate_report_content(report_data)` 函数（或调用 LLM 的位置）修改为：

```python
import os
from anthropic import Anthropic
from app.state import AgentState, ReportData
from datetime import datetime


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


def _call_llm(template: str, report_data: ReportData) -> str:
    client = Anthropic()
    data_json = {k: v for k, v in report_data.items() if v not in (None, "", 0, 0.0, [])}
    import json
    user_msg = f"{template}\n\n---\n以下是分析数据：\n```json\n{json.dumps(data_json, ensure_ascii=False, indent=2)}\n```"
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
```

> **注意**：`build_report_data` 函数主体保留现有实现，在末尾添加 `user_level` 和 `user_intent_summary`、`reasoning` 字段的填充即可。

- [ ] **Step 3: 验证模板加载**

```bash
uv run python -c "
from nodes.reporting import _load_template
g = _load_template('general')
e = _load_template('expert')
assert '业务' in g
assert 'Dataset Overview' in e
print('templates OK')
"
```

- [ ] **Step 4: Commit**

```bash
git add nodes/reporting.py prompts/report_prompt.txt
git commit -m "feat: reporting supports user_level with general/expert dual templates"
```

---

## Task 11 — LangGraph Graph + CLI Main

**Files:**
- Create: `app/graph.py`
- Create: `app/main.py`

- [ ] **Step 1: 创建 graph.py**

```python
# app/graph.py
from langgraph.graph import StateGraph, END
from app.state import AgentState
from nodes.profiling      import profiling_node
from nodes.intent_routing import intent_routing_node
from nodes.processing     import data_processing
from nodes.eda            import run_eda
from nodes.model_routing  import model_routing_node
from nodes.modeling       import modeling_node
from nodes.evaluation     import evaluation_node
from nodes.reporting      import generate_report


def _route_after_model_routing(state: AgentState) -> str:
    """correlation_analysis 跳过 Modeling + Evaluation，直接进 Reporting。"""
    if state.get("task_type") == "correlation_analysis":
        return "reporting"
    return "modeling"


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("profiling",      profiling_node)
    graph.add_node("intent_routing", intent_routing_node)
    graph.add_node("processing",     data_processing)
    graph.add_node("eda",            run_eda)
    graph.add_node("model_routing",  model_routing_node)
    graph.add_node("modeling",       modeling_node)
    graph.add_node("evaluation",     evaluation_node)
    graph.add_node("reporting",      generate_report)

    graph.set_entry_point("profiling")
    graph.add_edge("profiling",      "intent_routing")
    graph.add_edge("intent_routing", "processing")
    graph.add_edge("processing",     "eda")
    graph.add_edge("eda",            "model_routing")
    graph.add_conditional_edges(
        "model_routing",
        _route_after_model_routing,
        {"modeling": "modeling", "reporting": "reporting"},
    )
    graph.add_edge("modeling",       "evaluation")
    graph.add_edge("evaluation",     "reporting")
    graph.add_edge("reporting",      END)

    return graph.compile()
```

- [ ] **Step 2: 验证 graph 可编译**

```bash
uv run python -c "from app.graph import build_graph; g = build_graph(); print('graph OK')"
```

期望：`graph OK`

- [ ] **Step 3: 创建 main.py**

```python
# app/main.py
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser(
        description="AutoInsight — Automated Data Mining Agent"
    )
    parser.add_argument("--file",   required=True,  help="Path to input CSV file")
    parser.add_argument("--prompt", required=True,  help="Describe your business question in plain language")
    parser.add_argument("--target", default="",     help="Target column name (optional; inferred if omitted)")
    parser.add_argument("--level",  default="general", choices=["general", "expert"],
                        help="Report language level (default: general)")
    parser.add_argument("--tune",   action="store_true",
                        help="Enable RandomizedSearchCV hyperparameter tuning (slower)")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"Error: file not found — {args.file}")
        sys.exit(1)

    os.makedirs("outputs/charts",  exist_ok=True)
    os.makedirs("outputs/reports", exist_ok=True)

    from app.graph import build_graph

    initial_state = {
        "user_query":    args.prompt,
        "file_path":     args.file,
        "target_column": args.target,
        "user_level":    args.level,
        "tune":          args.tune,
        "logs":          [],
        # 以下字段由节点写入，此处给默认值避免 TypedDict KeyError
        "schema":               {},
        "quality_issues":       [],
        "task_category":        "",
        "task_type":            "",
        "user_intent_summary":  "",
        "selected_algorithms":  [],
        "reasoning":            "",
        "model_results":        {},
        "metrics":              {},
        "best_model":           "",
        "charts":               [],
        "eda_summary":          {},
        "modeling_hints":       {},
        "feature_names":        [],
        "X_train": None, "X_test": None,
        "y_train": None, "y_test": None,
        "report_path":          "",
    }

    print(f"[AutoInsight] 开始分析: {args.file}")
    graph = build_graph()
    final_state = graph.invoke(initial_state)

    print("\n=== AutoInsight 分析完成 ===")
    print(f"任务类型:   {final_state.get('task_type')}")
    print(f"最优模型:   {final_state.get('best_model') or 'N/A'}")
    print(f"报告路径:   {final_state.get('report_path')}")
    print(f"图表数量:   {len(final_state.get('charts', []))}")
    print("\n运行日志:")
    for log in final_state.get("logs", []):
        print(f"  {log}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Commit**

```bash
git add app/graph.py app/main.py
git commit -m "feat: add LangGraph graph with conditional routing and CLI main entry point"
```

---

## Task 12 — nodes/__init__.py 更新 + 集成冒烟测试

**Files:**
- Modify: `nodes/__init__.py`
- Create: `tests/test_integration.py`

- [ ] **Step 1: 更新 nodes/__init__.py**

```python
from .profiling      import profiling_node
from .intent_routing import intent_routing_node
from .processing     import data_processing
from .eda            import run_eda
from .model_routing  import model_routing_node
from .modeling       import modeling_node
from .evaluation     import evaluation_node
from .reporting      import generate_report

__all__ = [
    "profiling_node",
    "intent_routing_node",
    "data_processing",
    "run_eda",
    "model_routing_node",
    "modeling_node",
    "evaluation_node",
    "generate_report",
]
```

- [ ] **Step 2: 创建集成冒烟测试**

创建 `tests/test_integration.py`：

```python
"""
端到端冒烟测试：使用合成 CSV，mock LLM 节点，跑通完整 graph。
不测试 LLM 内容质量，只验证 graph 可以从头跑到尾且关键 state 字段被写入。
"""
import os
import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch


@pytest.fixture
def iris_csv(tmp_path):
    """简单的三分类合成数据集。"""
    np.random.seed(42)
    n = 150
    df = pd.DataFrame({
        "sepal_length": np.random.randn(n) + 5.0,
        "sepal_width":  np.random.randn(n) + 3.0,
        "petal_length": np.random.randn(n) + 4.0,
        "species":      np.random.choice(["setosa", "versicolor", "virginica"], n),
    })
    p = str(tmp_path / "iris.csv")
    df.to_csv(p, index=False)
    return p


def _mock_intent(target, category, task_type, summary):
    from nodes.intent_routing import IntentOutput
    out = IntentOutput(target_column=target, task_category=category,
                       task_type=task_type, user_intent_summary=summary)
    ms = MagicMock(); ms.invoke.return_value = out
    ml = MagicMock(); ml.with_structured_output.return_value = ms
    return ml


def _mock_model_routing(task_type, algorithms):
    from nodes.model_routing import ModelRoutingOutput
    out = ModelRoutingOutput(task_type=task_type,
                             selected_algorithms=algorithms,
                             reasoning="mock reasoning")
    ms = MagicMock(); ms.invoke.return_value = out
    ml = MagicMock(); ml.with_structured_output.return_value = ms
    return ml


@patch("nodes.reporting.Anthropic")
@patch("nodes.model_routing.ChatOpenAI")
@patch("nodes.intent_routing.ChatAnthropic")
def test_classification_pipeline_end_to_end(
    mock_intent_chat, mock_model_chat, mock_anthropic, iris_csv
):
    mock_intent_chat.return_value = _mock_intent(
        "species", "supervised", "classification", "预测鸢尾花种类"
    )
    mock_model_chat.return_value = _mock_model_routing(
        "classification", ["LogisticRegression", "RandomForestClassifier"]
    )
    # mock Anthropic reporting
    mock_resp = MagicMock()
    mock_resp.content = [MagicMock(text="# Mock Report\nTest report content.")]
    mock_anthropic.return_value.messages.create.return_value = mock_resp

    from app.graph import build_graph

    initial_state = {
        "user_query":   "预测鸢尾花种类",
        "file_path":    iris_csv,
        "target_column": "",
        "user_level":   "general",
        "tune":         False,
        "logs":         [],
        "schema": {}, "quality_issues": [], "task_category": "",
        "task_type": "", "user_intent_summary": "",
        "selected_algorithms": [], "reasoning": "",
        "model_results": {}, "metrics": {}, "best_model": "",
        "charts": [], "eda_summary": {}, "modeling_hints": {},
        "feature_names": [], "X_train": None, "X_test": None,
        "y_train": None, "y_test": None, "report_path": "",
    }

    graph = build_graph()
    final = graph.invoke(initial_state)

    assert final["task_type"] == "classification"
    assert final["best_model"] in ("LogisticRegression", "RandomForestClassifier")
    assert len(final["metrics"]) >= 1
    assert final["report_path"] != ""
    assert os.path.exists(final["report_path"])
    assert len(final["logs"]) >= 6   # 每个节点至少一条 log
```

- [ ] **Step 3: 运行集成测试**

```bash
uv run pytest tests/test_integration.py -v
```

期望：PASS

- [ ] **Step 4: 运行全量测试套件**

```bash
uv run pytest tests/ -v --tb=short
```

期望：所有测试通过，无 FAIL

- [ ] **Step 5: 最终 Commit**

```bash
git add nodes/__init__.py tests/test_integration.py
git commit -m "feat: update nodes __init__ exports and add end-to-end integration smoke test"
```

---

## 执行顺序总结

```
Task 1  → Task 2 → Task 3 → Task 4 → Task 5
              ↓
Task 6 (model_tools) ─→ Task 7 (model_routing)
                    ─→ Task 8 (modeling)
                    ─→ Task 9 (evaluation)
              ↓
Task 10 (reporting) → Task 11 (graph + main) → Task 12 (integration)
```

Tasks 6、7、8、9 可并行（均依赖 model_tools 但互不依赖）。

---

## 自检：Spec 覆盖验证

| PRD 需求 | 覆盖任务 |
|---|---|
| 五类任务类型 | Task 3, 7, 8, 9 |
| target_column 自动推断 | Task 3 |
| 三角度模型选择 | Task 5 (hints), Task 7 (model_routing) |
| 无监督 Processing 分支 | Task 4 |
| EDA modeling_hints | Task 5 |
| user_level 报告风格 | Task 10 |
| --tune 超参调优 | Task 6, 8 |
| correlation_analysis 条件跳转 | Task 11 (graph) |
| Profiling 节点 | Task 2 |
| 完整 pipeline 串联 | Task 11, 12 |
