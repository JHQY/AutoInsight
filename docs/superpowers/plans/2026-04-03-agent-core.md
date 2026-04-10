# AutoInsight Agent Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the full AutoInsight pipeline — 7 LangGraph nodes, tool utilities, LangGraph graph orchestration, and CLI entry point, producing a working end-to-end system that takes a CSV + target column and outputs charts + a markdown report.

**Architecture:** Linear LangGraph DAG: profiling → routing → processing → EDA → modeling → evaluation → reporting. All nodes share `AgentState` TypedDict. Node 4 (routing) uses `ChatAnthropic` + `.with_structured_output()` to classify task type and select algorithms. Node 7 (reporting) uses `ChatAnthropic` to generate the final markdown report from a structured prompt.

**Tech Stack:** LangGraph, langchain-anthropic, anthropic, pandas, scikit-learn, XGBoost, LightGBM, matplotlib, seaborn, pytest, uv (Python ≥ 3.11)

---

## File Map

| Action | Path | Responsibility |
|--------|------|---------------|
| Modify | `pyproject.toml` | Add xgboost, lightgbm, pytest dev deps |
| Modify | `app/state.py` | Add `eda_results: dict` field |
| Create | `tools/__init__.py` | Empty — makes tools a package |
| Create | `tools/data_tools.py` | CSV loading, column type inference, train/test split |
| Create | `tools/plot_tools.py` | Chart generation (distribution, heatmap, feature-target) |
| Create | `tools/model_tools.py` | Algorithm registry, classification/regression metrics |
| Create | `nodes/__init__.py` | Empty — makes nodes a package |
| Create | `nodes/profiling.py` | Node 1: schema + quality_issues + schema["_meta"] |
| Create | `nodes/routing.py` | Node 4: LLM selects task_type + selected_algorithms + reasoning |
| Create | `nodes/processing.py` | Node 2: clean, encode, scale, split → X/y train/test |
| Create | `nodes/eda.py` | Node 3: correlations, stats, charts → eda_results |
| Create | `nodes/modeling.py` | Node 5: train each selected algorithm, store predictions |
| Create | `nodes/evaluation.py` | Node 6: compute metrics, pick best_model |
| Create | `nodes/reporting.py` | Node 7: build ReportData, call LLM, write markdown |
| Create | `prompts/report_prompt.txt` | Prompt template with {variable} placeholders |
| Create | `app/graph.py` | Build and compile LangGraph StateGraph |
| Create | `app/main.py` | CLI: argparse → initial_state → graph.invoke() |
| Create | `outputs/charts/.gitkeep` | Ensure output dirs exist in repo |
| Create | `outputs/reports/.gitkeep` | Ensure output dirs exist in repo |
| Create | `tests/__init__.py` | Empty |
| Create | `tests/conftest.py` | Shared fixtures: CSV files, sample DataFrames, tmp dirs |
| Create | `tests/test_data_tools.py` | Unit tests for data_tools |
| Create | `tests/test_plot_tools.py` | Unit tests for plot_tools |
| Create | `tests/test_model_tools.py` | Unit tests for model_tools registry and metrics |
| Create | `tests/test_profiling.py` | Unit tests for profiling_node |
| Create | `tests/test_routing.py` | Unit tests for routing_node (mock LLM) |
| Create | `tests/test_processing.py` | Unit tests for processing_node |
| Create | `tests/test_eda.py` | Unit tests for eda_node |
| Create | `tests/test_modeling.py` | Unit tests for modeling_node |
| Create | `tests/test_evaluation.py` | Unit tests for evaluation_node |
| Create | `tests/test_reporting.py` | Unit tests for reporting_node (mock LLM) |
| Create | `tests/test_graph.py` | Integration smoke test (mock LLM nodes) |

---

## Task 1: Update Dependencies and AgentState

**Files:**
- Modify: `pyproject.toml`
- Modify: `app/state.py`

- [ ] **Step 1: Add xgboost, lightgbm, and pytest to pyproject.toml**

Replace the contents of `pyproject.toml` with:

```toml
[project]
name = "autoinsight"
version = "0.1.0"
description = "LangGraph-based automated data mining agent system"
requires-python = ">=3.11"
dependencies = [
    "langgraph>=0.2.0",
    "langchain-anthropic>=0.3.0",
    "anthropic>=0.40.0",
    "pandas>=2.0.0",
    "scikit-learn>=1.4.0",
    "matplotlib>=3.8.0",
    "seaborn>=0.13.0",
    "numpy>=1.26.0",
    "xgboost>=2.0.0",
    "lightgbm>=4.0.0",
    "pydantic>=2.0.0",
]

[dependency-groups]
dev = [
    "pytest>=8.0.0",
]
```

- [ ] **Step 2: Add `eda_results` field to AgentState in `app/state.py`**

Add the following field to `AgentState` after the `charts` field (around line 63):

```python
    # ── EDA Skill — structured results for reporting ─────────────────
    eda_results:          dict
    # 示例：
    # {
    #   "top3_features": "feature_a,feature_b,feature_c",
    #   "distribution_desc": "中位数:5000;均值:5800;70%分位数:8000",
    #   "layer_desc": "低:4000;中:6000;高:9000",
    #   "abnormal_desc": "age有5个异常值(5.0%)",
    #   "key_charts": "outputs/charts/hist_age.png;outputs/charts/correlation_heatmap.png",
    #   "feature_1": "feature_a",  "feature_1_corr": 0.85,
    #   "feature_2": "feature_b",  "feature_2_corr": 0.72,
    #   "feature_3": "feature_c",  "feature_3_corr": 0.68,
    # }
```

- [ ] **Step 3: Install updated dependencies**

```bash
uv sync --dev
```

Expected: Resolves and installs xgboost, lightgbm, pytest with no errors.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml app/state.py
git commit -m "chore: add xgboost/lightgbm deps and eda_results to AgentState"
```

---

## Task 2: Tool — `tools/data_tools.py`

**Files:**
- Create: `tests/conftest.py`
- Create: `tests/__init__.py`
- Create: `tools/__init__.py`
- Create: `tools/data_tools.py`
- Create: `tests/test_data_tools.py`

- [ ] **Step 1: Create empty `__init__.py` files**

```bash
touch tools/__init__.py nodes/__init__.py tests/__init__.py
```

- [ ] **Step 2: Create `tests/conftest.py` with shared fixtures**

```python
# tests/conftest.py
import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def sample_classification_df():
    """100-row binary classification dataset."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "age": np.random.randint(18, 65, n).astype(float),
        "income": np.random.randint(20000, 100000, n).astype(float),
        "category": np.random.choice(["A", "B", "C"], n),
        "target": np.random.choice([0, 1], n),
    })


@pytest.fixture
def sample_regression_df():
    """100-row regression dataset with continuous target."""
    np.random.seed(42)
    n = 100
    X = np.random.randn(n, 3)
    y = X[:, 0] * 2 + X[:, 1] * 0.5 + np.random.randn(n) * 0.1
    return pd.DataFrame({
        "feature_a": X[:, 0],
        "feature_b": X[:, 1],
        "feature_c": X[:, 2],
        "price": y,
    })


@pytest.fixture
def csv_classification(sample_classification_df, tmp_path):
    path = tmp_path / "classification.csv"
    sample_classification_df.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def csv_regression(sample_regression_df, tmp_path):
    path = tmp_path / "regression.csv"
    sample_regression_df.to_csv(path, index=False)
    return str(path)


@pytest.fixture(autouse=True)
def redirect_output_dirs(tmp_path, monkeypatch):
    """Redirect chart and report output to tmp_path for all tests."""
    import tools.plot_tools as pt
    import nodes.reporting as rn
    monkeypatch.setattr(pt, "CHARTS_DIR", str(tmp_path / "charts"))
    monkeypatch.setattr(rn, "REPORTS_DIR", str(tmp_path / "reports"))
```

- [ ] **Step 3: Write failing tests in `tests/test_data_tools.py`**

```python
# tests/test_data_tools.py
import pandas as pd
import pytest
from tools.data_tools import load_csv, infer_column_types, stratified_split


def test_load_csv_returns_dataframe(csv_classification):
    df = load_csv(csv_classification)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 100
    assert "target" in df.columns


def test_infer_column_types_detects_numeric(sample_classification_df):
    schema = infer_column_types(sample_classification_df)
    assert schema["age"]["type"] == "numeric"
    assert schema["income"]["type"] == "numeric"


def test_infer_column_types_detects_categorical(sample_classification_df):
    schema = infer_column_types(sample_classification_df)
    assert schema["category"]["type"] == "categorical"


def test_infer_column_types_null_rate(tmp_path):
    df = pd.DataFrame({"a": [1.0, None, 3.0, 4.0], "b": ["x", "y", None, "z"]})
    schema = infer_column_types(df)
    assert schema["a"]["null_rate"] == 0.25
    assert schema["b"]["null_rate"] == 0.25


def test_stratified_split_sizes(sample_classification_df):
    X = sample_classification_df.drop(columns=["target"])
    y = sample_classification_df["target"]
    X_train, X_test, y_train, y_test = stratified_split(X, y, test_size=0.2)
    assert len(X_train) == 80
    assert len(X_test) == 20
    assert len(y_train) == 80
    assert len(y_test) == 20


def test_stratified_split_regression(sample_regression_df):
    X = sample_regression_df.drop(columns=["price"])
    y = sample_regression_df["price"]
    X_train, X_test, y_train, y_test = stratified_split(X, y, test_size=0.2)
    assert len(X_train) == 80
    assert len(X_test) == 20
```

- [ ] **Step 4: Run tests to confirm they fail**

```bash
uv run pytest tests/test_data_tools.py -v
```

Expected: `ModuleNotFoundError: No module named 'tools.data_tools'`

- [ ] **Step 5: Create `tools/data_tools.py`**

```python
# tools/data_tools.py
import pandas as pd
from sklearn.model_selection import train_test_split


def load_csv(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)


def infer_column_types(df: pd.DataFrame) -> dict:
    """
    Returns {col: {"type": "numeric"|"categorical"|"datetime", "null_rate": float}}.
    Tries datetime parsing for string columns before falling back to categorical.
    """
    result = {}
    for col in df.columns:
        null_rate = round(float(df[col].isnull().mean()), 4)
        if pd.api.types.is_numeric_dtype(df[col]):
            col_type = "numeric"
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            col_type = "datetime"
        else:
            try:
                pd.to_datetime(df[col].dropna().head(10), infer_datetime_format=True)
                col_type = "datetime"
            except Exception:
                col_type = "categorical"
        result[col] = {"type": col_type, "null_rate": null_rate}
    return result


def stratified_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Stratified split for classification (< 20 unique target values), plain split otherwise.
    Returns X_train, X_test, y_train, y_test.
    """
    try:
        if y.nunique() < 20:
            return train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
    except ValueError:
        pass
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
```

- [ ] **Step 6: Run tests to confirm they pass**

```bash
uv run pytest tests/test_data_tools.py -v
```

Expected: 6 tests PASSED.

- [ ] **Step 7: Commit**

```bash
git add tools/__init__.py nodes/__init__.py tests/__init__.py tests/conftest.py tools/data_tools.py tests/test_data_tools.py
git commit -m "feat: add data_tools with CSV loading, type inference, and split"
```

---

## Task 3: Tool — `tools/plot_tools.py`

**Files:**
- Create: `tools/plot_tools.py`
- Create: `tests/test_plot_tools.py`

- [ ] **Step 1: Write failing tests in `tests/test_plot_tools.py`**

```python
# tests/test_plot_tools.py
import os
import pytest
from tools.plot_tools import (
    save_distribution_plots,
    save_correlation_heatmap,
    save_feature_target_plots,
    CHARTS_DIR,
)


def test_save_distribution_plots_creates_files(sample_regression_df, tmp_path):
    paths = save_distribution_plots(sample_regression_df, "price")
    assert len(paths) > 0
    for p in paths:
        assert os.path.exists(p), f"Expected file: {p}"


def test_save_correlation_heatmap_creates_file(sample_regression_df):
    path = save_correlation_heatmap(sample_regression_df, "price")
    assert os.path.exists(path)
    assert "correlation_heatmap" in path


def test_save_feature_target_plots_creates_files(sample_regression_df):
    paths = save_feature_target_plots(
        sample_regression_df, "price", ["feature_a", "feature_b"]
    )
    assert len(paths) == 2
    for p in paths:
        assert os.path.exists(p)


def test_save_distribution_plots_categorical_target(sample_classification_df):
    # Should not crash when target is an int column
    paths = save_distribution_plots(sample_classification_df, "target")
    assert len(paths) > 0
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
uv run pytest tests/test_plot_tools.py -v
```

Expected: `ModuleNotFoundError: No module named 'tools.plot_tools'`

- [ ] **Step 3: Create `tools/plot_tools.py`**

```python
# tools/plot_tools.py
import os
import matplotlib
matplotlib.use("Agg")  # Non-interactive — safe for server/test environments
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

CHARTS_DIR = "outputs/charts"


def save_distribution_plots(df: pd.DataFrame, target_col: str) -> list[str]:
    """Histograms for every numeric column. Returns list of saved file paths."""
    os.makedirs(CHARTS_DIR, exist_ok=True)
    paths = []
    for col in df.select_dtypes(include="number").columns:
        path = os.path.join(CHARTS_DIR, f"hist_{col}.png")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(df[col].dropna(), bins=30, edgecolor="black")
        ax.set_title(f"Distribution of {col}")
        ax.set_xlabel(col)
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def save_correlation_heatmap(df: pd.DataFrame, target_col: str) -> str:
    """Correlation heatmap for all numeric columns. Returns saved file path."""
    os.makedirs(CHARTS_DIR, exist_ok=True)
    path = os.path.join(CHARTS_DIR, "correlation_heatmap.png")
    numeric_df = df.select_dtypes(include="number")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", ax=ax, cmap="coolwarm")
    ax.set_title("Feature Correlation Heatmap")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def save_feature_target_plots(
    df: pd.DataFrame, target_col: str, feature_cols: list[str]
) -> list[str]:
    """Scatter (numeric) or bar (categorical/low-cardinality) plots of feature vs target."""
    os.makedirs(CHARTS_DIR, exist_ok=True)
    paths = []
    for col in feature_cols:
        if not col or col not in df.columns:
            continue
        path = os.path.join(CHARTS_DIR, f"feature_{col}_vs_{target_col}.png")
        fig, ax = plt.subplots(figsize=(6, 4))
        if df[col].dtype == "object" or df[col].nunique() <= 20:
            df.groupby(col)[target_col].mean().plot(kind="bar", ax=ax)
            ax.set_title(f"{col} vs {target_col} (mean)")
            ax.set_xlabel(col)
        else:
            ax.scatter(df[col], df[target_col], alpha=0.3, s=10)
            ax.set_xlabel(col)
            ax.set_ylabel(target_col)
            ax.set_title(f"{col} vs {target_col}")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
uv run pytest tests/test_plot_tools.py -v
```

Expected: 4 tests PASSED.

- [ ] **Step 5: Commit**

```bash
git add tools/plot_tools.py tests/test_plot_tools.py
git commit -m "feat: add plot_tools with distribution, heatmap, and feature-target charts"
```

---

## Task 4: Tool — `tools/model_tools.py`

**Files:**
- Create: `tools/model_tools.py`
- Create: `tests/test_model_tools.py`

- [ ] **Step 1: Write failing tests in `tests/test_model_tools.py`**

```python
# tests/test_model_tools.py
import pytest
import numpy as np
from tools.model_tools import (
    ALGORITHM_REGISTRY,
    get_model,
    compute_classification_metrics,
    compute_regression_metrics,
)


def test_registry_contains_base_algorithms():
    required = {
        "LogisticRegression", "RandomForestClassifier",
        "LinearRegression", "Ridge", "RandomForestRegressor",
    }
    assert required.issubset(set(ALGORITHM_REGISTRY.keys()))


def test_get_model_returns_fresh_instance():
    m1 = get_model("LogisticRegression")
    m2 = get_model("LogisticRegression")
    assert m1 is not m2  # Each call returns a new instance


def test_get_model_raises_for_unknown():
    with pytest.raises(KeyError, match="not in registry"):
        get_model("UndefinedModel")


def test_classification_metrics_perfect():
    y = [0, 1, 0, 1, 0]
    metrics = compute_classification_metrics(y, y)
    assert metrics["accuracy"] == 1.0
    assert metrics["f1"] == 1.0


def test_classification_metrics_keys():
    y_true = [0, 1, 0, 1]
    y_pred = [0, 0, 0, 1]
    metrics = compute_classification_metrics(y_true, y_pred)
    assert set(metrics.keys()) == {"accuracy", "precision", "recall", "f1"}


def test_regression_metrics_keys():
    y_true = [1.0, 2.0, 3.0, 4.0]
    y_pred = [1.1, 1.9, 3.1, 3.9]
    metrics = compute_regression_metrics(y_true, y_pred)
    assert set(metrics.keys()) == {"mae", "rmse", "r2"}


def test_regression_metrics_perfect():
    y = [1.0, 2.0, 3.0]
    metrics = compute_regression_metrics(y, y)
    assert metrics["mae"] == 0.0
    assert metrics["r2"] == 1.0
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
uv run pytest tests/test_model_tools.py -v
```

Expected: `ModuleNotFoundError: No module named 'tools.model_tools'`

- [ ] **Step 3: Create `tools/model_tools.py`**

```python
# tools/model_tools.py
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score,
)

try:
    from xgboost import XGBClassifier, XGBRegressor
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False


ALGORITHM_REGISTRY: dict = {
    "LogisticRegression":    lambda: LogisticRegression(max_iter=1000, random_state=42),
    "RandomForestClassifier": lambda: RandomForestClassifier(n_estimators=100, random_state=42),
    "LinearRegression":      lambda: LinearRegression(),
    "Ridge":                 lambda: Ridge(),
    "RandomForestRegressor": lambda: RandomForestRegressor(n_estimators=100, random_state=42),
}

if _HAS_XGB:
    ALGORITHM_REGISTRY.update({
        "XGBClassifier": lambda: XGBClassifier(
            n_estimators=100, random_state=42, eval_metric="logloss", verbosity=0
        ),
        "XGBRegressor": lambda: XGBRegressor(
            n_estimators=100, random_state=42, verbosity=0
        ),
    })

if _HAS_LGB:
    ALGORITHM_REGISTRY.update({
        "LGBMClassifier": lambda: LGBMClassifier(
            n_estimators=100, random_state=42, verbosity=-1
        ),
        "LGBMRegressor": lambda: LGBMRegressor(
            n_estimators=100, random_state=42, verbosity=-1
        ),
    })


def get_model(name: str):
    """Return a fresh (unfitted) model instance by registry name."""
    if name not in ALGORITHM_REGISTRY:
        raise KeyError(
            f"Algorithm '{name}' not in registry. "
            f"Available: {sorted(ALGORITHM_REGISTRY.keys())}"
        )
    return ALGORITHM_REGISTRY[name]()


def compute_classification_metrics(y_true, y_pred) -> dict:
    return {
        "accuracy":  round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "recall":    round(float(recall_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "f1":        round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
    }


def compute_regression_metrics(y_true, y_pred) -> dict:
    return {
        "mae":  round(float(mean_absolute_error(y_true, y_pred)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "r2":   round(float(r2_score(y_true, y_pred)), 4),
    }
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
uv run pytest tests/test_model_tools.py -v
```

Expected: 7 tests PASSED.

- [ ] **Step 5: Commit**

```bash
git add tools/model_tools.py tests/test_model_tools.py
git commit -m "feat: add model_tools with algorithm registry and metrics calculators"
```

---

## Task 5: Node 1 — `nodes/profiling.py`

**Files:**
- Create: `nodes/profiling.py`
- Create: `tests/test_profiling.py`

- [ ] **Step 1: Write failing tests in `tests/test_profiling.py`**

```python
# tests/test_profiling.py
import pytest
from nodes.profiling import profiling_node


def _base_state(csv_path, target="target"):
    return {
        "file_path": csv_path,
        "target_column": target,
        "user_query": "",
        "logs": [],
    }


def test_profiling_returns_schema(csv_classification):
    result = profiling_node(_base_state(csv_classification))
    schema = result["schema"]
    assert "age" in schema
    assert "income" in schema
    assert schema["age"]["type"] == "numeric"
    assert schema["category"]["type"] == "categorical"


def test_profiling_returns_meta(csv_classification):
    result = profiling_node(_base_state(csv_classification))
    meta = result["schema"]["_meta"]
    assert meta["row_count"] == 100
    assert meta["col_count"] == 4
    assert "classification.csv" in meta["file_name"]
    assert isinstance(meta["mean_target"], float)
    assert "age" in meta["core_features"]


def test_profiling_detects_missing_values(tmp_path):
    import pandas as pd
    df = pd.DataFrame({
        "a": [1.0, None, 3.0, 4.0],
        "b": ["x", "y", "z", "w"],
        "target": [0, 1, 0, 1],
    })
    path = str(tmp_path / "missing.csv")
    df.to_csv(path, index=False)
    result = profiling_node({"file_path": path, "target_column": "target", "user_query": "", "logs": []})
    issues = result["quality_issues"]
    assert any("a" in issue and "缺失" in issue for issue in issues)


def test_profiling_detects_duplicates(tmp_path):
    import pandas as pd
    df = pd.DataFrame({
        "a": [1.0, 1.0, 3.0],
        "b": ["x", "x", "z"],
        "target": [0, 0, 1],
    })
    path = str(tmp_path / "dupes.csv")
    df.to_csv(path, index=False)
    result = profiling_node({"file_path": path, "target_column": "target", "user_query": "", "logs": []})
    assert any("重复" in issue for issue in result["quality_issues"])


def test_profiling_appends_to_logs(csv_classification):
    result = profiling_node(_base_state(csv_classification))
    assert any("[profiling]" in log for log in result["logs"])
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
uv run pytest tests/test_profiling.py -v
```

Expected: `ModuleNotFoundError: No module named 'nodes.profiling'`

- [ ] **Step 3: Create `nodes/profiling.py`**

```python
# nodes/profiling.py
import os
import pandas as pd
from app.state import AgentState
from tools.data_tools import load_csv, infer_column_types


def profiling_node(state: AgentState) -> dict:
    df = load_csv(state["file_path"])
    target_col = state["target_column"]

    column_info = infer_column_types(df)

    # Quality issues: missing values per column
    quality_issues = []
    for col, info in column_info.items():
        if info["null_rate"] > 0:
            quality_issues.append(
                f"{col} 列缺失率 {info['null_rate'] * 100:.1f}%"
            )

    # Duplicate rows
    dup_count = int(df.duplicated().sum())
    if dup_count > 0:
        quality_issues.append(f"发现 {dup_count} 行重复数据")

    # Target statistics
    target_series = df[target_col]
    mean_target = (
        round(float(target_series.mean()), 4)
        if pd.api.types.is_numeric_dtype(target_series)
        else 0.0
    )

    file_name = os.path.basename(state["file_path"])
    feature_cols = [c for c in df.columns if c != target_col]

    schema = dict(column_info)
    schema["_meta"] = {
        "file_name": file_name,
        "row_count": len(df),
        "col_count": len(df.columns),
        "data_scope": f"{len(df)} rows from {file_name}",
        "unit": "",
        "mean_target": mean_target,
        "core_features": ",".join(feature_cols),
    }

    return {
        "schema": schema,
        "quality_issues": quality_issues,
        "logs": list(state.get("logs", [])) + ["[profiling] 完成数据质量检测"],
    }
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
uv run pytest tests/test_profiling.py -v
```

Expected: 5 tests PASSED.

- [ ] **Step 5: Commit**

```bash
git add nodes/__init__.py nodes/profiling.py tests/test_profiling.py
git commit -m "feat: add Node 1 profiling — schema, quality_issues, and metadata"
```

---

## Task 6: Node 4 — `nodes/routing.py` (LLM)

**Files:**
- Create: `nodes/routing.py`
- Create: `tests/test_routing.py`

- [ ] **Step 1: Write failing tests in `tests/test_routing.py`**

```python
# tests/test_routing.py
from unittest.mock import MagicMock, patch
import pytest
from nodes.routing import routing_node, RoutingOutput


def _state_with_schema(csv_path, task_type, target="target"):
    """Build state as profiling_node would produce it."""
    schema = {
        "age":      {"type": "numeric",     "null_rate": 0.0},
        "income":   {"type": "numeric",     "null_rate": 0.0},
        "category": {"type": "categorical", "null_rate": 0.0},
        "_meta": {
            "file_name": "test.csv",
            "row_count": 100,
            "col_count": 4,
            "data_scope": "100 rows from test.csv",
            "unit": "",
            "mean_target": 0.5,
            "core_features": "age,income,category",
        },
    }
    return {
        "file_path": csv_path,
        "target_column": target,
        "user_query": "Predict customer churn",
        "schema": schema,
        "logs": [],
    }


def _mock_llm_response(task_type, algorithms, reasoning):
    mock_output = RoutingOutput(
        task_type=task_type,
        selected_algorithms=algorithms,
        reasoning=reasoning,
    )
    mock_structured = MagicMock()
    mock_structured.invoke.return_value = mock_output
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured
    return mock_llm


@patch("nodes.routing.ChatAnthropic")
def test_routing_classification(mock_chat, csv_classification):
    mock_chat.return_value = _mock_llm_response(
        "classification",
        ["LogisticRegression", "RandomForestClassifier"],
        "Binary target with few unique values.",
    )
    result = routing_node(_state_with_schema(csv_classification, "classification"))
    assert result["task_type"] == "classification"
    assert "LogisticRegression" in result["selected_algorithms"]
    assert isinstance(result["reasoning"], str) and len(result["reasoning"]) > 0


@patch("nodes.routing.ChatAnthropic")
def test_routing_regression(mock_chat, csv_regression):
    mock_chat.return_value = _mock_llm_response(
        "regression",
        ["Ridge", "RandomForestRegressor"],
        "Continuous target column.",
    )
    result = routing_node(_state_with_schema(csv_regression, "regression", target="price"))
    assert result["task_type"] == "regression"
    assert "Ridge" in result["selected_algorithms"]


@patch("nodes.routing.ChatAnthropic")
def test_routing_appends_to_logs(mock_chat, csv_classification):
    mock_chat.return_value = _mock_llm_response(
        "classification", ["LogisticRegression"], "reason"
    )
    result = routing_node(_state_with_schema(csv_classification, "classification"))
    assert any("[routing]" in log for log in result["logs"])
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
uv run pytest tests/test_routing.py -v
```

Expected: `ModuleNotFoundError: No module named 'nodes.routing'`

- [ ] **Step 3: Create `nodes/routing.py`**

```python
# nodes/routing.py
from pydantic import BaseModel
from langchain_anthropic import ChatAnthropic
from app.state import AgentState


class RoutingOutput(BaseModel):
    task_type: str              # "classification" | "regression"
    selected_algorithms: list[str]
    reasoning: str


_ALGORITHM_CHOICES = """
Classification algorithms: LogisticRegression, RandomForestClassifier, XGBClassifier, LGBMClassifier
Regression algorithms:     LinearRegression, Ridge, RandomForestRegressor, XGBRegressor, LGBMRegressor
"""


def routing_node(state: AgentState) -> dict:
    schema = state["schema"]
    target_col = state["target_column"]
    user_query = state.get("user_query", "")

    col_lines = [
        f"  - {col}: {info['type']}, null_rate={info['null_rate']}"
        for col, info in schema.items()
        if col != "_meta"
    ]
    col_text = "\n".join(col_lines)

    prompt = f"""You are a data science assistant. Analyze this dataset and select ML algorithms.

Dataset columns:
{col_text}

Target column: {target_col}
User task description: {user_query or "(not provided)"}

Available algorithms:
{_ALGORITHM_CHOICES}

Rules:
- task_type: "classification" if target has fewer than 20 unique values, else "regression"
- selected_algorithms: choose 2-3 algorithms appropriate for the task_type
- reasoning: one sentence explaining the algorithm choices
"""

    llm = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0)
    structured_llm = llm.with_structured_output(RoutingOutput)
    result: RoutingOutput = structured_llm.invoke(prompt)

    return {
        "task_type": result.task_type,
        "selected_algorithms": result.selected_algorithms,
        "reasoning": result.reasoning,
        "logs": list(state.get("logs", [])) + [
            f"[routing] task_type={result.task_type}, algorithms={result.selected_algorithms}"
        ],
    }
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
uv run pytest tests/test_routing.py -v
```

Expected: 3 tests PASSED.

- [ ] **Step 5: Commit**

```bash
git add nodes/routing.py tests/test_routing.py
git commit -m "feat: add Node 4 routing — LLM-based task type and algorithm selection"
```

---

## Task 7: Node 2 — `nodes/processing.py`

**Files:**
- Create: `nodes/processing.py`
- Create: `tests/test_processing.py`

- [ ] **Step 1: Write failing tests in `tests/test_processing.py`**

```python
# tests/test_processing.py
import pandas as pd
import numpy as np
import pytest
from nodes.processing import processing_node


def _base_state(csv_path, target, task_type, schema):
    return {
        "file_path": csv_path,
        "target_column": target,
        "task_type": task_type,
        "schema": schema,
        "logs": [],
    }


def _classification_schema():
    return {
        "age":      {"type": "numeric",     "null_rate": 0.0},
        "income":   {"type": "numeric",     "null_rate": 0.0},
        "category": {"type": "categorical", "null_rate": 0.0},
        "_meta": {},
    }


def test_processing_split_sizes(csv_classification):
    result = processing_node(
        _base_state(csv_classification, "target", "classification", _classification_schema())
    )
    assert len(result["X_train"]) == 80
    assert len(result["X_test"]) == 20


def test_processing_no_missing_values(csv_classification):
    result = processing_node(
        _base_state(csv_classification, "target", "classification", _classification_schema())
    )
    assert result["X_train"].isnull().sum().sum() == 0
    assert result["X_test"].isnull().sum().sum() == 0


def test_processing_encodes_categoricals(csv_classification):
    result = processing_node(
        _base_state(csv_classification, "target", "classification", _classification_schema())
    )
    # "category" column should now be numeric
    assert result["X_train"]["category"].dtype in [np.int64, np.float64, np.int32]


def test_processing_feature_names(csv_classification):
    result = processing_node(
        _base_state(csv_classification, "target", "classification", _classification_schema())
    )
    assert set(result["feature_names"]) == {"age", "income", "category"}


def test_processing_appends_log(csv_classification):
    result = processing_node(
        _base_state(csv_classification, "target", "classification", _classification_schema())
    )
    assert any("[processing]" in log for log in result["logs"])


def test_processing_fills_nulls_with_median(tmp_path):
    df = pd.DataFrame({
        "a": [1.0, None, 3.0, 4.0, 5.0],
        "target": [0, 1, 0, 1, 0],
    })
    path = str(tmp_path / "nulls.csv")
    df.to_csv(path, index=False)
    schema = {"a": {"type": "numeric", "null_rate": 0.2}, "_meta": {}}
    result = processing_node({"file_path": path, "target_column": "target",
                               "task_type": "classification", "schema": schema, "logs": []})
    assert result["X_train"].isnull().sum().sum() == 0
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
uv run pytest tests/test_processing.py -v
```

Expected: `ModuleNotFoundError: No module named 'nodes.processing'`

- [ ] **Step 3: Create `nodes/processing.py`**

```python
# nodes/processing.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from app.state import AgentState
from tools.data_tools import load_csv, stratified_split


def processing_node(state: AgentState) -> dict:
    df = load_csv(state["file_path"])
    schema = state["schema"]
    target_col = state["target_column"]
    task_type = state["task_type"]

    # Drop duplicate rows
    df = df.drop_duplicates().reset_index(drop=True)

    # Impute missing values
    for col in df.columns:
        if col == target_col:
            continue
        col_info = schema.get(col, {})
        if col_info.get("type") == "numeric":
            df[col] = df[col].fillna(df[col].median())
        else:
            mode = df[col].mode()
            df[col] = df[col].fillna(mode.iloc[0] if not mode.empty else "unknown")

    feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # Encode categorical features
    for col in X.columns:
        col_info = schema.get(col, {})
        if col_info.get("type") == "categorical":
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))

    # Encode categorical target for classification
    if task_type == "classification" and y.dtype == object:
        le_target = LabelEncoder()
        y = pd.Series(
            le_target.fit_transform(y.astype(str)), name=target_col, index=y.index
        )

    # Scale numeric features
    numeric_cols = [c for c in X.columns if schema.get(c, {}).get("type") == "numeric"]
    if numeric_cols:
        scaler = StandardScaler()
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    X_train, X_test, y_train, y_test = stratified_split(X, y)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": list(X.columns),
        "logs": list(state.get("logs", [])) + [
            f"[processing] 完成数据处理，训练集 {len(X_train)} 行，测试集 {len(X_test)} 行"
        ],
    }
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
uv run pytest tests/test_processing.py -v
```

Expected: 6 tests PASSED.

- [ ] **Step 5: Commit**

```bash
git add nodes/processing.py tests/test_processing.py
git commit -m "feat: add Node 2 processing — imputation, encoding, scaling, and train/test split"
```

---

## Task 8: Node 3 — `nodes/eda.py`

**Files:**
- Create: `nodes/eda.py`
- Create: `tests/test_eda.py`

- [ ] **Step 1: Write failing tests in `tests/test_eda.py`**

```python
# tests/test_eda.py
import os
import pytest
from nodes.eda import eda_node


def _state(csv_path, target, schema):
    return {
        "file_path": csv_path,
        "target_column": target,
        "schema": schema,
        "quality_issues": [],
        "logs": [],
    }


def _regression_schema():
    return {
        "feature_a": {"type": "numeric", "null_rate": 0.0},
        "feature_b": {"type": "numeric", "null_rate": 0.0},
        "feature_c": {"type": "numeric", "null_rate": 0.0},
        "_meta": {},
    }


def test_eda_returns_chart_paths(csv_regression):
    result = eda_node(_state(csv_regression, "price", _regression_schema()))
    assert len(result["charts"]) > 0
    for path in result["charts"]:
        assert os.path.exists(path), f"Chart file missing: {path}"


def test_eda_returns_eda_results_keys(csv_regression):
    result = eda_node(_state(csv_regression, "price", _regression_schema()))
    eda = result["eda_results"]
    required = {
        "top3_features", "distribution_desc", "layer_desc", "abnormal_desc",
        "key_charts", "feature_1", "feature_2", "feature_3",
        "feature_1_corr", "feature_2_corr", "feature_3_corr",
    }
    assert required.issubset(set(eda.keys()))


def test_eda_feature_correlations_are_floats(csv_regression):
    result = eda_node(_state(csv_regression, "price", _regression_schema()))
    eda = result["eda_results"]
    assert isinstance(eda["feature_1_corr"], float)
    assert 0.0 <= eda["feature_1_corr"] <= 1.0


def test_eda_top_feature_has_highest_correlation(csv_regression):
    result = eda_node(_state(csv_regression, "price", _regression_schema()))
    eda = result["eda_results"]
    # feature_a has coefficient 2.0, should be top correlated
    assert eda["feature_1"] == "feature_a"


def test_eda_appends_log(csv_regression):
    result = eda_node(_state(csv_regression, "price", _regression_schema()))
    assert any("[eda]" in log for log in result["logs"])
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
uv run pytest tests/test_eda.py -v
```

Expected: `ModuleNotFoundError: No module named 'nodes.eda'`

- [ ] **Step 3: Create `nodes/eda.py`**

```python
# nodes/eda.py
import pandas as pd
import numpy as np
from app.state import AgentState
from tools.data_tools import load_csv
from tools.plot_tools import (
    save_distribution_plots,
    save_correlation_heatmap,
    save_feature_target_plots,
)


def eda_node(state: AgentState) -> dict:
    df = load_csv(state["file_path"])
    target_col = state["target_column"]
    schema = state["schema"]

    # Prepare data (same cleaning as processing for consistent EDA)
    df = df.drop_duplicates().reset_index(drop=True)
    for col in df.columns:
        col_info = schema.get(col, {})
        if col_info.get("type") == "numeric":
            df[col] = df[col].fillna(df[col].median())
        else:
            mode = df[col].mode()
            df[col] = df[col].fillna(mode.iloc[0] if not mode.empty else "unknown")

    # Compute feature-target correlations (numeric only)
    numeric_df = df.select_dtypes(include="number")
    if target_col in numeric_df.columns and len(numeric_df.columns) > 1:
        corr = numeric_df.corr()[target_col].drop(target_col).abs().sort_values(ascending=False)
        top3_names = list(corr.head(3).index)
        top3_corrs = [round(float(corr.get(f, 0.0)), 4) for f in top3_names]
    else:
        top3_names = [c for c in df.columns if c != target_col][:3]
        top3_corrs = [0.0, 0.0, 0.0]

    # Pad to exactly 3 entries
    while len(top3_names) < 3:
        top3_names.append("")
        top3_corrs.append(0.0)

    # Distribution description for target
    if pd.api.types.is_numeric_dtype(df[target_col]):
        s = df[target_col]
        dist_desc = (
            f"中位数:{s.median():.2f};均值:{s.mean():.2f};"
            f"25%分位数:{s.quantile(0.25):.2f};75%分位数:{s.quantile(0.75):.2f}"
        )
    else:
        counts = df[target_col].value_counts()
        dist_desc = ";".join(f"{k}:{v}" for k, v in counts.head(5).items())

    # Layer description: group target by top feature
    layer_desc = ""
    top_feat = top3_names[0]
    if top_feat and pd.api.types.is_numeric_dtype(df[target_col]):
        if df[top_feat].nunique() <= 10:
            grouped = df.groupby(top_feat)[target_col].mean()
            layer_desc = ";".join(f"{k}均值:{v:.2f}" for k, v in grouped.items())
        else:
            df["_bin"] = pd.cut(df[top_feat], bins=3, labels=["低", "中", "高"])
            grouped = df.groupby("_bin", observed=True)[target_col].mean()
            layer_desc = ";".join(f"{k}:{v:.2f}" for k, v in grouped.items())
            df.drop(columns=["_bin"], inplace=True)

    # Abnormal value detection (IQR method)
    abnormal_parts = []
    for col in numeric_df.columns:
        if col == target_col:
            continue
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        n_outliers = int(((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum())
        if n_outliers > 0:
            pct = n_outliers / len(df) * 100
            abnormal_parts.append(f"{col}有{n_outliers}个异常值({pct:.1f}%)")
    abnormal_desc = ";".join(abnormal_parts)

    # Generate charts
    chart_paths = []
    chart_paths.extend(save_distribution_plots(df, target_col))
    chart_paths.append(save_correlation_heatmap(df, target_col))
    chart_paths.extend(
        save_feature_target_plots(df, target_col, [f for f in top3_names if f])
    )

    key_charts = ";".join(chart_paths[:5])

    # Append EDA quality findings to existing issues
    updated_issues = list(state.get("quality_issues", []))
    if abnormal_desc:
        updated_issues.append(f"EDA发现异常值: {abnormal_desc}")

    eda_results = {
        "top3_features":  ",".join(top3_names),
        "distribution_desc": dist_desc,
        "layer_desc":     layer_desc,
        "abnormal_desc":  abnormal_desc,
        "key_charts":     key_charts,
        "feature_1":      top3_names[0],
        "feature_2":      top3_names[1],
        "feature_3":      top3_names[2],
        "feature_1_corr": top3_corrs[0],
        "feature_2_corr": top3_corrs[1],
        "feature_3_corr": top3_corrs[2],
    }

    return {
        "charts": chart_paths,
        "quality_issues": updated_issues,
        "eda_results": eda_results,
        "logs": list(state.get("logs", [])) + [
            f"[eda] 生成 {len(chart_paths)} 张图表，TOP3特征: {top3_names}"
        ],
    }
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
uv run pytest tests/test_eda.py -v
```

Expected: 5 tests PASSED.

- [ ] **Step 5: Commit**

```bash
git add nodes/eda.py tests/test_eda.py
git commit -m "feat: add Node 3 EDA — correlations, stats, and chart generation"
```

---

## Task 9: Node 5 — `nodes/modeling.py`

**Files:**
- Create: `nodes/modeling.py`
- Create: `tests/test_modeling.py`

- [ ] **Step 1: Write failing tests in `tests/test_modeling.py`**

```python
# tests/test_modeling.py
import numpy as np
import pandas as pd
import pytest
from nodes.modeling import modeling_node


def _state(X_train, X_test, y_train, algorithms):
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "selected_algorithms": algorithms,
        "logs": [],
    }


@pytest.fixture
def classification_arrays():
    np.random.seed(0)
    n = 80
    X_train = pd.DataFrame({"a": np.random.randn(n), "b": np.random.randn(n)})
    y_train = pd.Series(np.random.choice([0, 1], n))
    X_test  = pd.DataFrame({"a": np.random.randn(20), "b": np.random.randn(20)})
    return X_train, X_test, y_train


def test_modeling_runs_each_algorithm(classification_arrays):
    X_train, X_test, y_train = classification_arrays
    result = modeling_node(_state(X_train, X_test, y_train,
                                  ["LogisticRegression", "RandomForestClassifier"]))
    assert "LogisticRegression" in result["model_results"]
    assert "RandomForestClassifier" in result["model_results"]


def test_modeling_predictions_are_list(classification_arrays):
    X_train, X_test, y_train = classification_arrays
    result = modeling_node(_state(X_train, X_test, y_train, ["LogisticRegression"]))
    y_pred = result["model_results"]["LogisticRegression"]["y_pred"]
    assert isinstance(y_pred, list)
    assert len(y_pred) == 20


def test_modeling_stores_fitted_model(classification_arrays):
    X_train, X_test, y_train = classification_arrays
    result = modeling_node(_state(X_train, X_test, y_train, ["LogisticRegression"]))
    model = result["model_results"]["LogisticRegression"]["model"]
    assert hasattr(model, "predict")


def test_modeling_handles_unknown_algorithm_gracefully(classification_arrays):
    X_train, X_test, y_train = classification_arrays
    result = modeling_node(_state(X_train, X_test, y_train, ["UnknownModel"]))
    assert "error" in result["model_results"]["UnknownModel"]


def test_modeling_appends_log(classification_arrays):
    X_train, X_test, y_train = classification_arrays
    result = modeling_node(_state(X_train, X_test, y_train, ["LogisticRegression"]))
    assert any("[modeling]" in log for log in result["logs"])
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
uv run pytest tests/test_modeling.py -v
```

Expected: `ModuleNotFoundError: No module named 'nodes.modeling'`

- [ ] **Step 3: Create `nodes/modeling.py`**

```python
# nodes/modeling.py
from app.state import AgentState
from tools.model_tools import get_model


def modeling_node(state: AgentState) -> dict:
    X_train = state["X_train"]
    y_train = state["y_train"]
    X_test  = state["X_test"]
    selected_algorithms = state["selected_algorithms"]

    model_results = {}
    for algo_name in selected_algorithms:
        try:
            model = get_model(algo_name)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            model_results[algo_name] = {
                "y_pred": y_pred.tolist() if hasattr(y_pred, "tolist") else list(y_pred),
                "model": model,
            }
        except Exception as exc:
            model_results[algo_name] = {"error": str(exc)}

    return {
        "model_results": model_results,
        "logs": list(state.get("logs", [])) + [
            f"[modeling] 训练完成: {list(model_results.keys())}"
        ],
    }
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
uv run pytest tests/test_modeling.py -v
```

Expected: 5 tests PASSED.

- [ ] **Step 5: Commit**

```bash
git add nodes/modeling.py tests/test_modeling.py
git commit -m "feat: add Node 5 modeling — train algorithms from registry, store predictions"
```

---

## Task 10: Node 6 — `nodes/evaluation.py`

**Files:**
- Create: `nodes/evaluation.py`
- Create: `tests/test_evaluation.py`

- [ ] **Step 1: Write failing tests in `tests/test_evaluation.py`**

```python
# tests/test_evaluation.py
import numpy as np
import pandas as pd
import pytest
from nodes.evaluation import evaluation_node


def _state(model_results, y_test, task_type):
    return {
        "model_results": model_results,
        "y_test": y_test,
        "task_type": task_type,
        "logs": [],
    }


def test_evaluation_classification_metrics_keys():
    y_test = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    model_results = {
        "ModelA": {"y_pred": [0, 1, 0, 1, 0, 1, 0, 1, 0, 0], "model": None},
        "ModelB": {"y_pred": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1], "model": None},
    }
    result = evaluation_node(_state(model_results, y_test, "classification"))
    for model in result["metrics"]:
        assert set(result["metrics"][model].keys()) == {"accuracy", "precision", "recall", "f1"}


def test_evaluation_selects_best_by_f1():
    y_test = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    model_results = {
        "BadModel":  {"y_pred": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], "model": None},
        "GoodModel": {"y_pred": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1], "model": None},
    }
    result = evaluation_node(_state(model_results, y_test, "classification"))
    assert result["best_model"] == "GoodModel"


def test_evaluation_regression_metrics_keys():
    y_test = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    model_results = {
        "ModelA": {"y_pred": [1.1, 1.9, 3.1, 3.9, 5.1], "model": None},
    }
    result = evaluation_node(_state(model_results, y_test, "regression"))
    assert set(result["metrics"]["ModelA"].keys()) == {"mae", "rmse", "r2"}


def test_evaluation_selects_best_by_r2():
    y_test = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    model_results = {
        "WeakModel":   {"y_pred": [3.0, 3.0, 3.0, 3.0, 3.0], "model": None},
        "StrongModel": {"y_pred": [1.1, 1.9, 3.1, 3.9, 5.1], "model": None},
    }
    result = evaluation_node(_state(model_results, y_test, "regression"))
    assert result["best_model"] == "StrongModel"


def test_evaluation_skips_error_models():
    y_test = pd.Series([0, 1, 0, 1, 0])
    model_results = {
        "BrokenModel": {"error": "something went wrong"},
        "WorkingModel": {"y_pred": [0, 1, 0, 1, 0], "model": None},
    }
    result = evaluation_node(_state(model_results, y_test, "classification"))
    assert "BrokenModel" not in result["metrics"]
    assert result["best_model"] == "WorkingModel"


def test_evaluation_appends_log():
    y_test = pd.Series([0, 1, 0, 1])
    model_results = {"M": {"y_pred": [0, 1, 0, 1], "model": None}}
    result = evaluation_node(_state(model_results, y_test, "classification"))
    assert any("[evaluation]" in log for log in result["logs"])
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
uv run pytest tests/test_evaluation.py -v
```

Expected: `ModuleNotFoundError: No module named 'nodes.evaluation'`

- [ ] **Step 3: Create `nodes/evaluation.py`**

```python
# nodes/evaluation.py
from app.state import AgentState
from tools.model_tools import compute_classification_metrics, compute_regression_metrics


def evaluation_node(state: AgentState) -> dict:
    model_results = state["model_results"]
    y_test = state["y_test"]
    task_type = state["task_type"]

    metrics = {}
    for algo_name, result in model_results.items():
        if "error" in result:
            continue
        y_pred = result["y_pred"]
        if task_type == "classification":
            metrics[algo_name] = compute_classification_metrics(y_test, y_pred)
        else:
            metrics[algo_name] = compute_regression_metrics(y_test, y_pred)

    if not metrics:
        raise RuntimeError("All models failed — cannot select best_model.")

    if task_type == "classification":
        best_model = max(metrics, key=lambda k: metrics[k]["f1"])
    else:
        best_model = max(metrics, key=lambda k: metrics[k]["r2"])

    return {
        "metrics": metrics,
        "best_model": best_model,
        "logs": list(state.get("logs", [])) + [
            f"[evaluation] 最优模型: {best_model}, 指标: {metrics[best_model]}"
        ],
    }
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
uv run pytest tests/test_evaluation.py -v
```

Expected: 6 tests PASSED.

- [ ] **Step 5: Commit**

```bash
git add nodes/evaluation.py tests/test_evaluation.py
git commit -m "feat: add Node 6 evaluation — metrics per model and best model selection"
```

---

## Task 11: Node 7 — `nodes/reporting.py` + Prompt

**Files:**
- Create: `prompts/report_prompt.txt`
- Create: `nodes/reporting.py`
- Create: `tests/test_reporting.py`

- [ ] **Step 1: Write failing tests in `tests/test_reporting.py`**

```python
# tests/test_reporting.py
import os
from unittest.mock import MagicMock, patch
import pytest
from nodes.reporting import reporting_node, _build_report_data


def _full_state(csv_path, tmp_path):
    return {
        "file_path": csv_path,
        "target_column": "price",
        "task_type": "regression",
        "user_query": "",
        "schema": {
            "feature_a": {"type": "numeric", "null_rate": 0.0},
            "feature_b": {"type": "numeric", "null_rate": 0.0},
            "_meta": {
                "file_name": "regression.csv",
                "row_count": 100,
                "col_count": 4,
                "data_scope": "100 rows from regression.csv",
                "unit": "",
                "mean_target": 0.5,
                "core_features": "feature_a,feature_b",
            },
        },
        "quality_issues": ["feature_a 列缺失率 2.0%"],
        "eda_results": {
            "top3_features": "feature_a,feature_b,feature_c",
            "distribution_desc": "中位数:0.5;均值:0.5;25%分位数:-0.7;75%分位数:0.7",
            "layer_desc": "低:-1.0;中:0.0;高:1.0",
            "abnormal_desc": "",
            "key_charts": "outputs/charts/hist_feature_a.png",
            "feature_1": "feature_a",
            "feature_2": "feature_b",
            "feature_3": "feature_c",
            "feature_1_corr": 0.95,
            "feature_2_corr": 0.45,
            "feature_3_corr": 0.10,
        },
        "metrics": {
            "Ridge": {"mae": 0.12, "rmse": 0.15, "r2": 0.97},
        },
        "best_model": "Ridge",
        "logs": [],
    }


def test_build_report_data_has_required_fields(csv_regression, tmp_path):
    state = _full_state(csv_regression, tmp_path)
    data = _build_report_data(state)
    required = {"timestamp", "file_name", "target_column", "task_type",
                 "row_count", "best_model", "mae", "rmse", "r2"}
    assert required.issubset(set(data.keys()))


def test_build_report_data_metrics_populated(csv_regression, tmp_path):
    state = _full_state(csv_regression, tmp_path)
    data = _build_report_data(state)
    assert data["r2"] == 0.97
    assert data["best_model"] == "Ridge"


@patch("nodes.reporting.ChatAnthropic")
def test_reporting_node_writes_file(mock_chat, csv_regression, tmp_path):
    mock_msg = MagicMock()
    mock_msg.content = "# Test Report\nThis is a generated report."
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_msg
    mock_chat.return_value = mock_llm

    state = _full_state(csv_regression, tmp_path)
    result = reporting_node(state)

    assert os.path.exists(result["report_path"])
    with open(result["report_path"], encoding="utf-8") as f:
        content = f.read()
    assert "Test Report" in content


@patch("nodes.reporting.ChatAnthropic")
def test_reporting_node_appends_log(mock_chat, csv_regression, tmp_path):
    mock_msg = MagicMock()
    mock_msg.content = "# Report"
    mock_chat.return_value.invoke.return_value = mock_msg

    state = _full_state(csv_regression, tmp_path)
    result = reporting_node(state)
    assert any("[reporting]" in log for log in result["logs"])
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
uv run pytest tests/test_reporting.py -v
```

Expected: `ModuleNotFoundError: No module named 'nodes.reporting'`

- [ ] **Step 3: Create `prompts/report_prompt.txt`**

```
You are a professional data analyst. Generate a complete, business-oriented Markdown analysis report in Chinese.
Use the following structured data to write the report. Be precise with numbers. Do not invent data not provided.

---DATA---
生成时间: {timestamp}
数据文件: {file_name}
目标列: {target_column}
任务类型: {task_type}

数据规模: {data_scope}
样本数: {row_count}，特征数: {col_count}
核心特征: {core_features}
目标列均值: {mean_target}
数据质量问题: {quality_issues}

EDA - TOP3相关特征: {top3_features}
目标列分布: {distribution_desc}
分层分析: {layer_desc}
异常值: {abnormal_desc}
关键图表路径: {key_charts}

特征相关性:
  - {feature_1}: 相关系数 {feature_1_corr}
  - {feature_2}: 相关系数 {feature_2_corr}
  - {feature_3}: 相关系数 {feature_3_corr}

最优模型: {best_model}
业务背景: {business_context}
特征-目标关联: {feature_relation_desc}
模型价值: {model_value}

核心结论:
  1. {conclusion_1}
  2. {conclusion_2}
  3. {conclusion_3}

短期建议 (1-3个月):
  1. {short_suggest_1}
  2. {short_suggest_2}

长期建议 (3-12个月):
  1. {long_suggest_1}
  2. {long_suggest_2}

风险提示:
  数据风险: {data_risk}
  执行风险: {exec_risk}
---END DATA---

Generate a professional Markdown report with these sections:
1. # 分析报告 — 标题 (include file_name and timestamp)
2. ## 一、数据概览 (data_scope, row_count, col_count, core_features, mean_target, quality_issues)
3. ## 二、探索性分析 (distribution_desc, layer_desc, abnormal_desc, and reference key_charts)
4. ## 三、特征重要性 (feature correlations with target, table format)
5. ## 四、模型评估 (best_model, all metrics, model_value)
6. ## 五、核心结论 (conclusion_1, conclusion_2, conclusion_3)
7. ## 六、行动建议 (short and long term suggestions)
8. ## 七、风险提示 (data_risk, exec_risk)

Use formal Chinese. Include all provided numbers. Format metrics as tables where appropriate.
```

- [ ] **Step 4: Create `nodes/reporting.py`**

```python
# nodes/reporting.py
import os
import datetime
from app.state import AgentState, ReportData
from langchain_anthropic import ChatAnthropic

REPORTS_DIR = "outputs/reports"


def _build_report_data(state: AgentState) -> ReportData:
    meta = state["schema"].get("_meta", {})
    eda = state.get("eda_results", {})
    metrics = state.get("metrics", {})
    best_model = state.get("best_model", "")
    task_type = state.get("task_type", "")
    target_col = state["target_column"]

    data: ReportData = {
        "timestamp":   datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "file_name":   meta.get("file_name", ""),
        "target_column": target_col,
        "task_type":   task_type,
        "data_scope":  meta.get("data_scope", ""),
        "row_count":   meta.get("row_count", 0),
        "col_count":   meta.get("col_count", 0),
        "core_features": meta.get("core_features", ""),
        "unit":        meta.get("unit", ""),
        "mean_target": meta.get("mean_target", 0.0),
        "quality_issues": ";".join(state.get("quality_issues", [])) or "无",
        "top3_features":  eda.get("top3_features", ""),
        "distribution_desc": eda.get("distribution_desc", ""),
        "layer_desc":    eda.get("layer_desc", ""),
        "abnormal_desc": eda.get("abnormal_desc", "") or "无",
        "key_charts":    eda.get("key_charts", ""),
        "feature_1":     eda.get("feature_1", ""),
        "feature_2":     eda.get("feature_2", ""),
        "feature_3":     eda.get("feature_3", ""),
        "feature_1_corr": eda.get("feature_1_corr", 0.0),
        "feature_2_corr": eda.get("feature_2_corr", 0.0),
        "feature_3_corr": eda.get("feature_3_corr", 0.0),
        "best_model": best_model,
        "business_context": (
            f"基于{meta.get('file_name','')}的{task_type}分析，"
            f"数据覆盖{meta.get('row_count',0)}条样本"
        ),
        "feature_relation_desc": (
            f"{eda.get('feature_1','')}与{target_col}强相关"
            f"（相关系数{eda.get('feature_1_corr',0.0):.2f}）"
            if eda.get("feature_1") else ""
        ),
        "model_value": f"{best_model}模型在{task_type}任务中表现最优",
        "conclusion_1": "",
        "conclusion_2": "",
        "conclusion_3": "",
        "short_suggest_1": "",
        "short_suggest_2": "",
        "long_suggest_1": "",
        "long_suggest_2": "",
        "data_risk": f"数据集样本量{meta.get('row_count', 0)}条",
        "exec_risk": f"模型{best_model}仅基于当前数据训练，决策需结合业务经验",
    }

    # Fill metric-dependent fields
    bm_metrics = metrics.get(best_model, {})
    if task_type == "classification":
        data["accuracy"]  = bm_metrics.get("accuracy", 0.0)
        data["precision"] = bm_metrics.get("precision", 0.0)
        data["recall"]    = bm_metrics.get("recall", 0.0)
        data["f1"]        = bm_metrics.get("f1", 0.0)
        data["conclusion_1"] = f"{eda.get('feature_1', '')}是影响{target_col}的核心因素（相关系数{eda.get('feature_1_corr',0):.2f}）"
        data["conclusion_2"] = f"{best_model}模型F1达到{bm_metrics.get('f1', 0.0):.2f}"
        data["conclusion_3"] = f"数据质量状况：{data['quality_issues']}"
    else:
        data["mae"]  = bm_metrics.get("mae", 0.0)
        data["rmse"] = bm_metrics.get("rmse", 0.0)
        data["r2"]   = bm_metrics.get("r2", 0.0)
        data["conclusion_1"] = f"{eda.get('feature_1', '')}是影响{target_col}的核心因素（相关系数{eda.get('feature_1_corr',0):.2f}）"
        data["conclusion_2"] = f"{best_model}模型R²达到{bm_metrics.get('r2', 0.0):.2f}"
        data["conclusion_3"] = f"数据质量状况：{data['quality_issues']}"

    data["short_suggest_1"] = f"优先优化{eda.get('feature_1', '')}相关业务策略"
    data["short_suggest_2"] = f"处理数据质量问题：{data['quality_issues']}"
    data["long_suggest_1"]  = f"补充与{eda.get('feature_1', '')}相关的衍生特征，提升模型解释力"
    data["long_suggest_2"]  = f"定期更新数据集，持续优化{best_model}模型"

    return data


def reporting_node(state: AgentState) -> dict:
    data = _build_report_data(state)

    prompt_path = "prompts/report_prompt.txt"
    with open(prompt_path, encoding="utf-8") as f:
        prompt_template = f.read()

    # Build safe format dict (convert all values to str for .format())
    format_data = {k: str(v) if v is not None else "" for k, v in data.items()}
    prompt = prompt_template.format(**format_data)

    llm = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0)
    response = llm.invoke(prompt)
    report_content = response.content

    os.makedirs(REPORTS_DIR, exist_ok=True)
    report_path = os.path.join(REPORTS_DIR, "analysis_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    return {
        "report_path": report_path,
        "logs": list(state.get("logs", [])) + [f"[reporting] 报告已生成: {report_path}"],
    }
```

- [ ] **Step 5: Run tests to confirm they pass**

```bash
uv run pytest tests/test_reporting.py -v
```

Expected: 4 tests PASSED.

- [ ] **Step 6: Commit**

```bash
git add prompts/report_prompt.txt nodes/reporting.py tests/test_reporting.py
git commit -m "feat: add Node 7 reporting — ReportData builder and LLM report generation"
```

---

## Task 12: Graph + CLI — `app/graph.py` and `app/main.py`

**Files:**
- Create: `app/graph.py`
- Create: `app/main.py`
- Create: `outputs/charts/.gitkeep`
- Create: `outputs/reports/.gitkeep`

- [ ] **Step 1: Write failing tests in `tests/test_graph.py`**

```python
# tests/test_graph.py
"""
Integration smoke test: run the full graph with mocked LLM nodes.
Uses a real CSV and all real non-LLM nodes; mocks routing and reporting.
"""
from unittest.mock import MagicMock, patch
import pytest
from nodes.routing import RoutingOutput
from app.graph import build_graph


def _mock_routing_llm(task_type, algorithms):
    mock_output = RoutingOutput(
        task_type=task_type,
        selected_algorithms=algorithms,
        reasoning="Mocked for integration test.",
    )
    mock_structured = MagicMock()
    mock_structured.invoke.return_value = mock_output
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured
    return mock_llm


def _mock_reporting_llm():
    mock_msg = MagicMock()
    mock_msg.content = "# 分析报告\n集成测试生成报告。"
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_msg
    return mock_llm


@patch("nodes.reporting.ChatAnthropic")
@patch("nodes.routing.ChatAnthropic")
def test_full_graph_regression(mock_routing_chat, mock_reporting_chat, csv_regression):
    mock_routing_chat.return_value = _mock_routing_llm(
        "regression", ["Ridge", "RandomForestRegressor"]
    )
    mock_reporting_chat.return_value = _mock_reporting_llm()

    graph = build_graph()
    initial_state = {
        "user_query": "Predict price",
        "file_path": csv_regression,
        "target_column": "price",
        "schema": {}, "quality_issues": [], "task_type": "",
        "selected_algorithms": [], "reasoning": "", "model_results": {},
        "metrics": {}, "best_model": "", "X_train": None, "X_test": None,
        "y_train": None, "y_test": None, "feature_names": [],
        "charts": [], "eda_results": {}, "report_path": "", "logs": [],
    }

    result = graph.invoke(initial_state)

    assert result["task_type"] == "regression"
    assert result["best_model"] in ["Ridge", "RandomForestRegressor"]
    assert len(result["metrics"]) > 0
    assert result["report_path"].endswith(".md")
    assert len(result["logs"]) >= 7  # one log per node


@patch("nodes.reporting.ChatAnthropic")
@patch("nodes.routing.ChatAnthropic")
def test_full_graph_classification(mock_routing_chat, mock_reporting_chat, csv_classification):
    mock_routing_chat.return_value = _mock_routing_llm(
        "classification", ["LogisticRegression", "RandomForestClassifier"]
    )
    mock_reporting_chat.return_value = _mock_reporting_llm()

    graph = build_graph()
    initial_state = {
        "user_query": "", "file_path": csv_classification, "target_column": "target",
        "schema": {}, "quality_issues": [], "task_type": "",
        "selected_algorithms": [], "reasoning": "", "model_results": {},
        "metrics": {}, "best_model": "", "X_train": None, "X_test": None,
        "y_train": None, "y_test": None, "feature_names": [],
        "charts": [], "eda_results": {}, "report_path": "", "logs": [],
    }

    result = graph.invoke(initial_state)
    assert result["task_type"] == "classification"
    assert result["best_model"] in ["LogisticRegression", "RandomForestClassifier"]
    assert "f1" in result["metrics"][result["best_model"]]
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
uv run pytest tests/test_graph.py -v
```

Expected: `ModuleNotFoundError: No module named 'app.graph'`

- [ ] **Step 3: Create `app/graph.py`**

```python
# app/graph.py
from langgraph.graph import StateGraph, END
from app.state import AgentState
from nodes.profiling import profiling_node
from nodes.routing import routing_node
from nodes.processing import processing_node
from nodes.eda import eda_node
from nodes.modeling import modeling_node
from nodes.evaluation import evaluation_node
from nodes.reporting import reporting_node


def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("profiling",  profiling_node)
    workflow.add_node("routing",    routing_node)
    workflow.add_node("processing", processing_node)
    workflow.add_node("eda",        eda_node)
    workflow.add_node("modeling",   modeling_node)
    workflow.add_node("evaluation", evaluation_node)
    workflow.add_node("reporting",  reporting_node)

    workflow.set_entry_point("profiling")
    workflow.add_edge("profiling",  "routing")
    workflow.add_edge("routing",    "processing")
    workflow.add_edge("processing", "eda")
    workflow.add_edge("eda",        "modeling")
    workflow.add_edge("modeling",   "evaluation")
    workflow.add_edge("evaluation", "reporting")
    workflow.add_edge("reporting",  END)

    return workflow.compile()
```

- [ ] **Step 4: Create `app/main.py`**

```python
# app/main.py
import argparse
from app.graph import build_graph


def main():
    parser = argparse.ArgumentParser(
        description="AutoInsight — Automated Data Mining Agent"
    )
    parser.add_argument("--file",   required=True, help="Path to input CSV file")
    parser.add_argument("--target", required=True, help="Target column name")
    parser.add_argument("--query",  default="",    help="Natural language task description")
    args = parser.parse_args()

    graph = build_graph()
    initial_state = {
        "user_query":        args.query,
        "file_path":         args.file,
        "target_column":     args.target,
        "schema":            {},
        "quality_issues":    [],
        "task_type":         "",
        "selected_algorithms": [],
        "reasoning":         "",
        "model_results":     {},
        "metrics":           {},
        "best_model":        "",
        "X_train":           None,
        "X_test":            None,
        "y_train":           None,
        "y_test":            None,
        "feature_names":     [],
        "charts":            [],
        "eda_results":       {},
        "report_path":       "",
        "logs":              [],
    }

    result = graph.invoke(initial_state)

    print(f"\nAnalysis complete!")
    print(f"Report:     {result['report_path']}")
    print(f"Charts:     {len(result['charts'])} files in outputs/charts/")
    print(f"Best model: {result['best_model']}")
    print(f"Task type:  {result['task_type']}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Create output placeholder files**

```bash
mkdir -p outputs/charts outputs/reports
touch outputs/charts/.gitkeep outputs/reports/.gitkeep
```

- [ ] **Step 6: Run integration tests**

```bash
uv run pytest tests/test_graph.py -v
```

Expected: 2 tests PASSED.

- [ ] **Step 7: Run full test suite to confirm nothing is broken**

```bash
uv run pytest tests/ -v
```

Expected: All tests PASSED. No failures or errors.

- [ ] **Step 8: Commit**

```bash
git add app/graph.py app/main.py outputs/charts/.gitkeep outputs/reports/.gitkeep tests/test_graph.py
git commit -m "feat: add LangGraph graph, CLI entry point, and integration smoke tests"
```

---

## Self-Review

### Spec Coverage Check

| Spec requirement | Covered by |
|---|---|
| Node 1: detect schema, types, nulls, duplicates | Task 5 — `profiling_node` |
| Node 4: LLM selects task_type + algorithms + reasoning | Task 6 — `routing_node` with `with_structured_output` |
| Node 4: returns `{task_type, selected_algorithms, reasoning}` | Task 6 — `RoutingOutput` Pydantic model |
| Algorithm registry (sklearn + XGBoost + LightGBM) | Task 4 — `ALGORITHM_REGISTRY` |
| Node 2: impute, encode, scale, 80/20 split | Task 7 — `processing_node` |
| Node 3: distribution plots, heatmap, feature-target plots | Task 8 — `eda_node` |
| Node 3: top3_features, distribution_desc, layer_desc, etc. | Task 8 — `eda_results` dict |
| Node 5: train selected algorithms from registry | Task 9 — `modeling_node` |
| Node 6: classification metrics (acc/prec/rec/f1) | Task 10 — `compute_classification_metrics` |
| Node 6: regression metrics (MAE/RMSE/R²) | Task 10 — `compute_regression_metrics` |
| Node 6: select best model by F1 or R² | Task 10 — `evaluation_node` |
| Node 7: LLM-generated markdown report | Task 11 — `reporting_node` + `report_prompt.txt` |
| Node 7: all 43 ReportData variables populated | Task 11 — `_build_report_data` |
| `state.py`: `selected_algorithms`, `reasoning`, `X_train/y_train/X_test/y_test`, `feature_names` | Task 1 — already in state.py, `eda_results` added |
| `pyproject.toml`: xgboost, lightgbm | Task 1 |
| `app/graph.py`: LangGraph orchestration | Task 12 |
| `app/main.py`: CLI | Task 12 |
| Outputs to `outputs/charts/` and `outputs/reports/` | Tasks 3, 11, 12 |
| `ChatAnthropic` + `langchain-anthropic` for all LLM calls | Tasks 6, 11 |

### Placeholder Scan

No TBD, TODO, or "similar to Task N" patterns in this plan. Every step contains the complete code needed.

### Type Consistency

- `RoutingOutput.selected_algorithms: list[str]` — matches `AgentState.selected_algorithms: list[str]` ✓
- `RoutingOutput.task_type: str` — matches `AgentState.task_type: str` ✓
- `RoutingOutput.reasoning: str` — matches `AgentState.reasoning: str` ✓
- `modeling_node` reads `state["X_train"]` (pd.DataFrame) — set by `processing_node` ✓
- `evaluation_node` reads `state["model_results"]` dict — set by `modeling_node` ✓
- `_build_report_data` reads `state["eda_results"]` dict — set by `eda_node` ✓
- `schema["_meta"]` dict — set by `profiling_node`, read by `reporting_node._build_report_data` ✓
- `CHARTS_DIR` module variable in `plot_tools` — patched by `conftest.py` `autouse` fixture ✓
- `REPORTS_DIR` module variable in `reporting` — patched by `conftest.py` `autouse` fixture ✓
