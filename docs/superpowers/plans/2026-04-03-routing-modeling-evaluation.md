# AutoInsight: Routing + Modeling + Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement Node 4 (LLM-based task routing + algorithm selection), Node 5 (model training), and Node 6 (evaluation + best model selection) for the AutoInsight pipeline.

**Architecture:** These three nodes sit in the middle of the pipeline. Node 4 receives `schema` from profiling and uses `ChatAnthropic` + `.with_structured_output()` to output `task_type`, `selected_algorithms`, `reasoning`. Nodes 5–6 receive train/test splits from `data_processing` and produce `model_results`, `metrics`, `best_model`.

**Tech Stack:** langchain-anthropic, scikit-learn, XGBoost, LightGBM, pydantic, pytest, uv

---

## Critical Interface Contracts

Read before writing a single line of code:

### Input from `data_processing` (upstream — already merged)

```python
# nodes/processing.py — data_processing() output
state["X_train"]  # pd.DataFrame — scaled features, 80% rows
state["X_test"]   # pd.DataFrame — scaled features, 20% rows
state["y_train"]  # pd.DataFrame — ONE COLUMN, shape (n, 1)  ← NOT a Series
state["y_test"]   # pd.DataFrame — ONE COLUMN, shape (n, 1)  ← NOT a Series
state["feature_names"]  # list[str]
```

**`y_train` / `y_test` are DataFrames, not Series.** Always call:
```python
model.fit(X_train, y_train.values.ravel())
```
Never `model.fit(X_train, y_train)` — XGBoost/LightGBM will raise a DataConversionWarning or shape error.

### Output expected by downstream nodes

Node 4 must write to state:
```python
state["task_type"]            # str: "classification" | "regression"
state["selected_algorithms"]  # list[str]: e.g. ["Ridge", "XGBRegressor"]
state["reasoning"]            # str: one sentence for the report
```

Node 5 must write to state:
```python
state["model_results"]  # dict: {algo_name: {"y_pred": list, "model": fitted_model}}
```

Node 6 must write to state:
```python
state["metrics"]     # dict: {algo_name: {"accuracy":..., "f1":...}} or {algo_name: {"mae":..., "r2":...}}
state["best_model"]  # str: name of the best algorithm
```

---

## File Map

| Action | Path | Owner |
|--------|------|-------|
| Modify | `pyproject.toml` | Add xgboost, lightgbm, pydantic |
| Create | `tools/__init__.py` | Empty package marker |
| Create | `tools/model_tools.py` | Algorithm registry + metric calculators |
| Create | `tests/__init__.py` | Empty package marker |
| Create | `tests/conftest.py` | Shared fixtures |
| Create | `nodes/routing.py` | Node 4: LLM task type + algorithm selection |
| Create | `nodes/modeling.py` | Node 5: train each selected algorithm |
| Create | `nodes/evaluation.py` | Node 6: compute metrics, pick best model |
| Create | `tests/test_model_tools.py` | Unit tests |
| Create | `tests/test_routing.py` | Unit tests (mock LLM) |
| Create | `tests/test_modeling.py` | Unit tests |
| Create | `tests/test_evaluation.py` | Unit tests |

**Branch strategy:** Work on `feature/modeling`. After all tests pass, open a PR to `main`.

---

## Task 1: Update Dependencies

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add xgboost, lightgbm, pydantic to pyproject.toml**

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

- [ ] **Step 2: Install**

```bash
uv sync --dev
```

Expected: Resolves and installs xgboost, lightgbm, pydantic, pytest with no errors.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "chore: add xgboost, lightgbm, pydantic deps"
```

---

## Task 2: `tools/model_tools.py` — Algorithm Registry + Metrics

**Files:**
- Create: `tools/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `tools/model_tools.py`
- Create: `tests/test_model_tools.py`

- [ ] **Step 1: Create empty package markers**

```bash
touch tools/__init__.py tests/__init__.py
```

- [ ] **Step 2: Create `tests/conftest.py`**

```python
# tests/conftest.py
import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def clf_arrays():
    """Binary classification arrays matching data_processing output shape."""
    np.random.seed(42)
    n_train, n_test = 80, 20
    X_train = pd.DataFrame({"a": np.random.randn(n_train), "b": np.random.randn(n_train)})
    X_test  = pd.DataFrame({"a": np.random.randn(n_test),  "b": np.random.randn(n_test)})
    # y as DataFrame — same as data_processing produces
    y_train = pd.DataFrame({"target": np.random.choice([0, 1], n_train)})
    y_test  = pd.DataFrame({"target": np.random.choice([0, 1], n_test)})
    return X_train, X_test, y_train, y_test


@pytest.fixture
def reg_arrays():
    """Regression arrays matching data_processing output shape."""
    np.random.seed(42)
    n_train, n_test = 80, 20
    X = np.random.randn(100, 3)
    y = X[:, 0] * 2.0 + X[:, 1] * 0.5 + np.random.randn(100) * 0.05
    X_train = pd.DataFrame(X[:n_train], columns=["fa", "fb", "fc"])
    X_test  = pd.DataFrame(X[n_train:], columns=["fa", "fb", "fc"])
    y_train = pd.DataFrame(y[:n_train], columns=["price"])
    y_test  = pd.DataFrame(y[n_train:], columns=["price"])
    return X_train, X_test, y_train, y_test
```

- [ ] **Step 3: Write failing tests in `tests/test_model_tools.py`**

```python
# tests/test_model_tools.py
import pytest
from tools.model_tools import (
    ALGORITHM_REGISTRY,
    get_model,
    compute_classification_metrics,
    compute_regression_metrics,
)


def test_registry_contains_required_algorithms():
    required = {
        "LogisticRegression", "RandomForestClassifier",
        "LinearRegression", "Ridge", "RandomForestRegressor",
    }
    assert required.issubset(set(ALGORITHM_REGISTRY.keys()))


def test_registry_contains_xgb_lgbm():
    assert "XGBClassifier" in ALGORITHM_REGISTRY
    assert "XGBRegressor" in ALGORITHM_REGISTRY
    assert "LGBMClassifier" in ALGORITHM_REGISTRY
    assert "LGBMRegressor" in ALGORITHM_REGISTRY


def test_get_model_returns_fresh_instance():
    m1 = get_model("LogisticRegression")
    m2 = get_model("LogisticRegression")
    assert m1 is not m2


def test_get_model_raises_for_unknown():
    with pytest.raises(KeyError, match="not in registry"):
        get_model("NoSuchModel")


def test_classification_metrics_keys():
    y_true = [0, 1, 0, 1]
    y_pred = [0, 0, 0, 1]
    metrics = compute_classification_metrics(y_true, y_pred)
    assert set(metrics.keys()) == {"accuracy", "precision", "recall", "f1"}


def test_classification_metrics_perfect():
    y = [0, 1, 0, 1, 0]
    metrics = compute_classification_metrics(y, y)
    assert metrics["accuracy"] == 1.0
    assert metrics["f1"] == 1.0


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

- [ ] **Step 4: Run to confirm failure**

```bash
uv run pytest tests/test_model_tools.py -v
```

Expected: `ModuleNotFoundError: No module named 'tools.model_tools'`

- [ ] **Step 5: Create `tools/model_tools.py`**

```python
# tools/model_tools.py
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score,
)
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor


ALGORITHM_REGISTRY: dict = {
    "LogisticRegression":     lambda: LogisticRegression(max_iter=1000, random_state=42),
    "RandomForestClassifier": lambda: RandomForestClassifier(n_estimators=100, random_state=42),
    "LinearRegression":       lambda: LinearRegression(),
    "Ridge":                  lambda: Ridge(),
    "RandomForestRegressor":  lambda: RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBClassifier":          lambda: XGBClassifier(
        n_estimators=100, random_state=42, eval_metric="logloss", verbosity=0
    ),
    "XGBRegressor":           lambda: XGBRegressor(
        n_estimators=100, random_state=42, verbosity=0
    ),
    "LGBMClassifier":         lambda: LGBMClassifier(
        n_estimators=100, random_state=42, verbosity=-1
    ),
    "LGBMRegressor":          lambda: LGBMRegressor(
        n_estimators=100, random_state=42, verbosity=-1
    ),
}


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

- [ ] **Step 6: Run tests to confirm pass**

```bash
uv run pytest tests/test_model_tools.py -v
```

Expected: 8 tests PASSED.

- [ ] **Step 7: Commit**

```bash
git add tools/__init__.py tests/__init__.py tests/conftest.py tools/model_tools.py tests/test_model_tools.py
git commit -m "feat: add model_tools — algorithm registry (sklearn+XGB+LGB) and metrics"
```

---

## Task 3: Node 4 — `nodes/routing.py`

**Files:**
- Create: `nodes/routing.py`
- Create: `tests/test_routing.py`

- [ ] **Step 1: Write failing tests in `tests/test_routing.py`**

```python
# tests/test_routing.py
from unittest.mock import MagicMock, patch
import pytest
from nodes.routing import routing_node, RoutingOutput


def _schema():
    return {
        "age":      {"type": "numeric",     "null_rate": 0.0},
        "income":   {"type": "numeric",     "null_rate": 0.0},
        "category": {"type": "categorical", "null_rate": 0.0},
        "_meta": {
            "file_name": "test.csv",
            "row_count": 100,
            "col_count": 4,
            "data_scope": "100 rows",
            "unit": "",
            "mean_target": 0.5,
            "core_features": "age,income,category",
        },
    }


def _mock_llm(task_type, algorithms, reasoning="test reason"):
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
def test_routing_classification_task_type(mock_chat):
    mock_chat.return_value = _mock_llm(
        "classification", ["LogisticRegression", "RandomForestClassifier"]
    )
    result = routing_node({
        "target_column": "target",
        "user_query": "Predict churn",
        "schema": _schema(),
        "logs": [],
    })
    assert result["task_type"] == "classification"


@patch("nodes.routing.ChatAnthropic")
def test_routing_returns_algorithms_list(mock_chat):
    mock_chat.return_value = _mock_llm(
        "regression", ["Ridge", "XGBRegressor"]
    )
    result = routing_node({
        "target_column": "price",
        "user_query": "",
        "schema": _schema(),
        "logs": [],
    })
    assert isinstance(result["selected_algorithms"], list)
    assert len(result["selected_algorithms"]) >= 1
    assert "Ridge" in result["selected_algorithms"]


@patch("nodes.routing.ChatAnthropic")
def test_routing_returns_reasoning_string(mock_chat):
    mock_chat.return_value = _mock_llm(
        "classification", ["LogisticRegression"], "Continuous target with many unique values."
    )
    result = routing_node({
        "target_column": "target",
        "user_query": "",
        "schema": _schema(),
        "logs": [],
    })
    assert isinstance(result["reasoning"], str)
    assert len(result["reasoning"]) > 0


@patch("nodes.routing.ChatAnthropic")
def test_routing_appends_to_logs(mock_chat):
    mock_chat.return_value = _mock_llm("classification", ["LogisticRegression"])
    result = routing_node({
        "target_column": "target",
        "user_query": "",
        "schema": _schema(),
        "logs": ["[profiling] done"],
    })
    assert len(result["logs"]) == 2
    assert any("[routing]" in log for log in result["logs"])
```

- [ ] **Step 2: Run to confirm failure**

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

    prompt = f"""You are a data science assistant. Analyze this dataset schema and select the best ML algorithms.

Dataset columns:
{chr(10).join(col_lines)}

Target column: {target_col}
User task description: {user_query or "(not provided)"}

Available algorithms:
{_ALGORITHM_CHOICES}

Rules:
- task_type: "classification" if target has fewer than 20 unique values or the column type is categorical, else "regression"
- selected_algorithms: choose 2-3 algorithms appropriate for the task_type from the list above
- reasoning: one concise sentence explaining the algorithm selection
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

- [ ] **Step 4: Run tests to confirm pass**

```bash
uv run pytest tests/test_routing.py -v
```

Expected: 4 tests PASSED.

- [ ] **Step 5: Commit**

```bash
git add nodes/routing.py tests/test_routing.py
git commit -m "feat: add Node 4 routing — LLM task classification and algorithm selection"
```

---

## Task 4: Node 5 — `nodes/modeling.py`

**Files:**
- Create: `nodes/modeling.py`
- Create: `tests/test_modeling.py`

**Key constraint:** `y_train` arriving from `data_processing` is a **pd.DataFrame** with shape `(n, 1)`. Always call `y_train.values.ravel()` before passing to any model's `.fit()`.

- [ ] **Step 1: Write failing tests in `tests/test_modeling.py`**

```python
# tests/test_modeling.py
import pytest
import pandas as pd
from nodes.modeling import modeling_node


def _state(X_train, X_test, y_train, algorithms):
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,   # pd.DataFrame shape (n, 1) — same as data_processing output
        "selected_algorithms": algorithms,
        "logs": [],
    }


def test_modeling_runs_each_algorithm(clf_arrays):
    X_train, X_test, y_train, _ = clf_arrays
    result = modeling_node(_state(X_train, X_test, y_train,
                                  ["LogisticRegression", "RandomForestClassifier"]))
    assert "LogisticRegression" in result["model_results"]
    assert "RandomForestClassifier" in result["model_results"]


def test_modeling_predictions_are_list(clf_arrays):
    X_train, X_test, y_train, _ = clf_arrays
    result = modeling_node(_state(X_train, X_test, y_train, ["LogisticRegression"]))
    y_pred = result["model_results"]["LogisticRegression"]["y_pred"]
    assert isinstance(y_pred, list)
    assert len(y_pred) == 20


def test_modeling_stores_fitted_model(clf_arrays):
    X_train, X_test, y_train, _ = clf_arrays
    result = modeling_node(_state(X_train, X_test, y_train, ["LogisticRegression"]))
    model = result["model_results"]["LogisticRegression"]["model"]
    assert hasattr(model, "predict")


def test_modeling_works_with_dataframe_y(clf_arrays):
    """Explicitly test that DataFrame y_train does not crash any algorithm."""
    X_train, X_test, y_train, _ = clf_arrays
    assert isinstance(y_train, pd.DataFrame), "fixture must return DataFrame"
    result = modeling_node(_state(X_train, X_test, y_train,
                                  ["LogisticRegression", "XGBClassifier"]))
    assert "LogisticRegression" in result["model_results"]
    assert "XGBClassifier" in result["model_results"]
    # Neither should be an error entry
    assert "error" not in result["model_results"]["LogisticRegression"]
    assert "error" not in result["model_results"]["XGBClassifier"]


def test_modeling_handles_unknown_algorithm_gracefully(clf_arrays):
    X_train, X_test, y_train, _ = clf_arrays
    result = modeling_node(_state(X_train, X_test, y_train, ["NoSuchModel"]))
    assert "error" in result["model_results"]["NoSuchModel"]


def test_modeling_regression(reg_arrays):
    X_train, X_test, y_train, _ = reg_arrays
    result = modeling_node(_state(X_train, X_test, y_train,
                                  ["Ridge", "RandomForestRegressor"]))
    assert "Ridge" in result["model_results"]
    assert "error" not in result["model_results"]["Ridge"]
    assert len(result["model_results"]["Ridge"]["y_pred"]) == 20


def test_modeling_appends_log(clf_arrays):
    X_train, X_test, y_train, _ = clf_arrays
    result = modeling_node(_state(X_train, X_test, y_train, ["LogisticRegression"]))
    assert any("[modeling]" in log for log in result["logs"])
```

- [ ] **Step 2: Run to confirm failure**

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

    # y_train from data_processing is a DataFrame with shape (n, 1).
    # All sklearn/XGB/LGB models expect a 1-D array — ravel() converts safely.
    y_fit = y_train.values.ravel() if hasattr(y_train, "values") else y_train

    model_results = {}
    for algo_name in selected_algorithms:
        try:
            model = get_model(algo_name)
            model.fit(X_train, y_fit)
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

- [ ] **Step 4: Run tests to confirm pass**

```bash
uv run pytest tests/test_modeling.py -v
```

Expected: 7 tests PASSED.

- [ ] **Step 5: Commit**

```bash
git add nodes/modeling.py tests/test_modeling.py
git commit -m "feat: add Node 5 modeling — train algorithms from registry, handle DataFrame y"
```

---

## Task 5: Node 6 — `nodes/evaluation.py`

**Files:**
- Create: `nodes/evaluation.py`
- Create: `tests/test_evaluation.py`

**Key constraint:** `y_test` from `data_processing` is a **pd.DataFrame** with shape `(n, 1)`. Pass `y_test.values.ravel()` to all metric functions.

- [ ] **Step 1: Write failing tests in `tests/test_evaluation.py`**

```python
# tests/test_evaluation.py
import pandas as pd
import pytest
from nodes.evaluation import evaluation_node


def _state(model_results, y_test, task_type):
    # y_test as DataFrame — same as data_processing output
    if not isinstance(y_test, pd.DataFrame):
        y_test = pd.DataFrame({"target": y_test})
    return {
        "model_results": model_results,
        "y_test": y_test,
        "task_type": task_type,
        "logs": [],
    }


def test_evaluation_classification_metrics_keys():
    y_test = pd.DataFrame({"t": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]})
    model_results = {
        "ModelA": {"y_pred": [0, 1, 0, 1, 0, 1, 0, 1, 0, 0], "model": None},
        "ModelB": {"y_pred": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1], "model": None},
    }
    result = evaluation_node(_state(model_results, y_test, "classification"))
    for model in result["metrics"]:
        assert set(result["metrics"][model].keys()) == {"accuracy", "precision", "recall", "f1"}


def test_evaluation_selects_best_by_f1():
    y_test = pd.DataFrame({"t": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]})
    model_results = {
        "BadModel":  {"y_pred": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], "model": None},
        "GoodModel": {"y_pred": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1], "model": None},
    }
    result = evaluation_node(_state(model_results, y_test, "classification"))
    assert result["best_model"] == "GoodModel"


def test_evaluation_regression_metrics_keys():
    y_test = pd.DataFrame({"p": [1.0, 2.0, 3.0, 4.0, 5.0]})
    model_results = {
        "ModelA": {"y_pred": [1.1, 1.9, 3.1, 3.9, 5.1], "model": None},
    }
    result = evaluation_node(_state(model_results, y_test, "regression"))
    assert set(result["metrics"]["ModelA"].keys()) == {"mae", "rmse", "r2"}


def test_evaluation_selects_best_by_r2():
    y_test = pd.DataFrame({"p": [1.0, 2.0, 3.0, 4.0, 5.0]})
    model_results = {
        "WeakModel":   {"y_pred": [3.0, 3.0, 3.0, 3.0, 3.0], "model": None},
        "StrongModel": {"y_pred": [1.1, 1.9, 3.1, 3.9, 5.1], "model": None},
    }
    result = evaluation_node(_state(model_results, y_test, "regression"))
    assert result["best_model"] == "StrongModel"


def test_evaluation_skips_error_models():
    y_test = pd.DataFrame({"t": [0, 1, 0, 1, 0]})
    model_results = {
        "BrokenModel":  {"error": "something went wrong"},
        "WorkingModel": {"y_pred": [0, 1, 0, 1, 0], "model": None},
    }
    result = evaluation_node(_state(model_results, y_test, "classification"))
    assert "BrokenModel" not in result["metrics"]
    assert result["best_model"] == "WorkingModel"


def test_evaluation_appends_log():
    y_test = pd.DataFrame({"t": [0, 1, 0, 1]})
    model_results = {"M": {"y_pred": [0, 1, 0, 1], "model": None}}
    result = evaluation_node(_state(model_results, y_test, "classification"))
    assert any("[evaluation]" in log for log in result["logs"])


def test_evaluation_raises_if_all_models_failed():
    y_test = pd.DataFrame({"t": [0, 1]})
    model_results = {"Broken": {"error": "failed"}}
    with pytest.raises(RuntimeError, match="All models failed"):
        evaluation_node(_state(model_results, y_test, "classification"))
```

- [ ] **Step 2: Run to confirm failure**

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

    # y_test from data_processing is a DataFrame with shape (n, 1).
    y_true = y_test.values.ravel() if hasattr(y_test, "values") else y_test

    metrics = {}
    for algo_name, result in model_results.items():
        if "error" in result:
            continue
        y_pred = result["y_pred"]
        if task_type == "classification":
            metrics[algo_name] = compute_classification_metrics(y_true, y_pred)
        else:
            metrics[algo_name] = compute_regression_metrics(y_true, y_pred)

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

- [ ] **Step 4: Run tests to confirm pass**

```bash
uv run pytest tests/test_evaluation.py -v
```

Expected: 7 tests PASSED.

- [ ] **Step 5: Run the full test suite**

```bash
uv run pytest tests/ -v
```

Expected: All tests PASSED.

- [ ] **Step 6: Commit**

```bash
git add nodes/evaluation.py tests/test_evaluation.py
git commit -m "feat: add Node 6 evaluation — metrics per model, best model by F1/R²"
```

---

## Self-Review

### Spec Coverage

| Requirement | Covered by |
|---|---|
| Node 4: LLM with `with_structured_output()` | Task 3 — `RoutingOutput` Pydantic model |
| Node 4: returns `task_type`, `selected_algorithms`, `reasoning` | Task 3 — `routing_node` dict return |
| Algorithm registry — sklearn + XGBoost + LightGBM | Task 2 — `ALGORITHM_REGISTRY` |
| Node 5: train from registry | Task 4 — `modeling_node` |
| Node 5: y_train is DataFrame → `.values.ravel()` | Task 4 — `y_fit = y_train.values.ravel()` |
| Node 6: classification metrics (accuracy/precision/recall/f1) | Task 5 — `compute_classification_metrics` |
| Node 6: regression metrics (MAE/RMSE/R²) | Task 5 — `compute_regression_metrics` |
| Node 6: best model by F1 (classification) or R² (regression) | Task 5 — `evaluation_node` |
| Node 6: y_test is DataFrame → `.values.ravel()` | Task 5 — `y_true = y_test.values.ravel()` |
| Error models are skipped gracefully | Task 4 + Task 5 |

### Placeholder Scan

No TBD, TODO, or "similar to Task N" in this plan.

### Type Consistency

- `RoutingOutput.selected_algorithms: list[str]` → `AgentState.selected_algorithms: list[str]` ✓
- `modeling_node` input `y_train: pd.DataFrame` → `.values.ravel()` before fit ✓
- `evaluation_node` input `y_test: pd.DataFrame` → `.values.ravel()` before metrics ✓
- `model_results[name]["y_pred"]` is `list` (`.tolist()`) — consistent between nodes 5 and 6 ✓
- `metrics[name]` keys are `{"f1", ...}` or `{"r2", ...}` — `best_model` selection matches ✓
