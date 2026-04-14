import os
from typing import Literal
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from app.state import AgentState


class ModelRoutingOutput(BaseModel):
    task_type:           Literal["classification", "regression", "clustering", "anomaly_detection", "correlation_analysis"]
    selected_algorithms: list[str]  # 2-3 algorithms; empty for correlation_analysis
    reasoning:           str        # one-sentence explanation


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
- outlier_ratio > 0.05  → confirm anomaly_detection; prefer IsolationForest or LocalOutlierFactor
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

IMPORTANT — task_type must be EXACTLY one of these 5 strings (not a category like "supervised"/"unsupervised"):
  classification | regression | clustering | anomaly_detection | correlation_analysis

Output:
- task_type: confirm or refine the initial task type — use ONLY the 5 strings above
- selected_algorithms: list of 2-3 algorithm names from the menu above; [] for correlation_analysis
- reasoning: one concise English sentence explaining your algorithm selection
"""

    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=os.environ.get("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
        temperature=0,
    )
    structured_llm = llm.with_structured_output(ModelRoutingOutput, method="function_calling")
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
