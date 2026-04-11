import os
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
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

    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=os.environ.get("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
        temperature=0,
    )
    structured_llm = llm.with_structured_output(IntentOutput, method="function_calling")
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
