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
