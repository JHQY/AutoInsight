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
        "data_scope": f"{len(df)} rows x {len(df.columns)} columns",
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
            f"[profiling] {len(df)} 行 x {len(df.columns)} 列，质量问题 {len(quality_issues)} 条"
        ],
    }
