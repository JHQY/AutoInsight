# nodes/processing.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from app.state import AgentState

def data_processing(state: AgentState) -> AgentState:
    """
    数据预处理节点（严格匹配 AutoInsight AgentState）
    已适配：y_train / y_test = pd.DataFrame
    """
    # 从状态读取输入
    file_path = state["file_path"]
    task_category = state.get("task_category", "supervised")
    target_col = state.get("target_column", "")
    schema = state["schema"]
    quality_issues = state.get("quality_issues", [])
    logs = state.get("logs", [])

    # 读取数据
    df = pd.read_csv(file_path)
    logs.append("[processing] 成功读取 CSV 文件")

    # Split features from target
    if task_category in ("unsupervised", "analytical") or not target_col:
        X = df.copy()
    else:
        X = df.drop(columns=[target_col])
    feature_names = list(X.columns)

    # 缺失值填充（无警告版）
    for col in X.columns:
        if col not in schema:
            continue
        dtype = schema[col]["type"]

        if dtype == "numeric":
            fill_val = X[col].median()
            X[col] = X[col].fillna(fill_val)
            logs.append(f"[processing] {col} 数值缺失值使用中位数填充")

        elif dtype == "categorical":
            fill_val = X[col].mode()[0]
            X[col] = X[col].fillna(fill_val)
            logs.append(f"[processing] {col} 分类缺失值使用众数填充")

    # 分类列编码
    for col in X.columns:
        if schema[col]["type"] == "categorical":
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            logs.append(f"[processing] {col} 完成 LabelEncoder 编码")

    # 数值列标准化
    numeric_cols = [c for c in X.columns if schema[c]["type"] == "numeric"]
    if numeric_cols:
        scaler = StandardScaler()
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
        logs.append(f"[processing] 数值列 {numeric_cols} 标准化完成")

    # Save full processed feature matrix for downstream inference
    X_full = X.copy()

    # Train/test split — unsupervised: no y
    if task_category in ("unsupervised", "analytical") or not target_col:
        from sklearn.model_selection import train_test_split as _split
        X_train, X_test = _split(X, test_size=0.2, random_state=42)
        state["X_full"] = X_full
        state["X_train"] = X_train
        state["X_test"] = X_test
        state["y_train"] = None
        state["y_test"] = None
        state["feature_names"] = feature_names
        state["logs"] = logs + ["[processing] 无监督分支，跳过 y 拆分"]
        return state

    # Supervised: split with y
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

    state["X_full"] = X_full
    state["X_train"] = X_train
    state["X_test"] = X_test
    state["y_train"] = y_train
    state["y_test"] = y_test
    state["feature_names"] = feature_names
    state["logs"] = logs
    return state

# ===================== 本地测试代码=====================
if __name__ == "__main__":
    # 自动生成测试CSV
    test_df = pd.DataFrame({
        "feature1": [10, 20, None, 40, 50, 60, None, 80, 90, 100],
        "feature2": ["M", "F", "M", None, "F", "M", "F", "M", "F", "M"],
        "target": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    })
    test_df.to_csv("test_data.csv", index=False)
    print("✅ 自动生成 test_data.csv 成功")

    test_state: AgentState = {
        "user_query": "测试数据预处理",
        "file_path": "test_data.csv",
        "target_column": "target",
        "schema": {
            "feature1": {"type": "numeric", "null_rate": 0.05},
            "feature2": {"type": "categorical", "null_rate": 0.02},
            "target": {"type": "numeric", "null_rate": 0.0}
        },
        "quality_issues": [],
        "task_type": "regression",
        "logs": [],
        "X_train": pd.DataFrame(),
        "X_test": pd.DataFrame(),
        "y_train": pd.DataFrame(),
        "y_test": pd.DataFrame(),
        "feature_names": []
    }

    result_state = data_processing(test_state)

    print("\n=== 预处理日志 ===")
    print("\n".join(result_state["logs"]))
    print("\n=== 处理后数据集形状 ===")
    print(f"X_train: {result_state['X_train'].shape}")
    print(f"X_test: {result_state['X_test'].shape}")
    print(f"y_train: {result_state['y_train'].shape}    (类型：{type(result_state['y_train'])})")
    print(f"y_test: {result_state['y_test'].shape}      (类型：{type(result_state['y_test'])})")