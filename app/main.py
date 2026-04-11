# app/main.py
import argparse
import os
import sys

from dotenv import load_dotenv
load_dotenv()

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
        # Fields written by nodes; initialize to safe defaults
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
