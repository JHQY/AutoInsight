"""
AutoInsight Real-Data Test Runner
==================================
Phase 1: single dataset (housing, regression) — availability check
Phase 2: all 5 datasets — full scale test

Outputs per run:
  outputs/test_logs/<run_id>/log_<dataset_id>.json   — structured node log
  outputs/test_logs/<run_id>/summary_<dataset_id>.md — human-readable summary
  outputs/test_logs/<run_id>/report_<dataset_id>.md  — final analysis report (copied)
  outputs/test_logs/<run_id>/run_report.md           — cross-dataset comparison
"""

import os
import sys
import json
import time
import shutil
import argparse
import traceback
from datetime import datetime
from pathlib import Path

import psutil
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv()


# ── Helpers ─────────────────────────────────────────────────────────────────

def _mem_mb() -> float:
    return psutil.Process().memory_info().rss / 1024 / 1024


def _build_initial_state(case: dict) -> dict:
    return {
        "user_query":          case["prompt"],
        "file_path":           case["file"],
        "target_column":       case.get("target", ""),
        "user_level":          case.get("level", "general"),
        "tune":                False,
        "logs":                [],
        "schema":              {},
        "quality_issues":      [],
        "task_category":       "",
        "task_type":           "",
        "user_intent_summary": "",
        "selected_algorithms": [],
        "reasoning":           "",
        "model_results":       {},
        "metrics":             {},
        "best_model":          "",
        "charts":              [],
        "eda_summary":         {},
        "modeling_hints":      {},
        "feature_names":       [],
        "X_train": None, "X_test": None,
        "y_train": None, "y_test": None,
        "report_path":         "",
    }


NODE_ORDER = [
    "profiling", "intent_routing", "processing",
    "eda", "model_routing", "modeling", "evaluation", "reporting",
]

NODE_DESCRIPTIONS = {
    "profiling":      "数据结构分析",
    "intent_routing": "意图理解与任务识别",
    "processing":     "数据清洗与特征工程",
    "eda":            "探索性数据分析",
    "model_routing":  "模型选择决策",
    "modeling":       "模型训练",
    "evaluation":     "模型评估",
    "reporting":      "报告生成",
}


# ── Core runner ─────────────────────────────────────────────────────────────

def run_single(case: dict, out_dir: Path) -> dict:
    """Run one test case, return structured result dict."""
    from app.graph import build_graph

    case_id   = case["id"]
    print(f"\n{'='*60}")
    print(f"[{case_id}] {case['prompt'][:60]}...")
    print(f"  File : {case['file']}")
    print(f"  Level: {case['level']}  |  Expected: {case['task_type_expected']}")

    graph         = build_graph()
    initial_state = _build_initial_state(case)

    node_records  = {}          # node_name → timing / log / resource
    wall_start    = time.time()
    mem_start     = _mem_mb()
    prev_time     = wall_start
    prev_mem      = mem_start
    error         = None
    final_state   = {}

    try:
        for chunk in graph.stream(initial_state, stream_mode="updates"):
            now      = time.time()
            mem_now  = _mem_mb()
            node_name = next(iter(chunk))
            node_state = chunk[node_name]

            # Collect all logs from state up to this point
            all_logs = node_state.get("logs", [])
            # Find logs that likely belong to this node (last appended)
            node_logs = [l for l in all_logs if f"[{node_name.upper()}]" in l.upper()
                         or f"[{node_name}]" in l]
            if not node_logs and all_logs:
                # fallback: pick last log entry
                node_logs = [all_logs[-1]]

            record = {
                "node":        node_name,
                "description": NODE_DESCRIPTIONS.get(node_name, node_name),
                "duration_s":  round(now - prev_time, 3),
                "mem_delta_mb": round(mem_now - prev_mem, 1),
                "mem_rss_mb":  round(mem_now, 1),
                "logs":        node_logs,
            }

            # Node-specific structured info
            if node_name == "profiling":
                schema = node_state.get("schema", {})
                meta   = schema.get("_meta", {})
                record["detail"] = {
                    "row_count":     meta.get("row_count"),
                    "col_count":     meta.get("col_count"),
                    "quality_issues": node_state.get("quality_issues", []),
                }

            elif node_name == "intent_routing":
                record["detail"] = {
                    "task_category":       node_state.get("task_category"),
                    "task_type":           node_state.get("task_type"),
                    "user_intent_summary": node_state.get("user_intent_summary"),
                    "target_column":       node_state.get("target_column"),
                }

            elif node_name == "processing":
                x_train = node_state.get("X_train")
                record["detail"] = {
                    "train_shape": list(x_train.shape) if x_train is not None else None,
                    "features":    node_state.get("feature_names", []),
                    "has_y":       node_state.get("y_train") is not None,
                }

            elif node_name == "eda":
                hints = node_state.get("modeling_hints", {})
                summ  = node_state.get("eda_summary", {})
                record["detail"] = {
                    "charts_generated": len(node_state.get("charts", [])),
                    "top3_features":    summ.get("top3_features"),
                    "distribution":     summ.get("distribution_desc"),
                    "modeling_hints":   hints,
                }

            elif node_name == "model_routing":
                record["detail"] = {
                    "task_type":           node_state.get("task_type"),
                    "selected_algorithms": node_state.get("selected_algorithms"),
                    "reasoning":           node_state.get("reasoning"),
                }

            elif node_name == "modeling":
                results = node_state.get("model_results", {})
                record["detail"] = {
                    "models_trained": list(results.keys()),
                    "errors": [k for k, v in results.items()
                               if isinstance(v, dict) and "error" in v],
                }

            elif node_name == "evaluation":
                metrics = node_state.get("metrics", {})
                record["detail"] = {
                    "best_model": node_state.get("best_model"),
                    "metrics":    metrics,
                }

            elif node_name == "reporting":
                record["detail"] = {
                    "report_path": node_state.get("report_path"),
                }

            node_records[node_name] = record
            final_state = node_state
            prev_time = now
            prev_mem  = mem_now

            print(f"  [ok] {node_name:<16} {record['duration_s']:>6.2f}s  "
                  f"mem={record['mem_rss_mb']:.0f}MB (+{record['mem_delta_mb']:+.0f})")

    except Exception as exc:
        error = traceback.format_exc()
        print(f"  [FAIL] {exc}")

    total_s   = round(time.time() - wall_start, 2)
    mem_peak  = round(_mem_mb(), 1)

    result = {
        "case_id":          case_id,
        "timestamp":        datetime.now().isoformat(),
        "file":             case["file"],
        "prompt":           case["prompt"],
        "level":            case["level"],
        "task_type_expected": case["task_type_expected"],
        "task_type_actual": final_state.get("task_type", ""),
        "best_model":       final_state.get("best_model", ""),
        "report_path":      final_state.get("report_path", ""),
        "total_duration_s": total_s,
        "mem_start_mb":     round(mem_start, 1),
        "mem_peak_mb":      mem_peak,
        "nodes":            node_records,
        "all_logs":         final_state.get("logs", []),
        "error":            error,
        "success":          error is None,
    }

    # ── Write JSON log ────────────────────────────────────────────────────
    log_path = out_dir / f"log_{case_id}.json"
    log_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str))

    # ── Write markdown summary ────────────────────────────────────────────
    _write_summary(result, out_dir / f"summary_{case_id}.md")

    # ── Copy final report ─────────────────────────────────────────────────
    rp = final_state.get("report_path", "")
    if rp and os.path.exists(rp):
        dest = out_dir / f"report_{case_id}.md"
        shutil.copy2(rp, dest)

    print(f"  -> total {total_s}s | peak mem {mem_peak}MB | "
          f"{'SUCCESS' if result['success'] else 'FAILED'}")
    return result


# ── Markdown writers ────────────────────────────────────────────────────────

def _write_summary(result: dict, path: Path):
    lines = []
    lines.append(f"# Test Log — {result['case_id']}")
    lines.append(f"\n**时间:** {result['timestamp']}")
    lines.append(f"**数据集:** `{os.path.basename(result['file'])}`")
    lines.append(f"**用户诉求:** {result['prompt']}")
    lines.append(f"**报告等级:** {result['level']}")
    lines.append(f"**预期任务:** {result['task_type_expected']} | **实际任务:** {result['task_type_actual']}")
    lines.append(f"**最优模型:** {result['best_model'] or 'N/A'}")
    lines.append(f"**总耗时:** {result['total_duration_s']}s | "
                 f"**内存起始:** {result['mem_start_mb']}MB | **内存峰值:** {result['mem_peak_mb']}MB")
    lines.append(f"**状态:** {'✅ 成功' if result['success'] else '❌ 失败'}")

    lines.append("\n---\n## 节点执行记录\n")
    for node_name in NODE_ORDER:
        rec = result["nodes"].get(node_name)
        if not rec:
            continue
        lines.append(f"### {rec['description']} (`{node_name}`)")
        lines.append(f"- **耗时:** {rec['duration_s']}s")
        lines.append(f"- **内存:** {rec['mem_rss_mb']}MB (Δ{rec['mem_delta_mb']:+.0f}MB)")

        detail = rec.get("detail", {})
        if node_name == "profiling" and detail:
            lines.append(f"- **数据规模:** {detail.get('row_count')} 行 × {detail.get('col_count')} 列")
            issues = detail.get("quality_issues", [])
            lines.append(f"- **数据质量问题:** {len(issues)} 项")
            for iss in issues:
                lines.append(f"  - {iss}")

        elif node_name == "intent_routing" and detail:
            lines.append(f"- **任务类别:** {detail.get('task_category')}")
            lines.append(f"- **任务类型:** {detail.get('task_type')}")
            lines.append(f"- **目标列:** `{detail.get('target_column')}`")
            lines.append(f"- **意图摘要:** {detail.get('user_intent_summary')}")

        elif node_name == "processing" and detail:
            lines.append(f"- **训练集形状:** {detail.get('train_shape')}")
            lines.append(f"- **特征列:** {', '.join(detail.get('features', [])[:10])}")
            lines.append(f"- **含目标列:** {'是' if detail.get('has_y') else '否（无监督）'}")

        elif node_name == "eda" and detail:
            lines.append(f"- **生成图表:** {detail.get('charts_generated')} 张")
            lines.append(f"- **Top3 特征:** {detail.get('top3_features')}")
            lines.append(f"- **分布描述:** {detail.get('distribution')}")
            hints = detail.get("modeling_hints", {})
            if hints:
                lines.append("- **Modeling Hints:**")
                for k, v in hints.items():
                    lines.append(f"  - {k}: {v}")

        elif node_name == "model_routing" and detail:
            lines.append(f"- **最终任务类型:** {detail.get('task_type')}")
            lines.append(f"- **选中算法:** {', '.join(detail.get('selected_algorithms', []))}")
            lines.append(f"- **选择理由:** {detail.get('reasoning')}")

        elif node_name == "modeling" and detail:
            lines.append(f"- **训练模型:** {', '.join(detail.get('models_trained', []))}")
            errs = detail.get("errors", [])
            if errs:
                lines.append(f"- **训练失败:** {', '.join(errs)}")

        elif node_name == "evaluation" and detail:
            lines.append(f"- **最优模型:** {detail.get('best_model')}")
            metrics = detail.get("metrics", {})
            for model, m in metrics.items():
                if isinstance(m, dict):
                    m_str = " | ".join(f"{k}={v:.4f}" for k, v in m.items()
                                       if isinstance(v, float))
                    lines.append(f"  - {model}: {m_str}")

        elif node_name == "reporting" and detail:
            lines.append(f"- **报告路径:** `{detail.get('report_path')}`")

        node_logs = rec.get("logs", [])
        if node_logs:
            lines.append("- **节点日志:**")
            for log in node_logs:
                lines.append(f"  - {log}")
        lines.append("")

    if result.get("error"):
        lines.append("\n---\n## 错误信息\n```")
        lines.append(result["error"])
        lines.append("```")

    lines.append("\n---\n## 完整运行日志\n")
    for log in result.get("all_logs", []):
        lines.append(f"- {log}")

    path.write_text("\n".join(lines), encoding="utf-8")


def _write_run_report(results: list, out_dir: Path, phase: int):
    path = out_dir / "run_report.md"
    lines = []
    lines.append(f"# AutoInsight 测试报告 — Phase {phase}")
    lines.append(f"\n**执行时间:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**测试用例数:** {len(results)}")
    success_count = sum(1 for r in results if r["success"])
    lines.append(f"**通过:** {success_count}/{len(results)}\n")

    lines.append("## 汇总表\n")
    lines.append("| 数据集 | 预期任务 | 实际任务 | 最优模型 | 耗时(s) | 峰值内存(MB) | 状态 |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in results:
        match = "✅" if r["task_type_expected"] == r["task_type_actual"] else "⚠️"
        status = "✅" if r["success"] else "❌"
        lines.append(
            f"| {r['case_id']} | {r['task_type_expected']} | "
            f"{r['task_type_actual']} {match} | {r['best_model'] or 'N/A'} | "
            f"{r['total_duration_s']} | {r['mem_peak_mb']} | {status} |"
        )

    lines.append("\n## 节点耗时对比 (秒)\n")
    lines.append("| 节点 | " + " | ".join(r["case_id"] for r in results) + " |")
    lines.append("|---|" + "---|" * len(results))
    for node in NODE_ORDER:
        row = f"| {node} |"
        for r in results:
            rec = r["nodes"].get(node)
            row += f" {rec['duration_s'] if rec else '-'} |"
        lines.append(row)

    lines.append("\n## 各用例结论\n")
    for r in results:
        lines.append(f"### {r['case_id']}")
        if r["success"]:
            lines.append(f"- 任务识别: {r['task_type_actual']}")
            lines.append(f"- 最优模型: {r['best_model'] or 'N/A'}")
            eval_rec = r["nodes"].get("evaluation", {})
            metrics = eval_rec.get("detail", {}).get("metrics", {})
            best = r["best_model"]
            if best and best in metrics and isinstance(metrics[best], dict):
                m_str = " | ".join(f"{k}={v:.4f}" for k, v in metrics[best].items()
                                   if isinstance(v, float))
                lines.append(f"- 关键指标: {m_str}")
        else:
            lines.append(f"- ❌ 失败")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[run_report] → {path}")


# ── Entry point ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AutoInsight Real-Data Test Runner")
    parser.add_argument("--phase", type=int, choices=[1, 2], default=1,
                        help="1 = single dataset (housing); 2 = all 5 datasets")
    parser.add_argument("--prompts", default="test_data/prompts.json",
                        help="Path to prompts JSON file")
    args = parser.parse_args()

    prompts_path = Path(args.prompts)
    if not prompts_path.exists():
        print(f"Error: prompts file not found — {prompts_path}")
        sys.exit(1)

    all_cases = json.loads(prompts_path.read_text(encoding="utf-8"))
    cases = [c for c in all_cases if c["phase"] <= args.phase]

    run_id  = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_phase{args.phase}"
    out_dir = Path("outputs/test_logs") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"AutoInsight Test Runner — Phase {args.phase}")
    print(f"Output dir: {out_dir}")
    print(f"Cases: {[c['id'] for c in cases]}")

    results = []
    for case in cases:
        result = run_single(case, out_dir)
        results.append(result)

    _write_run_report(results, out_dir, args.phase)

    print(f"\n{'='*60}")
    print(f"Phase {args.phase} complete: {sum(r['success'] for r in results)}/{len(results)} passed")
    print(f"Results: {out_dir}")


if __name__ == "__main__":
    main()
