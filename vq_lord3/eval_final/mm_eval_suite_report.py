import _eval_final_bootstrap  # noqa: F401

import argparse
import json
import os
from typing import List

from mm_eval_suite_utils import save_json


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_markdown_report(benchmark_payload: dict, control_payload: dict) -> str:
    lines: List[str] = []
    lines.append("# Multimodal Eval Suite Report")
    lines.append("")

    if benchmark_payload:
        metrics = benchmark_payload.get("metrics", {})
        lines.append("## Special Benchmarks")
        lines.append("")
        lines.append(f"Overall accuracy: {metrics.get('overall_accuracy', 0.0):.4f} ({metrics.get('overall_correct', 0)}/{metrics.get('overall_total', 0)})")
        lines.append("")
        lines.append("| Benchmark | Accuracy | Correct | Total |")
        lines.append("|---|---:|---:|---:|")
        for run in benchmark_payload.get("benchmark_runs", []):
            run_metrics = run.get("metrics", {})
            lines.append(
                f"| {run.get('benchmark_name', 'unknown')} | {run_metrics.get('accuracy', 0.0):.4f} | {run_metrics.get('correct', 0)} | {run_metrics.get('total', 0)} |"
            )
        lines.append("")

    if control_payload:
        summary = control_payload.get("metrics", {}).get("control_summary", {})
        lines.append("## ScienceQA Controls")
        lines.append("")
        lines.append("| Control | Accuracy | Delta vs Baseline |")
        lines.append("|---|---:|---:|")
        for control_name, row in summary.items():
            delta = row.get("delta_vs_baseline")
            delta_text = "n/a" if delta is None else f"{delta:+.4f}"
            lines.append(f"| {control_name} | {row.get('accuracy', 0.0):.4f} | {delta_text} |")
        lines.append("")

        lines.append("## Failure Samples")
        lines.append("")
        for run in control_payload.get("control_runs", []):
            failures = run.get("failure_cases", [])[:3]
            if not failures:
                continue
            lines.append(f"### {run.get('control_name', 'unknown')}")
            lines.append("")
            for case in failures:
                lines.append(f"- sample_id: `{case.get('sample_id', 'unknown')}`")
                lines.append(f"  question: {case.get('question', '').replace(chr(10), ' ')[:240]}")
                lines.append(f"  output: {case.get('output', '').replace(chr(10), ' ')[:240]}")
            lines.append("")

    return "\n".join(lines).strip() + "\n"


def parse_args():
    parser = argparse.ArgumentParser(description="Merge multimodal benchmark and ScienceQA control results into a compact report.")
    parser.add_argument("--benchmark_result", type=str, default="", help="Path to mm_special_benchmark_eval.py output")
    parser.add_argument("--control_result", type=str, default="", help="Path to mm_scienceqa_control_eval.py output")
    parser.add_argument("--save_json", type=str, default="./vq_lord_test_results/mm_eval_suite_report.json", help="Path to merged JSON summary")
    parser.add_argument("--save_md", type=str, default="./vq_lord_test_results/mm_eval_suite_report.md", help="Path to markdown report")
    return parser.parse_args()


def main():
    args = parse_args()

    benchmark_payload = load_json(args.benchmark_result) if args.benchmark_result else {}
    control_payload = load_json(args.control_result) if args.control_result else {}

    payload = {
        "benchmark_summary": benchmark_payload.get("metrics", {}),
        "control_summary": control_payload.get("metrics", {}),
        "benchmark_runs": [
            {
                "benchmark_name": run.get("benchmark_name"),
                "metrics": run.get("metrics", {}),
                "failure_cases": run.get("failure_cases", [])[:10],
            }
            for run in benchmark_payload.get("benchmark_runs", [])
        ],
        "control_runs": [
            {
                "control_name": run.get("control_name"),
                "metrics": run.get("metrics", {}),
                "failure_cases": run.get("failure_cases", [])[:10],
            }
            for run in control_payload.get("control_runs", [])
        ],
    }

    save_json(payload, args.save_json)

    markdown = build_markdown_report(benchmark_payload=benchmark_payload, control_payload=control_payload)
    save_dir = os.path.dirname(args.save_md)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    with open(args.save_md, "w", encoding="utf-8") as f:
        f.write(markdown)

    print(f"Saved merged JSON summary to: {os.path.abspath(args.save_json)}")
    print(f"Saved markdown report to: {os.path.abspath(args.save_md)}")


if __name__ == "__main__":
    main()
