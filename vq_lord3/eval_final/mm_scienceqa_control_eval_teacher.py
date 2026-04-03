import _eval_final_bootstrap  # noqa: F401

import argparse
import os
import time

from data_collector2 import _parse_optional_bool
from mm_eval_suite_utils import collect_failure_cases, compute_accuracy, save_json, summarize_by_field
from mm_eval_teacher_utils import (
    build_teacher_control_variant,
    load_teacher_collector,
    load_teacher_scienceqa_samples,
    run_teacher_mcq_samples,
    teacher_model_info,
)
from mm_scienceqa_control_eval import CONTROL_ORDER


def build_payload(args, load_info, requested_controls, control_runs, baseline_accuracy, control_summary, status):
    return {
        "status": status,
        "run_config": {
            "victim_model": args.victim_model,
            "scienceqa_path": args.scienceqa_path,
            "split": args.split,
            "max_samples": args.max_samples,
            "controls": requested_controls,
            "prompt_style": args.prompt_style,
            "teacher_api_base": args.teacher_api_base,
            "teacher_enable_thinking": args.teacher_enable_thinking,
            "shuffle_seed": args.shuffle_seed,
            "max_retries": args.max_retries,
            "sleep_sec": args.sleep_sec,
            "max_concurrency": args.max_concurrency,
        },
        "model_load_info": load_info,
        "metrics": {
            "baseline_accuracy": baseline_accuracy,
            "control_summary": control_summary,
        },
        "completed_controls": [run["control_name"] for run in control_runs],
        "control_runs": control_runs,
    }


def evaluate_control_teacher(
    collector,
    control_name: str,
    samples,
    prompt_style: str,
    shuffle_seed: int,
    max_new_tokens: int,
    sleep_sec: float,
    max_concurrency: int,
) -> dict:
    variant_samples = [
        build_teacher_control_variant(
            control_name=control_name,
            sample=sample,
            sample_idx=idx,
            samples=samples,
            shuffle_seed=shuffle_seed,
        )
        for idx, sample in enumerate(samples)
    ]
    print(f"[Control] {control_name}: total_samples={len(variant_samples)}", flush=True)
    infer_rows = run_teacher_mcq_samples(
        collector=collector,
        samples=variant_samples,
        max_new_tokens=max_new_tokens,
        prompt_style=prompt_style,
        sleep_sec=sleep_sec,
        max_concurrency=max_concurrency,
        progress_desc=f"scienceqa teacher {control_name}",
    )
    infer_map = {row["sample_id"]: row for row in infer_rows}

    results = []
    api_failures = 0
    for sample in variant_samples:
        infer = infer_map[sample.sample_id]
        api_failures += int(bool(infer.get("api_failed")))
        results.append(
            {
                "sample_id": sample.sample_id,
                "question": sample.question,
                "choices": sample.choices,
                "answer_idx": sample.answer_idx,
                "pred_idx": infer["pred_idx"],
                "output": infer["output"],
                "correct": infer["correct"],
                "metadata": sample.metadata or {},
                "api_failed": infer.get("api_failed", False),
            }
        )

    correct, total, accuracy = compute_accuracy(results)
    return {
        "control_name": control_name,
        "metrics": {
            "correct": correct,
            "total": total,
            "accuracy": accuracy,
            "api_failures": api_failures,
        },
        "by_subject": summarize_by_field(results, "subject"),
        "by_grade": summarize_by_field(results, "grade"),
        "failure_cases": collect_failure_cases(results, limit=25),
        "results": results,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Run ScienceQA control experiments with a teacher model through the OpenAI-compatible API.")
    parser.add_argument("--scienceqa_path", type=str, default="ScienceQA", help="ScienceQA dataset path or HF name")
    parser.add_argument("--split", type=str, default="test", help="ScienceQA split")
    parser.add_argument("--max_samples", type=int, default=0, help="Maximum number of image samples, 0 for full split")
    parser.add_argument("--controls", type=str, default=",".join(CONTROL_ORDER), help="Comma-separated controls to run")
    parser.add_argument("--max_new_tokens", type=int, default=64, help="Generation length")
    parser.add_argument("--prompt_style", type=str, default="legacy", choices=["legacy", "simple"], help="Prompt style")
    parser.add_argument("--shuffle_seed", type=int, default=1234, help="Seed for deterministic option shuffling")
    parser.add_argument("--victim_model", type=str, default="qwen3.5-flash-2026-02-23", help="Teacher model name")
    parser.add_argument("--teacher_api_base", type=str, default=os.environ.get("TEACHER_API_BASE", os.environ.get("OPENAI_API_BASE", "")), help="Teacher API Base URL")
    parser.add_argument("--teacher_api_key", type=str, default=os.environ.get("TEACHER_API_KEY", os.environ.get("OPENAI_API_KEY", "")), help="Teacher API Key")
    parser.add_argument("--teacher_enable_thinking", type=str, default=os.environ.get("TEACHER_ENABLE_THINKING", ""), help="Teacher thinking switch")
    parser.add_argument("--max_retries", type=int, default=3, help="API max retries")
    parser.add_argument("--sleep_sec", type=float, default=0.0, help="Sleep between samples")
    parser.add_argument("--max_concurrency", type=int, default=4, help="Maximum in-flight teacher API requests")
    parser.add_argument("--save_path", type=str, default="./vq_lord_test_results/scienceqa_control_suite_teacher.json", help="Where to save the control-suite report")
    return parser.parse_args()


def main():
    args = parse_args()
    requested_controls = [name.strip() for name in args.controls.split(",") if name.strip()]
    unsupported = [name for name in requested_controls if name not in CONTROL_ORDER]
    if unsupported:
        raise ValueError(f"Unsupported controls: {unsupported}")

    samples = load_teacher_scienceqa_samples(
        scienceqa_path=args.scienceqa_path,
        split=args.split,
        max_samples=args.max_samples,
    )
    if not samples:
        raise RuntimeError("No ScienceQA image samples found for control evaluation.")
    print(f"[Data] ScienceQA image samples={len(samples)}", flush=True)

    collector = load_teacher_collector(
        victim_model=args.victim_model,
        teacher_api_base=args.teacher_api_base,
        teacher_api_key=args.teacher_api_key,
        max_retries=args.max_retries,
        enable_thinking=_parse_optional_bool(args.teacher_enable_thinking),
        save_dir=os.path.dirname(args.save_path) or "./vq_lord_test_results",
    )
    load_info = teacher_model_info(collector)
    print("[Model] Teacher API collector ready.", flush=True)

    control_runs = []
    baseline_accuracy = None
    control_summary = {}

    for control_name in requested_controls:
        stage_start = time.perf_counter()
        print(f"[Stage] Starting control={control_name}", flush=True)
        control_run = evaluate_control_teacher(
            collector=collector,
            control_name=control_name,
            samples=samples,
            prompt_style=args.prompt_style,
            shuffle_seed=args.shuffle_seed,
            max_new_tokens=args.max_new_tokens,
            sleep_sec=args.sleep_sec,
            max_concurrency=max(1, args.max_concurrency),
        )
        control_runs.append(control_run)

        if control_name == "baseline":
            baseline_accuracy = control_run["metrics"]["accuracy"]

        acc = control_run["metrics"]["accuracy"]
        control_summary[control_name] = {
            "accuracy": acc,
            "delta_vs_baseline": (acc - baseline_accuracy) if baseline_accuracy is not None else None,
        }
        elapsed = time.perf_counter() - stage_start
        delta = control_summary[control_name]["delta_vs_baseline"]
        delta_text = "n/a" if delta is None else f"{delta:+.4f}"
        print(
            f"[Stage] Finished control={control_name} acc={acc:.4f} "
            f"({control_run['metrics']['correct']}/{control_run['metrics']['total']}) "
            f"api_failures={control_run['metrics']['api_failures']} "
            f"delta_vs_baseline={delta_text} elapsed_sec={elapsed:.2f}",
            flush=True,
        )
        save_json(
            build_payload(args, load_info, requested_controls, control_runs, baseline_accuracy, control_summary, status="running"),
            args.save_path,
        )
        print(f"[Checkpoint] Saved partial control results to: {os.path.abspath(args.save_path)}", flush=True)

    payload = build_payload(args, load_info, requested_controls, control_runs, baseline_accuracy, control_summary, status="completed")
    save_json(payload, args.save_path)
    print(f"[Done] Saved ScienceQA control suite to: {os.path.abspath(args.save_path)}", flush=True)
    if baseline_accuracy is not None:
        print(f"[Done] Baseline accuracy: {baseline_accuracy:.4f}", flush=True)
        for control_name in requested_controls:
            acc = control_summary[control_name]["accuracy"]
            delta = control_summary[control_name]["delta_vs_baseline"]
            delta_text = "n/a" if delta is None else f"{delta:+.4f}"
            print(f"[Done] {control_name}: acc={acc:.4f}, delta_vs_baseline={delta_text}", flush=True)


if __name__ == "__main__":
    main()
