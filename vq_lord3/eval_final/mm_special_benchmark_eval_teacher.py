import _eval_final_bootstrap  # noqa: F401

import argparse
import os
import time
from typing import Dict, List

from data_collector2 import _parse_optional_bool
from mm_eval_suite_utils import collect_failure_cases, compute_accuracy, save_json
from mm_eval_teacher_utils import (
    load_teacher_ai2d_samples,
    load_teacher_chartqa_samples,
    load_teacher_collector,
    run_teacher_mcq_samples,
    run_teacher_open_samples,
    teacher_model_info,
)


BENCHMARK_LOADERS: Dict[str, tuple] = {
    "ai2d": (load_teacher_ai2d_samples, "lmms-lab/ai2d", "test"),
    "chartqa": (load_teacher_chartqa_samples, "HuggingFaceM4/ChartQA", "test"),
}


def build_payload(args, load_info, benchmark_names, benchmark_runs, status):
    overall_correct = sum(run["metrics"]["correct"] for run in benchmark_runs)
    overall_total = sum(run["metrics"]["total"] for run in benchmark_runs)
    overall_accuracy = overall_correct / overall_total if overall_total else 0.0
    return {
        "status": status,
        "run_config": {
            "victim_model": args.victim_model,
            "benchmarks": benchmark_names,
            "max_samples_per_benchmark": args.max_samples_per_benchmark,
            "max_new_tokens": args.max_new_tokens,
            "prompt_style": args.prompt_style,
            "teacher_api_base": args.teacher_api_base,
            "teacher_enable_thinking": args.teacher_enable_thinking,
            "max_retries": args.max_retries,
            "sleep_sec": args.sleep_sec,
            "max_concurrency": args.max_concurrency,
        },
        "model_load_info": load_info,
        "metrics": {
            "overall_correct": overall_correct,
            "overall_total": overall_total,
            "overall_accuracy": overall_accuracy,
        },
        "completed_benchmarks": [run["benchmark_name"] for run in benchmark_runs],
        "benchmark_runs": benchmark_runs,
    }


def evaluate_benchmark_teacher(
    collector,
    benchmark_name: str,
    dataset_name: str,
    split: str,
    max_samples: int,
    max_new_tokens: int,
    prompt_style: str,
    sleep_sec: float,
    max_concurrency: int,
) -> dict:
    loader, _, _ = BENCHMARK_LOADERS[benchmark_name]
    samples = loader(dataset_name=dataset_name, split=split, max_samples=max_samples)

    mcq_samples = [sample for sample in samples if sample.task_type == "mcq"]
    open_samples = [sample for sample in samples if sample.task_type != "mcq"]

    print(
        f"[Benchmark] {benchmark_name}: total_samples={len(samples)}, mcq={len(mcq_samples)}, open={len(open_samples)}",
        flush=True,
    )

    result_map = {}
    if mcq_samples:
        for row in run_teacher_mcq_samples(
            collector=collector,
            samples=mcq_samples,
            max_new_tokens=max_new_tokens,
            prompt_style=prompt_style,
            sleep_sec=sleep_sec,
            max_concurrency=max_concurrency,
            progress_desc=f"{benchmark_name} teacher mcq",
        ):
            result_map[row["sample_id"]] = row

    if open_samples:
        for row in run_teacher_open_samples(
            collector=collector,
            samples=open_samples,
            max_new_tokens=max_new_tokens,
            sleep_sec=sleep_sec,
            max_concurrency=max_concurrency,
            progress_desc=f"{benchmark_name} teacher open",
        ):
            result_map[row["sample_id"]] = row

    results: List[dict] = []
    api_failures = 0
    for sample in samples:
        infer = result_map[sample.sample_id]
        api_failures += int(bool(infer.get("api_failed")))
        if sample.task_type == "mcq":
            row = {
                "sample_id": sample.sample_id,
                "dataset_name": sample.dataset_name,
                "task_type": sample.task_type,
                "question": sample.question,
                "choices": sample.choices,
                "answer_idx": sample.answer_idx,
                "pred_idx": infer["pred_idx"],
                "output": infer["output"],
                "correct": infer["correct"],
                "metadata": sample.metadata or {},
                "api_failed": infer.get("api_failed", False),
            }
        else:
            row = {
                "sample_id": sample.sample_id,
                "dataset_name": sample.dataset_name,
                "task_type": sample.task_type,
                "question": sample.question,
                "answers": sample.answers,
                "output": infer["output"],
                "correct": infer["correct"],
                "metadata": sample.metadata or {},
                "api_failed": infer.get("api_failed", False),
            }
        results.append(row)

    correct, total, accuracy = compute_accuracy(results)
    return {
        "benchmark_name": benchmark_name,
        "dataset_name": dataset_name,
        "split": split,
        "metrics": {
            "correct": correct,
            "total": total,
            "accuracy": accuracy,
            "api_failures": api_failures,
        },
        "failure_cases": collect_failure_cases(results, limit=25),
        "results": results,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Run teacher-model multimodal special benchmarks through the OpenAI-compatible API.")
    parser.add_argument("--victim_model", type=str, default="qwen3.5-flash-2026-02-23", help="Teacher model name")
    parser.add_argument("--teacher_api_base", type=str, default=os.environ.get("TEACHER_API_BASE", os.environ.get("OPENAI_API_BASE", "")), help="Teacher API Base URL")
    parser.add_argument("--teacher_api_key", type=str, default=os.environ.get("TEACHER_API_KEY", os.environ.get("OPENAI_API_KEY", "")), help="Teacher API Key")
    parser.add_argument("--teacher_enable_thinking", type=str, default=os.environ.get("TEACHER_ENABLE_THINKING", ""), help="Teacher thinking switch")
    parser.add_argument("--max_retries", type=int, default=3, help="API max retries")
    parser.add_argument("--sleep_sec", type=float, default=0.0, help="Sleep between samples")
    parser.add_argument("--max_concurrency", type=int, default=4, help="Maximum in-flight teacher API requests")
    parser.add_argument("--benchmarks", type=str, default="ai2d,chartqa", help="Comma-separated benchmark list from: ai2d,chartqa")
    parser.add_argument("--ai2d_dataset", type=str, default=BENCHMARK_LOADERS["ai2d"][1], help="AI2D dataset name or local path")
    parser.add_argument("--ai2d_split", type=str, default=BENCHMARK_LOADERS["ai2d"][2], help="AI2D split")
    parser.add_argument("--chartqa_dataset", type=str, default=BENCHMARK_LOADERS["chartqa"][1], help="ChartQA dataset name or local path")
    parser.add_argument("--chartqa_split", type=str, default=BENCHMARK_LOADERS["chartqa"][2], help="ChartQA split")
    parser.add_argument("--max_samples_per_benchmark", type=int, default=0, help="Max samples per benchmark, 0 for full split")
    parser.add_argument("--max_new_tokens", type=int, default=64, help="Generation length")
    parser.add_argument("--prompt_style", type=str, default="legacy", choices=["legacy", "simple"], help="Prompt style")
    parser.add_argument("--save_path", type=str, default="./vq_lord_test_results/mm_special_benchmarks_teacher.json", help="Where to save the combined benchmark report")
    return parser.parse_args()


def main():
    args = parse_args()
    benchmark_names = [name.strip() for name in args.benchmarks.split(",") if name.strip()]
    unsupported = [name for name in benchmark_names if name not in BENCHMARK_LOADERS]
    if unsupported:
        raise ValueError(f"Unsupported benchmarks: {unsupported}")

    collector = load_teacher_collector(
        victim_model=args.victim_model,
        teacher_api_base=args.teacher_api_base,
        teacher_api_key=args.teacher_api_key,
        max_retries=args.max_retries,
        enable_thinking=_parse_optional_bool(args.teacher_enable_thinking),
        save_dir=os.path.dirname(args.save_path) or "./vq_lord_test_results",
    )
    load_info = teacher_model_info(collector)

    print(f"[Start] Benchmarks={benchmark_names}", flush=True)
    print("[Model] Teacher API collector ready.", flush=True)

    benchmark_runs = []
    for benchmark_name in benchmark_names:
        if benchmark_name == "ai2d":
            dataset_name = args.ai2d_dataset
            split = args.ai2d_split
        elif benchmark_name == "chartqa":
            dataset_name = args.chartqa_dataset
            split = args.chartqa_split
        else:
            raise ValueError(f"Unexpected benchmark: {benchmark_name}")

        stage_start = time.perf_counter()
        print(f"[Stage] Starting benchmark={benchmark_name}", flush=True)
        benchmark_run = evaluate_benchmark_teacher(
            collector=collector,
            benchmark_name=benchmark_name,
            dataset_name=dataset_name,
            split=split,
            max_samples=args.max_samples_per_benchmark,
            max_new_tokens=args.max_new_tokens,
            prompt_style=args.prompt_style,
            sleep_sec=args.sleep_sec,
            max_concurrency=max(1, args.max_concurrency),
        )
        benchmark_runs.append(benchmark_run)
        elapsed = time.perf_counter() - stage_start
        print(
            f"[Stage] Finished benchmark={benchmark_name} acc={benchmark_run['metrics']['accuracy']:.4f} "
            f"({benchmark_run['metrics']['correct']}/{benchmark_run['metrics']['total']}) "
            f"api_failures={benchmark_run['metrics']['api_failures']} elapsed_sec={elapsed:.2f}",
            flush=True,
        )
        save_json(build_payload(args, load_info, benchmark_names, benchmark_runs, status="running"), args.save_path)
        print(f"[Checkpoint] Saved partial benchmark results to: {os.path.abspath(args.save_path)}", flush=True)

    payload = build_payload(args, load_info, benchmark_names, benchmark_runs, status="completed")
    save_json(payload, args.save_path)
    print(
        f"[Done] Special benchmark overall accuracy: {payload['metrics']['overall_accuracy']:.4f} "
        f"({payload['metrics']['overall_correct']}/{payload['metrics']['overall_total']})",
        flush=True,
    )
    print(f"[Done] Saved to: {os.path.abspath(args.save_path)}", flush=True)


if __name__ == "__main__":
    main()
