import _eval_final_bootstrap  # noqa: F401

import argparse
import os
import time
from typing import Callable, Dict, List, Tuple

from mm_eval_suite_fast_utils import run_mcq_batches_logits, run_open_batches_generate
from mm_eval_suite_utils import (
    collect_failure_cases,
    compute_accuracy,
    load_ai2d_samples,
    load_chartqa_samples,
    load_suite_model,
    save_json,
)


BENCHMARK_LOADERS: Dict[str, Tuple[Callable, str, str]] = {
    "ai2d": (load_ai2d_samples, "lmms-lab/ai2d", "test"),
    "chartqa": (load_chartqa_samples, "HuggingFaceM4/ChartQA", "test"),
}


def build_payload(args, load_info, benchmark_names, benchmark_runs, status):
    overall_correct = sum(run["metrics"]["correct"] for run in benchmark_runs)
    overall_total = sum(run["metrics"]["total"] for run in benchmark_runs)
    overall_accuracy = overall_correct / overall_total if overall_total else 0.0
    return {
        "status": status,
        "run_config": {
            "model_path": args.model_path,
            "adapter_path": args.adapter_path,
            "benchmarks": benchmark_names,
            "max_samples_per_benchmark": args.max_samples_per_benchmark,
            "max_new_tokens": args.max_new_tokens,
            "prompt_style": args.prompt_style,
            "mcq_batch_size": args.mcq_batch_size,
            "open_batch_size": args.open_batch_size,
            "effective_mcq_mode": "logits_batch",
            "effective_open_mode": "generate_batch",
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


def evaluate_benchmark_fast(
    model,
    processor,
    benchmark_name: str,
    dataset_name: str,
    split: str,
    max_samples: int,
    max_new_tokens: int,
    prompt_style: str,
    mcq_batch_size: int,
    open_batch_size: int,
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
        for row in run_mcq_batches_logits(
            model=model,
            processor=processor,
            samples=mcq_samples,
            batch_size=mcq_batch_size,
            prompt_style=prompt_style,
            progress_desc=f"{benchmark_name} mcq",
            show_progress=True,
        ):
            result_map[row["sample_id"]] = row

    if open_samples:
        for row in run_open_batches_generate(
            model=model,
            processor=processor,
            samples=open_samples,
            batch_size=open_batch_size,
            max_new_tokens=max_new_tokens,
            prompt_style=prompt_style,
            progress_desc=f"{benchmark_name} open",
            show_progress=True,
        ):
            result_map[row["sample_id"]] = row

    results: List[dict] = []
    for sample in samples:
        infer = result_map[sample.sample_id]
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
        },
        "failure_cases": collect_failure_cases(results, limit=25),
        "results": results,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Run accelerated multimodal special benchmarks without teacher-model API calls.")
    parser.add_argument("--model_path", type=str, required=True, help="Base model path")
    parser.add_argument("--adapter_path", type=str, default="", help="Optional LoRA adapter path")
    parser.add_argument("--use_4bit", type=int, default=1, help="Whether to use 4bit loading")
    parser.add_argument("--use_vq", type=int, default=1, help="Whether to enable VQ inference hook")
    parser.add_argument("--vq_codebook_size", type=int, default=1024, help="VQ codebook size")
    parser.add_argument("--freeze_vision_tower", type=int, default=0, help="Consistency flag for the VQ wrapper")
    parser.add_argument("--vq_codebook_path", type=str, default="", help="Optional vq_codebook.pt path")
    parser.add_argument("--benchmarks", type=str, default="ai2d,chartqa", help="Comma-separated benchmark list from: ai2d,chartqa")
    parser.add_argument("--ai2d_dataset", type=str, default=BENCHMARK_LOADERS["ai2d"][1], help="AI2D dataset name or local path")
    parser.add_argument("--ai2d_split", type=str, default=BENCHMARK_LOADERS["ai2d"][2], help="AI2D split")
    parser.add_argument("--chartqa_dataset", type=str, default=BENCHMARK_LOADERS["chartqa"][1], help="ChartQA dataset name or local path")
    parser.add_argument("--chartqa_split", type=str, default=BENCHMARK_LOADERS["chartqa"][2], help="ChartQA split")
    parser.add_argument("--max_samples_per_benchmark", type=int, default=0, help="Max samples per benchmark, 0 for full split")
    parser.add_argument("--max_new_tokens", type=int, default=32, help="Generation length for open-ended benchmarks")
    parser.add_argument("--prompt_style", type=str, default="legacy", choices=["legacy", "simple"], help="Prompt style")
    parser.add_argument("--mcq_batch_size", type=int, default=8, help="Batch size for MCQ benchmarks")
    parser.add_argument("--open_batch_size", type=int, default=4, help="Batch size for open-ended benchmarks")
    parser.add_argument("--save_path", type=str, default="./vq_lord_test_results/mm_special_benchmarks_fast.json", help="Where to save the combined benchmark report")
    return parser.parse_args()


def main():
    args = parse_args()
    benchmark_names = [name.strip() for name in args.benchmarks.split(",") if name.strip()]
    unsupported = [name for name in benchmark_names if name not in BENCHMARK_LOADERS]
    if unsupported:
        raise ValueError(f"Unsupported benchmarks: {unsupported}")

    print(f"[Start] Benchmarks={benchmark_names}", flush=True)
    model, processor, load_info = load_suite_model(args)
    print("[Model] Loaded benchmark model and processor.", flush=True)

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
        benchmark_run = evaluate_benchmark_fast(
            model=model,
            processor=processor,
            benchmark_name=benchmark_name,
            dataset_name=dataset_name,
            split=split,
            max_samples=args.max_samples_per_benchmark,
            max_new_tokens=args.max_new_tokens,
            prompt_style=args.prompt_style,
            mcq_batch_size=args.mcq_batch_size,
            open_batch_size=args.open_batch_size,
        )
        benchmark_runs.append(benchmark_run)
        elapsed = time.perf_counter() - stage_start
        print(
            f"[Stage] Finished benchmark={benchmark_name} acc={benchmark_run['metrics']['accuracy']:.4f} "
            f"({benchmark_run['metrics']['correct']}/{benchmark_run['metrics']['total']}) elapsed_sec={elapsed:.2f}",
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
