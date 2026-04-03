import _eval_final_bootstrap  # noqa: F401

import argparse
import os
import time
from typing import List

from mm_eval_suite_fast_utils import run_scienceqa_mcq_samples_exact
from mm_eval_suite_utils import (
    StandardSample,
    collect_failure_cases,
    compute_accuracy,
    load_scienceqa_image_samples,
    save_json,
    summarize_by_field,
)
from mm_scienceqa_control_eval import CONTROL_ORDER, build_control_variant
from sciqa_process import load_model_and_processor as load_stage2_model_and_processor
from sciqa_process2 import load_model_and_processor as load_stage3_model_and_processor


def build_variant_samples(control_name: str, samples: List[StandardSample], shuffle_seed: int) -> List[StandardSample]:
    variant_samples: List[StandardSample] = []
    for idx, sample in enumerate(samples):
        variant = build_control_variant(
            control_name=control_name,
            sample=sample,
            sample_idx=idx,
            samples=samples,
            shuffle_seed=shuffle_seed,
        )
        variant_samples.append(
            StandardSample(
                sample_id=sample.sample_id,
                dataset_name=sample.dataset_name,
                question=variant["question"],
                image=variant["image"],
                task_type="mcq",
                choices=list(variant["choices"]),
                answer_idx=int(variant["answer_idx"]),
                hint=variant["hint"],
                metadata=variant["metadata"],
            )
        )
    return variant_samples


def load_original_scienceqa_model(args):
    if args.prompt_style == "simple":
        loader = load_stage2_model_and_processor
        source = "sciqa_process.py"
    else:
        loader = load_stage3_model_and_processor
        source = "sciqa_process2.py"

    model, processor, load_info = loader(
        args.model_path,
        args.adapter_path,
        args.use_4bit,
        args.use_vq,
        args.vq_codebook_size,
        args.freeze_vision_tower,
        args.vq_codebook_path,
    )
    load_info = dict(load_info or {})
    load_info["scienceqa_eval_source"] = source
    return model, processor, load_info


def build_payload(args, load_info, requested_controls, control_runs, baseline_accuracy, control_summary, status):
    return {
        "status": status,
        "run_config": {
            "model_path": args.model_path,
            "adapter_path": args.adapter_path,
            "scienceqa_path": args.scienceqa_path,
            "split": args.split,
            "max_samples": args.max_samples,
            "controls": requested_controls,
            "prompt_style": args.prompt_style,
            "answer_mode": args.answer_mode,
            "max_new_tokens": args.max_new_tokens,
            "shuffle_seed": args.shuffle_seed,
            "mcq_batch_size": args.mcq_batch_size,
            "use_4bit": args.use_4bit,
            "effective_mcq_mode": "scienceqa_original_exact",
        },
        "model_load_info": load_info,
        "metrics": {
            "baseline_accuracy": baseline_accuracy,
            "control_summary": control_summary,
        },
        "completed_controls": [run["control_name"] for run in control_runs],
        "control_runs": control_runs,
    }


def evaluate_control_fast(
    model,
    processor,
    control_name: str,
    samples: List[StandardSample],
    prompt_style: str,
    answer_mode: str,
    max_new_tokens: int,
    shuffle_seed: int,
    batch_size: int,
) -> dict:
    del batch_size  # kept only for CLI compatibility with previous versions
    variant_samples = build_variant_samples(control_name=control_name, samples=samples, shuffle_seed=shuffle_seed)
    print(f"[Control] {control_name}: total_samples={len(variant_samples)}", flush=True)
    infer_rows = run_scienceqa_mcq_samples_exact(
        model=model,
        processor=processor,
        samples=variant_samples,
        prompt_style=prompt_style,
        answer_mode=answer_mode,
        max_new_tokens=max_new_tokens,
        progress_desc=f"scienceqa {control_name}",
        show_progress=True,
    )
    infer_map = {row["sample_id"]: row for row in infer_rows}

    results = []
    for sample in variant_samples:
        infer = infer_map[sample.sample_id]
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
            }
        )

    correct, total, accuracy = compute_accuracy(results)
    return {
        "control_name": control_name,
        "metrics": {
            "correct": correct,
            "total": total,
            "accuracy": accuracy,
        },
        "by_subject": summarize_by_field(results, "subject"),
        "by_grade": summarize_by_field(results, "grade"),
        "failure_cases": collect_failure_cases(results, limit=25),
        "results": results,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Run ScienceQA control experiments with the exact original stage2/stage3 evaluation path.")
    parser.add_argument("--model_path", type=str, required=True, help="Base model path")
    parser.add_argument("--adapter_path", type=str, default="", help="Optional LoRA adapter path")
    parser.add_argument("--scienceqa_path", type=str, default="ScienceQA", help="ScienceQA dataset path or HF name")
    parser.add_argument("--split", type=str, default="test", help="ScienceQA split")
    parser.add_argument("--max_samples", type=int, default=0, help="Maximum number of image samples, 0 for full split")
    parser.add_argument("--controls", type=str, default=",".join(CONTROL_ORDER), help="Comma-separated controls to run")
    parser.add_argument("--prompt_style", type=str, default="legacy", choices=["legacy", "simple"], help="Prompt style")
    parser.add_argument("--answer_mode", type=str, default="logits", choices=["generate", "logits", "hybrid"], help="Answer mode, kept identical to the original sciqa scripts")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Generation length, kept identical to the original sciqa scripts")
    parser.add_argument("--shuffle_seed", type=int, default=1234, help="Seed for deterministic option shuffling")
    parser.add_argument("--mcq_batch_size", type=int, default=8, help="Compatibility argument; exact ScienceQA mode runs sequentially like the original scripts")
    parser.add_argument("--use_4bit", type=int, default=0, help="Whether to use 4bit loading; original stage2/stage3 test scripts use 0")
    parser.add_argument("--use_vq", type=int, default=1, help="Whether to enable VQ inference hook")
    parser.add_argument("--vq_codebook_size", type=int, default=1024, help="VQ codebook size")
    parser.add_argument("--freeze_vision_tower", type=int, default=0, help="Consistency flag for the VQ wrapper")
    parser.add_argument("--vq_codebook_path", type=str, default="", help="Optional vq_codebook.pt path")
    parser.add_argument("--save_path", type=str, default="./vq_lord_test_results/scienceqa_control_suite_fast.json", help="Where to save the control-suite report")
    return parser.parse_args()


def main():
    args = parse_args()
    requested_controls = [name.strip() for name in args.controls.split(",") if name.strip()]
    unsupported = [name for name in requested_controls if name not in CONTROL_ORDER]
    if unsupported:
        raise ValueError(f"Unsupported controls: {unsupported}")

    print(f"[Start] ScienceQA controls={requested_controls}", flush=True)
    samples = load_scienceqa_image_samples(
        scienceqa_path=args.scienceqa_path,
        split=args.split,
        max_samples=args.max_samples,
    )
    if not samples:
        raise RuntimeError("No ScienceQA image samples found for control evaluation.")
    print(f"[Data] ScienceQA image samples={len(samples)}", flush=True)

    model, processor, load_info = load_original_scienceqa_model(args)
    print(f"[Model] Loaded ScienceQA control model via {load_info.get('scienceqa_eval_source')}", flush=True)

    control_runs = []
    baseline_accuracy = None
    control_summary = {}

    for control_name in requested_controls:
        stage_start = time.perf_counter()
        print(f"[Stage] Starting control={control_name}", flush=True)
        control_run = evaluate_control_fast(
            model=model,
            processor=processor,
            control_name=control_name,
            samples=samples,
            prompt_style=args.prompt_style,
            answer_mode=args.answer_mode,
            max_new_tokens=args.max_new_tokens,
            shuffle_seed=args.shuffle_seed,
            batch_size=args.mcq_batch_size,
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
