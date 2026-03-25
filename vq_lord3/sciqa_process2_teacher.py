"""
======================================================================
SCIQA_PROCESS2_TEACHER ---

ScienceQA 教师模型验证脚本（旧 Stage3 prompt 口径）

功能:
1. 使用与 sciqa_process2.py 一致的 legacy prompt 模板
2. 通过 OpenAI 兼容 API 调用教师模型进行多模态问答
3. 解析教师输出并计算 ScienceQA 选择题准确率
======================================================================
"""

import argparse
import json
import os
import time
from typing import Optional

from datasets import load_dataset
from tqdm import tqdm

from data_collector2 import GPT4VDataCollector, _parse_optional_bool
from sciqa_process2 import build_legacy_instruction, extract_choice_from_output


def run_eval(
    collector: GPT4VDataCollector,
    scienceqa_path: str,
    split: str,
    max_samples: int,
    max_new_tokens: int,
    save_path: str,
    sleep_sec: float,
    run_config: Optional[dict] = None,
):
    dataset = load_dataset(scienceqa_path, split=split)
    dataset_with_images = [item for item in dataset if item.get("image") is not None]

    if max_samples > 0 and len(dataset_with_images) > max_samples:
        dataset_with_images = dataset_with_images[:max_samples]

    results = []
    correct = 0
    total = 0
    api_failures = 0

    for item in tqdm(dataset_with_images, desc="ScienceQA Teacher Eval"):
        question = item.get("question", "")
        choices = item.get("choices", [])
        answer_idx = item.get("answer", 0)
        hint = item.get("hint", "")
        image = item.get("image")

        prompt = build_legacy_instruction(question, choices, hint)
        output_text = collector.query_gpt4v_image(
            image=image,
            prompt=prompt,
            max_tokens=max_new_tokens,
            image_format="PNG",
        )
        if sleep_sec > 0:
            time.sleep(sleep_sec)

        if output_text is None:
            output_text = "[api_failed]"
            api_failures += 1

        pred_idx = extract_choice_from_output(output_text, choices)
        is_correct = pred_idx == answer_idx

        total += 1
        if is_correct:
            correct += 1

        results.append({
            "question": question,
            "choices": choices,
            "answer_idx": answer_idx,
            "pred_idx": pred_idx,
            "output": output_text,
            "correct": is_correct,
        })

    accuracy = correct / total if total else 0.0
    metrics = {
        "accuracy": accuracy,
        "total": total,
        "correct": correct,
        "api_failures": api_failures,
    }

    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    payload = {
        "metrics": metrics,
        "run_config": run_config or {},
        "results": results,
    }
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")
    print(f"API failures: {api_failures}")
    print(f"Results saved to: {save_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="ScienceQA 教师模型验证脚本（旧 Stage3 prompt 口径）")
    parser.add_argument("--scienceqa_path", type=str, default="ScienceQA", help="ScienceQA 数据集路径（本地目录或数据集名）")
    parser.add_argument("--split", type=str, default="test", help="ScienceQA split")
    parser.add_argument("--max_samples", type=int, default=0, help="最大评测样本数，0 表示全量")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="教师回答最大生成长度")
    parser.add_argument("--victim_model", type=str, default="qwen3.5-flash-2026-02-23", help="教师模型名称")
    parser.add_argument("--teacher_api_base", type=str, default=os.environ.get("OPENAI_API_BASE", ""), help="教师 API Base URL")
    parser.add_argument("--teacher_api_key", type=str, default=os.environ.get("OPENAI_API_KEY", ""), help="教师 API Key")
    parser.add_argument("--teacher_enable_thinking", type=str, default=os.environ.get("TEACHER_ENABLE_THINKING", ""), help="教师 thinking 开关")
    parser.add_argument("--max_retries", type=int, default=3, help="API 最大重试次数")
    parser.add_argument("--sleep_sec", type=float, default=0.0, help="样本间 sleep 秒数")
    parser.add_argument("--save_path", type=str, default="./scienceqa_teacher_eval.json", help="保存结果路径")
    return parser.parse_args()


def main():
    args = parse_args()

    collector = GPT4VDataCollector(
        api_key=(args.teacher_api_key if args.teacher_api_key else None),
        base_url=(args.teacher_api_base if args.teacher_api_base else None),
        model=args.victim_model,
        save_dir=os.path.dirname(args.save_path) or "./vq_lord_test_results",
        max_retries=int(args.max_retries),
        enable_thinking=_parse_optional_bool(args.teacher_enable_thinking),
    )
    if not collector.api_key:
        raise RuntimeError("缺少教师 API Key，请通过 --teacher_api_key 或 OPENAI_API_KEY 提供。")

    run_config = {
        "scienceqa_path": args.scienceqa_path,
        "split": args.split,
        "max_samples": int(args.max_samples),
        "max_new_tokens": int(args.max_new_tokens),
        "victim_model": args.victim_model,
        "teacher_api_base": args.teacher_api_base,
        "teacher_enable_thinking": args.teacher_enable_thinking,
        "max_retries": int(args.max_retries),
        "sleep_sec": float(args.sleep_sec),
        "prompt_style": "stage3_legacy_teacher",
        "answer_mode": "generate",
    }
    run_eval(
        collector=collector,
        scienceqa_path=args.scienceqa_path,
        split=args.split,
        max_samples=args.max_samples,
        max_new_tokens=args.max_new_tokens,
        save_path=args.save_path,
        sleep_sec=args.sleep_sec,
        run_config=run_config,
    )


if __name__ == "__main__":
    main()
