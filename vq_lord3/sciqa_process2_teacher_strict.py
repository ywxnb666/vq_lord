"""
======================================================================
SCIQA_PROCESS2_TEACHER_STRICT ---

ScienceQA 教师模型严格验证脚本

目标:
1. 使用严格输出约束评测教师模型多选准确率
2. 只接受 `Answer: X` 或单独字母 `X` 形式
3. 避免自由生成解释、截断和中途 `(A)` 分析污染准确率
======================================================================
"""

import argparse
import json
import os
import time
from typing import List, Optional

from datasets import load_dataset
from tqdm import tqdm

from data_collector2 import GPT4VDataCollector, _parse_optional_bool


STRICT_SYSTEM_PROMPT = (
    "You are solving a visual multiple-choice question. "
    "Think silently. Do not reveal reasoning. "
    "Reply with exactly one line in the format `Answer: X` where X is one capital letter."
)


def build_strict_instruction(question: str, choices: List[str], hint: str = "") -> str:
    choice_lines = [f"({chr(65 + idx)}) {choice}" for idx, choice in enumerate(choices)]
    hint_block = f"Hint: {hint}\n" if hint else ""
    return (
        f"Question: {question}\n"
        f"{hint_block}"
        f"Options:\n" + "\n".join(choice_lines) + "\n"
        "Return exactly one line in this format:\n"
        "Answer: X"
    )


def extract_strict_answer(output_text: str, num_choices: int) -> Optional[int]:
    if not output_text:
        return None

    max_letter = chr(65 + num_choices - 1) if num_choices > 0 else "A"
    letter_class = f"A-{max_letter}" if max_letter >= "A" else "A"
    raw = output_text.strip()

    patterns = [
        rf"^\s*Answer\s*[:：]\s*([{letter_class}])\s*$",
        rf"^\s*答案\s*[:：]\s*([{letter_class}])\s*$",
        rf"^\s*\(?\s*([{letter_class}])\s*\)?\s*$",
    ]
    for pattern in patterns:
        match = __import__("re").search(pattern, raw, flags=__import__("re").IGNORECASE)
        if match:
            idx = ord(match.group(1).upper()) - 65
            if 0 <= idx < num_choices:
                return idx
    return None


def query_teacher_strict(
    collector: GPT4VDataCollector,
    image,
    prompt: str,
    max_tokens: int,
    temperature: float,
) -> Optional[str]:
    client = collector._get_openai_client()
    if client is None:
        return None

    image_data = collector.encode_pil_image_base64(image, image_format="PNG")
    mime_type = "image/png"

    for attempt in range(collector.max_retries):
        try:
            request_kwargs = {
                "model": collector.model,
                "messages": [
                    {"role": "system", "content": STRICT_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{image_data}",
                                    "detail": "high",
                                },
                            },
                            {
                                "type": "text",
                                "text": prompt,
                            },
                        ],
                    },
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            extra_body = collector._build_model_extra_body()
            if extra_body is not None:
                request_kwargs["extra_body"] = extra_body

            response = client.chat.completions.create(**request_kwargs)
            return response.choices[0].message.content
        except Exception as e:
            print(f"API 调用失败 (尝试 {attempt + 1}/{collector.max_retries}): {e}")
            if attempt < collector.max_retries - 1:
                time.sleep(2 * attempt)

    return None


def run_eval(
    collector: GPT4VDataCollector,
    scienceqa_path: str,
    split: str,
    max_samples: int,
    max_new_tokens: int,
    save_path: str,
    sleep_sec: float,
    temperature: float,
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
    format_valid_count = 0

    for item in tqdm(dataset_with_images, desc="ScienceQA Teacher Strict Eval"):
        question = item.get("question", "")
        choices = item.get("choices", [])
        answer_idx = item.get("answer", 0)
        hint = item.get("hint", "")
        image = item.get("image")

        prompt = build_strict_instruction(question, choices, hint)
        output_text = query_teacher_strict(
            collector=collector,
            image=image,
            prompt=prompt,
            max_tokens=max_new_tokens,
            temperature=temperature,
        )
        if sleep_sec > 0:
            time.sleep(sleep_sec)

        if output_text is None:
            output_text = "[api_failed]"
            api_failures += 1

        pred_idx = extract_strict_answer(output_text, len(choices))
        format_valid = pred_idx is not None
        if format_valid:
            format_valid_count += 1

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
            "format_valid": format_valid,
        })

    accuracy = correct / total if total else 0.0
    format_valid_rate = format_valid_count / total if total else 0.0
    metrics = {
        "accuracy": accuracy,
        "total": total,
        "correct": correct,
        "api_failures": api_failures,
        "format_valid_count": format_valid_count,
        "format_valid_rate": format_valid_rate,
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
    print(f"Format valid rate: {format_valid_rate:.4f} ({format_valid_count}/{total})")
    print(f"API failures: {api_failures}")
    print(f"Results saved to: {save_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="ScienceQA 教师模型严格验证脚本")
    parser.add_argument("--scienceqa_path", type=str, default="ScienceQA", help="ScienceQA 数据集路径（本地目录或数据集名）")
    parser.add_argument("--split", type=str, default="test", help="ScienceQA split")
    parser.add_argument("--max_samples", type=int, default=0, help="最大评测样本数，0 表示全量")
    parser.add_argument("--max_new_tokens", type=int, default=16, help="教师回答最大生成长度")
    parser.add_argument("--temperature", type=float, default=0.0, help="教师采样温度")
    parser.add_argument("--victim_model", type=str, default="qwen3.5-flash-2026-02-23", help="教师模型名称")
    parser.add_argument("--teacher_api_base", type=str, default=os.environ.get("OPENAI_API_BASE", ""), help="教师 API Base URL")
    parser.add_argument("--teacher_api_key", type=str, default=os.environ.get("OPENAI_API_KEY", ""), help="教师 API Key")
    parser.add_argument("--teacher_enable_thinking", type=str, default=os.environ.get("TEACHER_ENABLE_THINKING", ""), help="教师 thinking 开关")
    parser.add_argument("--max_retries", type=int, default=3, help="API 最大重试次数")
    parser.add_argument("--sleep_sec", type=float, default=0.0, help="样本间 sleep 秒数")
    parser.add_argument("--save_path", type=str, default="./scienceqa_teacher_strict_eval.json", help="保存结果路径")
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
        "temperature": float(args.temperature),
        "victim_model": args.victim_model,
        "teacher_api_base": args.teacher_api_base,
        "teacher_enable_thinking": args.teacher_enable_thinking,
        "max_retries": int(args.max_retries),
        "sleep_sec": float(args.sleep_sec),
        "prompt_style": "stage3_legacy_teacher_strict",
        "answer_mode": "strict_generate",
    }
    run_eval(
        collector=collector,
        scienceqa_path=args.scienceqa_path,
        split=args.split,
        max_samples=args.max_samples,
        max_new_tokens=args.max_new_tokens,
        save_path=args.save_path,
        sleep_sec=args.sleep_sec,
        temperature=args.temperature,
        run_config=run_config,
    )


if __name__ == "__main__":
    main()
