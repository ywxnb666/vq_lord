"""
======================================================================
SCIQA_PROCESS_PARALLEL ---

ScienceQA 多模态并行验证脚本（按样本分片 + 多卡并行）

功能:
1. 复用 sciqa_process.py 的模型加载、prompt、答案解析和 logits 判分逻辑
2. 按过滤后的评测样本索引进行分片，多卡并行评测
3. 支持合并多个 shard 结果，生成最终汇总 JSON
======================================================================
"""

import argparse
import json
import os
from typing import List, Optional

import torch
from datasets import load_dataset
from tqdm import tqdm

from sciqa_process import (
    build_prompt,
    extract_choice_from_output,
    load_model_and_processor,
    predict_choice_with_next_token_logits,
)


def build_eval_samples(scienceqa_path: str, split: str, max_samples: int) -> List[dict]:
    dataset = load_dataset(scienceqa_path, split=split)
    dataset_with_images = [item for item in dataset if item.get("image") is not None]

    if max_samples > 0 and len(dataset_with_images) > max_samples:
        dataset_with_images = dataset_with_images[:max_samples]

    samples = []
    for source_index, item in enumerate(dataset_with_images):
        samples.append({
            "source_index": int(source_index),
            "question": item.get("question", ""),
            "choices": item.get("choices", []),
            "answer_idx": item.get("answer", 0),
            "image": item.get("image"),
        })
    return samples


def shard_eval_samples(samples: List[dict], num_shards: int, shard_id: int) -> List[dict]:
    if num_shards <= 0:
        raise ValueError(f"num_shards must be positive, got {num_shards}")
    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError(f"shard_id must be in [0, {num_shards}), got {shard_id}")

    return [sample for sample in samples if int(sample["source_index"]) % num_shards == shard_id]


def run_eval_shard(
    model,
    processor,
    scienceqa_path: str,
    split: str,
    max_samples: int,
    max_new_tokens: int,
    save_path: str,
    answer_mode: str,
    num_shards: int,
    shard_id: int,
    run_config: Optional[dict] = None,
):
    samples = build_eval_samples(scienceqa_path, split, max_samples)
    shard_samples = shard_eval_samples(samples, num_shards=num_shards, shard_id=shard_id)

    results = []
    correct = 0
    total = 0

    for item in tqdm(shard_samples, desc=f"ScienceQA Eval Shard {shard_id}/{num_shards}"):
        question = item["question"]
        choices = item["choices"]
        answer_idx = item["answer_idx"]
        image = item["image"]

        prompt = build_prompt(question, choices)
        inputs = processor(
            text=prompt,
            images=image,
            return_tensors="pt",
            padding="longest",
            truncation=False,
        )

        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs["attention_mask"].to(model.device)
        pixel_values = inputs["pixel_values"].to(model.device)
        image_sizes = inputs.get("image_sizes")
        if image_sizes is not None:
            image_sizes = image_sizes.to(model.device)

        output_text = ""
        pred_idx = None

        if answer_mode in ("generate", "hybrid"):
            with torch.no_grad():
                generated = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    image_sizes=image_sizes,
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=1,
                    do_sample=False,
                    pad_token_id=model.config.pad_token_id or processor.tokenizer.pad_token_id,
                    eos_token_id=processor.tokenizer.eos_token_id,
                )

            prompt_len = input_ids.shape[-1]
            gen_tokens = generated[0][prompt_len:]
            output_text = processor.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

            if not output_text:
                output_text = "[empty_generation]"

            pred_idx = extract_choice_from_output(output_text, choices)

        if pred_idx is None and answer_mode in ("logits", "hybrid"):
            pred_idx = predict_choice_with_next_token_logits(
                model=model,
                tokenizer=processor.tokenizer,
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_sizes=image_sizes,
                num_choices=len(choices),
            )
            if answer_mode == "logits":
                output_text = "[logits_mode]"
            elif not output_text:
                output_text = "[hybrid_fallback_to_logits]"
            else:
                output_text = f"{output_text}\n[hybrid_fallback_to_logits]"

        is_correct = pred_idx == answer_idx

        total += 1
        if is_correct:
            correct += 1

        results.append({
            "source_index": int(item["source_index"]),
            "question": question,
            "choices": choices,
            "answer_idx": answer_idx,
            "pred_idx": pred_idx,
            "output": output_text,
            "correct": is_correct,
        })

    results.sort(key=lambda item: int(item["source_index"]))

    accuracy = correct / total if total else 0.0
    metrics = {"accuracy": accuracy, "total": total, "correct": correct}

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

    print(f"[Shard {shard_id}] Accuracy: {accuracy:.4f} ({correct}/{total})")
    print(f"[Shard {shard_id}] Results saved to: {save_path}")


def merge_shard_results(
    shard_result_dir: str,
    num_shards: int,
    save_path: str,
    run_config: Optional[dict] = None,
):
    shard_paths = [
        os.path.join(shard_result_dir, f"shard_{shard_id:02d}.json")
        for shard_id in range(num_shards)
    ]
    for shard_path in shard_paths:
        if not os.path.exists(shard_path):
            raise FileNotFoundError(f"missing shard result: {shard_path}")

    merged_results = []
    for shard_path in shard_paths:
        with open(shard_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        merged_results.extend(payload.get("results", []))

    merged_results.sort(key=lambda item: int(item.get("source_index", 10**18)))

    final_results = []
    for item in merged_results:
        merged_item = dict(item)
        merged_item.pop("source_index", None)
        final_results.append(merged_item)

    correct = sum(1 for item in final_results if item.get("correct"))
    total = len(final_results)
    accuracy = correct / total if total else 0.0

    metrics = {"accuracy": accuracy, "total": total, "correct": correct}

    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    payload = {
        "metrics": metrics,
        "run_config": run_config or {},
        "results": final_results,
    }
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"[Merge] Accuracy: {accuracy:.4f} ({correct}/{total})")
    print(f"[Merge] Results saved to: {save_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="ScienceQA 多模态并行验证脚本")
    parser.add_argument("--model_path", type=str, required=True, help="基础模型路径")
    parser.add_argument("--adapter_path", type=str, default="", help="LoRA 适配器路径")
    parser.add_argument("--scienceqa_path", type=str, default="ScienceQA", help="ScienceQA 数据集路径（本地目录或数据集名）")
    parser.add_argument("--split", type=str, default="validation", help="ScienceQA split")
    parser.add_argument("--max_samples", type=int, default=200, help="最大评测样本数")
    parser.add_argument("--max_new_tokens", type=int, default=64, help="生成长度")
    parser.add_argument("--use_4bit", type=int, default=1, help="是否使用4bit加载")
    parser.add_argument("--use_vq", type=int, default=1, help="是否启用训练同款VQ视觉量化路径")
    parser.add_argument("--vq_codebook_size", type=int, default=8192, help="VQ codebook 大小")
    parser.add_argument("--freeze_vision_tower", type=int, default=0, help="是否冻结视觉塔参数（仅构图一致性参数）")
    parser.add_argument("--vq_codebook_path", type=str, default="", help="训练得到的 vq_codebook.pt 路径")
    parser.add_argument(
        "--answer_mode",
        type=str,
        default="hybrid",
        choices=["generate", "logits", "hybrid"],
        help="答案获取方式：generate=仅生成解析, logits=仅下一token打分, hybrid=生成失败时回退logits",
    )
    parser.add_argument("--num_shards", type=int, default=4, help="总分片数")
    parser.add_argument("--shard_id", type=int, default=0, help="当前分片编号")
    parser.add_argument("--shard_result_dir", type=str, default="", help="分片结果目录")
    parser.add_argument("--merge_only", type=int, default=0, help="是否只执行 shard 结果合并")
    parser.add_argument("--save_path", type=str, default="./sciqa_eval_parallel.json", help="保存结果路径")
    return parser.parse_args()


def main():
    args = parse_args()

    if int(args.merge_only) == 1:
        merge_run_config = {
            "model_path": args.model_path,
            "adapter_path": args.adapter_path,
            "scienceqa_path": args.scienceqa_path,
            "split": args.split,
            "max_samples": int(args.max_samples),
            "max_new_tokens": int(args.max_new_tokens),
            "use_4bit": int(args.use_4bit),
            "use_vq": int(args.use_vq),
            "vq_codebook_size": int(args.vq_codebook_size),
            "vq_codebook_path": args.vq_codebook_path,
            "answer_mode": args.answer_mode,
            "parallel_eval": True,
            "parallel_shard_strategy": "sample_index_modulo",
            "num_shards": int(args.num_shards),
            "merged_from": args.shard_result_dir,
        }
        merge_shard_results(
            shard_result_dir=args.shard_result_dir,
            num_shards=int(args.num_shards),
            save_path=args.save_path,
            run_config=merge_run_config,
        )
        return

    model, processor, load_info = load_model_and_processor(
        args.model_path,
        args.adapter_path,
        args.use_4bit,
        args.use_vq,
        args.vq_codebook_size,
        args.freeze_vision_tower,
        args.vq_codebook_path,
    )
    run_config = {
        "model_path": args.model_path,
        "adapter_path": args.adapter_path,
        "adapter_loaded": load_info.get("adapter_loaded", False),
        "adapter_fingerprint": load_info.get("adapter_fingerprint"),
        "scienceqa_path": args.scienceqa_path,
        "split": args.split,
        "max_samples": int(args.max_samples),
        "max_new_tokens": int(args.max_new_tokens),
        "use_4bit": int(args.use_4bit),
        "use_vq": int(args.use_vq),
        "vq_codebook_size": int(args.vq_codebook_size),
        "vq_codebook_path": args.vq_codebook_path,
        "vq_codebook_loaded": load_info.get("vq_codebook_loaded", False),
        "projector_state_found": load_info.get("projector_state_found", False),
        "projector_state_loaded": load_info.get("projector_state_loaded", 0),
        "projector_state_skipped": load_info.get("projector_state_skipped", 0),
        "trainable_state_found": load_info.get("trainable_state_found", False),
        "trainable_state_loaded": load_info.get("trainable_state_loaded", 0),
        "trainable_state_skipped": load_info.get("trainable_state_skipped", 0),
        "answer_mode": args.answer_mode,
        "parallel_eval": True,
        "parallel_shard_strategy": "sample_index_modulo",
        "num_shards": int(args.num_shards),
        "shard_id": int(args.shard_id),
    }
    run_eval_shard(
        model=model,
        processor=processor,
        scienceqa_path=args.scienceqa_path,
        split=args.split,
        max_samples=int(args.max_samples),
        max_new_tokens=int(args.max_new_tokens),
        save_path=args.save_path,
        answer_mode=args.answer_mode,
        num_shards=int(args.num_shards),
        shard_id=int(args.shard_id),
        run_config=run_config,
    )


if __name__ == "__main__":
    main()
