"""
======================================================================
SCIQA_PROCESS2_PARALLEL_BUCKETED ---

ScienceQA 多模态并行验证脚本（按 patch 分桶 + 多卡分片）

功能:
1. 读取 data_preprocess/sciqa_preprocess.py 生成的 bucket plan JSON
2. 按 patch_count 构造稳定 batch，降低多模态 generate 的显存波动
3. 支持按 batch_id 分片到多卡并行评测
4. 支持合并多个 shard 结果，生成最终汇总 JSON
======================================================================
"""

import argparse
import json
import os
import re
from typing import Dict, List, Optional, Tuple

import torch
from datasets import load_dataset
from tqdm import tqdm

from sciqa_process2 import (
    build_prompt,
    extract_choice_from_output,
    load_model_and_processor,
)


def build_two_pass_structured_instruction(
    question: str,
    choices: List[str],
    hint: str = "",
    first_pass_answer: str = "",
    first_pass_output: str = "",
) -> str:
    choices_text = ""
    for idx, choice in enumerate(choices):
        choices_text += f"({chr(65 + idx)}) {choice}\n"
    hint_block = f"Hint: {hint}\n" if hint else ""
    return (
        "You are a rigorous visual science QA assistant.\n"
        "Return strict JSON only with exactly keys in this order:\n"
        "observed_facts_visual, context_textual, reasoning, answer.\n"
        "Rules:\n"
        "0) Every field value must be a plain string. Do not output arrays, lists, or objects.\n"
        "0.1) The output must be a valid JSON object.\n"
        "0.2) Use the first-pass answer as the final answer. Do not change the chosen option.\n"
        "0.3) The answer field must be the final field, preferably as '(A) option text'.\n"
        "0.4) The final answer must remain consistent with the first-pass answer and with your reasoning.\n"
        "1) observed_facts_visual: only image-observable evidence (OCR allowed); no inference, no option matching.\n"
        "2) context_textual: restate the full textual conditions from question, hint, and options.\n"
        "3) reasoning: one short sentence only; option comparison belongs here, not in observed_facts_visual.\n"
        "No markdown, no extra fields.\n\n"
        f"First-pass answer to preserve: {first_pass_answer or '[missing]'}\n"
        f"First-pass raw output:\n{first_pass_output.strip() or '[missing]'}\n\n"
        f"Question: {question}\n"
        f"{hint_block}"
        f"Options:\n{choices_text}"
    )


def build_two_pass_structured_prompt(
    processor,
    question: str,
    choices: List[str],
    hint: str = "",
    first_pass_answer: str = "",
    first_pass_output: str = "",
) -> str:
    instruction_text = build_two_pass_structured_instruction(
        question=question,
        choices=choices,
        hint=hint,
        first_pass_answer=first_pass_answer,
        first_pass_output=first_pass_output,
    )
    if hasattr(processor, "apply_chat_template"):
        prompt_conv = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction_text},
                    {"type": "image"},
                ],
            }
        ]
        return processor.apply_chat_template(prompt_conv, add_generation_prompt=True)
    return f"<image>\n{instruction_text}"


def _extract_json_object(text: str) -> Optional[dict]:
    if not isinstance(text, str):
        return None
    cleaned = text.strip()
    if not cleaned:
        return None
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()

    candidates = [cleaned]
    left = cleaned.find("{")
    right = cleaned.rfind("}")
    if left != -1 and right != -1 and right > left:
        candidates.append(cleaned[left:right + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _extract_struct_answer_field(text: str) -> str:
    payload = _extract_json_object(text)
    if isinstance(payload, dict):
        answer = payload.get("answer", "")
        if isinstance(answer, str):
            return answer.strip()

    match = re.search(r'"answer"\s*:\s*"((?:\\.|[^"\\])*)"', text or "")
    if not match:
        return ""
    raw_value = match.group(1)
    try:
        return json.loads(f'"{raw_value}"').strip()
    except Exception:
        return raw_value.strip()


def extract_choice_from_structured_output(output_text: str, choices: List[str]) -> Optional[int]:
    answer_field = _extract_struct_answer_field(output_text)
    if answer_field:
        pred_idx = extract_choice_from_output(answer_field, choices)
        if pred_idx is not None:
            return pred_idx
    return extract_choice_from_output(output_text, choices)


def build_canonical_answer_text(pred_idx: Optional[int], choices: List[str]) -> str:
    if pred_idx is None or pred_idx < 0 or pred_idx >= len(choices):
        return ""
    return f"({chr(65 + pred_idx)}) {choices[pred_idx]}"


def load_bucket_payload(bucket_plan_path: str) -> dict:
    if not os.path.exists(bucket_plan_path):
        raise FileNotFoundError(f"bucket plan not found: {bucket_plan_path}")
    with open(bucket_plan_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload


def validate_bucket_payload(payload: dict, scienceqa_path: str, split: str, max_samples: int) -> None:
    cfg = payload.get("config", {})
    payload_split = str(cfg.get("split", ""))
    payload_train_num = int(cfg.get("train_num", 0) or 0)
    payload_bucket_by = str(cfg.get("bucket_by", ""))

    if payload_split and payload_split != split:
        raise ValueError(
            f"bucket plan split mismatch: expected={split}, got={payload_split}"
        )
    if payload_train_num != int(max_samples):
        raise ValueError(
            f"bucket plan train_num mismatch: expected={max_samples}, got={payload_train_num}"
        )
    if payload_bucket_by != "patches":
        raise ValueError(
            f"bucket plan bucket_by mismatch: expected=patches, got={payload_bucket_by}"
        )


def build_eval_sample_map(scienceqa_path: str, split: str, bucket_payload: dict) -> Dict[int, dict]:
    dataset = load_dataset(scienceqa_path, split=split)
    dataset_with_images = [item for item in dataset if item.get("image") is not None]

    sample_map: Dict[int, dict] = {}
    for record in bucket_payload.get("samples", []):
        sample_id = int(record["sample_id"])
        source_index = int(record["source_index"])
        if source_index < 0 or source_index >= len(dataset_with_images):
            raise IndexError(
                f"source_index out of range in bucket plan: source_index={source_index}, "
                f"dataset_with_images={len(dataset_with_images)}"
            )

        item = dataset_with_images[source_index]
        sample_map[sample_id] = {
            "sample_id": sample_id,
            "source_index": source_index,
            "question": item.get("question", ""),
            "choices": item.get("choices", []),
            "answer_idx": item.get("answer", 0),
            "hint": item.get("hint", ""),
            "image": item.get("image"),
            "patch_count": int(record.get("patch_count", 0)),
            "image_size": record.get("image_size"),
            "bucket_key": str(record.get("bucket_key", "")),
        }

    return sample_map


def shard_batches(batch_plan: List[dict], num_shards: int, shard_id: int) -> List[dict]:
    if num_shards <= 0:
        raise ValueError(f"num_shards must be positive, got {num_shards}")
    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError(f"shard_id must be in [0, {num_shards}), got {shard_id}")
    return [batch for batch in batch_plan if int(batch["batch_id"]) % num_shards == shard_id]


def predict_choices_with_next_token_logits_batch(
    model,
    tokenizer,
    input_ids,
    attention_mask,
    pixel_values,
    image_sizes,
    num_choices_list: List[int],
) -> List[Optional[int]]:
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_sizes=image_sizes,
            use_cache=False,
        )

    next_token_logits = outputs.logits[:, -1, :]
    predictions: List[Optional[int]] = []

    for row_idx, num_choices in enumerate(num_choices_list):
        if num_choices <= 0:
            predictions.append(None)
            continue

        best_idx = None
        best_score = None
        for choice_idx in range(num_choices):
            letter = chr(65 + choice_idx)
            variants = [
                letter,
                f" {letter}",
                f"({letter})",
                f" ({letter})",
                f"{letter}.",
                f" {letter}.",
                f"答案:{letter}",
                f"Answer: {letter}",
            ]
            candidate_ids = set()
            for text in variants:
                ids = tokenizer.encode(text, add_special_tokens=False)
                if len(ids) == 1:
                    candidate_ids.add(ids[0])
            if not candidate_ids:
                ids = tokenizer.encode(letter, add_special_tokens=False)
                if ids:
                    candidate_ids.add(ids[0])
            if not candidate_ids:
                continue

            score = max(next_token_logits[row_idx, token_id].item() for token_id in candidate_ids)
            if best_score is None or score > best_score:
                best_score = score
                best_idx = choice_idx
        predictions.append(best_idx)

    return predictions


def run_eval_shard(
    model,
    processor,
    bucket_plan_path: str,
    scienceqa_path: str,
    split: str,
    max_samples: int,
    max_new_tokens: int,
    second_pass_max_new_tokens: int,
    save_path: str,
    answer_mode: str,
    num_shards: int,
    shard_id: int,
    run_config: Optional[dict] = None,
):
    bucket_payload = load_bucket_payload(bucket_plan_path)
    validate_bucket_payload(bucket_payload, scienceqa_path=scienceqa_path, split=split, max_samples=max_samples)

    sample_map = build_eval_sample_map(scienceqa_path, split, bucket_payload)
    batch_plan = bucket_payload.get("batch_plan", [])
    shard_plan = shard_batches(batch_plan, num_shards=num_shards, shard_id=shard_id)

    # Batched generation for decoder-only models is much more stable with left padding.
    if hasattr(processor, "tokenizer") and getattr(processor.tokenizer, "padding_side", None) != "left":
        processor.tokenizer.padding_side = "left"

    results = []
    correct = 0
    total = 0

    for batch in tqdm(shard_plan, desc=f"ScienceQA Eval Bucket Shard {shard_id}/{num_shards}"):
        sample_ids = [int(x) for x in batch.get("sample_ids", [])]
        batch_samples = [sample_map[sample_id] for sample_id in sample_ids]

        prompts = [
            build_prompt(processor, item["question"], item["choices"], item["hint"])
            for item in batch_samples
        ]
        images = [item["image"] for item in batch_samples]

        inputs = processor(
            text=prompts,
            images=images,
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

        batch_output_texts = [""] * len(batch_samples)
        batch_first_pass_output_texts = [""] * len(batch_samples)
        batch_second_pass_output_texts = [""] * len(batch_samples)
        batch_first_pass_answer_texts = [""] * len(batch_samples)
        batch_pred_idx: List[Optional[int]] = [None] * len(batch_samples)

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

            prompt_len = input_ids.shape[1]
            for row_idx, item in enumerate(batch_samples):
                gen_tokens = generated[row_idx][prompt_len:]
                output_text = processor.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
                if not output_text:
                    output_text = "[empty_generation]"
                batch_first_pass_output_texts[row_idx] = output_text
                batch_pred_idx[row_idx] = extract_choice_from_output(output_text, item["choices"])
                batch_first_pass_answer_texts[row_idx] = build_canonical_answer_text(
                    batch_pred_idx[row_idx],
                    item["choices"],
                )

            if answer_mode == "generate":
                struct_prompts = [
                    build_two_pass_structured_prompt(
                        processor=processor,
                        question=item["question"],
                        choices=item["choices"],
                        hint=item["hint"],
                        first_pass_answer=batch_first_pass_answer_texts[row_idx],
                        first_pass_output=batch_first_pass_output_texts[row_idx],
                    )
                    for row_idx, item in enumerate(batch_samples)
                ]
                struct_inputs = processor(
                    text=struct_prompts,
                    images=images,
                    return_tensors="pt",
                    padding="longest",
                    truncation=False,
                )
                struct_input_ids = struct_inputs["input_ids"].to(model.device)
                struct_attention_mask = struct_inputs["attention_mask"].to(model.device)
                struct_pixel_values = struct_inputs["pixel_values"].to(model.device)
                struct_image_sizes = struct_inputs.get("image_sizes")
                if struct_image_sizes is not None:
                    struct_image_sizes = struct_image_sizes.to(model.device)

                with torch.no_grad():
                    struct_generated = model.generate(
                        input_ids=struct_input_ids,
                        attention_mask=struct_attention_mask,
                        pixel_values=struct_pixel_values,
                        image_sizes=struct_image_sizes,
                        max_new_tokens=second_pass_max_new_tokens,
                        min_new_tokens=1,
                        do_sample=False,
                        pad_token_id=model.config.pad_token_id or processor.tokenizer.pad_token_id,
                        eos_token_id=processor.tokenizer.eos_token_id,
                    )

                struct_prompt_len = struct_input_ids.shape[1]
                for row_idx, item in enumerate(batch_samples):
                    struct_gen_tokens = struct_generated[row_idx][struct_prompt_len:]
                    struct_output_text = processor.tokenizer.decode(
                        struct_gen_tokens, skip_special_tokens=True
                    ).strip()
                    if not struct_output_text:
                        struct_output_text = "[empty_generation]"
                    batch_second_pass_output_texts[row_idx] = struct_output_text
                    batch_output_texts[row_idx] = struct_output_text
                    if batch_pred_idx[row_idx] is None:
                        batch_pred_idx[row_idx] = extract_choice_from_structured_output(
                            struct_output_text, item["choices"]
                        )
            else:
                batch_output_texts = list(batch_first_pass_output_texts)

        if answer_mode in ("logits", "hybrid"):
            unresolved_rows = [
                row_idx for row_idx, pred_idx in enumerate(batch_pred_idx)
                if pred_idx is None
            ] if answer_mode == "hybrid" else list(range(len(batch_samples)))

            if unresolved_rows:
                row_input_ids = input_ids[unresolved_rows]
                row_attention_mask = attention_mask[unresolved_rows]
                row_pixel_values = pixel_values[unresolved_rows]
                row_image_sizes = image_sizes[unresolved_rows] if image_sizes is not None else None
                row_num_choices = [len(batch_samples[row_idx]["choices"]) for row_idx in unresolved_rows]

                row_predictions = predict_choices_with_next_token_logits_batch(
                    model=model,
                    tokenizer=processor.tokenizer,
                    input_ids=row_input_ids,
                    attention_mask=row_attention_mask,
                    pixel_values=row_pixel_values,
                    image_sizes=row_image_sizes,
                    num_choices_list=row_num_choices,
                )

                for local_idx, row_idx in enumerate(unresolved_rows):
                    batch_pred_idx[row_idx] = row_predictions[local_idx]
                    if answer_mode == "logits":
                        batch_output_texts[row_idx] = "[logits_mode]"
                    elif not batch_output_texts[row_idx]:
                        batch_output_texts[row_idx] = "[hybrid_fallback_to_logits]"
                    else:
                        batch_output_texts[row_idx] = (
                            f"{batch_output_texts[row_idx]}\n[hybrid_fallback_to_logits]"
                        )

        for row_idx, item in enumerate(batch_samples):
            pred_idx = batch_pred_idx[row_idx]
            is_correct = pred_idx == item["answer_idx"]
            total += 1
            if is_correct:
                correct += 1

            results.append({
                "sample_id": int(item["sample_id"]),
                "source_index": int(item["source_index"]),
                "question": item["question"],
                "choices": item["choices"],
                "answer_idx": item["answer_idx"],
                "pred_idx": pred_idx,
                "output": batch_output_texts[row_idx],
                "first_pass_output": batch_first_pass_output_texts[row_idx],
                "second_pass_output": batch_second_pass_output_texts[row_idx],
                "first_pass_answer": batch_first_pass_answer_texts[row_idx],
                "correct": is_correct,
                "patch_count": int(item["patch_count"]),
                "bucket_key": item["bucket_key"],
            })

    results.sort(key=lambda item: int(item["sample_id"]))

    accuracy = correct / total if total else 0.0
    metrics = {
        "accuracy": accuracy,
        "total": total,
        "correct": correct,
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

    merged_results.sort(key=lambda item: int(item.get("sample_id", 10**18)))

    correct = sum(1 for item in merged_results if item.get("correct"))
    total = len(merged_results)
    accuracy = correct / total if total else 0.0

    metrics = {
        "accuracy": accuracy,
        "total": total,
        "correct": correct,
    }

    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    payload = {
        "metrics": metrics,
        "run_config": run_config or {},
        "results": merged_results,
    }
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"[Merge] Accuracy: {accuracy:.4f} ({correct}/{total})")
    print(f"[Merge] Results saved to: {save_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="ScienceQA patch 分桶并行验证脚本")
    parser.add_argument("--model_path", type=str, default="", help="基础模型路径")
    parser.add_argument("--adapter_path", type=str, default="", help="LoRA 适配器路径")
    parser.add_argument("--scienceqa_path", type=str, default="ScienceQA", help="ScienceQA 数据集路径（本地目录或数据集名）")
    parser.add_argument("--split", type=str, default="test", help="ScienceQA split")
    parser.add_argument("--max_samples", type=int, default=0, help="最大评测样本数，0 表示全量")
    parser.add_argument("--max_new_tokens", type=int, default=16, help="第一轮答案生成长度")
    parser.add_argument("--second_pass_max_new_tokens", type=int, default=256, help="第二轮结构化整理的生成长度")
    parser.add_argument("--use_4bit", type=int, default=0, help="是否使用4bit加载")
    parser.add_argument("--use_vq", type=int, default=1, help="是否启用训练同款VQ视觉量化路径")
    parser.add_argument("--vq_codebook_size", type=int, default=1024, help="VQ codebook 大小")
    parser.add_argument("--freeze_vision_tower", type=int, default=0, help="是否冻结视觉塔参数")
    parser.add_argument("--vq_codebook_path", type=str, default="", help="训练得到的 vq_codebook.pt 路径")
    parser.add_argument(
        "--answer_mode",
        type=str,
        default="generate",
        choices=["generate", "logits", "hybrid"],
        help="答案获取方式：generate=仅生成解析, logits=仅下一token打分, hybrid=生成失败时回退logits",
    )
    parser.add_argument("--bucket_plan_path", type=str, required=True, help="预处理分桶 JSON 路径")
    parser.add_argument("--num_shards", type=int, default=4, help="总分片数")
    parser.add_argument("--shard_id", type=int, default=0, help="当前分片编号")
    parser.add_argument("--shard_result_dir", type=str, default="", help="分片结果目录")
    parser.add_argument("--merge_only", type=int, default=0, help="是否只执行 shard 结果合并")
    parser.add_argument("--save_path", type=str, default="./sciqa_eval_parallel_bucketed.json", help="保存结果路径")
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
            "second_pass_max_new_tokens": int(args.second_pass_max_new_tokens),
            "use_4bit": int(args.use_4bit),
            "use_vq": int(args.use_vq),
            "vq_codebook_size": int(args.vq_codebook_size),
            "vq_codebook_path": args.vq_codebook_path,
            "answer_mode": args.answer_mode,
            "prompt_style": "student_two_pass_legacy_then_structured_4field" if args.answer_mode == "generate" else "stage3_legacy",
            "parallel_eval": True,
            "bucketed_eval": True,
            "num_shards": int(args.num_shards),
            "bucket_plan_path": args.bucket_plan_path,
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
        "second_pass_max_new_tokens": int(args.second_pass_max_new_tokens),
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
        "prompt_style": "student_two_pass_legacy_then_structured_4field" if args.answer_mode == "generate" else "stage3_legacy",
        "parallel_eval": True,
        "bucketed_eval": True,
        "num_shards": int(args.num_shards),
        "shard_id": int(args.shard_id),
        "bucket_plan_path": args.bucket_plan_path,
    }
    run_eval_shard(
        model=model,
        processor=processor,
        bucket_plan_path=args.bucket_plan_path,
        scienceqa_path=args.scienceqa_path,
        split=args.split,
        max_samples=int(args.max_samples),
        max_new_tokens=int(args.max_new_tokens),
        second_pass_max_new_tokens=int(args.second_pass_max_new_tokens),
        save_path=args.save_path,
        answer_mode=args.answer_mode,
        num_shards=int(args.num_shards),
        shard_id=int(args.shard_id),
        run_config=run_config,
    )


if __name__ == "__main__":
    main()
