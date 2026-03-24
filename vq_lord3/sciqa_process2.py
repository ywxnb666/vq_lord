"""
======================================================================
SCIQA_PROCESS2 ---

ScienceQA 多模态验证脚本（旧 Stage3 prompt 口径）

功能:
1. 加载 LLaVA/LLaVA-Next 学生模型（可选 LoRA 适配器）
2. 使用旧 Stage3/ScienceQADataset prompt 模板进行推理
3. 解析模型输出并计算选择题准确率

Author: VQ-LoRD Project
Created: January 2026
======================================================================
"""

import argparse
import hashlib
import json
import os
import re
from typing import List, Optional

import torch
from datasets import load_dataset
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm
from vq_module2 import VQVisionEncoder


def file_md5(path: str) -> str:
    md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            md5.update(chunk)
    return md5.hexdigest()


def probe_adapter_fingerprint(adapter_path: str) -> dict:
    info = {
        "adapter_path": adapter_path,
        "exists": os.path.exists(adapter_path),
        "adapter_model_file": "",
        "adapter_model_md5": "",
        "adapter_config_md5": "",
    }
    if not info["exists"]:
        return info

    safetensors_path = os.path.join(adapter_path, "adapter_model.safetensors")
    bin_path = os.path.join(adapter_path, "adapter_model.bin")
    config_path = os.path.join(adapter_path, "adapter_config.json")

    if os.path.exists(safetensors_path):
        info["adapter_model_file"] = safetensors_path
        info["adapter_model_md5"] = file_md5(safetensors_path)
    elif os.path.exists(bin_path):
        info["adapter_model_file"] = bin_path
        info["adapter_model_md5"] = file_md5(bin_path)

    if os.path.exists(config_path):
        info["adapter_config_md5"] = file_md5(config_path)

    return info


def build_legacy_instruction(question: str, choices: List[str], hint: str = "") -> str:
    choices_text = ""
    for idx, choice in enumerate(choices):
        choices_text += f"({chr(65 + idx)}) {choice}\n"
    hint_block = f"Hint: {hint}\n" if hint else ""
    return f"Question: {question}\n{hint_block}Options:\n{choices_text}Answer:"


def build_prompt(processor, question: str, choices: List[str], hint: str = "") -> str:
    instruction_text = build_legacy_instruction(question, choices, hint)
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


def normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def extract_choice_from_output(output_text: str, choices: List[str]) -> Optional[int]:
    """尽量从输出中抽取选择题答案索引。"""
    if not output_text:
        return None

    raw = output_text.strip()
    normalized = normalize_text(raw)

    # 1) 最优先：开头直接给选项字母，如 "A" / "(B)" / "答案：C"
    max_letter = chr(65 + len(choices) - 1) if choices else "A"
    letter_class = f"A-{max_letter}" if max_letter >= "A" else "A"

    match = re.search(rf"^\s*\(?\s*([{letter_class}])\s*\)?\b", raw, flags=re.IGNORECASE)
    if match:
        idx = ord(match.group(1).upper()) - 65
        if 0 <= idx < len(choices):
            return idx

    # 2) 其次：显式 answer/答案 字段
    match = re.search(rf"(?:answer|答案)\s*[:：]\s*\(?\s*([{letter_class}])\s*\)?", raw, flags=re.IGNORECASE)
    if match:
        letter = match.group(1).upper()
        idx = ord(letter) - 65
        if 0 <= idx < len(choices):
            return idx

    # 3) 再次：文本中出现了唯一可用字母选项
    letter_hits = re.findall(rf"\b([{letter_class}])\b", raw, flags=re.IGNORECASE)
    if letter_hits:
        unique_hits = {ord(x.upper()) - 65 for x in letter_hits if 0 <= ord(x.upper()) - 65 < len(choices)}
        if len(unique_hits) == 1:
            return list(unique_hits)[0]

    # 4) 最后：按选项文本匹配（仅当唯一命中）
    choice_hits = []
    for idx, choice in enumerate(choices):
        if not choice:
            continue
        if normalize_text(choice) in normalized:
            choice_hits.append(idx)

    if len(choice_hits) == 1:
        return choice_hits[0]

    return None


def _get_letter_token_candidates(tokenizer, letter: str) -> List[int]:
    """收集表示某个选项字母的候选 token id（单 token 形式）。"""
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

    return list(candidate_ids)


def predict_choice_with_next_token_logits(
    model,
    tokenizer,
    input_ids,
    attention_mask,
    pixel_values,
    image_sizes,
    num_choices: int,
) -> Optional[int]:
    """基于 Answer: 之后的下一 token 概率直接选择 A/B/C/..."""
    if num_choices <= 0:
        return None

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_sizes=image_sizes,
            use_cache=False,
        )

    next_token_logits = outputs.logits[0, -1, :]

    best_idx = None
    best_score = None
    for idx in range(num_choices):
        letter = chr(65 + idx)
        cand_ids = _get_letter_token_candidates(tokenizer, letter)
        if not cand_ids:
            continue
        score = max(next_token_logits[token_id].item() for token_id in cand_ids)
        if best_score is None or score > best_score:
            best_score = score
            best_idx = idx

    return best_idx


def add_vq_inference_hook(model, vq_codebook_size: int, freeze_vision_tower: int):
    """给模型注入与训练一致的 VQ 视觉量化包装器（推理版）。"""
    base_model = model.get_base_model() if hasattr(model, "get_base_model") else model
    original_vision_tower = base_model.vision_tower

    vq_vision_encoder = VQVisionEncoder(
        vision_tower=original_vision_tower,
        num_embeddings=vq_codebook_size,
        freeze_vision_tower=bool(freeze_vision_tower),
    )

    try:
        vision_device = next(original_vision_tower.parameters()).device
        vq_vision_encoder.to(vision_device)
    except StopIteration:
        pass

    base_model.vq_vision_encoder = vq_vision_encoder
    base_model.vision_tower = vq_vision_encoder
    base_model._vq_loss_container = vq_vision_encoder.vq_cache
    base_model._vq_hook_handle = None
    return model


def load_vq_codebook_for_inference(model, vq_codebook_path: str):
    """加载训练得到的 VQ codebook。"""
    if not vq_codebook_path:
        return False
    if not os.path.exists(vq_codebook_path):
        print(f"[Warning] vq_codebook_path does not exist, skip VQ loading: {vq_codebook_path}")
        return False

    base_model = model.get_base_model() if hasattr(model, "get_base_model") else model
    try:
        codebook = torch.load(vq_codebook_path, map_location="cpu", weights_only=True)
    except TypeError:
        # 兼容旧版 PyTorch
        codebook = torch.load(vq_codebook_path, map_location="cpu")

    emb = base_model.vq_vision_encoder.vq.embedding.weight
    codebook = codebook.to(device=emb.device, dtype=emb.dtype)
    emb.data.copy_(codebook)

    # 与训练侧保持一致：除 codebook 外，还需恢复 pre/post quant 与量化器状态。
    vq_state_path = os.path.join(os.path.dirname(vq_codebook_path), "vq_encoder_state.pt")
    if os.path.exists(vq_state_path):
        try:
            vq_state = torch.load(vq_state_path, map_location="cpu", weights_only=True)
        except TypeError:
            # 兼容旧版 PyTorch
            vq_state = torch.load(vq_state_path, map_location="cpu")
        base_model.vq_vision_encoder.load_vq_state(vq_state)
    else:
        print(f"[Warning] vq_encoder_state not found, pre/post quant stays random: {vq_state_path}")

    return True


def load_model_and_processor(
    model_path: str,
    adapter_path: str,
    use_4bit: int,
    use_vq: int,
    vq_codebook_size: int,
    freeze_vision_tower: int,
    vq_codebook_path: str,
):
    load_info = {
        "model_path": model_path,
        "adapter_path": adapter_path,
        "adapter_loaded": False,
        "adapter_fingerprint": probe_adapter_fingerprint(adapter_path) if adapter_path else None,
        "use_vq": bool(use_vq),
        "vq_codebook_size": int(vq_codebook_size),
        "vq_codebook_path": vq_codebook_path,
        "vq_codebook_loaded": False,
    }

    if use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_path,
            device_map="auto",
            quantization_config=quantization_config,
            trust_remote_code=True,
        )
    else:
        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )

    if adapter_path:
        if os.path.exists(adapter_path):
            model = PeftModel.from_pretrained(model, adapter_path)
            load_info["adapter_loaded"] = True
        else:
            raise FileNotFoundError(
                f"adapter_path does not exist, refusing to fallback to base model: {adapter_path}"
            )

    if use_vq:
        model = add_vq_inference_hook(
            model,
            vq_codebook_size=vq_codebook_size,
            freeze_vision_tower=freeze_vision_tower,
        )
        loaded = load_vq_codebook_for_inference(model, vq_codebook_path)
        load_info["vq_codebook_loaded"] = bool(loaded)
        if loaded:
            print(f"[Info] VQ enabled with codebook: {vq_codebook_path}")
        else:
            print("[Warning] VQ enabled but codebook not loaded, behavior may mismatch training")

    model.eval()

    processor = LlavaNextProcessor.from_pretrained(model_path, trust_remote_code=True)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    return model, processor, load_info


def run_eval(
    model,
    processor,
    scienceqa_path: str,
    split: str,
    max_samples: int,
    max_new_tokens: int,
    save_path: str,
    answer_mode: str,
    run_config: Optional[dict] = None,
):
    dataset = load_dataset(scienceqa_path, split=split)
    dataset_with_images = [item for item in dataset if item.get("image") is not None]

    if max_samples > 0 and len(dataset_with_images) > max_samples:
        dataset_with_images = dataset_with_images[:max_samples]

    results = []
    correct = 0
    total = 0

    for item in tqdm(dataset_with_images, desc="ScienceQA Eval"):
        question = item.get("question", "")
        choices = item.get("choices", [])
        answer_idx = item.get("answer", 0)
        hint = item.get("hint", "")
        image = item.get("image")

        prompt = build_prompt(processor, question, choices, hint)
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

            # 仅解码新生成部分，避免把整段 prompt 回显当成答案解析。
            prompt_len = input_ids.shape[-1]
            gen_tokens = generated[0][prompt_len:]
            output_text = processor.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

            if not output_text:
                # 空生成时不要回退解码整段序列；那会把 prompt 误记成模型输出。
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
            "question": question,
            "choices": choices,
            "answer_idx": answer_idx,
            "pred_idx": pred_idx,
            "output": output_text,
            "correct": is_correct,
        })

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

    print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")
    print(f"Results saved to: {save_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="ScienceQA 多模态验证脚本（旧 Stage3 prompt 口径）")
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
    parser.add_argument("--save_path", type=str, default="./sciqa_eval.json", help="保存结果路径")
    return parser.parse_args()


def main():
    args = parse_args()
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
        "max_samples": args.max_samples,
        "max_new_tokens": args.max_new_tokens,
        "use_4bit": int(args.use_4bit),
        "use_vq": int(args.use_vq),
        "vq_codebook_size": int(args.vq_codebook_size),
        "vq_codebook_path": args.vq_codebook_path,
        "vq_codebook_loaded": load_info.get("vq_codebook_loaded", False),
        "answer_mode": args.answer_mode,
        "prompt_style": "stage3_legacy",
    }
    run_eval(
        model=model,
        processor=processor,
        scienceqa_path=args.scienceqa_path,
        split=args.split,
        max_samples=args.max_samples,
        max_new_tokens=args.max_new_tokens,
        save_path=args.save_path,
        answer_mode=args.answer_mode,
        run_config=run_config,
    )


if __name__ == "__main__":
    main()
