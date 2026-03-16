"""
======================================================================
TRAIN_VQ_LORD ---

VQ-LoRD 主训练脚本

结合 VQ 离散化和 LoRD 蒸馏，窃取多模态模型的图像识别能力

训练流程:
1. 加载预训练 LLaVA 模型
2. 添加 VQ 层到 Vision Encoder
3. 加载教师模型收集的视觉数据
4. 三阶段训练: VQ预训练 → 视觉蒸馏 → LoRD联合训练

    Author: VQ-LoRD Project
    Created: January 2026
======================================================================
"""

import os
import json
import hashlib
import random
import torch
import argparse
from collections import defaultdict
from typing import Optional, List, Dict
from pprint import pprint
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from torch.utils.tensorboard import SummaryWriter
from datasets import load_dataset

from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ==========================================
# Monkey Patch: 修复 transformers 版本兼容性问题
# 自动忽略旧版 config 中的 image_token 参数
# ==========================================
_original_init = LlavaNextProcessor.__init__

def _new_init(self, *args, **kwargs):
    kwargs.pop("image_token", None)  # 移除不支持的参数
    _original_init(self, *args, **kwargs)

LlavaNextProcessor.__init__ = _new_init
# ==========================================

from vq_module2 import VQVisionEncoder
from data_collector import GPT4VDataCollector, VQLORDDataset

import torch.nn.functional as F


def build_scienceqa_samples(
    scienceqa_path: str,
    split: str,
    train_num: int,
    seed: int,
) -> List[dict]:
    try:
        dataset = load_dataset(scienceqa_path, split=split)
    except Exception as exc:
        offline = os.environ.get("HF_DATASETS_OFFLINE", "0") == "1"
        if offline and not os.path.exists(scienceqa_path):
            raise RuntimeError(
                f"无法加载 ScienceQA: scienceqa_path={scienceqa_path}。当前处于离线模式，且该路径不是本地目录。"
            ) from exc
        raise RuntimeError(
            f"无法加载 ScienceQA: scienceqa_path={scienceqa_path}, split={split}"
        ) from exc
    dataset_with_images = [item for item in dataset if item.get("image") is not None]

    all_indices = list(range(len(dataset_with_images)))
    random.seed(seed)
    random.shuffle(all_indices)
    if train_num > 0 and len(all_indices) > train_num:
        all_indices = all_indices[:train_num]

    samples = []
    for sampled_pos, dataset_idx in enumerate(all_indices):
        item = dataset_with_images[dataset_idx]
        question = item.get("question", "")
        choices = item.get("choices", [])
        choices_text = ""
        for choice_idx, choice in enumerate(choices):
            choices_text += f"({chr(65 + choice_idx)}) {choice}\n"

        answer_idx = int(item.get("answer", 0))
        answer = choices[answer_idx] if choices and answer_idx < len(choices) else ""
        answer_letter = chr(65 + answer_idx) if 0 <= answer_idx < 26 else "A"
        lecture = item.get("lecture", "")
        solution = item.get("solution", "")

        instruction = f"<image>\nQuestion: {question}\nOptions:\n{choices_text}Answer:"
        if lecture:
            response = f"Explanation: {lecture}\nSolution: {solution}\nAnswer: {answer}"
        else:
            response = f"Solution: {solution}\nAnswer: {answer}"

        samples.append({
            "sample_id": sampled_pos,
            "source_index": dataset_idx,
            "image": item.get("image"),
            "question": question,
            "choices": choices,
            "answer_idx": answer_idx,
            "answer_letter": answer_letter,
            "answer_text": answer,
            "instruction": instruction,
            "response": response,
        })

    return samples


def _safe_name(text: str) -> str:
    keep = []
    for ch in text:
        if ch.isalnum() or ch in ("-", "_", "."):
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep)


def _sample_key(sample: dict) -> str:
    instruction = sample.get("instruction", "")
    response = sample.get("response", "")
    image = sample.get("image")
    size = ""
    if image is not None and hasattr(image, "size"):
        size = f"{image.size[0]}x{image.size[1]}"
    raw = f"{instruction}\n{response}\n{size}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def normalize_teacher_response(text: str, lang: str = "zh") -> str:
    """规范教师回答字段标签，尽量减少中英文模板混杂。"""
    if not isinstance(text, str):
        return ""

    text = text.strip()
    # 防止教师回答里回显 <image>，破坏 LLaVA-Next 的图文 token 对齐
    text = text.replace("<image>", "")
    text = text.replace("< image >", "")
    text = text.replace("<Image>", "")

    if lang == "zh":
        text = text.replace("Explanation:", "解释：")
        text = text.replace("Answer:", "答案：")
    else:
        text = text.replace("解释：", "Explanation:")
        text = text.replace("答案：", "Answer:")
    return text


def _extract_teacher_rationale(text: str) -> str:
    if not isinstance(text, str):
        return ""

    cleaned = text.strip()
    if not cleaned:
        return ""

    explanation_markers = ["Explanation:", "解释："]
    answer_markers = ["\nAnswer:", "\n答案：", "Answer:", "答案："]

    rationale = cleaned
    for marker in explanation_markers:
        if marker in rationale:
            rationale = rationale.split(marker, 1)[1].strip()
            break

    for marker in answer_markers:
        if marker in rationale:
            rationale = rationale.split(marker, 1)[0].strip()

    return rationale.strip()


def attach_gpt4v_teacher_responses(samples: List[dict], args) -> List[dict]:
    """
    为 ScienceQA 样本补充教师模型回答（带缓存）

    说明：
    - 若缓存存在，优先读取缓存；
    - 若启用采集且存在 API key，则补齐缺失项；
    - 若 strict_teacher_distill=1 且仍有缺失，将直接报错。
    """
    os.makedirs(args.data_dir, exist_ok=True)

    cache_path = args.teacher_cache_path
    if not cache_path:
        victim_tag = _safe_name(args.victim_model)
        cache_name = (
            f"scienceqa_teacher_{victim_tag}_{args.scienceqa_split}_"
            f"n{args.train_num}_seed{args.scienceqa_seed}.json"
        )
        cache_path = os.path.join(args.data_dir, cache_name)

    cache_data = {}
    teacher_lang = getattr(args, "teacher_lang", "zh")
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            cache_data = payload.get("samples", {}) if isinstance(payload, dict) else {}
            print(f"已加载教师缓存: {cache_path}, 条目数={len(cache_data)}")
        except Exception as e:
            print(f"警告: 教师缓存读取失败，将忽略缓存 ({e})")
            cache_data = {}

    need_collect = []
    for i, sample in enumerate(samples):
        key = _sample_key(sample)
        teacher_response = cache_data.get(key)
        if teacher_response:
            sample["teacher_response"] = normalize_teacher_response(teacher_response, teacher_lang)
            sample["teacher_source"] = args.victim_model
        else:
            need_collect.append((i, key))

    if need_collect and args.collect_teacher_data:
        collector = GPT4VDataCollector(
            api_key=(args.teacher_api_key if args.teacher_api_key else None),
            base_url=(args.teacher_api_base if args.teacher_api_base else None),
            model=args.victim_model,
            save_dir=args.data_dir,
        )

        if not collector.api_key:
            msg = (
                "需要采集教师模型回答，但未检测到 API Key。"
                "可通过 --teacher_api_key 或 OPENAI_API_KEY 提供。"
            )
            if args.strict_teacher_distill:
                raise RuntimeError(msg)
            print(f"警告: {msg}")
        else:
            print(f"开始采集教师回答，待补齐样本数: {len(need_collect)}")
            for rank, (idx, key) in enumerate(tqdm(need_collect, desc="采集教师模型回答"), start=1):
                sample = samples[idx]
                image = sample.get("image")
                instruction = sample.get("instruction", "")

                if teacher_lang == "en":
                    teacher_prompt = (
                        "You are a rigorous visual science QA assistant. "
                        "Please answer the following multiple-choice question based on the image. "
                        "Use only English. "
                        "Output format must be:\n"
                        "Explanation: ...\n"
                        "Answer: ...\n\n"
                        f"{instruction}"
                    )
                else:
                    teacher_prompt = (
                        "你是严谨的视觉科学题解答助手。"
                        "请基于图片回答下面的选择题。"
                        "请仅使用中文回答。"
                        "输出格式必须为：\n"
                        "解释：...\n"
                        "答案：...\n\n"
                        f"{instruction}"
                    )

                teacher_response = None
                if image is not None:
                    teacher_response = collector.query_gpt4v_image(
                        image=image,
                        prompt=teacher_prompt,
                        max_tokens=args.max_new_tokens,
                        image_format="PNG",
                    )

                if teacher_response:
                    teacher_response = normalize_teacher_response(teacher_response, teacher_lang)
                    sample["teacher_response"] = teacher_response
                    sample["teacher_source"] = args.victim_model
                    cache_data[key] = teacher_response
                else:
                    # 非 strict 模式下回退到原始 response
                    sample["teacher_response"] = normalize_teacher_response(sample.get("response", ""), teacher_lang)
                    sample["teacher_source"] = "fallback_dataset"

                if rank % 10 == 0 or rank == len(need_collect):
                    with open(cache_path, "w", encoding="utf-8") as f:
                        json.dump(
                            {
                                "victim_model": args.victim_model,
                                "scienceqa_split": args.scienceqa_split,
                                "train_num": args.train_num,
                                "scienceqa_seed": args.scienceqa_seed,
                                "samples": cache_data,
                            },
                            f,
                            ensure_ascii=False,
                            indent=2,
                        )

    # 兜底：保证每个样本存在 teacher_response 字段
    missing = 0
    for sample in samples:
        if "teacher_response" not in sample:
            sample["teacher_response"] = normalize_teacher_response(sample.get("response", ""), teacher_lang)
            sample["teacher_source"] = "fallback_dataset"
        if sample.get("teacher_source") != args.victim_model:
            missing += 1

    print(
        f"教师样本统计: total={len(samples)}, "
        f"from_{args.victim_model}={len(samples)-missing}, fallback={missing}"
    )

    if args.strict_teacher_distill and missing > 0:
        raise RuntimeError(
            f"严格蒸馏模式开启，但仍有 {missing} 条样本未使用 {args.victim_model} 教师回答。"
        )

    return samples


class ScienceQADataset(torch.utils.data.Dataset):
    """ScienceQA 多模态数据集包装（兼容 Stage2 双监督与 Stage3 单监督）。"""

    def __init__(
        self,
        processor,
        scienceqa_path: str = "ScienceQA",
        split: str = "train",
        train_num: int = 500,
        max_length: int = 512,
        samples: Optional[List[dict]] = None,
        seed: int = 20240306,
        teacher_lang: str = "zh",
    ):
        self.processor = processor
        self.max_length = max_length
        self.teacher_lang = teacher_lang

        if samples is None:
            samples = build_scienceqa_samples(
                scienceqa_path=scienceqa_path,
                split=split,
                train_num=train_num,
                seed=seed,
            )

        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def _build_targets(self, item: dict, instruction_text: str) -> tuple[str, str, bool]:
        answer_letter = item.get("answer_letter", "A")
        if not isinstance(answer_letter, str) or len(answer_letter) == 0:
            answer_letter = "A"
        answer_letter = answer_letter.strip().upper()[0]

        # 固定答案锚点，确保 Stage2/Stage3 指标口径一致。
        answer_target = f"Answer: {answer_letter}"

        teacher_response = item.get("teacher_response", item.get("response", ""))
        teacher_response = teacher_response.replace("<image>", "").replace("< image >", "").replace("<Image>", "")
        rationale = _extract_teacher_rationale(teacher_response)
        has_rationale = bool(rationale)
        rationale_prefix = ""

        if has_rationale:
            if self.teacher_lang == "zh":
                rationale_prefix = f"解释：{rationale}"
            else:
                rationale_prefix = f"Explanation: {rationale}"
            rationale_target = f"{rationale_prefix}\n{answer_target}"
        else:
            rationale_target = answer_target

        # 预截断 rationale 文本，保留图文 token 对齐。
        max_text_tokens = self.max_length
        instr_tokens = self.processor.tokenizer.encode(instruction_text, add_special_tokens=False)
        ratio_tokens = self.processor.tokenizer.encode(rationale_target, add_special_tokens=False)
        budget = max_text_tokens - len(instr_tokens) - 2
        if budget > 0 and len(ratio_tokens) > budget:
            answer_tokens = self.processor.tokenizer.encode(answer_target, add_special_tokens=False)
            if has_rationale and len(answer_tokens) < budget:
                prefix_tokens = self.processor.tokenizer.encode(rationale_prefix, add_special_tokens=False)
                keep_prefix_tokens = prefix_tokens[:max(0, budget - len(answer_tokens) - 1)]
                trimmed_prefix = self.processor.tokenizer.decode(keep_prefix_tokens, skip_special_tokens=True).strip()
                rationale_target = f"{trimmed_prefix}\n{answer_target}" if trimmed_prefix else answer_target
                has_rationale = bool(trimmed_prefix)
            else:
                rationale_target = answer_target
                has_rationale = False

        return answer_target, rationale_target, has_rationale

    def __getitem__(self, idx):
        item = self.samples[idx]
        image = item["image"]
        image_sizes_default = torch.tensor([image.height, image.width], dtype=torch.long)

        instruction = item.get("instruction", "")
        instruction_text = instruction.replace("<image>", "").replace("< image >", "").replace("<Image>", "").strip()
        answer_target, rationale_target, has_rationale = self._build_targets(item, instruction_text)

        if hasattr(self.processor, "apply_chat_template"):
            prompt_conv = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": instruction_text},
                        {"type": "image"},
                    ],
                }
            ]
            full_conv = prompt_conv + [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": rationale_target},
                    ],
                }
            ]
            answer_conv = prompt_conv + [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": answer_target},
                    ],
                }
            ]

            prompt_text = self.processor.apply_chat_template(prompt_conv, add_generation_prompt=True)
            full_text = self.processor.apply_chat_template(full_conv, add_generation_prompt=False)
            answer_text = self.processor.apply_chat_template(answer_conv, add_generation_prompt=False)
        else:
            prompt_text = f"<image>\n{instruction_text}"
            full_text = f"{prompt_text}\n{rationale_target}"
            answer_text = f"{prompt_text}\n{answer_target}"

        prompt_inputs = self.processor(
            text=prompt_text,
            images=image,
            return_tensors="pt",
            padding="longest",
            truncation=False,
        )
        full_inputs = self.processor(
            text=full_text,
            images=image,
            return_tensors="pt",
            padding="longest",
            truncation=False,
        )
        answer_inputs = self.processor(
            text=answer_text,
            images=image,
            return_tensors="pt",
            padding="longest",
            truncation=False,
        )

        image_sizes = image_sizes_default
        pad_id = self.processor.tokenizer.pad_token_id

        full_labels = full_inputs["input_ids"].squeeze(0).clone()
        answer_labels = answer_inputs["input_ids"].squeeze(0).clone()

        prompt_len = prompt_inputs["input_ids"].shape[1]
        prompt_len = min(prompt_len, full_labels.shape[0], answer_labels.shape[0])

        full_labels[:prompt_len] = -100
        answer_labels[:prompt_len] = -100

        full_labels = full_labels.masked_fill(full_labels == pad_id, -100)
        answer_labels = answer_labels.masked_fill(answer_labels == pad_id, -100)
        answer_letter = str(item.get("answer_letter", "A") or "A").strip().upper()[0]

        return {
            # 兼容 Stage3/旧逻辑
            "input_ids": full_inputs["input_ids"].squeeze(0),
            "attention_mask": full_inputs["attention_mask"].squeeze(0),
            "labels": full_labels,
            # Stage2 双监督
            "prompt_input_ids": prompt_inputs["input_ids"].squeeze(0),
            "prompt_attention_mask": prompt_inputs["attention_mask"].squeeze(0),
            "full_input_ids": full_inputs["input_ids"].squeeze(0),
            "full_attention_mask": full_inputs["attention_mask"].squeeze(0),
            "full_labels": full_labels,
            "answer_input_ids": answer_inputs["input_ids"].squeeze(0),
            "answer_attention_mask": answer_inputs["attention_mask"].squeeze(0),
            "answer_labels": answer_labels,
            "pixel_values": full_inputs["pixel_values"].squeeze(0),
            "image_sizes": image_sizes,
            "has_rationale": int(has_rationale),
            "answer_idx": int(item.get("answer_idx", 0)),
            "answer_letter": answer_letter,
            "data_type": "scienceqa",
        }


class ScienceQABucketBatchSampler(torch.utils.data.Sampler):
    """基于预处理结果中的 bucket_key 生成 batch。"""

    def __init__(
        self,
        sample_to_bucket: Dict[int, str],
        batch_size: int,
        drop_last: bool = False,
        shuffle: bool = True,
        seed: int = 0,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")

        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.bucket_to_samples = defaultdict(list)

        for sample_id, bucket_key in sorted(sample_to_bucket.items()):
            self.bucket_to_samples[str(bucket_key)].append(int(sample_id))

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        batches = []

        for bucket_key in sorted(self.bucket_to_samples.keys()):
            sample_ids = list(self.bucket_to_samples[bucket_key])
            if self.shuffle:
                rng.shuffle(sample_ids)

            for start in range(0, len(sample_ids), self.batch_size):
                batch = sample_ids[start:start + self.batch_size]
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                batches.append(batch)

        if self.shuffle:
            rng.shuffle(batches)

        return iter(batches)

    def __len__(self):
        total = 0
        for sample_ids in self.bucket_to_samples.values():
            if self.drop_last:
                total += len(sample_ids) // self.batch_size
            else:
                total += (len(sample_ids) + self.batch_size - 1) // self.batch_size
        return total


def load_scienceqa_preprocessed_buckets(preprocessed_path: str, expected_len: int, args) -> Optional[dict]:
    if not preprocessed_path:
        return None
    if not os.path.exists(preprocessed_path):
        raise FileNotFoundError(f"未找到 ScienceQA 预处理文件: {preprocessed_path}")

    with open(preprocessed_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    config = payload.get("config", {})
    records = payload.get("samples", [])
    if not isinstance(records, list) or not records:
        raise RuntimeError(f"预处理文件缺少 samples 记录: {preprocessed_path}")

    sample_to_bucket = {}
    sample_ids = []
    for record in records:
        sample_id = int(record["sample_id"])
        sample_to_bucket[sample_id] = str(record["bucket_key"])
        sample_ids.append(sample_id)

    sorted_ids = sorted(sample_ids)
    expected_ids = list(range(expected_len))
    if sorted_ids != expected_ids:
        raise RuntimeError(
            f"预处理文件 sample_id 与当前数据集不一致: expected 0..{expected_len-1}, "
            f"got range {sorted_ids[:3]}...{sorted_ids[-3:]}"
        )

    preprocess_split = config.get("split")
    preprocess_train_num = int(config.get("train_num", -1)) if config.get("train_num") is not None else -1
    preprocess_seed = int(config.get("seed", -1)) if config.get("seed") is not None else -1
    if preprocess_split not in (None, args.scienceqa_split):
        raise RuntimeError(
            f"预处理 split={preprocess_split} 与训练 split={args.scienceqa_split} 不一致"
        )
    if preprocess_train_num not in (-1, args.train_num):
        raise RuntimeError(
            f"预处理 train_num={preprocess_train_num} 与训练 train_num={args.train_num} 不一致"
        )
    if preprocess_seed not in (-1, args.scienceqa_seed):
        raise RuntimeError(
            f"预处理 seed={preprocess_seed} 与训练 seed={args.scienceqa_seed} 不一致"
        )

    print(
        f"已加载 ScienceQA 预处理文件: {preprocessed_path}, "
        f"bucket_by={config.get('bucket_by')}, batch_size={config.get('bucket_batch_size')}"
    )

    return {
        "path": preprocessed_path,
        "config": config,
        "sample_to_bucket": sample_to_bucket,
    }


def maybe_set_dataloader_epoch(dataloader, epoch: int):
    sampler = getattr(dataloader, "batch_sampler", None)
    if sampler is not None and hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)
        return

    sampler = getattr(dataloader, "sampler", None)
    if sampler is not None and hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)


def collect_dataloader_batch_stats(dataloader) -> dict:
    stats = {
        "num_batches": 0,
        "num_samples": 0,
        "batch_size_hist": {},
        "bucket_hist": {},
        "mean_batch_size": 0.0,
        "full_batch_ratio": 0.0,
    }

    batch_sampler = getattr(dataloader, "batch_sampler", None)
    if batch_sampler is not None and isinstance(batch_sampler, ScienceQABucketBatchSampler):
        batch_size_hist = defaultdict(int)
        bucket_hist = defaultdict(int)
        full_batches = 0
        total_batch_size = 0

        for bucket_key, sample_ids in batch_sampler.bucket_to_samples.items():
            n_samples = len(sample_ids)
            stats["num_samples"] += n_samples
            bucket_hist[str(bucket_key)] = n_samples

            for start in range(0, n_samples, batch_sampler.batch_size):
                cur_batch = sample_ids[start:start + batch_sampler.batch_size]
                if batch_sampler.drop_last and len(cur_batch) < batch_sampler.batch_size:
                    continue
                cur_size = len(cur_batch)
                batch_size_hist[cur_size] += 1
                stats["num_batches"] += 1
                total_batch_size += cur_size
                if cur_size == batch_sampler.batch_size:
                    full_batches += 1

        stats["batch_size_hist"] = dict(sorted(batch_size_hist.items()))
        stats["bucket_hist"] = dict(sorted(bucket_hist.items(), key=lambda item: (-item[1], item[0])))
        if stats["num_batches"] > 0:
            stats["mean_batch_size"] = total_batch_size / stats["num_batches"]
            stats["full_batch_ratio"] = full_batches / stats["num_batches"]
        return stats

    batch_size = getattr(dataloader, "batch_size", None)
    dataset_len = len(getattr(dataloader, "dataset", []))
    if batch_size is None or batch_size <= 0:
        return stats

    n_full = dataset_len // batch_size
    tail = dataset_len % batch_size
    batch_size_hist = defaultdict(int)
    if n_full > 0:
        batch_size_hist[batch_size] = n_full
    if tail > 0 and not getattr(dataloader, "drop_last", False):
        batch_size_hist[tail] += 1

    stats["num_samples"] = dataset_len
    stats["num_batches"] = sum(batch_size_hist.values())
    stats["batch_size_hist"] = dict(sorted(batch_size_hist.items()))
    if stats["num_batches"] > 0:
        total_batch_size = sum(size * count for size, count in batch_size_hist.items())
        stats["mean_batch_size"] = total_batch_size / stats["num_batches"]
        stats["full_batch_ratio"] = batch_size_hist.get(batch_size, 0) / stats["num_batches"]
    return stats


def log_dataloader_batch_stats(stage_tag: str, dataloader, tb_writer, top_k_buckets: int = 8):
    stats = collect_dataloader_batch_stats(dataloader)
    hist_str = ", ".join(f"bs{size}:{count}" for size, count in stats["batch_size_hist"].items()) or "empty"
    print(
        f"[{stage_tag}] batch 统计: batches={stats['num_batches']}, samples={stats['num_samples']}, "
        f"mean_bs={stats['mean_batch_size']:.2f}, full_batch_ratio={stats['full_batch_ratio']:.3f}, hist={hist_str}"
    )

    if stats["bucket_hist"]:
        top_buckets = list(stats["bucket_hist"].items())[:top_k_buckets]
        bucket_str = ", ".join(f"{key}:{count}" for key, count in top_buckets)
        print(f"[{stage_tag}] top bucket: {bucket_str}")

    if tb_writer is not None:
        tb_writer.add_scalar(f"{stage_tag}/num_batches", stats["num_batches"], 0)
        tb_writer.add_scalar(f"{stage_tag}/num_samples", stats["num_samples"], 0)
        tb_writer.add_scalar(f"{stage_tag}/mean_batch_size", stats["mean_batch_size"], 0)
        tb_writer.add_scalar(f"{stage_tag}/full_batch_ratio", stats["full_batch_ratio"], 0)
        for size, count in stats["batch_size_hist"].items():
            tb_writer.add_scalar(f"{stage_tag}/batch_hist_bs{size}", count, 0)


def sanitize_image_sizes(image_sizes: Optional[torch.Tensor], batch_size: int) -> Optional[torch.Tensor]:
    """规范化 image_sizes 为 CPU int64 [bs, 2]。"""
    if image_sizes is None:
        return None
    if not isinstance(image_sizes, torch.Tensor):
        image_sizes = torch.tensor(image_sizes, dtype=torch.long)
    if image_sizes.dim() == 1:
        image_sizes = image_sizes.unsqueeze(0)
    image_sizes = image_sizes[:, :2].to(dtype=torch.long).clamp(min=1)
    if image_sizes.is_cuda:
        image_sizes = image_sizes.cpu()
    return image_sizes


def _get_image_token_id(model) -> int:
    """从 model.config 获取 image_token_index（已在 load_model_and_processor 中统一设置）。"""
    cfg = getattr(model, "config", None)
    if cfg is None:
        # PeftModel: 穿透到 base model
        base = getattr(model, "get_base_model", lambda: None)()
        cfg = getattr(base, "config", None) if base is not None else None
    val = getattr(cfg, "image_token_index", None)
    if val is None:
        raise RuntimeError("model.config 中未找到 image_token_index，请检查模型加载")
    return int(val)


def _resolve_torch_dtype(dtype_name: str) -> torch.dtype:
    dtype_name = str(dtype_name).lower()
    mapping = {
        "auto": torch.bfloat16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if dtype_name not in mapping:
        raise ValueError(f"不支持的 model_dtype={dtype_name}")
    return mapping[dtype_name]


def _iter_use_cache_holders(model):
    seen = set()
    holders = []

    def _add(obj):
        if obj is None or not hasattr(obj, "use_cache"):
            return
        obj_id = id(obj)
        if obj_id in seen:
            return
        seen.add(obj_id)
        holders.append(obj)

    _add(getattr(model, "config", None))
    _add(getattr(model, "generation_config", None))

    get_base_model = getattr(model, "get_base_model", None)
    if callable(get_base_model):
        base_model = get_base_model()
        _add(getattr(base_model, "config", None))
        _add(getattr(base_model, "generation_config", None))
        _add(getattr(getattr(base_model, "model", None), "config", None))
        _add(getattr(getattr(base_model, "model", None), "generation_config", None))

    return holders


def _set_model_use_cache(model, enabled: bool):
    prev_states = []
    for holder in _iter_use_cache_holders(model):
        prev_states.append((holder, holder.use_cache))
        holder.use_cache = enabled
    return prev_states


def _restore_model_use_cache(prev_states):
    for holder, prev_value in prev_states:
        holder.use_cache = prev_value


def _iter_gradient_checkpointing_targets(model):
    seen = set()
    targets = []

    def _add(obj):
        if obj is None:
            return
        obj_id = id(obj)
        if obj_id in seen:
            return
        seen.add(obj_id)
        targets.append(obj)

    _add(model)
    _add(getattr(model, "model", None))
    _add(getattr(model, "language_model", None))
    _add(getattr(model, "vision_tower", None))

    get_base_model = getattr(model, "get_base_model", None)
    if callable(get_base_model):
        base_model = get_base_model()
        _add(base_model)
        _add(getattr(base_model, "model", None))
        _add(getattr(base_model, "language_model", None))
        _add(getattr(base_model, "vision_tower", None))
        inner_model = getattr(base_model, "model", None)
        _add(getattr(inner_model, "language_model", None))
        _add(getattr(inner_model, "vision_tower", None))

    return targets


def _set_model_gradient_checkpointing(model, enabled: bool):
    prev_states = []

    for target in _iter_gradient_checkpointing_targets(model):
        has_gc_attr = hasattr(target, "gradient_checkpointing")
        prev_enabled = bool(
            getattr(target, "is_gradient_checkpointing", False)
            or getattr(target, "gradient_checkpointing", False)
        )
        prev_states.append((target, prev_enabled))

        if enabled:
            if hasattr(target, "gradient_checkpointing_enable"):
                try:
                    target.gradient_checkpointing_enable(
                        gradient_checkpointing_kwargs={"use_reentrant": False}
                    )
                except TypeError:
                    target.gradient_checkpointing_enable()
            elif has_gc_attr:
                target.gradient_checkpointing = True
        else:
            if hasattr(target, "gradient_checkpointing_disable"):
                target.gradient_checkpointing_disable()
            elif has_gc_attr:
                target.gradient_checkpointing = False

            if has_gc_attr:
                target.gradient_checkpointing = False

    return prev_states


def _restore_model_gradient_checkpointing(prev_states):
    for target, prev_enabled in prev_states:
        if prev_enabled:
            if hasattr(target, "gradient_checkpointing_enable"):
                try:
                    target.gradient_checkpointing_enable(
                        gradient_checkpointing_kwargs={"use_reentrant": False}
                    )
                except TypeError:
                    target.gradient_checkpointing_enable()
            elif hasattr(target, "gradient_checkpointing"):
                target.gradient_checkpointing = True
        else:
            if hasattr(target, "gradient_checkpointing_disable"):
                target.gradient_checkpointing_disable()
            elif hasattr(target, "gradient_checkpointing"):
                target.gradient_checkpointing = False


def _enable_input_require_grads_if_needed(model):
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
        return

    input_embeddings = model.get_input_embeddings()
    if input_embeddings is None:
        return

    def _make_inputs_require_grad(module, inputs, output):
        output.requires_grad_(True)

    input_embeddings.register_forward_hook(_make_inputs_require_grad)


def _capture_rng_state() -> dict:
    state = {
        "python_random_state": random.getstate(),
        "torch_rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: Optional[dict]):
    if not state:
        return

    python_random_state = state.get("python_random_state")
    if python_random_state is not None:
        random.setstate(python_random_state)

    torch_rng_state = state.get("torch_rng_state")
    if torch_rng_state is not None:
        torch.set_rng_state(torch_rng_state)

    cuda_rng_state_all = state.get("torch_cuda_rng_state_all")
    if cuda_rng_state_all is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_rng_state_all)


def _get_trainable_parameter_state(model) -> dict:
    trainable_state = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_state[name] = param.detach().cpu()
    return trainable_state


def _load_parameter_state(model, parameter_state: dict, state_name: str):
    if not parameter_state:
        print(f"[Resume] {state_name} 为空，跳过加载")
        return

    model_state = model.state_dict()
    loaded_names = []
    skipped_names = []

    with torch.no_grad():
        for name, tensor in parameter_state.items():
            target = model_state.get(name)
            if target is None:
                skipped_names.append(name)
                continue
            target.copy_(tensor.to(device=target.device, dtype=target.dtype))
            loaded_names.append(name)

    print(
        f"[Resume] 已加载 {state_name}: loaded={len(loaded_names)}, skipped={len(skipped_names)}"
    )
    if skipped_names:
        preview = ", ".join(skipped_names[:8])
        print(f"[Resume] 未匹配参数示例: {preview}")


def save_stage3_checkpoint(model, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir, safe_serialization=False)

    save_vq_codebook(model, os.path.join(save_dir, "vq_codebook.pt"))
    torch.save(
        _get_trainable_parameter_state(model),
        os.path.join(save_dir, "trainable_model_state.pt")
    )
    print(f"Stage3 检查点已保存至 {save_dir}")


def load_stage3_checkpoint(model, ckpt_dir: str, device: str):
    if not os.path.exists(ckpt_dir):
        raise FileNotFoundError(f"未找到 Stage3 checkpoint 目录: {ckpt_dir}")

    trainable_state_path = os.path.join(ckpt_dir, "trainable_model_state.pt")
    if os.path.exists(trainable_state_path):
        parameter_state = torch.load(trainable_state_path, map_location="cpu")
        _load_parameter_state(model, parameter_state, "Stage3 trainable_model_state")
    else:
        print(f"[Resume] 缺少 trainable_model_state.pt: {ckpt_dir}")

    vq_resume_path = os.path.join(ckpt_dir, "vq_codebook.pt")
    if os.path.exists(vq_resume_path):
        load_vq_codebook(model, vq_resume_path)
        print(f"[Resume] 已加载 Stage3 VQ stack: {vq_resume_path}")

    return model


def setup_args():
    """设置训练参数"""
    parser = argparse.ArgumentParser(description="VQ-LoRD 训练脚本")
    
    # 模型参数
    parser.add_argument("--model_path", type=str, 
                       default="/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/luye/align_vq/downloads/models/llama3-llava-next-8b-hf",
                       help="LLaVA 模型路径")
    parser.add_argument("--victim_model", type=str,
                       default="gpt-4-vision-preview",
                       help="教师模型 (API)")
    parser.add_argument("--teacher_api_base", type=str, default="",
                       help="教师 API Base URL（OpenAI 兼容）")
    parser.add_argument("--teacher_api_key", type=str, default="",
                       help="教师 API Key（留空则回退到 OPENAI_API_KEY）")
    
    # VQ 参数
    parser.add_argument("--vq_codebook_size", type=int, default=8192,
                       help="VQ codebook 大小")
    parser.add_argument("--vq_commitment_cost", type=float, default=0.25,
                       help="VQ commitment loss 权重")
    parser.add_argument("--vq_dead_code_threshold", type=float, default=1.0,
                       help="EMA 使用率低于该阈值的 code 视为 dead code")
    parser.add_argument("--vq_usage_decay", type=float, default=0.99,
                       help="code 使用率 EMA 衰减系数")
    parser.add_argument("--vq_dead_code_reset_interval", type=int, default=20,
                       help="每隔多少 step 尝试重置 dead code；<=0 表示关闭")
    parser.add_argument("--vq_legacy_loss", type=int, default=0,
                       help="是否使用 taming 的 legacy VQ loss（1=legacy,0=修正版）")
    parser.add_argument("--freeze_vision_tower", type=int, default=0,
                       help="是否冻结原始 vision tower")
    
    # 损失权重
    parser.add_argument("--beta", type=float, default=0.25,
                       help="VQ 损失权重")
    parser.add_argument("--temperature", type=float, default=1.5,
                       help="Stage3 采样温度")
    
    # 训练参数
    parser.add_argument("--stage", type=int, default=3,
                       help="训练阶段 (1=VQ预训练, 2=视觉蒸馏, 3=联合训练)")
    parser.add_argument("--epochs", type=int, default=3,
                       help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="批次大小")
    parser.add_argument("--lr", type=float, default=3e-5,
                       help="学习率")
    parser.add_argument("--stage1_lr", type=float, default=0.0,
                       help="Stage1 学习率，<=0 时使用 lr*5")
    parser.add_argument("--stage1_recon_weight", type=float, default=1.0,
                       help="Stage1 特征重建损失权重")
    parser.add_argument("--stage1_cosine_weight", type=float, default=0.25,
                       help="Stage1 特征余弦损失权重")
    parser.add_argument("--stage1_vq_weight", type=float, default=1.0,
                       help="Stage1 VQ 损失权重")
    parser.add_argument("--stage1_grad_clip", type=float, default=5.0,
                       help="Stage1 梯度裁剪阈值；<=0 关闭")
    parser.add_argument("--max_length", type=int, default=512,
                       help="最大序列长度")
    
    # LoRD 超参数
    parser.add_argument("--tau1", type=float, default=0.01,
                       help="LoRD 冷启动阈值: exp(avg_lp) < tau1 时用教师兜底")
    parser.add_argument("--max_new_tokens", type=int, default=128,
                       help="LoRD 生成最大新 token 数")
    parser.add_argument("--grad_accum", type=int, default=4,
                       help="梯度累积步数 (等效增大 batch size)")
    parser.add_argument("--stage2_grad_accum", type=int, default=0,
                       help="Stage2 梯度累积步数，0 表示回退到 grad_accum")
    parser.add_argument("--stage2_answer_weight", type=float, default=1.0,
                       help="Stage2 答案监督权重")
    parser.add_argument("--stage2_rationale_weight", type=float, default=0.3,
                       help="Stage2 解释监督权重")
    parser.add_argument("--stage2_prepost_lr_scale", type=float, default=0.5,
                       help="Stage2 pre/post quant 学习率缩放")
    parser.add_argument("--stage2_vision_lr_scale", type=float, default=0.2,
                       help="Stage2 视觉塔学习率缩放")
    parser.add_argument("--stage2_grad_clip", type=float, default=1.0,
                       help="Stage2 梯度裁剪阈值；<=0 关闭")
    parser.add_argument("--stage3_grad_accum", type=int, default=0,
                       help="Stage3 梯度累积步数，0 表示回退到 grad_accum")
    parser.add_argument("--stage3_lr_scale", type=float, default=0.2,
                       help="Stage3 学习率缩放，默认低于 Stage2")
    parser.add_argument("--stage3_train_projector", type=int, default=0,
                       help="Stage3 是否继续训练 projector，默认冻结以保护 Stage2 建立的视觉桥")
    
    # LoRA 参数
    parser.add_argument("--use_lora", type=int, default=1,
                       help="是否使用 LoRA")
    parser.add_argument("--lora_rank", type=int, default=32,
                       help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=64,
                       help="LoRA alpha")
    
    # 量化参数
    parser.add_argument("--use_4bit", type=int, default=1,
                       help="是否使用 4-bit 量化")
    parser.add_argument("--model_dtype", type=str, default="bfloat16",
                       choices=["auto", "bfloat16", "bf16", "float16", "fp16", "float32", "fp32"],
                       help="非 4bit 模式下的模型 dtype")
    
    # 数据参数
    parser.add_argument("--data_dir", type=str, default="./vq_lord_data",
                       help="数据目录")
    parser.add_argument("--train_num", type=int, default=500,
                       help="训练样本数")
    parser.add_argument("--dataset_name", type=str, default="vq_lord",
                       choices=["vq_lord", "scienceqa"],
                       help="训练数据来源 (vq_lord 或 scienceqa)")
    parser.add_argument("--scienceqa_split", type=str, default="train",
                       help="ScienceQA 数据集 split")
    parser.add_argument("--scienceqa_path", type=str, default="ScienceQA",
                       help="ScienceQA 数据集路径（本地路径或 HuggingFace 数据集名）")
    parser.add_argument("--scienceqa_seed", type=int, default=20240306,
                       help="ScienceQA 划分随机种子")
    parser.add_argument("--teacher_cache_path", type=str, default="",
                       help="ScienceQA 教师回答缓存路径 (json)")
    parser.add_argument("--collect_teacher_data", type=int, default=1,
                       help="是否自动采集缺失的教师模型回答")
    parser.add_argument("--strict_teacher_distill", type=int, default=1,
                       help="严格蒸馏模式：若缺失教师模型回答则报错")
    parser.add_argument("--teacher_lang", type=str, default="zh", choices=["zh", "en"],
                       help="教师回答统一语言：zh 或 en")
    parser.add_argument("--reuse_vq_codebook", type=int, default=1,
                       help="若存在已保存的VQ codebook则复用并跳过Stage1")
    parser.add_argument("--vq_codebook_path", type=str, default="",
                       help="VQ codebook 保存/加载路径 (为空则使用save_path/stage1_vq)")
    parser.add_argument("--reuse_stage2", type=int, default=1,
                       help="若存在已保存的Stage2模型则复用并跳过Stage2")
    parser.add_argument("--stage2_ckpt_path", type=str, default="",
                       help="Stage2 checkpoint 保存/加载路径 (为空则使用save_path/stage2_vision)")
    parser.add_argument("--scienceqa_preprocessed_path", type=str, default="",
                       help="ScienceQA 预处理结果 JSON 路径，供 Stage1/2 分桶采样使用")
    parser.add_argument("--bucket_batch_size", type=int, default=0,
                       help="分桶采样 batch size 覆盖值，0 表示使用预处理 JSON 中的 batch size")
    parser.add_argument("--stage3_bucket_batch_size", type=int, default=0,
                       help="Stage3 分桶采样 batch size，0 表示沿用 bucket_batch_size 或预处理 JSON")
    parser.add_argument("--disable_bucket_for_stage3", type=int, default=1,
                       help="是否在 Stage3 禁用分桶采样，1=禁用，0=启用")
    
    # 保存参数
    parser.add_argument("--save_path", type=str, default="./vq_lord_ckpts",
                       help="模型保存路径")
    parser.add_argument("--log_step", type=int, default=10,
                       help="日志间隔")
    parser.add_argument("--save_step", type=int, default=100,
                       help="保存间隔")
    parser.add_argument("--save_each_epoch", type=int, default=0,
                       help="是否在 Stage1/2/3 每个 epoch 结束时额外保存一次检查点")
    
    parser.add_argument("--device", type=str, default="cuda",
                       help="训练设备")
    
    return parser.parse_args()


def load_model_and_processor(args):
    """加载模型和处理器"""
    print(f"加载模型: {args.model_path}")
    
    # 量化配置
    if args.use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
        model = LlavaNextForConditionalGeneration.from_pretrained(
            args.model_path,
            device_map="auto",
            quantization_config=quantization_config,
            trust_remote_code=True,
        )
    else:
        model_dtype = _resolve_torch_dtype(args.model_dtype)
        print(f"[Load] use_4bit=0，使用非量化训练路径，dtype={model_dtype}, device={args.device}")
        model = LlavaNextForConditionalGeneration.from_pretrained(
            args.model_path,
            device_map=None,
            low_cpu_mem_usage=True,
            torch_dtype=model_dtype,
            trust_remote_code=True,
        )
        model.to(args.device)
    
    processor = LlavaNextProcessor.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )
    
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    # 确认 model.config.image_token_index 与 tokenizer 一致
    tok_img_id = processor.tokenizer.convert_tokens_to_ids("<image>")
    cfg_img_id = getattr(model.config, "image_token_index", None)
    if cfg_img_id != tok_img_id:
        print(f"[Info] 对齐 image_token_index: config={cfg_img_id} -> tokenizer={tok_img_id}")
        model.config.image_token_index = int(tok_img_id)

    # 训练场景下关闭 KV cache，避免与梯度检查点冲突
    if hasattr(model, "config"):
        model.config.use_cache = False

    # 显式设置 gradient checkpointing 的 use_reentrant，消除警告
    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            # 兼容不支持该参数的版本
            model.gradient_checkpointing_enable()
    
    return model, processor


def add_vq_to_model(model, args):
    """
    为模型添加 VQ 层
    
    关键：使用 VQVisionEncoder 直接包装 vision tower，
    将视觉编码与 VQ codebook 量化串成统一前向链路
    """
    print("添加 VQ 离散化层...")
    
    # 获取 vision tower
    original_vision_tower = model.vision_tower
    
    # 创建 VQ Vision Encoder
    vq_vision_encoder = VQVisionEncoder(
        vision_tower=original_vision_tower,
        num_embeddings=args.vq_codebook_size,
        commitment_cost=args.vq_commitment_cost,
        legacy=bool(args.vq_legacy_loss),
        dead_code_threshold=args.vq_dead_code_threshold,
        usage_decay=args.vq_usage_decay,
        dead_code_reset_interval=args.vq_dead_code_reset_interval,
        freeze_vision_tower=bool(args.freeze_vision_tower),
    )

    try:
        vision_device = next(original_vision_tower.parameters()).device
        vq_vision_encoder.to(vision_device)
    except StopIteration:
        pass
    
    # 保存到模型，并以包装器替换原始 vision tower
    model.vq_vision_encoder = vq_vision_encoder
    model.vision_tower = vq_vision_encoder
    model._vq_loss_container = vq_vision_encoder.vq_cache
    model._vq_hook_handle = None
    
    print(
        f"VQ 层已添加，codebook 大小: {args.vq_codebook_size}, "
        f"quantizer=VectorQuantizer2(legacy={bool(args.vq_legacy_loss)}), "
        f"dead_code_threshold={args.vq_dead_code_threshold}, "
        f"reset_interval={args.vq_dead_code_reset_interval}, chain=vision_tower->vq_codebook"
    )
    return model


def apply_lora(model, args):
    """应用 LoRA"""
    print("应用 LoRA...")
    
    if args.use_4bit:
        model = prepare_model_for_kbit_training(model)
    else:
        _enable_input_require_grads_if_needed(model)
    
    # LoRA 配置 - 包含视觉部分
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        target_modules=[
            # 语言模型层
            "q_proj", "v_proj", "k_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
            # 多模态投影层 (如果需要训练)
            # "multi_modal_projector.linear_1",
            # "multi_modal_projector.linear_2",
        ],
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model


def train_stage1_vq(model, dataloader, args, tb_writer):
    """
    阶段 1: VQ Codebook 预训练
    
    目标：参考 VQGAN 的 encode->quantize->decode 闭环，
    在固定 vision tower 的前提下训练出可重建原始视觉特征的 VQ stack。
    """
    print("\n" + "=" * 50)
    print("阶段 1: VQ Codebook 预训练")
    print("=" * 50)

    model.eval()
    vq_encoder = model.vq_vision_encoder

    for param in model.parameters():
        param.requires_grad = False
    for param in vq_encoder.stage1_parameters():
        if torch.is_floating_point(param):
            param.requires_grad = True

    vq_encoder.pre_quant.train()
    vq_encoder.vq.train()
    vq_encoder.post_quant.train()
    vq_encoder.vision_tower.eval()
    
    # 统计可训练参数
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"可训练参数量: {trainable_params:,}")

    stage1_lr = args.stage1_lr if args.stage1_lr > 0 else args.lr * 5
    print(
        f"[Stage1] lr={stage1_lr}, recon_w={args.stage1_recon_weight}, "
        f"cos_w={args.stage1_cosine_weight}, vq_w={args.stage1_vq_weight}, "
        f"grad_clip={args.stage1_grad_clip}"
    )

    # 对齐 taming VQGAN：Stage1 使用 Adam(无 weight decay)，避免 codebook 被 AdamW 拉向 0。
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=stage1_lr,
        betas=(0.5, 0.9),
    )
    
    log_dir = os.path.join(args.save_path, "logs")
    os.makedirs(log_dir, exist_ok=True)
    metrics_path = os.path.join(log_dir, "vq_metrics.csv")

    if not os.path.exists(metrics_path):
        with open(metrics_path, "w", encoding="utf-8") as f:
            f.write(
                "stage,step,total_loss,recon_loss,cosine_loss,vq_loss,loss_ema,"
                "codebook_used,perplexity,dead_code_resets,dead_code_count\n"
            )

    global_step = 0
    loss_ema = None
    for epoch in range(args.epochs):
        maybe_set_dataloader_epoch(dataloader, epoch)
        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_cosine = 0.0
        epoch_vq = 0.0
        for batch in tqdm(dataloader, desc=f"VQ预训练 Epoch {epoch+1}"):
            global_step += 1
            
            pixel_values = batch["pixel_values"].to(args.device)

            if pixel_values.dim() == 5:
                batch_size, num_patches, channels, height, width = pixel_values.shape
                pixel_values = pixel_values.view(batch_size * num_patches, channels, height, width)
            
            optimizer.zero_grad(set_to_none=True)
            stage1_out = vq_encoder.stage1_forward(pixel_values)

            target_features = stage1_out["target_features"]
            reconstructed_features = stage1_out["reconstructed_features"]
            vq_loss = stage1_out["vq_loss"]
            vq_indices = stage1_out.get("indices")

            recon_loss = F.mse_loss(reconstructed_features, target_features)
            cosine_loss = 1.0 - F.cosine_similarity(
                reconstructed_features.float(),
                target_features.float(),
                dim=-1,
            ).mean()
            total_loss = (
                args.stage1_recon_weight * recon_loss
                + args.stage1_cosine_weight * cosine_loss
                + args.stage1_vq_weight * vq_loss
            )

            if not torch.isfinite(total_loss):
                print(
                    f"[Stage1][Warn] step={global_step} 出现非有限损失，"
                    f"跳过该 batch: recon={recon_loss.item()}, cos={cosine_loss.item()}, "
                    f"vq={vq_loss.item() if isinstance(vq_loss, torch.Tensor) else vq_loss}"
                )
                optimizer.zero_grad(set_to_none=True)
                continue

            total_loss.backward()
            grad_norm = None
            if args.stage1_grad_clip > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_([
                    p for p in model.parameters() if p.requires_grad
                ], max_norm=args.stage1_grad_clip)
                if isinstance(grad_norm, torch.Tensor) and not torch.isfinite(grad_norm):
                    print(f"[Stage1][Warn] step={global_step} 梯度范数非有限，跳过 optimizer.step()")
                    optimizer.zero_grad(set_to_none=True)
                    continue
            optimizer.step()

            total_loss_val = total_loss.item()
            recon_loss_val = recon_loss.item()
            cosine_loss_val = cosine_loss.item()
            vq_loss_val = vq_loss.item() if vq_loss is not None else 0.0

            epoch_loss += total_loss_val
            epoch_recon += recon_loss_val
            epoch_cosine += cosine_loss_val
            epoch_vq += vq_loss_val
            loss_ema = total_loss_val if loss_ema is None else (0.95 * loss_ema + 0.05 * total_loss_val)

            if vq_indices is not None:
                codebook_used = torch.unique(vq_indices).numel()
            else:
                codebook_used = 0

            perplexity = vq_encoder.vq_cache.get("perplexity")
            if isinstance(perplexity, torch.Tensor):
                perplexity_val = float(perplexity.detach().item())
            else:
                perplexity_val = 0.0
            dead_code_resets = int(vq_encoder.vq_cache.get("dead_code_resets", 0))
            dead_code_count = int(vq_encoder.vq_cache.get("dead_code_count", 0))

            if global_step % args.log_step == 0:
                print(
                    f"Step {global_step}, Total: {total_loss_val:.4f}, "
                    f"Recon: {recon_loss_val:.4f}, Cos: {cosine_loss_val:.4f}, VQ: {vq_loss_val:.4f}, "
                    f"used={codebook_used}, ppl={perplexity_val:.2f}, reset={dead_code_resets}, dead={dead_code_count}"
                )
                tb_writer.add_scalar("stage1/total_loss", total_loss_val, global_step)
                tb_writer.add_scalar("stage1/recon_loss", recon_loss_val, global_step)
                tb_writer.add_scalar("stage1/cosine_loss", cosine_loss_val, global_step)
                tb_writer.add_scalar("stage1/vq_loss", vq_loss_val, global_step)
                tb_writer.add_scalar("stage1/loss_ema", loss_ema, global_step)
                tb_writer.add_scalar("stage1/codebook_used", codebook_used, global_step)
                tb_writer.add_scalar("stage1/perplexity", perplexity_val, global_step)
                tb_writer.add_scalar("stage1/dead_code_resets", dead_code_resets, global_step)
                tb_writer.add_scalar("stage1/dead_code_count", dead_code_count, global_step)
                with open(metrics_path, "a", encoding="utf-8") as f:
                    f.write(
                        f"stage1,{global_step},{total_loss_val:.6f},{recon_loss_val:.6f},"
                        f"{cosine_loss_val:.6f},{vq_loss_val:.6f},{loss_ema:.6f},{codebook_used},"
                        f"{perplexity_val:.6f},{dead_code_resets},{dead_code_count}\n"
                    )
        
        num_batches = len(dataloader)
        avg_loss = epoch_loss / num_batches
        print(
            f"Epoch {epoch+1} 平均损失: Total={avg_loss:.4f}, "
            f"Recon={epoch_recon / num_batches:.4f}, Cos={epoch_cosine / num_batches:.4f}, "
            f"VQ={epoch_vq / num_batches:.4f}"
        )
        if int(getattr(args, "save_each_epoch", 0)) == 1:
            epoch_dir = os.path.join(args.save_path, f"stage1_vq_epoch{epoch+1}")
            epoch_codebook_path = os.path.join(epoch_dir, "vq_codebook.pt")
            save_vq_codebook(model, epoch_codebook_path)
            print(f"Stage1 Epoch 检查点已保存至 {epoch_dir}")
    
    return model


def _compute_answer_slot_top1(
    logits: torch.Tensor,
    labels: torch.Tensor,
    eos_token_id: Optional[int] = None,
) -> float:
    """统计答案字母位点的 top-1 命中率（忽略末尾 eos）。"""
    if logits is None or labels is None:
        return 0.0
    if logits.dim() != 3 or labels.dim() != 2:
        return 0.0

    pred_ids = logits.argmax(dim=-1)
    total = 0
    correct = 0
    batch_size = min(pred_ids.shape[0], labels.shape[0])

    for i in range(batch_size):
        valid_positions = torch.nonzero(labels[i] != -100, as_tuple=False)
        if valid_positions.numel() == 0:
            continue

        candidate_positions = valid_positions
        if eos_token_id is not None:
            non_eos_mask = labels[i, valid_positions] != int(eos_token_id)
            non_eos_positions = valid_positions[non_eos_mask]
            if non_eos_positions.numel() > 0:
                candidate_positions = non_eos_positions

        target_pos = int(candidate_positions[-1].item())
        if target_pos <= 0 or target_pos >= pred_ids.shape[1]:
            continue

        target_token = int(labels[i, target_pos].item())
        pred_token = int(pred_ids[i, target_pos - 1].item())
        correct += int(pred_token == target_token)
        total += 1

    if total == 0:
        return 0.0
    return float(correct) / float(total)


def _get_stage2_vq_loss(model, device: str, dtype: torch.dtype) -> torch.Tensor:
    vq_loss = model._vq_loss_container.get("loss", None)
    if isinstance(vq_loss, torch.Tensor):
        return vq_loss
    return torch.zeros((), device=device, dtype=dtype)


def _get_stage2_vq_stats(model) -> tuple[float, int, int]:
    perplexity = model._vq_loss_container.get("perplexity", None)
    if isinstance(perplexity, torch.Tensor):
        perplexity_val = float(perplexity.detach().item())
    else:
        perplexity_val = 0.0
    dead_code_resets = int(model._vq_loss_container.get("dead_code_resets", 0))
    dead_code_count = int(model._vq_loss_container.get("dead_code_count", 0))
    return perplexity_val, dead_code_resets, dead_code_count


def train_stage2_vision(model, dataloader, args, tb_writer):
    """
    阶段 2: 视觉能力蒸馏

    目标：在固定 Stage1 codebook 的前提下，利用答案强监督 + 解释弱监督
    将视觉能力迁移到可被 Stage3 继续优化的生成策略。
    """
    print("\n" + "=" * 50)
    print("阶段 2: 视觉能力蒸馏")
    print("=" * 50)

    image_token_id = _get_image_token_id(model)
    model.train()

    main_params = []
    prepost_params = []
    vision_params = []

    for name, param in model.named_parameters():
        if not torch.is_floating_point(param):
            param.requires_grad = False
            continue

        name_l = name.lower()
        is_lora = "lora_" in name_l or "modules_to_save" in name_l
        is_projector = "projector" in name_l or "multi_modal_projector" in name_l
        is_prepost = "pre_quant" in name_l or "post_quant" in name_l
        is_vq_embedding = "vq.embedding.weight" in name_l
        is_vision = "vision" in name_l and not args.freeze_vision_tower

        if is_vq_embedding:
            param.requires_grad = False
            continue

        if is_lora or is_projector:
            param.requires_grad = True
            main_params.append(param)
        elif is_prepost:
            param.requires_grad = True
            prepost_params.append(param)
        elif is_vision:
            param.requires_grad = True
            vision_params.append(param)
        else:
            param.requires_grad = False

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"可训练参数量: {trainable_params:,}")

    param_groups = []
    if main_params:
        param_groups.append({"params": main_params, "lr": args.lr})
    if prepost_params:
        param_groups.append({"params": prepost_params, "lr": args.lr * float(args.stage2_prepost_lr_scale)})
    if vision_params:
        param_groups.append({"params": vision_params, "lr": args.lr * float(args.stage2_vision_lr_scale)})

    if not param_groups:
        raise RuntimeError("Stage2 没有可训练参数，请检查冻结策略")

    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    grad_accum = max(1, int(getattr(args, "stage2_grad_accum", 0) or getattr(args, "grad_accum", 1)))
    print(
        f"[Stage2] grad_accum={grad_accum}, "
        f"answer_w={args.stage2_answer_weight}, rationale_w={args.stage2_rationale_weight}, "
        f"prepost_lr_scale={args.stage2_prepost_lr_scale}, "
        f"vision_lr_scale={args.stage2_vision_lr_scale}, grad_clip={args.stage2_grad_clip}"
    )

    global_step = 0
    for epoch in range(args.epochs):
        maybe_set_dataloader_epoch(dataloader, epoch)
        epoch_total = 0.0
        epoch_answer = 0.0
        epoch_rationale = 0.0
        epoch_vq = 0.0
        epoch_vq_ratio = 0.0
        epoch_vq_perplexity = 0.0
        epoch_acc = 0.0
        epoch_count = 0

        optimizer.zero_grad(set_to_none=True)
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"视觉蒸馏 Epoch {epoch+1}")):
            global_step += 1

            pixel_values = batch["pixel_values"].to(args.device)
            image_sizes = sanitize_image_sizes(batch.get("image_sizes"), batch_size=pixel_values.shape[0])

            # 新版双监督输入（ScienceQA）
            use_dual_targets = all(
                key in batch for key in (
                    "answer_input_ids", "answer_attention_mask", "answer_labels",
                    "full_input_ids", "full_attention_mask", "full_labels",
                )
            )

            if use_dual_targets:
                answer_input_ids = batch["answer_input_ids"].to(args.device)
                answer_attention_mask = batch["answer_attention_mask"].to(args.device)
                answer_labels = batch["answer_labels"].to(args.device)

                full_input_ids = batch["full_input_ids"].to(args.device)
                full_attention_mask = batch["full_attention_mask"].to(args.device)
                full_labels = batch["full_labels"].to(args.device)
                has_rationale = batch.get("has_rationale")
                if has_rationale is None:
                    rationale_labels = full_labels
                    has_any_rationale = True
                else:
                    has_rationale = has_rationale.to(args.device).bool()
                    rationale_labels = full_labels.masked_fill(~has_rationale.unsqueeze(1), -100)
                    has_any_rationale = bool(has_rationale.any().item())
            else:
                # 兼容旧路径（vq_lord 数据）
                full_input_ids = batch["input_ids"].to(args.device)
                full_attention_mask = batch["attention_mask"].to(args.device)
                full_labels = batch["labels"].to(args.device)
                rationale_labels = full_labels
                has_any_rationale = True
                answer_input_ids = full_input_ids
                answer_attention_mask = full_attention_mask
                answer_labels = full_labels

            if global_step == 1:
                n_img = (full_input_ids == image_token_id).sum(dim=1)
                if (n_img == 0).any():
                    raise RuntimeError(
                        f"首个 batch 缺少 image token: counts={n_img.tolist()}, id={image_token_id}"
                    )

            # 前向1：答案监督
            outputs_answer = model(
                input_ids=answer_input_ids,
                attention_mask=answer_attention_mask,
                pixel_values=pixel_values,
                image_sizes=image_sizes,
                labels=answer_labels,
            )
            answer_loss = outputs_answer.loss
            answer_vq_loss = _get_stage2_vq_loss(model, args.device, answer_loss.dtype)

            # 前向2：解释/完整回答监督（无解释样本可退化为 0）
            if has_any_rationale:
                outputs_full = model(
                    input_ids=full_input_ids,
                    attention_mask=full_attention_mask,
                    pixel_values=pixel_values,
                    image_sizes=image_sizes,
                    labels=rationale_labels,
                )
                rationale_loss = outputs_full.loss
                # 同一张图的 VQ 路径应近似一致；有第二次图像前向时取最新值。
                vq_loss = _get_stage2_vq_loss(model, args.device, answer_loss.dtype)
            else:
                rationale_loss = torch.zeros((), device=args.device, dtype=answer_loss.dtype)
                vq_loss = answer_vq_loss

            total_loss = (
                float(args.stage2_answer_weight) * answer_loss
                + float(args.stage2_rationale_weight) * rationale_loss
                + args.beta * vq_loss
            )

            if not torch.isfinite(total_loss):
                print(
                    f"[Stage2][Warn] step={global_step} 非有限损失，跳过。"
                )
                optimizer.zero_grad(set_to_none=True)
                continue

            (total_loss / grad_accum).backward()

            if (batch_idx + 1) % grad_accum == 0 or batch_idx == len(dataloader) - 1:
                if args.stage2_grad_clip > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        max_norm=float(args.stage2_grad_clip),
                    )
                    if isinstance(grad_norm, torch.Tensor) and not torch.isfinite(grad_norm):
                        print(f"[Stage2][Warn] step={global_step} 梯度范数非有限，跳过 optimizer.step()")
                        optimizer.zero_grad(set_to_none=True)
                        continue
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            answer_loss_val = float(answer_loss.item())
            rationale_loss_val = float(rationale_loss.item())
            vq_loss_val = float(vq_loss.item())
            total_loss_val = float(total_loss.item())
            weighted_vq_ratio = float(args.beta) * vq_loss_val / max(answer_loss_val, 1e-8)
            perplexity_val, dead_code_resets, dead_code_count = _get_stage2_vq_stats(model)
            answer_acc_proxy = _compute_answer_slot_top1(
                outputs_answer.logits,
                answer_labels,
                eos_token_id=getattr(model.config, "eos_token_id", None),
            )

            epoch_total += total_loss_val
            epoch_answer += answer_loss_val
            epoch_rationale += rationale_loss_val
            epoch_vq += vq_loss_val
            epoch_vq_ratio += weighted_vq_ratio
            epoch_vq_perplexity += perplexity_val
            epoch_acc += answer_acc_proxy
            epoch_count += 1

            if global_step % args.log_step == 0:
                print(
                    f"Step {global_step}, Total: {total_loss_val:.4f}, "
                    f"Answer: {answer_loss_val:.4f}, Rationale: {rationale_loss_val:.4f}, "
                    f"VQ: {vq_loss_val:.4f}, VQ/Answer: {weighted_vq_ratio:.4f}, "
                    f"PPL: {perplexity_val:.2f}, Dead: {dead_code_count}, "
                    f"AnswerAccProxy: {answer_acc_proxy:.4f}"
                )
                tb_writer.add_scalar("stage2/total_loss", total_loss_val, global_step)
                tb_writer.add_scalar("stage2/answer_loss", answer_loss_val, global_step)
                tb_writer.add_scalar("stage2/rationale_loss", rationale_loss_val, global_step)
                tb_writer.add_scalar("stage2/vq_loss", vq_loss_val, global_step)
                tb_writer.add_scalar("stage2/vq_answer_ratio", weighted_vq_ratio, global_step)
                tb_writer.add_scalar("stage2/vq_perplexity", perplexity_val, global_step)
                tb_writer.add_scalar("stage2/dead_code_resets", dead_code_resets, global_step)
                tb_writer.add_scalar("stage2/dead_code_count", dead_code_count, global_step)
                tb_writer.add_scalar("stage2/answer_only_accuracy_proxy", answer_acc_proxy, global_step)

        denom = max(1, epoch_count)
        print(
            f"Epoch {epoch+1} 平均损失: Total={epoch_total/denom:.4f}, "
            f"Answer={epoch_answer/denom:.4f}, Rationale={epoch_rationale/denom:.4f}, "
            f"VQ={epoch_vq/denom:.4f}, VQ/Answer={epoch_vq_ratio/denom:.4f}, "
            f"PPL={epoch_vq_perplexity/denom:.2f}, AnswerAccProxy={epoch_acc/denom:.4f}"
        )
        tb_writer.add_scalar("stage2_epoch/total_loss", epoch_total / denom, epoch + 1)
        tb_writer.add_scalar("stage2_epoch/answer_loss", epoch_answer / denom, epoch + 1)
        tb_writer.add_scalar("stage2_epoch/rationale_loss", epoch_rationale / denom, epoch + 1)
        tb_writer.add_scalar("stage2_epoch/vq_loss", epoch_vq / denom, epoch + 1)
        tb_writer.add_scalar("stage2_epoch/vq_answer_ratio", epoch_vq_ratio / denom, epoch + 1)
        tb_writer.add_scalar("stage2_epoch/vq_perplexity", epoch_vq_perplexity / denom, epoch + 1)
        tb_writer.add_scalar("stage2_epoch/answer_only_accuracy_proxy", epoch_acc / denom, epoch + 1)
        if int(getattr(args, "save_each_epoch", 0)) == 1:
            epoch_ckpt_path = os.path.join(args.save_path, f"stage2_vision_epoch{epoch+1}")
            save_stage2_checkpoint(model, args, epoch_ckpt_path)

    return model


def log_clip(tnsr, epsilon=1.0):
    """对数裁剪函数，防止数值不稳定。
    将输入裁剪到 [log(1-epsilon), log(1+epsilon)] 范围内。
    epsilon=1.0 对应 [-inf, log(2)] ≈ [-inf, 0.693]，
    实际 max 侧裁剪为 log(1+epsilon)=0.693，min 侧裁剪为 -10（防 -inf）。
    """
    upper = torch.log(torch.tensor(1.0 + epsilon, device=tnsr.device, dtype=tnsr.dtype))
    lower = torch.tensor(-10.0, device=tnsr.device, dtype=tnsr.dtype)  # 下界不用 log(1-1)=-inf
    return torch.clamp(tnsr, min=lower.item(), max=upper.item())


def _compute_generation_log_prob(model, ids, mask, pixel_values, image_sizes, prompt_lens):
    """计算每个样本仅生成段的平均 log probability。"""
    out = model(
        input_ids=ids,
        attention_mask=mask,
        pixel_values=pixel_values,
        image_sizes=image_sizes,
    )
    logits = out.logits[:, :-1, :]
    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(dim=-1, index=ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    token_mask = mask[:, 1:].float()

    for batch_idx in range(ids.shape[0]):
        gen_start = max(int(prompt_lens[batch_idx].item()) - 1, 0)
        if gen_start > 0:
            token_mask[batch_idx, :gen_start] = 0.0

    return (token_log_probs * token_mask).sum(dim=-1) / token_mask.sum(dim=-1).clamp(min=1)


def train_stage3_lord(model, dataloader, args, tb_writer):
    """
    阶段 3: LoRD 联合训练 (真正的 LoRD 方法)
    
    实现 LoRD 核心算法：
    1. 双样本生成：学生模型生成两个独立序列 S1, S2
    2. 局部性排序：根据 log prob 确定正负样本 y+, y-
    3. 冷启动策略：置信度低时使用教师标签 y_vic
    4. LoRD 损失：对比损失 + 正则化损失
    
    损失公式:
    L_obj = -log(P(y+|x) / P(y-|x)) = -(log P(y+) - log P(y-))
    L_reg = -clip(log(P(y_vic|x) / P(y-|x)))
    L_total = L_obj + L_reg + beta * L_vq
    """
    print("\n" + "=" * 50)
    print("阶段 3: LoRD 联合训练 (真正的 LoRD 方法)")
    print("=" * 50)
    
    # LoRD 超参数
    tau1 = getattr(args, 'tau1', 0.01)  # 冷启动置信度阈值
    max_new_tokens = getattr(args, 'max_new_tokens', 64)  # 生成长度
    
    grad_accum = max(1, int(getattr(args, 'stage3_grad_accum', 0) or getattr(args, 'grad_accum', 1)))
    stage3_lr = float(args.lr) * float(getattr(args, "stage3_lr_scale", 1.0))
    
    print(
        f"LoRD 参数: tau1={tau1}, max_new_tokens={max_new_tokens}, grad_accum={grad_accum}, "
        f"lr={stage3_lr}, train_projector={int(bool(args.stage3_train_projector))}"
    )

    image_token_id = _get_image_token_id(model)

    # Stage3 重新显式设置可训练参数，避免沿用 Stage2 冻结状态
    model.train()
    for name, param in model.named_parameters():
        if not torch.is_floating_point(param):
            param.requires_grad = False
            continue

        name_l = name.lower()
        is_lora = "lora_" in name_l or "modules_to_save" in name_l
        # VQ codebook 在 Stage3 冻结 — Stage1+2 已训练好，Stage3 专注 LoRD 对比学习
        is_vq = False
        is_projector = ("projector" in name_l or "multi_modal_projector" in name_l) and bool(args.stage3_train_projector)
        is_vision = ("vision" in name_l) and (not args.freeze_vision_tower)

        param.requires_grad = bool(is_lora or is_vq or is_projector or is_vision)

    print("[Stage3] VQ codebook 通过 requires_grad=False 冻结；无 EMA/dead-code 状态需要切换")

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Stage3 可训练参数量: {trainable_params:,}")
    
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=stage3_lr,
    )
    
    # ========== 断点续训：加载已有的 Stage3 checkpoint ==========
    global_step = 0
    start_epoch = 0
    resume_batch_idx = 0
    stage3_resume_dir = os.path.join(args.save_path, "stage3_resume")
    stage3_state_path = os.path.join(stage3_resume_dir, "training_state.pt")

    def _save_stage3_resume_state(next_epoch: int, next_batch_idx: int):
        os.makedirs(stage3_resume_dir, exist_ok=True)
        resume_model_dir = os.path.join(stage3_resume_dir, "model")
        save_stage3_checkpoint(model, resume_model_dir)
        torch.save({
            "epoch": next_epoch,
            "next_batch_idx": next_batch_idx,
            "global_step": global_step,
            "optimizer_state_dict": optimizer.state_dict(),
            "rng_state": _capture_rng_state(),
        }, stage3_state_path)
        print(
            f"Stage3 续训状态已保存: epoch={next_epoch}, next_batch_idx={next_batch_idx}, "
            f"global_step={global_step}"
        )
    
    if os.path.exists(stage3_state_path):
        print(f"检测到 Stage3 断点，正在恢复: {stage3_state_path}")
        state = torch.load(stage3_state_path, map_location="cpu")
        start_epoch = state["epoch"]
        resume_batch_idx = int(state.get("next_batch_idx", 0))
        global_step = state["global_step"]
        
        # 加载模型权重（需要在 optimizer 之前加载）
        resume_model_dir = os.path.join(stage3_resume_dir, "model")
        if os.path.exists(resume_model_dir):
            model = load_stage3_checkpoint(model, resume_model_dir, args.device)
        
        # 加载 optimizer state（加载到与参数一致的设备上）
        optimizer.load_state_dict(state["optimizer_state_dict"])
        _restore_rng_state(state.get("rng_state"))
        
        # 如果已经训练完成，跳过
        if start_epoch >= args.epochs:
            print(f"训练已完成 (epoch={start_epoch}/{args.epochs})，无需继续")
            return model
        
        print(
            f"已恢复: epoch={start_epoch}, next_batch_idx={resume_batch_idx}, "
            f"global_step={global_step}"
        )
    
    # pad_token_id 统一获取，用于 mask 构建
    _pad_id = model.config.pad_token_id
    if _pad_id is None:
        _pad_id = model.config.eos_token_id or 0

    for epoch in range(start_epoch, args.epochs):
        maybe_set_dataloader_epoch(dataloader, epoch)
        epoch_loss = 0.0
        epoch_obj_loss = 0.0
        epoch_reg_loss = 0.0
        epoch_vq_loss = 0.0
        cold_start_count = 0
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"LoRD Epoch {epoch+1}")):
            if epoch == start_epoch and batch_idx < resume_batch_idx:
                continue
            if batch is None:
                continue
            global_step += 1

            # 获取输入数据
            input_ids = batch["input_ids"].to(args.device)  # prompt + y_vic
            attention_mask = batch["attention_mask"].to(args.device)
            pixel_values = batch["pixel_values"].to(args.device)
            image_sizes = batch.get("image_sizes")
            image_sizes = sanitize_image_sizes(image_sizes, batch_size=input_ids.size(0))
            labels = batch["labels"].to(args.device)  # y_vic

            bs = input_ids.shape[0]

            # ========== Step 1: 提取 prompt，双样本生成 ==========
            # Stage3 在 bs>1 时，样本之间 prompt 长度可能不同。
            # 若直接取一个全局 prompt_len，会截断某些样本的 image tokens，
            # 进而触发 Llava-Next 的 "image features and image tokens do not match"。
            prompt_lens = []
            prompt_ids_list = []
            prompt_mask_list = []

            for i in range(bs):
                cur_prompt_len = int((labels[i] == -100).sum().item())
                cur_prompt_len = max(1, min(cur_prompt_len, input_ids.shape[1]))
                cur_prompt_ids = input_ids[i:i+1, :cur_prompt_len]
                cur_prompt_mask = attention_mask[i:i+1, :cur_prompt_len]

                if image_token_id is not None and not (cur_prompt_ids == image_token_id).any().item():
                    cur_prompt_ids = input_ids[i:i+1]
                    cur_prompt_mask = attention_mask[i:i+1]
                    cur_prompt_len = input_ids.shape[1]

                prompt_lens.append(cur_prompt_len)
                prompt_ids_list.append(cur_prompt_ids)
                prompt_mask_list.append(cur_prompt_mask)

            prompt_lens = torch.tensor(prompt_lens, device=input_ids.device, dtype=torch.long)

            # 防止生成 <image> token
            bad_words_ids = [[int(image_token_id)]] if image_token_id is not None else None

            model.eval()
            with torch.no_grad():
                gc_enabled = getattr(model, "is_gradient_checkpointing", False)
                use_cache_states = _set_model_use_cache(model, True)
                gc_states = _set_model_gradient_checkpointing(model, False) if gc_enabled else []
                try:
                    gen_output_1_list = []
                    gen_output_2_list = []
                    eos_or_pad = model.config.eos_token_id or _pad_id

                    def _strip_extra_img_single(seq_ids, allowed_count):
                        if image_token_id is None:
                            return seq_ids
                        seq_ids = seq_ids.clone()
                        pos = (seq_ids[0] == image_token_id).nonzero(as_tuple=False).view(-1)
                        if pos.numel() > allowed_count:
                            seq_ids[0, pos[allowed_count:]] = eos_or_pad
                        return seq_ids

                    for i in range(bs):
                        cur_prompt_ids = prompt_ids_list[i]
                        cur_prompt_mask = prompt_mask_list[i]
                        cur_pixel_values = pixel_values[i:i+1]
                        cur_image_sizes = image_sizes[i:i+1] if image_sizes is not None else None

                        _gen_kwargs = dict(
                            input_ids=cur_prompt_ids,
                            attention_mask=cur_prompt_mask,
                            pixel_values=cur_pixel_values,
                            image_sizes=cur_image_sizes,
                            do_sample=True,
                            max_new_tokens=max_new_tokens,
                            temperature=args.temperature,
                            bad_words_ids=bad_words_ids,
                            pad_token_id=_pad_id,
                        )
                        cur_gen_output_1 = model.generate(**_gen_kwargs)
                        cur_gen_output_2 = model.generate(**_gen_kwargs)

                        allowed_img_count = int((cur_prompt_ids == image_token_id).sum().item()) if image_token_id is not None else 0
                        cur_gen_output_1 = _strip_extra_img_single(cur_gen_output_1, allowed_img_count)
                        cur_gen_output_2 = _strip_extra_img_single(cur_gen_output_2, allowed_img_count)
                        gen_output_1_list.append(cur_gen_output_1)
                        gen_output_2_list.append(cur_gen_output_2)

                    gen_output_1 = torch.nn.utils.rnn.pad_sequence(
                        [tensor.squeeze(0) for tensor in gen_output_1_list],
                        batch_first=True,
                        padding_value=_pad_id,
                    )
                    gen_output_2 = torch.nn.utils.rnn.pad_sequence(
                        [tensor.squeeze(0) for tensor in gen_output_2_list],
                        batch_first=True,
                        padding_value=_pad_id,
                    )
                finally:
                    if gc_enabled:
                        _restore_model_gradient_checkpointing(gc_states)
                    _restore_model_use_cache(use_cache_states)

            model.train()

            # ========== 对齐序列长度 ==========
            max_len = max(gen_output_1.shape[1], gen_output_2.shape[1], input_ids.shape[1])

            def pad_to_len(tensor, length, pad_val=_pad_id):
                if tensor.shape[1] < length:
                    p = torch.full((tensor.shape[0], length - tensor.shape[1]),
                                   pad_val, dtype=tensor.dtype, device=tensor.device)
                    return torch.cat([tensor, p], dim=1)
                return tensor[:, :length]

            s1_ids = pad_to_len(gen_output_1, max_len)
            s2_ids = pad_to_len(gen_output_2, max_len)
            y_vic_ids = pad_to_len(input_ids, max_len)

            # 修复4: mask 使用 pad_token_id 而非硬编码 0
            s1_mask = (s1_ids != _pad_id).long()
            s2_mask = (s2_ids != _pad_id).long()
            y_vic_mask = pad_to_len(attention_mask, max_len, pad_val=0)

            # ========== Step 2+3: 无梯度排序 + 冷启动（复用同一批 forward） ==========
            # 合并排序和损失计算，避免额外冗余 forward
            # 只用 no_grad forward 做排序/冷启动判断，然后在 Step 4 中带梯度计算损失。
            with torch.no_grad():
                lp_s1 = _compute_generation_log_prob(
                    model, s1_ids, s1_mask, pixel_values, image_sizes, prompt_lens
                )
                lp_s2 = _compute_generation_log_prob(
                    model, s2_ids, s2_mask, pixel_values, image_sizes, prompt_lens
                )

            # 局部性排序 + 冷启动
            y_plus_ids = s1_ids.clone()
            y_minus_ids = s2_ids.clone()
            y_plus_mask = s1_mask.clone()
            y_minus_mask = s2_mask.clone()
            # 记录 y_minus 对应的 no_grad lp，供 step 4 L_reg 直接复用（省 1 次 forward）
            lp_minus_detached = lp_s2.clone()  # 默认 s1=y+, s2=y-

            for i in range(bs):
                lp1 = lp_s1[i].item()
                lp2 = lp_s2[i].item()

                # 排序：log prob 更高的为 y+
                if lp2 > lp1:
                    y_plus_ids[i] = s2_ids[i]
                    y_minus_ids[i] = s1_ids[i]
                    y_plus_mask[i] = s2_mask[i]
                    y_minus_mask[i] = s1_mask[i]
                    lp_minus_detached[i] = lp_s1[i]  # y- 换成 s1
                # else: 默认 s1=y+, s2=y-, lp_minus_detached[i] 已是 lp_s2[i]

                # 冷启动策略：仅当 y⁺ 置信度极低（生成质量太差、无法作为有效正样本）时
                # 用教师回答 y_vic 替代。tau1=0.01 → 仅 avg per-token prob < 1% 时触发。
                # 去掉 delta 条件，避免 epochs>1 时因 delta 过小导致全部退化为 SFT。
                p_best = torch.exp(torch.tensor(max(lp1, lp2))).item()
                if p_best < tau1:
                    y_plus_ids[i] = y_vic_ids[i]
                    y_plus_mask[i] = y_vic_mask[i]
                    cold_start_count += 1

            del lp_s1, lp_s2, s1_ids, s2_ids, s1_mask, s2_mask
            del gen_output_1, gen_output_2

            # ========== Step 4: LoRD 损失（带梯度，分拆 backward 省显存） ==========
            # (a) y- 的 detached log prob：直接复用排序阶段已算过的 lp_minus_detached
            seq_lp_minus_d = lp_minus_detached.detach()

            # (b) y+ forward + backward
            seq_lp_plus = _compute_generation_log_prob(
                model, y_plus_ids, y_plus_mask, pixel_values, image_sizes, prompt_lens
            )
            L_obj_plus = -torch.mean(seq_lp_plus)
            (L_obj_plus / grad_accum).backward()
            L_obj_plus_val = L_obj_plus.item()
            del seq_lp_plus, L_obj_plus

            # (c) y_vic forward + backward (L_reg)
            seq_lp_vic = _compute_generation_log_prob(
                model, y_vic_ids, y_vic_mask, pixel_values, image_sizes, prompt_lens
            )
            L_reg = -torch.mean(log_clip(seq_lp_vic - seq_lp_minus_d))
            (L_reg / grad_accum).backward()
            L_reg_val = L_reg.item()
            del seq_lp_vic, L_reg

            # (d) y- forward + backward (L_obj_minus + VQ loss)
            seq_lp_minus = _compute_generation_log_prob(
                model, y_minus_ids, y_minus_mask, pixel_values, image_sizes, prompt_lens
            )
            L_obj_minus = torch.mean(seq_lp_minus)

            vq_loss = model._vq_loss_container.get("loss", None)
            if vq_loss is None or not isinstance(vq_loss, torch.Tensor):
                vq_loss = torch.tensor(0.0, device=args.device)

            ((L_obj_minus + args.beta * vq_loss) / grad_accum).backward()
            L_obj_minus_val = L_obj_minus.item()
            vq_loss_val = vq_loss.item() if isinstance(vq_loss, torch.Tensor) else 0.0
            del seq_lp_minus, L_obj_minus, vq_loss, seq_lp_minus_d

            # (e) optimizer step
            did_optimizer_step = False
            if (batch_idx + 1) % grad_accum == 0 or batch_idx == len(dataloader) - 1:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                did_optimizer_step = True

            # 日志
            L_obj_val = L_obj_plus_val + L_obj_minus_val
            total_loss_val = L_obj_val + L_reg_val + args.beta * vq_loss_val

            epoch_loss += total_loss_val
            epoch_obj_loss += L_obj_val
            epoch_reg_loss += L_reg_val
            epoch_vq_loss += vq_loss_val

            if global_step % args.log_step == 0:
                print(f"Step {global_step}, Total: {total_loss_val:.4f}, "
                      f"L_obj: {L_obj_val:.4f}, L_reg: {L_reg_val:.4f}, "
                      f"VQ: {vq_loss_val:.4f}, cold_start: {cold_start_count}")
                tb_writer.add_scalar("stage3/total_loss", total_loss_val, global_step)
                tb_writer.add_scalar("stage3/L_obj", L_obj_val, global_step)
                tb_writer.add_scalar("stage3/L_reg", L_reg_val, global_step)
                tb_writer.add_scalar("stage3/vq_loss", vq_loss_val, global_step)

            # 修复5: 只在 step 末尾清理一次显存
            del input_ids, attention_mask, pixel_values, image_sizes, labels
            del y_plus_ids, y_minus_ids, y_plus_mask, y_minus_mask
            del y_vic_ids, y_vic_mask, prompt_ids_list, prompt_mask_list, prompt_lens, batch
            torch.cuda.empty_cache()

            if did_optimizer_step and args.save_step > 0 and global_step % args.save_step == 0:
                _save_stage3_resume_state(epoch, batch_idx + 1)
        
        # Epoch 统计
        num_batches = len(dataloader)
        print(f"Epoch {epoch+1} 平均损失: Total={epoch_loss/num_batches:.4f}, "
              f"L_obj={epoch_obj_loss/num_batches:.4f}, L_reg={epoch_reg_loss/num_batches:.4f}, "
              f"VQ={epoch_vq_loss/num_batches:.4f}")
        
        # ========== 每个 Epoch 结束保存断点 ==========
        print(f"保存 Stage3 断点 (epoch={epoch+1}, step={global_step})...")
        _save_stage3_resume_state(epoch + 1, 0)
        
        # 同时保存当前 epoch 的独立检查点
        save_stage3_checkpoint(model, os.path.join(args.save_path, f"stage3_epoch{epoch+1}"))

        resume_batch_idx = 0
    
    return model


def save_checkpoint(model, args, suffix=""):
    """保存检查点"""
    save_path = os.path.join(args.save_path, suffix)
    os.makedirs(save_path, exist_ok=True)
    
    model.save_pretrained(save_path, safe_serialization=False)
    print(f"模型已保存至 {save_path}")


def _get_vq_state_path(codebook_path: str) -> str:
    return os.path.join(os.path.dirname(codebook_path), "vq_encoder_state.pt")


def save_vq_codebook(model, codebook_path: str):
    os.makedirs(os.path.dirname(codebook_path), exist_ok=True)
    torch.save(model.vq_vision_encoder.vq.embedding.weight.detach().cpu(), codebook_path)
    torch.save(model.vq_vision_encoder.get_vq_state(), _get_vq_state_path(codebook_path))


def save_stage1_checkpoint(model, args):
    """Stage1 只保存后续蒸馏真正需要的 VQ stack，避免整模型落盘占用大量磁盘。"""
    save_dir = os.path.join(args.save_path, "stage1_vq")
    os.makedirs(save_dir, exist_ok=True)
    save_vq_codebook(model, args.vq_codebook_path)
    print(f"Stage1 VQ stack 已保存至 {save_dir}")


def load_vq_codebook(model, codebook_path: str):
    codebook = torch.load(codebook_path, map_location="cpu")
    model.vq_vision_encoder.vq.embedding.weight.data.copy_(codebook)
    vq_state_path = _get_vq_state_path(codebook_path)
    if os.path.exists(vq_state_path):
        vq_state = torch.load(vq_state_path, map_location="cpu")
        model.vq_vision_encoder.load_vq_state(vq_state)
    print(f"已加载 VQ codebook: {codebook_path}")


def save_stage2_checkpoint(model, args, ckpt_path: str):
    """保存 Stage2 检查点（包括 VQ codebook 和 LoRA 权重）"""
    os.makedirs(ckpt_path, exist_ok=True)
    
    save_vq_codebook(model, os.path.join(ckpt_path, "vq_codebook.pt"))
    
    # 保存 LoRA 适配器
    model.save_pretrained(ckpt_path, safe_serialization=False)
    
    # 保存配置信息
    import json
    config_info = {
        "vq_codebook_size": args.vq_codebook_size,
        "freeze_vision_tower": args.freeze_vision_tower,
        "beta": args.beta,
        "stage": 2,
        "stage2_answer_weight": float(args.stage2_answer_weight),
        "stage2_rationale_weight": float(args.stage2_rationale_weight),
        "stage2_train_lora": 1,
        "stage2_train_codebook": 0,
        "stage2_answer_format": "letter",
    }
    with open(os.path.join(ckpt_path, "stage2_config.json"), "w") as f:
        json.dump(config_info, f, indent=2)
    
    print(f"Stage2 检查点已保存至 {ckpt_path}")


def load_stage2_checkpoint(model, ckpt_path: str):
    """加载 Stage2 检查点"""
    vq_path = os.path.join(ckpt_path, "vq_codebook.pt")
    if os.path.exists(vq_path):
        load_vq_codebook(model, vq_path)
    
    # 加载 LoRA 适配器
    adapter_config = os.path.join(ckpt_path, "adapter_config.json")
    if os.path.exists(adapter_config):
        adapter_weights = os.path.join(ckpt_path, "adapter_model.safetensors")
        if os.path.exists(adapter_weights):
            from safetensors.torch import load_file
            state_dict = load_file(adapter_weights)
        else:
            adapter_weights_bin = os.path.join(ckpt_path, "adapter_model.bin")
            if os.path.exists(adapter_weights_bin):
                state_dict = torch.load(adapter_weights_bin, map_location="cpu")
            else:
                state_dict = None
        if state_dict is not None:
            # 直接覆盖现有 default adapter，避免额外 adapter name 干扰 Stage3 参数遍历。
            model.load_state_dict(state_dict, strict=False)
        print(f"已加载 LoRA 权重: {ckpt_path}")
    
    print(f"Stage2 检查点加载完成")
    return model


def main():
    args = setup_args()
    
    print("=" * 60)
    print("VQ-LoRD 训练")
    print("=" * 60)
    pprint(vars(args))
    print("=" * 60)

    if args.reuse_vq_codebook:
        print("[Info] reuse_vq_codebook=1，若存在 stage1 codebook 将跳过 Stage1。")
    if args.reuse_stage2:
        print("[Info] reuse_stage2=1，若存在 stage2 ckpt 将跳过 Stage2。")
    
    # 创建保存目录
    os.makedirs(args.save_path, exist_ok=True)
    if not args.vq_codebook_path:
        args.vq_codebook_path = os.path.join(args.save_path, "stage1_vq", "vq_codebook.pt")
    tb_writer = SummaryWriter(log_dir=os.path.join(args.save_path, "logs"))
    
    # 加载模型
    model, processor = load_model_and_processor(args)
    
    # 添加 VQ 层
    model = add_vq_to_model(model, args)
    
    # 应用 LoRA
    if args.use_lora:
        model = apply_lora(model, args)
    
    # 加载数据
    print("\n加载训练数据...")
    train_bucket_meta = None
    if args.dataset_name == "scienceqa":
        train_samples = build_scienceqa_samples(
            scienceqa_path=args.scienceqa_path,
            split=args.scienceqa_split,
            train_num=args.train_num,
            seed=args.scienceqa_seed,
        )

        # 关键：为每条样本补齐教师模型回答，供 Stage2/Stage3 蒸馏使用
        train_samples = attach_gpt4v_teacher_responses(train_samples, args)

        train_dataset = ScienceQADataset(
            processor=processor,
            scienceqa_path=args.scienceqa_path,
            max_length=args.max_length,
            samples=train_samples,
            seed=args.scienceqa_seed,
            teacher_lang=args.teacher_lang,
        )
        train_bucket_meta = load_scienceqa_preprocessed_buckets(
            args.scienceqa_preprocessed_path,
            expected_len=len(train_dataset),
            args=args,
        )
    else:
        collector = GPT4VDataCollector(save_dir=args.data_dir)

        # 尝试加载已收集的数据
        visual_qa_data = collector.load_collected_data("visual_qa_data.json")
        image_descriptions = collector.load_collected_data("image_descriptions.json")

        if not visual_qa_data and not image_descriptions:
            print("警告：未找到预收集的数据，请先运行数据收集脚本")
            print("示例：python data_collector.py --collect_data")
            return

        # 创建数据集
        from data_collector import VisualQAItem, ImageDescriptionItem
        visual_qa_items = [VisualQAItem(**item) for item in visual_qa_data]
        desc_items = [ImageDescriptionItem(**item) for item in image_descriptions]

        train_dataset = VQLORDDataset(
            visual_qa_data=visual_qa_items,
            image_descriptions=desc_items,
            processor=processor,
            max_length=args.max_length,
        )
    
    def vq_lord_collate(batch):
        batch = [b for b in batch if b is not None]
        if len(batch) == 0:
            return None

        # --- 处理 image_sizes ---
        image_sizes_list = [b.get("image_sizes") for b in batch]
        for idx, size in enumerate(image_sizes_list):
            if size is None:
                pixel_values = batch[idx].get("pixel_values")
                if isinstance(pixel_values, torch.Tensor) and pixel_values.dim() >= 3:
                    height = int(pixel_values.shape[-2])
                    width = int(pixel_values.shape[-1])
                    image_sizes_list[idx] = torch.tensor([height, width], dtype=torch.long)

        # --- 对不等长 1D 张量做 right-padding 对齐 ---
        pad_token_id = processor.tokenizer.pad_token_id or 0
        pad_keys = {
            "input_ids": pad_token_id,
            "attention_mask": 0,
            "labels": -100,
            "prompt_input_ids": pad_token_id,
            "prompt_attention_mask": 0,
            "full_input_ids": pad_token_id,
            "full_attention_mask": 0,
            "full_labels": -100,
            "answer_input_ids": pad_token_id,
            "answer_attention_mask": 0,
            "answer_labels": -100,
        }

        max_lens = {}
        for key in pad_keys:
            lengths = [b[key].shape[-1] for b in batch if key in b and isinstance(b[key], torch.Tensor)]
            if lengths:
                max_lens[key] = max(lengths)

        for b in batch:
            for key, pad_val in pad_keys.items():
                if key not in b or not isinstance(b[key], torch.Tensor):
                    continue
                t = b[key]
                target_len = max_lens.get(key, t.shape[-1])
                if t.shape[-1] < target_len:
                    pad_size = target_len - t.shape[-1]
                    b[key] = F.pad(t, (0, pad_size), value=pad_val)

        # pixel_values 可能形状不同（不同 patch 数），也需要对齐
        pv_list = [b.get("pixel_values") for b in batch]
        if all(isinstance(pv, torch.Tensor) for pv in pv_list):
            pv_shapes = [pv.shape for pv in pv_list]
            if len(set(pv_shapes)) > 1:
                max_patches = max(s[0] for s in pv_shapes)
                for i, pv in enumerate(pv_list):
                    if pv.shape[0] < max_patches:
                        pad_tensor = torch.zeros(
                            (max_patches - pv.shape[0],) + pv.shape[1:],
                            dtype=pv.dtype, device=pv.device,
                        )
                        batch[i]["pixel_values"] = torch.cat([pv, pad_tensor], dim=0)

        base = []
        for b in batch:
            item = dict(b)
            item.pop("image_sizes", None)
            base.append(item)

        collated = default_collate(base)

        if all(s is None for s in image_sizes_list):
            collated["image_sizes"] = None
        else:
            tensor_list = [s for s in image_sizes_list if s is not None]
            collated["image_sizes"] = torch.stack(tensor_list, dim=0)

        return collated

    def build_train_dataloader(stage_id: int):
        use_bucket = (
            args.dataset_name == "scienceqa"
            and train_bucket_meta is not None
            and not (stage_id == 3 and bool(args.disable_bucket_for_stage3))
        )

        if use_bucket:
            config = train_bucket_meta["config"]
            bucket_batch_size = int(config.get("bucket_batch_size", args.batch_size))
            if args.bucket_batch_size > 0:
                bucket_batch_size = int(args.bucket_batch_size)
            if stage_id == 3 and args.stage3_bucket_batch_size > 0:
                bucket_batch_size = int(args.stage3_bucket_batch_size)

            batch_sampler = ScienceQABucketBatchSampler(
                sample_to_bucket=train_bucket_meta["sample_to_bucket"],
                batch_size=bucket_batch_size,
                drop_last=bool(config.get("bucket_drop_last", False)),
                shuffle=bool(config.get("shuffle", True)),
                seed=int(config.get("seed", args.scienceqa_seed)),
            )
            print(
                f"Stage{stage_id} 使用预处理分桶采样: "
                f"path={train_bucket_meta['path']}, batch_size={batch_sampler.batch_size}, "
                f"num_batches={len(batch_sampler)}"
            )
            return DataLoader(
                train_dataset,
                batch_sampler=batch_sampler,
                num_workers=0,
                collate_fn=vq_lord_collate,
            )

        print(f"Stage{stage_id} 使用常规 DataLoader: batch_size={args.batch_size}, shuffle=True")
        return DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=vq_lord_collate,
        )
    
    print(f"训练数据量: {len(train_dataset)}")
    
    # 根据阶段训练
    if args.stage >= 1:
        stage1_dataloader = build_train_dataloader(stage_id=1)
        log_dataloader_batch_stats("stage1_loader", stage1_dataloader, tb_writer)
        if args.reuse_vq_codebook and os.path.exists(args.vq_codebook_path):
            print(f"检测到VQ codebook，跳过Stage1并加载: {args.vq_codebook_path}")
            load_vq_codebook(model, args.vq_codebook_path)
        else:
            model = train_stage1_vq(model, stage1_dataloader, args, tb_writer)
            save_stage1_checkpoint(model, args)
    
    if args.stage >= 2:
        stage2_dataloader = build_train_dataloader(stage_id=2)
        log_dataloader_batch_stats("stage2_loader", stage2_dataloader, tb_writer)
        # Stage2 复用逻辑
        if not args.stage2_ckpt_path:
            args.stage2_ckpt_path = os.path.join(args.save_path, "stage2_vision")
        
        if args.reuse_stage2 and os.path.exists(args.stage2_ckpt_path) and os.path.exists(os.path.join(args.stage2_ckpt_path, "vq_codebook.pt")):
            print(f"检测到Stage2检查点，跳过Stage2并加载: {args.stage2_ckpt_path}")
            model = load_stage2_checkpoint(model, args.stage2_ckpt_path)
        else:
            model = train_stage2_vision(model, stage2_dataloader, args, tb_writer)
            save_stage2_checkpoint(model, args, args.stage2_ckpt_path)
            save_checkpoint(model, args, "stage2_vision")
    
    if args.stage >= 3:
        stage3_dataloader = build_train_dataloader(stage_id=3)
        log_dataloader_batch_stats("stage3_loader", stage3_dataloader, tb_writer)
        model = train_stage3_lord(model, stage3_dataloader, args, tb_writer)
        save_stage3_checkpoint(model, os.path.join(args.save_path, "stage3_lord_final"))
    
    print("\n" + "=" * 60)
    print("VQ-LoRD 训练完成!")
    print("=" * 60)
    
    tb_writer.close()


if __name__ == "__main__":
    main()
