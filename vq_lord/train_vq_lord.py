"""
======================================================================
TRAIN_VQ_LORD ---

VQ-LoRD 主训练脚本

结合 VQ 离散化和 LoRD 蒸馏，窃取多模态模型的图像识别能力

训练流程:
1. 加载预训练 LLaVA 模型
2. 添加 VQ 层到 Vision Encoder
3. 加载 GPT-4V 收集的视觉数据
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

from vq_module import VQVisionEncoder
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

        answer_idx = item.get("answer", 0)
        answer = choices[answer_idx] if choices and answer_idx < len(choices) else ""
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


def attach_gpt4v_teacher_responses(samples: List[dict], args) -> List[dict]:
    """
    为 ScienceQA 样本补充 GPT-4V 教师回答（带缓存）

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
            model=args.victim_model,
            save_dir=args.data_dir,
        )

        if not collector.api_key:
            msg = (
                "需要采集 GPT-4V 教师回答，但未检测到 OPENAI_API_KEY。"
                "可设置环境变量，或提供已有 teacher cache。"
            )
            if args.strict_teacher_distill:
                raise RuntimeError(msg)
            print(f"警告: {msg}")
        else:
            print(f"开始采集教师回答，待补齐样本数: {len(need_collect)}")
            for rank, (idx, key) in enumerate(tqdm(need_collect, desc="采集 GPT-4V 教师回答"), start=1):
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
    """ScienceQA 多模态数据集包装"""

    def __init__(
        self,
        processor,
        split: str = "train",
        train_num: int = 500,
        max_length: int = 512,
        samples: Optional[List[dict]] = None,
        seed: int = 20240306,
    ):
        self.processor = processor
        self.max_length = max_length

        if samples is None:
            samples = build_scienceqa_samples(
                split=split,
                train_num=train_num,
                seed=seed,
            )

        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        image = item["image"]
        image_sizes_default = torch.tensor([image.height, image.width], dtype=torch.long)

        instruction = item['instruction']
        response = item.get('teacher_response', item.get('response', ''))
        response = response.replace("<image>", "").replace("< image >", "").replace("<Image>", "")
        instruction_text = instruction.replace("<image>", "").replace("< image >", "").replace("<Image>", "").strip()

        # 预截断文本部分（不能截断 processor 输出，否则会破坏 <image> token 对齐）
        # 先估算 response 的 token 数，超出预算则截断 response
        max_text_tokens = self.max_length  # 纯文本 token 预算
        instr_tokens = self.processor.tokenizer.encode(instruction, add_special_tokens=False)
        resp_tokens = self.processor.tokenizer.encode(response, add_special_tokens=False)
        budget = max_text_tokens - len(instr_tokens) - 2  # 留 2 给特殊 token
        if budget > 0 and len(resp_tokens) > budget:
            resp_tokens = resp_tokens[:budget]
            response = self.processor.tokenizer.decode(resp_tokens, skip_special_tokens=True)

        if hasattr(self.processor, "apply_chat_template"):
            full_conv = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": instruction_text},
                        {"type": "image"},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": response},
                    ],
                },
            ]
            prompt_conv = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": instruction_text},
                        {"type": "image"},
                    ],
                }
            ]

            full_text = self.processor.apply_chat_template(full_conv, add_generation_prompt=False)
            prompt_text = self.processor.apply_chat_template(prompt_conv, add_generation_prompt=True)
        else:
            # 兼容无 chat template 的老版本
            full_text = f"<image>\n{instruction_text}\n{response}"
            prompt_text = f"<image>\n{instruction_text}"

        inputs = self.processor(
            text=full_text,
            images=image,
            return_tensors="pt",
            padding="longest",
            truncation=False,
        )

        prompt_inputs = self.processor(
            text=prompt_text,
            images=image,
            return_tensors="pt",
            padding="longest",
            truncation=False,
        )

        # 对 LLaVA-Next，直接使用原图尺寸最稳妥，避免 processor 返回形状差异导致 patch 计算异常
        image_sizes = image_sizes_default

        pad_id = self.processor.tokenizer.pad_token_id
        labels = inputs["input_ids"].squeeze(0).clone()
        prompt_len = prompt_inputs["input_ids"].shape[1]
        prompt_len = min(prompt_len, labels.shape[0])
        labels[:prompt_len] = -100
        labels = labels.masked_fill(labels == pad_id, -100)

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "image_sizes": image_sizes,
            "labels": labels,
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
            elif prev_attr is not None:
                target.gradient_checkpointing = True
        else:
            if hasattr(target, "gradient_checkpointing_disable"):
                target.gradient_checkpointing_disable()
            elif prev_attr is not None:
                target.gradient_checkpointing = False

            if hasattr(target, "gradient_checkpointing"):
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
    
    # VQ 参数
    parser.add_argument("--vq_codebook_size", type=int, default=8192,
                       help="VQ codebook 大小")
    parser.add_argument("--vq_commitment_cost", type=float, default=0.25,
                       help="VQ commitment loss 权重")
    parser.add_argument("--vq_dead_code_threshold", type=int, default=100,
                       help="VQ dead code restart 阈值：连续多少步未使用后重置 code")
    parser.add_argument("--freeze_vision_tower", type=int, default=0,
                       help="是否冻结原始 vision tower")
    
    # 损失权重
    parser.add_argument("--alpha", type=float, default=1.0,
                       help="视觉蒸馏损失权重")
    parser.add_argument("--beta", type=float, default=0.25,
                       help="VQ 损失权重")
    parser.add_argument("--temperature", type=float, default=1.5,
                       help="蒸馏温度")
    
    # 训练参数
    parser.add_argument("--stage", type=int, default=3,
                       help="训练阶段 (1=VQ预训练, 2=视觉蒸馏, 3=联合训练)")
    parser.add_argument("--epochs", type=int, default=3,
                       help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="批次大小")
    parser.add_argument("--lr", type=float, default=3e-5,
                       help="学习率")
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
    parser.add_argument("--stage3_grad_accum", type=int, default=0,
                       help="Stage3 梯度累积步数，0 表示回退到 grad_accum")
    
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
                       help="是否自动采集缺失的 GPT-4V 教师回答")
    parser.add_argument("--strict_teacher_distill", type=int, default=1,
                       help="严格蒸馏模式：若缺失 GPT-4V 教师回答则报错")
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
    
    关键：使用 forward hook 将 VQ 量化后的特征替换原始视觉特征
    """
    print("添加 VQ 离散化层...")
    
    # 获取 vision tower
    vision_tower = model.vision_tower
    
    # 创建 VQ Vision Encoder
    vq_vision_encoder = VQVisionEncoder(
        vision_tower=vision_tower,
        num_embeddings=args.vq_codebook_size,
        commitment_cost=args.vq_commitment_cost,
        freeze_vision_tower=bool(args.freeze_vision_tower),
    )
    vq_vision_encoder.vq.dead_code_threshold = int(args.vq_dead_code_threshold)

    try:
        vision_device = next(vision_tower.parameters()).device
        vq_vision_encoder.to(vision_device)
    except StopIteration:
        pass
    
    # 保存到模型
    model.vq_vision_encoder = vq_vision_encoder
    
    # 保存 VQ 损失的容器 (用于训练时获取)
    model._vq_loss_container = {"loss": None, "logits": None}
    
    # 注册 forward hook，在 vision_tower 输出后应用 VQ
    def vq_hook(module, input, output):
        """在 vision tower 输出后应用 VQ 量化"""
        # 处理不同的输出格式
        if hasattr(output, "last_hidden_state"):
            vision_features = output.last_hidden_state
        elif isinstance(output, tuple):
            vision_features = output[0]
        else:
            vision_features = output
        
        # 应用 VQ 量化
        if model.vq_vision_encoder.vq.embedding.weight.device != vision_features.device:
            model.vq_vision_encoder.vq.to(vision_features.device)
        quantized, indices, vq_loss, logits = model.vq_vision_encoder.vq(
            vision_features, return_logits=True
        )
        
        # 保存 VQ 损失供训练使用
        model._vq_loss_container["loss"] = vq_loss
        # model._vq_loss_container["logits"] = logits  # 节省显存，不保存未使用的 logits
        model._vq_loss_container["indices"] = indices
        
        # 返回量化后的特征（替换原始特征）
        if hasattr(output, "last_hidden_state"):
            # 如果是 BaseModelOutputWithPooling 类型
            output.last_hidden_state = quantized
            return output
        elif isinstance(output, tuple):
            return (quantized,) + output[1:]
        else:
            return quantized
    
    # 注册 hook (注意：这会修改原模型行为)
    model._vq_hook_handle = vision_tower.register_forward_hook(vq_hook)
    
    print(
        f"VQ 层已添加，codebook 大小: {args.vq_codebook_size}, "
        f"dead_code_threshold: {vq_vision_encoder.vq.dead_code_threshold}"
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
    
    目标：训练 VQ 层学习好的图像离散表示
    注意：这个阶段只训练 VQ codebook，不训练其他组件
    """
    print("\n" + "=" * 50)
    print("阶段 1: VQ Codebook 预训练")
    print("=" * 50)

    # 注意：不再关闭 EMA。保留 EMA 更新让 codebook 跟随数据分布移动，
    # 配合 dead code restart 机制可避免 codebook collapse。
    
    # 设置模型为训练模式
    model.train()
    
    # 只训练 VQ 层
    for name, param in model.named_parameters():
        if "vq" in name.lower() and torch.is_floating_point(param):
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    # 统计可训练参数
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"可训练参数量: {trainable_params:,}")
    
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr * 5,  # VQ 预训练用更大学习率
    )
    
    log_dir = os.path.join(args.save_path, "logs")
    os.makedirs(log_dir, exist_ok=True)
    metrics_path = os.path.join(log_dir, "vq_metrics.csv")

    if not os.path.exists(metrics_path):
        with open(metrics_path, "w", encoding="utf-8") as f:
            f.write("stage,step,loss,loss_ema,codebook_used\n")

    global_step = 0
    loss_ema = None
    for epoch in range(args.epochs):
        maybe_set_dataloader_epoch(dataloader, epoch)
        epoch_loss = 0.0
        for batch in tqdm(dataloader, desc=f"VQ预训练 Epoch {epoch+1}"):
            global_step += 1
            
            pixel_values = batch["pixel_values"].to(args.device)

            if pixel_values.dim() == 5:
                batch_size, num_patches, channels, height, width = pixel_values.shape
                pixel_values = pixel_values.view(batch_size * num_patches, channels, height, width)
            
            # 前向传播（VQ hook 自动执行 EMA 更新 + dead code restart）
            _ = model.vision_tower(pixel_values)
            
            # 获取 hook 中缓存的 VQ 损失与索引
            vq_loss = model._vq_loss_container["loss"]
            vq_indices = model._vq_loss_container.get("indices")
            
            # EMA 模式: codebook 已在 hook 中通过 EMA 原地更新，
            # vision_tower 冻结 → loss 无梯度 → 跳过 backward。
            # 非 EMA 模式: 需要 backward + optimizer.step()。
            if vq_loss is not None and vq_loss.requires_grad:
                optimizer.zero_grad()
                vq_loss.backward()
                optimizer.step()
            
            vq_loss_val = vq_loss.item() if vq_loss is not None else 0.0
            epoch_loss += vq_loss_val
            loss_ema = vq_loss_val if loss_ema is None else (0.95 * loss_ema + 0.05 * vq_loss_val)

            if vq_indices is not None:
                codebook_used = torch.unique(vq_indices).numel()
            else:
                codebook_used = 0
            
            if global_step % args.log_step == 0:
                print(f"Step {global_step}, VQ Loss: {vq_loss_val:.4f}")
                tb_writer.add_scalar("stage1/vq_loss", vq_loss_val, global_step)
                tb_writer.add_scalar("stage1/vq_loss_ema", loss_ema, global_step)
                tb_writer.add_scalar("stage1/codebook_used", codebook_used, global_step)
                with open(metrics_path, "a", encoding="utf-8") as f:
                    f.write(f"stage1,{global_step},{vq_loss_val:.6f},{loss_ema:.6f},{codebook_used}\n")
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} 平均 VQ Loss: {avg_loss:.4f}")
    
    return model


def train_stage2_vision(model, dataloader, args, tb_writer):
    """
    阶段 2: 视觉能力蒸馏
    
    目标：通过视觉问答蒸馏，让学生模型学习 GPT-4V 的视觉理解能力
    训练：VQ 层 + Vision Encoder（如果未冻结）+ Projector
    """
    print("\n" + "=" * 50)
    print("阶段 2: 视觉能力蒸馏")
    print("=" * 50)
    
    image_token_id = _get_image_token_id(model)
    
    model.train()
    
    # 训练 VQ + Vision (如果未冻结) + Projector
    for name, param in model.named_parameters():
        if not torch.is_floating_point(param):
            param.requires_grad = False
            continue
        if any(key in name.lower() for key in ["vq", "projector", "multi_modal_projector"]):
            param.requires_grad = True
        elif "vision" in name.lower() and not args.freeze_vision_tower:
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"可训练参数量: {trainable_params:,}")
    
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
    )
    grad_accum = max(1, int(getattr(args, "stage2_grad_accum", 0) or getattr(args, "grad_accum", 1)))
    print(f"[Stage2] grad_accum={grad_accum}")
    
    global_step = 0
    for epoch in range(args.epochs):
        maybe_set_dataloader_epoch(dataloader, epoch)
        epoch_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"视觉蒸馏 Epoch {epoch+1}")):
            global_step += 1
            
            input_ids = batch["input_ids"].to(args.device)
            attention_mask = batch["attention_mask"].to(args.device)
            pixel_values = batch["pixel_values"].to(args.device)
            image_sizes = batch.get("image_sizes")
            image_sizes = sanitize_image_sizes(image_sizes, batch_size=input_ids.size(0))
            labels = batch["labels"].to(args.device)

            # 首个 batch 做一次 sanity check
            if global_step == 1:
                n_img = (input_ids == image_token_id).sum(dim=1)
                if (n_img == 0).any():
                    raise RuntimeError(
                        f"首个 batch 缺少 image token: counts={n_img.tolist()}, id={image_token_id}"
                    )
            
            # 前向传播（VQ hook 会自动应用）
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_sizes=image_sizes,
                labels=labels,
            )
            
            # 获取 VQ 损失
            vq_loss = model._vq_loss_container.get("loss", torch.tensor(0.0, device=args.device))
            if vq_loss is None:
                vq_loss = torch.tensor(0.0, device=args.device)
            
            # 计算总损失
            text_loss = outputs.loss
            total_loss = text_loss + args.beta * vq_loss
            total_loss_val = total_loss.item()

            (total_loss / grad_accum).backward()
            if (batch_idx + 1) % grad_accum == 0 or batch_idx == len(dataloader) - 1:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            
            epoch_loss += total_loss_val
            
            if global_step % args.log_step == 0:
                vq_val = vq_loss.item() if hasattr(vq_loss, 'item') else vq_loss
                print(f"Step {global_step}, Total: {total_loss_val:.4f}, "
                      f"Text: {text_loss.item():.4f}, VQ: {vq_val:.4f}")
                tb_writer.add_scalar("stage2/total_loss", total_loss_val, global_step)
                tb_writer.add_scalar("stage2/text_loss", text_loss.item(), global_step)
                tb_writer.add_scalar("stage2/vq_loss", vq_val, global_step)
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} 平均损失: {avg_loss:.4f}")
    
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
    
    print(f"LoRD 参数: tau1={tau1}, max_new_tokens={max_new_tokens}, grad_accum={grad_accum}")

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
        is_projector = "projector" in name_l or "multi_modal_projector" in name_l
        is_vision = ("vision" in name_l) and (not args.freeze_vision_tower)

        param.requires_grad = bool(is_lora or is_vq or is_projector or is_vision)

    # 冻结 VQ codebook：关闭 EMA 和 dead code restart，
    # 避免 Stage3 多次 forward 导致 codebook 过激进更新。
    # STE 仍正常传递梯度到 vision_tower，commitment loss 仍约束 z→codebook。
    _vq = model.vq_vision_encoder.vq
    _vq.use_ema = False
    _vq.dead_code_threshold = 10**9  # 实质禁用 dead code restart
    print(f"[Stage3] VQ codebook 已冻结 (use_ema=False, dead_code restart 禁用)")

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Stage3 可训练参数量: {trainable_params:,}")
    
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
    )
    
    # ========== 断点续训：加载已有的 Stage3 checkpoint ==========
    global_step = 0
    start_epoch = 0
    stage3_resume_dir = os.path.join(args.save_path, "stage3_resume")
    stage3_state_path = os.path.join(stage3_resume_dir, "training_state.pt")
    
    if os.path.exists(stage3_state_path):
        print(f"检测到 Stage3 断点，正在恢复: {stage3_state_path}")
        state = torch.load(stage3_state_path, map_location=args.device)
        start_epoch = state["epoch"]
        global_step = state["global_step"]
        
        # 加载模型权重（需要在 optimizer 之前加载）
        resume_model_dir = os.path.join(stage3_resume_dir, "model")
        if os.path.exists(resume_model_dir):
            from safetensors.torch import load_file as safe_load
            adapter_path = os.path.join(resume_model_dir, "adapter_model.safetensors")
            if os.path.exists(adapter_path):
                sd = safe_load(adapter_path, device=str(args.device))
                model.load_state_dict(sd, strict=False)
            else:
                adapter_bin = os.path.join(resume_model_dir, "adapter_model.bin")
                if os.path.exists(adapter_bin):
                    sd = torch.load(adapter_bin, map_location=args.device)
                    model.load_state_dict(sd, strict=False)
        
            # 加载 VQ codebook
            vq_resume_path = os.path.join(resume_model_dir, "vq_codebook.pt")
            if os.path.exists(vq_resume_path):
                cb = torch.load(vq_resume_path, map_location="cpu")
                model.vq_vision_encoder.vq.embedding.weight.data.copy_(cb)
                _mark_vq_codebook_loaded(model.vq_vision_encoder.vq)
        
        # 加载 optimizer state（加载到与参数一致的设备上）
        optimizer.load_state_dict(state["optimizer_state_dict"])
        
        # 如果已经训练完成，跳过
        if start_epoch >= args.epochs:
            print(f"训练已完成 (epoch={start_epoch}/{args.epochs})，无需继续")
            return model
        
        print(f"已恢复: epoch={start_epoch}, global_step={global_step}")
    
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
                            temperature=0.7,
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
            if (batch_idx + 1) % grad_accum == 0 or batch_idx == len(dataloader) - 1:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

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
        
        # Epoch 统计
        num_batches = len(dataloader)
        print(f"Epoch {epoch+1} 平均损失: Total={epoch_loss/num_batches:.4f}, "
              f"L_obj={epoch_obj_loss/num_batches:.4f}, L_reg={epoch_reg_loss/num_batches:.4f}, "
              f"VQ={epoch_vq_loss/num_batches:.4f}")
        
        # ========== 每个 Epoch 结束保存断点 ==========
        print(f"保存 Stage3 断点 (epoch={epoch+1}, step={global_step})...")
        os.makedirs(stage3_resume_dir, exist_ok=True)
        resume_model_dir = os.path.join(stage3_resume_dir, "model")
        os.makedirs(resume_model_dir, exist_ok=True)
        
        # 保存模型权重
        model.save_pretrained(resume_model_dir)
        # 保存 VQ codebook
        torch.save(
            model.vq_vision_encoder.vq.embedding.weight.detach().cpu(),
            os.path.join(resume_model_dir, "vq_codebook.pt")
        )
        # 保存训练状态
        torch.save({
            "epoch": epoch + 1,
            "global_step": global_step,
            "optimizer_state_dict": optimizer.state_dict(),
        }, stage3_state_path)
        print(f"断点已保存至 {stage3_resume_dir}")
        
        # 同时保存当前 epoch 的独立检查点
        save_checkpoint(model, args, f"stage3_epoch{epoch+1}")
    
    return model


def save_checkpoint(model, args, suffix=""):
    """保存检查点"""
    save_path = os.path.join(args.save_path, suffix)
    os.makedirs(save_path, exist_ok=True)
    
    model.save_pretrained(save_path)
    print(f"模型已保存至 {save_path}")


def save_vq_codebook(model, codebook_path: str):
    os.makedirs(os.path.dirname(codebook_path), exist_ok=True)
    torch.save(model.vq_vision_encoder.vq.embedding.weight.detach().cpu(), codebook_path)


def _mark_vq_codebook_loaded(vq_module):
    """加载 codebook 后，同步内部状态，避免训练首批再次数据驱动初始化。"""
    if hasattr(vq_module, "_initialized"):
        vq_module._initialized.fill_(1)

    if getattr(vq_module, "use_ema", False):
        if hasattr(vq_module, "ema_embedding_sum"):
            vq_module.ema_embedding_sum.data.copy_(vq_module.embedding.weight.data)
        if hasattr(vq_module, "ema_cluster_size"):
            vq_module.ema_cluster_size.data.clamp_(min=1.0)

    if hasattr(vq_module, "_code_idle_steps"):
        vq_module._code_idle_steps.zero_()


def load_vq_codebook(model, codebook_path: str):
    codebook = torch.load(codebook_path, map_location="cpu")
    model.vq_vision_encoder.vq.embedding.weight.data.copy_(codebook)
    _mark_vq_codebook_loaded(model.vq_vision_encoder.vq)
    print(f"已加载 VQ codebook: {codebook_path} (已禁用首批重初始化)")


def save_stage2_checkpoint(model, args, ckpt_path: str):
    """保存 Stage2 检查点（包括 VQ codebook 和 LoRA 权重）"""
    os.makedirs(ckpt_path, exist_ok=True)
    
    # 保存 VQ codebook
    vq_path = os.path.join(ckpt_path, "vq_codebook.pt")
    torch.save(model.vq_vision_encoder.vq.embedding.weight.detach().cpu(), vq_path)
    
    # 保存 LoRA 适配器
    model.save_pretrained(ckpt_path)
    
    # 保存配置信息
    import json
    config_info = {
        "vq_codebook_size": args.vq_codebook_size,
        "freeze_vision_tower": args.freeze_vision_tower,
        "alpha": args.alpha,
        "beta": args.beta,
        "stage": 2,
    }
    with open(os.path.join(ckpt_path, "stage2_config.json"), "w") as f:
        json.dump(config_info, f, indent=2)
    
    print(f"Stage2 检查点已保存至 {ckpt_path}")


def load_stage2_checkpoint(model, ckpt_path: str):
    """加载 Stage2 检查点"""
    # 加载 VQ codebook
    vq_path = os.path.join(ckpt_path, "vq_codebook.pt")
    if os.path.exists(vq_path):
        codebook = torch.load(vq_path, map_location="cpu")
        model.vq_vision_encoder.vq.embedding.weight.data.copy_(codebook)
        _mark_vq_codebook_loaded(model.vq_vision_encoder.vq)
        print(f"已加载 VQ codebook: {vq_path} (已禁用首批重初始化)")
    
    # 加载 LoRA 适配器
    adapter_config = os.path.join(ckpt_path, "adapter_config.json")
    if os.path.exists(adapter_config):
        # 如果模型已经是 PeftModel，则加载权重
        if hasattr(model, 'load_adapter'):
            model.load_adapter(ckpt_path, adapter_name="stage2")
            model.set_adapter("stage2")
        else:
            # 直接加载状态字典
            adapter_weights = os.path.join(ckpt_path, "adapter_model.safetensors")
            if os.path.exists(adapter_weights):
                from safetensors.torch import load_file
                state_dict = load_file(adapter_weights)
                model.load_state_dict(state_dict, strict=False)
            else:
                adapter_weights_bin = os.path.join(ckpt_path, "adapter_model.bin")
                if os.path.exists(adapter_weights_bin):
                    state_dict = torch.load(adapter_weights_bin, map_location="cpu")
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

        # 关键：为每条样本补齐 GPT-4V 教师回答，供 Stage2/Stage3 蒸馏使用
        train_samples = attach_gpt4v_teacher_responses(train_samples, args)

        train_dataset = ScienceQADataset(
            processor=processor,
            max_length=args.max_length,
            samples=train_samples,
            seed=args.scienceqa_seed,
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

        # --- 对不等长的 1D 张量做 right-padding 对齐 ---
        pad_token_id = processor.tokenizer.pad_token_id or 0
        # 需要 pad 的 key 及其填充值
        pad_keys = {
            "input_ids": pad_token_id,
            "attention_mask": 0,
            "labels": -100,
        }

        # 先找出每个需要 pad 的 key 的最大长度
        max_lens = {}
        for key in pad_keys:
            lengths = [b[key].shape[-1] for b in batch if key in b and isinstance(b[key], torch.Tensor)]
            if lengths:
                max_lens[key] = max(lengths)

        # 对每个样本做 padding
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
                # patch 数不同，pad 第 0 维到最大值
                max_patches = max(s[0] for s in pv_shapes)
                for i, pv in enumerate(pv_list):
                    if pv.shape[0] < max_patches:
                        pad_tensor = torch.zeros(
                            (max_patches - pv.shape[0],) + pv.shape[1:],
                            dtype=pv.dtype, device=pv.device,
                        )
                        batch[i]["pixel_values"] = torch.cat([pv, pad_tensor], dim=0)

        # --- 组装 collated dict ---
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
            save_checkpoint(model, args, "stage1_vq")
            save_vq_codebook(model, args.vq_codebook_path)
    
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
        save_checkpoint(model, args, "stage3_lord_final")
    
    print("\n" + "=" * 60)
    print("VQ-LoRD 训练完成!")
    print("=" * 60)
    
    tb_writer.close()


if __name__ == "__main__":
    main()
