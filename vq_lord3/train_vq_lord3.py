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
import math
import time
import re
import sys
from datetime import timedelta
from dataclasses import dataclass
import torch
import argparse
from collections import defaultdict
from typing import Optional, List, Dict, Tuple, Set
from pprint import pprint
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
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
from data_collector2 import (
    GPT4VDataCollector,
    VQLORDDataset,
    VisualQAItem,
    ImageDescriptionItem,
)
from textvqa_mcq import build_textvqa_mcq_samples, materialize_textvqa_image
from student_models import (
    LLAVA_NEXT,
    add_vq_to_student_model,
    encode_multimodal,
    get_image_token_id as get_student_image_token_id,
    is_projector_param,
    is_vision_param,
    load_student_model_and_processor,
    lora_target_modules,
    normalize_student_model_type,
    student_forward,
    student_generate,
)

import torch.nn.functional as F
import numpy as np


# Ensure stage modules can import this module by name even when executed as a script.
sys.modules.setdefault("train_vq_lord3", sys.modules[__name__])


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def barrier_if_distributed():
    if is_distributed():
        backend = dist.get_backend()
        if torch.cuda.is_available() and str(backend).lower() == "nccl":
            dist.barrier(device_ids=[torch.cuda.current_device()])
            return
        dist.barrier()


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def _get_reduce_device(device: Optional[str] = None) -> torch.device:
    if device and str(device).startswith("cuda") and torch.cuda.is_available():
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def reduce_numeric_dict(metrics: Dict[str, float], device: Optional[str] = None) -> Dict[str, float]:
    reduced = {str(k): float(v) for k, v in metrics.items()}
    if not reduced or not is_distributed():
        return reduced

    keys = sorted(reduced.keys())
    tensor = torch.tensor(
        [reduced[key] for key in keys],
        dtype=torch.float64,
        device=_get_reduce_device(device),
    )
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return {key: float(tensor[idx].item()) for idx, key in enumerate(keys)}


def init_distributed_mode(args):
    args.rank = 0
    args.world_size = 1
    args.local_rank = 0
    args.distributed = False

    world_size_env = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size_env <= 1:
        args.is_main_process = True
        return args

    if not torch.cuda.is_available():
        raise RuntimeError("检测到 WORLD_SIZE>1，但当前环境无可用 CUDA，无法启动 DDP 训练。")

    args.distributed = True
    args.rank = int(os.environ["RANK"])
    args.world_size = world_size_env
    args.local_rank = int(os.environ["LOCAL_RANK"])
    args.device = f"cuda:{args.local_rank}"
    torch.cuda.set_device(args.local_rank)
    ddp_timeout_sec = int(os.environ.get("TORCH_DDP_TIMEOUT_SEC", "1800"))
    if ddp_timeout_sec <= 0:
        ddp_timeout_sec = 1800
    dist.init_process_group(
        backend="nccl",
        timeout=timedelta(seconds=ddp_timeout_sec),
    )
    args.is_main_process = bool(args.rank == 0)
    if args.is_main_process:
        print(
            f"[DDP] backend=nccl, world_size={args.world_size}, "
            f"timeout={ddp_timeout_sec}s"
        )
    return args


def cleanup_distributed():
    if is_distributed():
        dist.destroy_process_group()


HF_MULTICHOICE_DATASET_NAMES = ("scienceqa", "aokvqa", "textvqa")


def _normalize_dataset_name(dataset_name: str) -> str:
    name = str(dataset_name or "").strip().lower()
    if not name:
        return "scienceqa"
    return name


def _is_hf_multichoice_dataset(dataset_name: str) -> bool:
    return _normalize_dataset_name(dataset_name) in HF_MULTICHOICE_DATASET_NAMES


def _resolve_dataset_answer_idx_for_eval(item: dict, dataset_name: str) -> int:
    dataset_name = _normalize_dataset_name(dataset_name)
    if dataset_name == "aokvqa":
        field_candidates = ("correct_choice_idx", "answer")
    else:
        field_candidates = ("answer", "correct_choice_idx")

    for field in field_candidates:
        raw_value = item.get(field)
        if raw_value is None:
            continue
        try:
            return int(raw_value)
        except (TypeError, ValueError):
            continue

    available = ",".join(sorted(item.keys()))
    raise RuntimeError(
        f"评估样本缺少有效答案字段。dataset_name={dataset_name}, "
        f"expected_fields={field_candidates}, available_fields={available}"
    )


def build_scienceqa_samples(
    scienceqa_path: str,
    split: str,
    train_num: int,
    seed: int,
    dataset_name: str = "scienceqa",
    allowed_source_indices: Optional[Set[int]] = None,
    include_dataset_answer: bool = False,
) -> List[dict]:
    dataset_name = _normalize_dataset_name(dataset_name)
    if dataset_name == "textvqa":
        return build_textvqa_mcq_samples(
            dataset_path=scienceqa_path,
            split=split,
            train_num=train_num,
            seed=seed,
            allowed_source_indices=allowed_source_indices,
        )

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

    if allowed_source_indices is not None:
        all_indices = [
            idx for idx in sorted(allowed_source_indices)
            if 0 <= int(idx) < len(dataset_with_images)
        ]
    else:
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
        if not isinstance(choices, list):
            raise RuntimeError(
                f"样本 choices 不是 list。dataset_name={dataset_name}, split={split}, source_index={dataset_idx}"
            )
        if not choices:
            raise RuntimeError(
                f"样本 choices 为空。dataset_name={dataset_name}, split={split}, source_index={dataset_idx}"
            )
        choices_text = ""
        for choice_idx, choice in enumerate(choices):
            choices_text += f"({chr(65 + choice_idx)}) {choice}\n"

        hint = item.get("hint", "")
        lecture = item.get("lecture", "")
        solution = item.get("solution", "")

        hint_block = f"Hint: {hint}\n" if hint else ""
        instruction = f"<image>\nQuestion: {question}\n{hint_block}Options:\n{choices_text}Answer:"
        if lecture:
            response = f"Explanation: {lecture}\nSolution: {solution}"
        else:
            response = f"Solution: {solution}"

        sample = {
            "sample_id": sampled_pos,
            "source_index": dataset_idx,
            "split": split,
            "dataset_name": dataset_name,
            "image": item.get("image"),
            "question": question,
            "hint": hint,
            "choices": choices,
            "instruction": instruction,
            "response": response,
        }
        if include_dataset_answer:
            answer_idx = _resolve_dataset_answer_idx_for_eval(item, dataset_name=dataset_name)
            if answer_idx < 0 or answer_idx >= len(choices):
                raise RuntimeError(
                    f"评估样本 answer_idx 越界。dataset_name={dataset_name}, split={split}, "
                    f"source_index={dataset_idx}, answer_idx={answer_idx}, num_choices={len(choices)}"
                )
            sample["answer_idx"] = int(answer_idx)
            sample["answer_letter"] = chr(ord("A") + int(answer_idx))
            sample["answer_text"] = str(choices[int(answer_idx)])

        samples.append(sample)

    return samples


def _load_cached_teacher_source_indices(
    cache_path: str,
    dataset_name: str,
    split: str,
) -> Set[int]:
    if not cache_path:
        raise RuntimeError("sample_only_cached_teacher=1 时必须提供 teacher_cache_path")
    if not os.path.exists(cache_path):
        raise RuntimeError(f"teacher_cache_path 不存在: {cache_path}")

    with open(cache_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    cache_samples = payload.get("samples")
    if not isinstance(cache_samples, dict):
        raise RuntimeError(f"教师缓存 samples 字段无效: {cache_path}")

    dataset_prefix = _normalize_dataset_name(dataset_name)
    stable_prefix = f"{dataset_prefix}::{split}::"

    allowed: Set[int] = set()
    for cache_key, teacher_payload in cache_samples.items():
        if not isinstance(cache_key, str) or not cache_key.startswith(stable_prefix):
            continue
        suffix = cache_key[len(stable_prefix):]
        try:
            source_index = int(suffix)
        except (TypeError, ValueError):
            continue
        if not isinstance(teacher_payload, dict):
            continue
        valid = True
        for field in TEACHER_REQUIRED_FIELDS:
            value = teacher_payload.get(field)
            if not isinstance(value, str) or len(value.strip()) == 0:
                valid = False
                break
        if valid:
            allowed.add(source_index)

    return allowed


def _safe_name(text: str) -> str:
    keep = []
    for ch in text:
        if ch.isalnum() or ch in ("-", "_", "."):
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep)


def _sample_key(sample: dict) -> str:
    source_index = sample.get("source_index")
    if source_index is not None:
        split_name = str(sample.get("split") or "unknown")
        dataset_prefix = _normalize_dataset_name(sample.get("dataset_name", "scienceqa"))
        return f"{dataset_prefix}::{split_name}::{int(source_index)}"

    instruction = sample.get("instruction", "")
    response = sample.get("response", "")
    image = sample.get("image")
    size = ""
    if image is not None and hasattr(image, "size"):
        size = f"{image.size[0]}x{image.size[1]}"
    raw = f"{instruction}\n{response}\n{size}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


TEACHER_SCHEMA_VERSION = "v2"
TEACHER_REQUIRED_FIELDS = (
    "observed_facts_visual",
    "context_textual",
    "reasoning",
    "answer",
)


def _strip_image_tokens(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.replace("<image>", "")
    text = text.replace("< image >", "")
    text = text.replace("<Image>", "")
    return text.strip()


def _normalize_choice_text_for_match(text: str) -> str:
    text = _strip_image_tokens(text)
    text = text.lower().strip()
    text = re.sub(r"^\(?\s*[a-z]\s*\)?[\.\):：、-]\s*", "", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"^[\"'`“”‘’]+|[\"'`“”‘’]+$", "", text)
    text = re.sub(r"[.;,!?。！？；，]+$", "", text).strip()
    return text


def _resolve_teacher_answer_idx(sample: dict) -> int:
    choices = sample.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError(
            f"教师答案映射失败：样本 choices 非法。sample_id={sample.get('sample_id')}, "
            f"source_index={sample.get('source_index')}"
        )

    ann = sample.get("teacher_annotation")
    if not isinstance(ann, dict):
        raise RuntimeError(
            f"教师答案映射失败：缺失 teacher_annotation。sample_id={sample.get('sample_id')}, "
            f"source_index={sample.get('source_index')}"
        )
    raw_answer = ann.get("answer")
    if not isinstance(raw_answer, str) or not raw_answer.strip():
        raise RuntimeError(
            f"教师答案映射失败：teacher_annotation.answer 为空。sample_id={sample.get('sample_id')}, "
            f"source_index={sample.get('source_index')}"
        )

    answer = _strip_image_tokens(raw_answer)
    option_patterns = (
        r"^\s*\(?\s*([A-Za-z])\s*\)?\s*(?:[\.\):：、-]|$)",
        r"(?:answer|option|答案|选项)\s*(?:is|为|是)?\s*[:：]?\s*\(?\s*([A-Za-z])\s*\)?",
    )
    for pattern in option_patterns:
        match = re.search(pattern, answer, flags=re.IGNORECASE)
        if not match:
            continue
        idx = ord(match.group(1).upper()) - ord("A")
        if 0 <= idx < len(choices):
            return idx
        raise RuntimeError(
            f"教师答案映射失败：选项字母越界。sample_id={sample.get('sample_id')}, "
            f"source_index={sample.get('source_index')}, answer={raw_answer!r}, "
            f"num_choices={len(choices)}"
        )

    normalized_answer = _normalize_choice_text_for_match(answer)
    matches = [
        idx for idx, choice in enumerate(choices)
        if _normalize_choice_text_for_match(str(choice)) == normalized_answer
    ]
    if len(matches) == 1:
        return int(matches[0])

    raise RuntimeError(
        f"教师答案映射失败：无法唯一映射到选项。sample_id={sample.get('sample_id')}, "
        f"source_index={sample.get('source_index')}, answer={raw_answer!r}, "
        f"choices={choices!r}, matches={matches}"
    )


def _apply_teacher_answer_labels(samples: List[dict]) -> None:
    for sample in samples:
        answer_idx = _resolve_teacher_answer_idx(sample)
        choices = sample["choices"]
        sample["answer_idx"] = int(answer_idx)
        sample["answer_letter"] = chr(ord("A") + int(answer_idx))
        sample["answer_text"] = str(choices[int(answer_idx)])


def _truncate_text_by_budget_estimate(text: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return _strip_image_tokens(text)
    text = _strip_image_tokens(text)
    if not text:
        return ""
    # 仅用于缓存落盘时的预算兜底；训练时会再按 tokenizer 精确裁剪。
    pieces = text.split()
    if len(pieces) >= max_tokens:
        return " ".join(pieces[:max_tokens]).strip()
    if len(pieces) <= 1:
        return text[:max_tokens].strip()
    return text.strip()


def _normalize_teacher_annotation(
    payload: Optional[dict],
    budget: dict,
) -> Optional[dict]:
    if not isinstance(payload, dict):
        return None

    canonical = {
        "format_version": TEACHER_SCHEMA_VERSION,
        "observed_facts_visual": payload.get("observed_facts_visual", ""),
        "context_textual": payload.get("context_textual", ""),
        "reasoning": payload.get("reasoning", ""),
        "answer": payload.get("answer", ""),
    }

    canonical["observed_facts_visual"] = _truncate_text_by_budget_estimate(
        canonical["observed_facts_visual"],
        int(budget.get("teacher_observed_max_tokens", 0)),
    )
    canonical["context_textual"] = _truncate_text_by_budget_estimate(
        canonical["context_textual"],
        int(budget.get("teacher_context_max_tokens", 0)),
    )
    canonical["reasoning"] = _truncate_text_by_budget_estimate(
        canonical["reasoning"],
        int(budget.get("teacher_reasoning_max_tokens", 0)),
    )
    canonical["answer"] = _truncate_text_by_budget_estimate(
        canonical["answer"],
        int(budget.get("teacher_answer_max_tokens", 0)),
    )

    for field in TEACHER_REQUIRED_FIELDS:
        if not isinstance(canonical.get(field), str) or len(canonical[field].strip()) == 0:
            return None

    return canonical


def _build_structured_teacher_prompt(instruction: str, lang: str) -> str:
    instruction = _strip_image_tokens(instruction)
    if lang == "en":
        return (
            "You are a rigorous visual science QA assistant.\n"
            "Read image and question carefully. Return strict JSON only with keys:\n"
            "observed_facts_visual, context_textual, reasoning, answer.\n"
            "Rules:\n"
            "1) observed_facts_visual: only visual evidence from image (OCR allowed).\n"
            "2) context_textual: textual conditions from question/options.\n"
            "3) reasoning: explicit reasoning based on (1)+(2).\n"
            "4) answer: concise final answer text.\n"
            "Do not output markdown or extra fields.\n\n"
            f"{instruction}"
        )
    return (
        "你是严谨的视觉科学题解答助手。\n"
        "请仅输出严格 JSON，必须且只包含以下键：\n"
        "observed_facts_visual, context_textual, reasoning, answer。\n"
        "规则：\n"
        "1) observed_facts_visual 只写图像可见证据（可含 OCR）。\n"
        "2) context_textual 只写题干与选项文本条件。\n"
        "3) reasoning 写基于前两者的推理。\n"
        "4) answer 写最终简洁答案。\n"
        "不要输出 markdown，不要额外字段。\n\n"
        f"{instruction}"
    )


def attach_gpt4v_teacher_responses(samples: List[dict], args) -> List[dict]:
    """
    为 ScienceQA 样本补充教师模型回答（带缓存）

    说明：
    - 若缓存存在，优先读取缓存；
    - 若启用采集且存在 API key，则补齐缺失项；
    - vq_lord3 对 teacher_annotation 四字段实行强门禁，仍有缺失将直接报错。
    """
    os.makedirs(args.data_dir, exist_ok=True)

    budget = {
        "teacher_observed_max_tokens": int(args.teacher_observed_max_tokens),
        "teacher_context_max_tokens": int(args.teacher_context_max_tokens),
        "teacher_reasoning_max_tokens": int(args.teacher_reasoning_max_tokens),
        "teacher_answer_max_tokens": int(args.teacher_answer_max_tokens),
        "teacher_max_new_tokens_total": int(args.teacher_max_new_tokens_total),
    }

    cache_path = args.teacher_cache_path
    if not cache_path:
        victim_tag = _safe_name(args.victim_model)
        dataset_prefix = _normalize_dataset_name(getattr(args, "dataset_name", "scienceqa"))
        cache_name = (
            f"{dataset_prefix}_teacher_{victim_tag}_{args.scienceqa_split}_"
            f"n{args.train_num}_seed{args.scienceqa_seed}_new.json"
        )
        cache_path = os.path.join(args.data_dir, cache_name)

    cache_data = {}
    teacher_lang = args.teacher_lang
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            raise RuntimeError(f"教师缓存格式无效: {cache_path}")
        if payload.get("format_version") not in (None, TEACHER_SCHEMA_VERSION):
            raise RuntimeError(
                f"教师缓存 format_version={payload.get('format_version')} 与当前期望 "
                f"{TEACHER_SCHEMA_VERSION} 不一致: {cache_path}"
            )
        raw_cache_data = payload.get("samples", {})
        if not isinstance(raw_cache_data, dict):
            raise RuntimeError(f"教师缓存 samples 字段无效: {cache_path}")
        dataset_prefix = _normalize_dataset_name(getattr(args, "dataset_name", "scienceqa"))
        stable_prefix = f"{dataset_prefix}::{args.scienceqa_split}::"
        for cache_key, cache_value in raw_cache_data.items():
            if not isinstance(cache_key, str) or not cache_key.startswith(stable_prefix):
                raise RuntimeError(f"教师缓存 key 非稳定格式: {cache_key}")
            if not isinstance(cache_value, dict):
                raise RuntimeError(f"教师缓存 value 非法: key={cache_key}")
        cache_data = raw_cache_data
        print(f"已加载教师缓存(v2): {cache_path}, 条目数={len(cache_data)}")

    need_collect = []
    from_teacher = 0

    for i, sample in enumerate(samples):
        key = _sample_key(sample)
        cached = cache_data.get(key)
        annotation = _normalize_teacher_annotation(cached, budget)
        if annotation is not None:
            sample["teacher_annotation"] = annotation
            sample["teacher_source"] = args.victim_model
            from_teacher += 1
        else:
            need_collect.append((i, key))

    if need_collect and int(args.collect_teacher_data):
        collector = GPT4VDataCollector(
            api_key=(args.teacher_api_key if args.teacher_api_key else None),
            base_url=(args.teacher_api_base if args.teacher_api_base else None),
            model=args.victim_model,
            save_dir=args.data_dir,
        )

        if not collector.api_key:
            raise RuntimeError(
                "需要采集四字段教师标注，但未检测到 API Key。"
                "可通过 --teacher_api_key 或 OPENAI_API_KEY 提供。"
            )
        else:
            print(f"开始采集四字段教师标注，待补齐样本数: {len(need_collect)}")
            for rank, (idx, key) in enumerate(tqdm(need_collect, desc="采集四字段教师标注"), start=1):
                sample = samples[idx]
                image = sample.get("image")
                instruction = sample.get("instruction", "")
                teacher_prompt = _build_structured_teacher_prompt(instruction, teacher_lang)

                teacher_response = None
                if image is not None:
                    teacher_response = collector.query_gpt4v_image(
                        image=image,
                        prompt=teacher_prompt,
                        max_tokens=budget["teacher_max_new_tokens_total"],
                        image_format="PNG",
                    )

                annotation = None
                if teacher_response:
                    parsed = json.loads(teacher_response.strip())
                    annotation = _normalize_teacher_annotation(parsed, budget)

                if annotation is not None:
                    sample["teacher_annotation"] = annotation
                    sample["teacher_source"] = args.victim_model
                    from_teacher += 1
                    cache_data[key] = {
                        "format_version": TEACHER_SCHEMA_VERSION,
                        "observed_facts_visual": annotation["observed_facts_visual"],
                        "context_textual": annotation["context_textual"],
                        "reasoning": annotation["reasoning"],
                        "answer": annotation["answer"],
                        "raw_teacher_response": _strip_image_tokens(teacher_response or ""),
                        "meta": {
                            "teacher_model": args.victim_model,
                            "teacher_lang": teacher_lang,
                            "budget": budget,
                        },
                    }

                if rank % 10 == 0 or rank == len(need_collect):
                    with open(cache_path, "w", encoding="utf-8") as f:
                        json.dump(
                            {
                                "format_version": TEACHER_SCHEMA_VERSION,
                                "victim_model": args.victim_model,
                                "dataset_name": _normalize_dataset_name(getattr(args, "dataset_name", "scienceqa")),
                                "scienceqa_split": args.scienceqa_split,
                                "train_num": args.train_num,
                                "scienceqa_seed": args.scienceqa_seed,
                                "budget": budget,
                                "samples": cache_data,
                            },
                            f,
                            ensure_ascii=False,
                            indent=2,
                        )

    # schema v2 强门禁：vq_lord3 不再兼容旧 explanation+answer 字符串路径。
    missing = 0
    for sample in samples:
        ann = _normalize_teacher_annotation(sample.get("teacher_annotation"), budget)
        if ann is not None:
            sample["teacher_annotation"] = ann
            continue
        missing += 1

    print(
        f"教师样本统计(v2): total={len(samples)}, "
        f"from_{args.victim_model}={from_teacher}, missing={missing}"
    )

    if missing > 0:
        raise RuntimeError(
            "存在样本缺失四字段 teacher_annotation。"
            "vq_lord3 不再兼容旧 explanation+answer 缓存；"
            "请提供完整 *_new.json 或开启采集补齐。"
        )

    _apply_teacher_answer_labels(samples)
    print("教师答案标签已应用：answer_idx/answer_letter/answer_text 均来自 teacher_annotation.answer")

    return samples


class ScienceQADataset(torch.utils.data.Dataset):
    """ScienceQA 多模态数据集包装（兼容 Stage2 双监督与 Stage3 单监督）。"""

    def __init__(
        self,
        processor,
        scienceqa_path: str = "ScienceQA",
        dataset_name: str = "scienceqa",
        split: str = "train",
        train_num: int = 500,
        max_length: int = 512,
        samples: Optional[List[dict]] = None,
        seed: int = 20240306,
        teacher_lang: str = "zh",
        teacher_observed_max_tokens: int = 256,
        teacher_context_max_tokens: int = 192,
        teacher_reasoning_max_tokens: int = 256,
        teacher_answer_max_tokens: int = 64,
        stage3_vic_include_context: bool = False,
        require_teacher_annotation: bool = False,
    ):
        self.processor = processor
        self.student_model_type = normalize_student_model_type(
            getattr(processor, "student_model_type", LLAVA_NEXT)
        )
        self.max_length = max_length
        self.dataset_name = _normalize_dataset_name(dataset_name)
        self.teacher_lang = teacher_lang
        self.teacher_observed_max_tokens = int(teacher_observed_max_tokens)
        self.teacher_context_max_tokens = int(teacher_context_max_tokens)
        self.teacher_reasoning_max_tokens = int(teacher_reasoning_max_tokens)
        self.teacher_answer_max_tokens = int(teacher_answer_max_tokens)
        self.stage3_vic_include_context = bool(stage3_vic_include_context)
        self.require_teacher_annotation = bool(require_teacher_annotation)

        if samples is None:
            samples = build_scienceqa_samples(
                scienceqa_path=scienceqa_path,
                dataset_name=self.dataset_name,
                split=split,
                train_num=train_num,
                seed=seed,
            )

        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def _truncate_field_with_tokenizer(self, text: str, max_tokens: int) -> str:
        text = _strip_image_tokens(text)
        if max_tokens <= 0:
            return text
        tok = self.processor.tokenizer.encode(text, add_special_tokens=False)
        if len(tok) <= max_tokens:
            return text
        keep = tok[:max_tokens]
        return self.processor.tokenizer.decode(keep, skip_special_tokens=True).strip()

    def _clip_target_by_budget(self, instruction_text: str, target_text: str, answer_target: str) -> tuple[str, bool]:
        instr_tokens = self.processor.tokenizer.encode(instruction_text, add_special_tokens=False)
        target_tokens = self.processor.tokenizer.encode(target_text, add_special_tokens=False)
        budget = int(self.max_length) - len(instr_tokens) - 2
        if budget <= 0:
            return answer_target, False
        if len(target_tokens) <= budget:
            has_rationale = target_text.strip() != answer_target.strip()
            return target_text, has_rationale

        answer_tokens = self.processor.tokenizer.encode(answer_target, add_special_tokens=False)
        if len(answer_tokens) >= budget:
            return answer_target, False

        if answer_target in target_text:
            prefix = target_text.rsplit(answer_target, 1)[0].strip()
        else:
            prefix = target_text.strip()
        prefix_tokens = self.processor.tokenizer.encode(prefix, add_special_tokens=False)
        keep_prefix_tokens = prefix_tokens[:max(0, budget - len(answer_tokens) - 1)]
        trimmed_prefix = self.processor.tokenizer.decode(keep_prefix_tokens, skip_special_tokens=True).strip()
        if trimmed_prefix:
            return f"{trimmed_prefix}\n{answer_target}", True
        return answer_target, False

    def _build_targets(self, item: dict, instruction_text: str) -> tuple[str, str, str, bool]:
        answer_letter = item.get("answer_letter")
        if not isinstance(answer_letter, str) or len(answer_letter.strip()) == 0:
            raise RuntimeError(
                f"样本缺失 answer_letter，无法构造选择题训练目标。sample_id={item.get('sample_id')}, "
                f"source_index={item.get('source_index')}"
            )
        answer_letter = answer_letter.strip().upper()[0]
        if answer_letter < "A" or answer_letter > "Z":
            raise RuntimeError(
                f"样本 answer_letter 非法，无法构造选择题训练目标。sample_id={item.get('sample_id')}, "
                f"source_index={item.get('source_index')}, answer_letter={answer_letter!r}"
            )

        # 固定答案锚点，确保 Stage2/Stage3 指标口径一致。
        answer_target = f"Answer: {answer_letter}"
        ann = item.get("teacher_annotation")
        if not isinstance(ann, dict):
            if self.require_teacher_annotation:
                sample_id = item.get("sample_id", "unknown")
                raise RuntimeError(
                    f"sample_id={sample_id} 缺失 teacher_annotation(v2)。"
                    "Stage2/3 训练要求四字段教师标注全覆盖。"
                )
            rationale_target = answer_target
            rationale_target, has_rationale = self._clip_target_by_budget(
                instruction_text=instruction_text,
                target_text=rationale_target,
                answer_target=answer_target,
            )
            return answer_target, rationale_target, rationale_target, has_rationale

        observed = self._truncate_field_with_tokenizer(
            ann.get("observed_facts_visual", ""),
            self.teacher_observed_max_tokens,
        )
        context = self._truncate_field_with_tokenizer(
            ann.get("context_textual", ""),
            self.teacher_context_max_tokens,
        )
        reasoning = self._truncate_field_with_tokenizer(
            ann.get("reasoning", ""),
            self.teacher_reasoning_max_tokens,
        )
        answer_field = self._truncate_field_with_tokenizer(
            ann.get("answer", ""),
            self.teacher_answer_max_tokens,
        )

        if not observed or not context or not reasoning or not answer_field:
            raise RuntimeError("teacher_annotation 四字段存在空值，无法构造训练目标。")

        rationale_lines = [
            f"Observed Facts: {observed}",
            f"Reasoning: {reasoning}",
            answer_target,
        ]
        rationale_target = "\n".join(rationale_lines)
        rationale_target, has_rationale = self._clip_target_by_budget(
            instruction_text=instruction_text,
            target_text=rationale_target,
            answer_target=answer_target,
        )

        vic_lines = [f"Observed Facts: {observed}"]
        if self.stage3_vic_include_context:
            vic_lines.append(f"Context: {context}")
        vic_lines.append(f"Reasoning: {reasoning}")
        vic_lines.append(answer_target)
        vic_target = "\n".join(vic_lines)
        vic_target, _ = self._clip_target_by_budget(
            instruction_text=instruction_text,
            target_text=vic_target,
            answer_target=answer_target,
        )

        return answer_target, rationale_target, vic_target, has_rationale

    def __getitem__(self, idx):
        item = self.samples[idx]
        image = materialize_textvqa_image(item["image"])
        image_sizes_default = torch.tensor([image.height, image.width], dtype=torch.long)

        instruction = item.get("instruction", "")
        instruction_text = instruction.replace("<image>", "").replace("< image >", "").replace("<Image>", "").strip()
        answer_target, rationale_target, _, has_rationale = self._build_targets(item, instruction_text)

        prompt_inputs = encode_multimodal(
            self.processor,
            self.student_model_type,
            instruction_text=instruction_text,
            image=image,
            target_text=None,
            add_generation_prompt=True,
        )
        full_inputs = encode_multimodal(
            self.processor,
            self.student_model_type,
            instruction_text=instruction_text,
            image=image,
            target_text=rationale_target,
            add_generation_prompt=False,
        )
        answer_inputs = encode_multimodal(
            self.processor,
            self.student_model_type,
            instruction_text=instruction_text,
            image=image,
            target_text=answer_target,
            add_generation_prompt=False,
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
        answer_letter = str(item.get("answer_letter") or "").strip().upper()[:1]
        if len(answer_letter) != 1 or answer_letter < "A" or answer_letter > "Z":
            raise RuntimeError(
                f"样本 answer_letter 非法，无法构造训练 batch。sample_id={item.get('sample_id')}, "
                f"source_index={item.get('source_index')}, answer_letter={item.get('answer_letter')!r}"
            )
        try:
            answer_idx = int(item["answer_idx"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"样本 answer_idx 非法，无法构造训练 batch。sample_id={item.get('sample_id')}, "
                f"source_index={item.get('source_index')}, answer_idx={item.get('answer_idx')!r}"
            ) from exc

        sample = {
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
            "answer_idx": answer_idx,
            "answer_letter": answer_letter,
            "data_type": "scienceqa",
        }
        for prefix, inputs in (("prompt", prompt_inputs), ("full", full_inputs), ("answer", answer_inputs)):
            if "image_grid_thw" in inputs:
                sample[f"{prefix}_image_grid_thw"] = inputs["image_grid_thw"].squeeze(0)
        if "image_grid_thw" in full_inputs:
            sample["image_grid_thw"] = full_inputs["image_grid_thw"].squeeze(0)
        return sample


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


def _pad_batches_for_distributed(batches: List[List[int]], world_size: int) -> List[List[int]]:
    if world_size <= 1 or not batches:
        return list(batches)
    padded = [list(batch) for batch in batches]
    remainder = len(padded) % world_size
    if remainder == 0:
        return padded
    extra = world_size - remainder
    padded.extend([list(batch) for batch in padded[:extra]])
    return padded


class ScienceQAPrecomputedBatchPlanBatchSampler(torch.utils.data.Sampler):
    """基于预处理 batch_plan 的 batch sampler，支持 DDP 按 batch 切分。"""

    def __init__(
        self,
        batch_plan: List[List[int]],
        shuffle: bool = True,
        seed: int = 0,
        rank: int = 0,
        world_size: int = 1,
    ):
        if not isinstance(batch_plan, list) or not batch_plan:
            raise ValueError("batch_plan 必须是非空列表")
        self.batch_plan = [list(map(int, batch)) for batch in batch_plan]
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.rank = int(rank)
        self.world_size = max(1, int(world_size))
        self.epoch = 0
        self._cached_epoch = None
        self._cached_batches = None

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)
        self._cached_epoch = None
        self._cached_batches = None

    def _build_epoch_batches(self) -> List[List[int]]:
        if self._cached_epoch == self.epoch and self._cached_batches is not None:
            return [list(batch) for batch in self._cached_batches]
        batches = [list(batch) for batch in self.batch_plan]
        if self.shuffle:
            rng = random.Random(self.seed + self.epoch)
            rng.shuffle(batches)
        batches = _pad_batches_for_distributed(batches, self.world_size)
        self._cached_epoch = self.epoch
        self._cached_batches = [list(batch) for batch in batches]
        return [list(batch) for batch in batches]

    def __iter__(self):
        batches = self._build_epoch_batches()
        return iter(batches[self.rank::self.world_size])

    def __len__(self):
        batches = self._build_epoch_batches()
        return len(batches[self.rank::self.world_size])


class DistributedScienceQABucketBatchSampler(torch.utils.data.Sampler):
    """基于 sample_to_bucket 的分布式 batch sampler。"""

    def __init__(
        self,
        sample_to_bucket: Dict[int, str],
        batch_size: int,
        drop_last: bool = False,
        shuffle: bool = True,
        seed: int = 0,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.inner = ScienceQABucketBatchSampler(
            sample_to_bucket=sample_to_bucket,
            batch_size=batch_size,
            drop_last=drop_last,
            shuffle=shuffle,
            seed=seed,
        )
        self.rank = int(rank)
        self.world_size = max(1, int(world_size))
        self._cached_epoch = None
        self._cached_batches = None

    @property
    def batch_size(self):
        return self.inner.batch_size

    @property
    def drop_last(self):
        return self.inner.drop_last

    @property
    def bucket_to_samples(self):
        return self.inner.bucket_to_samples

    def set_epoch(self, epoch: int):
        self.inner.set_epoch(epoch)
        self._cached_epoch = None
        self._cached_batches = None

    def _build_epoch_batches(self) -> List[List[int]]:
        if self._cached_epoch == self.inner.epoch and self._cached_batches is not None:
            return [list(batch) for batch in self._cached_batches]
        batches = [list(batch) for batch in iter(self.inner)]
        batches = _pad_batches_for_distributed(batches, self.world_size)
        self._cached_epoch = self.inner.epoch
        self._cached_batches = [list(batch) for batch in batches]
        return [list(batch) for batch in batches]

    def __iter__(self):
        batches = self._build_epoch_batches()
        return iter(batches[self.rank::self.world_size])

    def __len__(self):
        batches = self._build_epoch_batches()
        return len(batches[self.rank::self.world_size])


def load_scienceqa_preprocessed_buckets(preprocessed_path: str, expected_len: int, args) -> Optional[dict]:
    if not preprocessed_path:
        return None
    if not os.path.exists(preprocessed_path):
        raise FileNotFoundError(f"未找到 ScienceQA 预处理文件: {preprocessed_path}")

    with open(preprocessed_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    config = payload.get("config", {})
    records = payload.get("samples", [])
    raw_batch_plan = payload.get("batch_plan", [])
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
    preprocess_dataset_name = str(config.get("dataset_name", "") or "").strip().lower()
    expected_dataset_name = _normalize_dataset_name(getattr(args, "dataset_name", "scienceqa"))
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
    if preprocess_dataset_name and preprocess_dataset_name != expected_dataset_name:
        raise RuntimeError(
            f"预处理 dataset_name={preprocess_dataset_name} 与训练 dataset_name={expected_dataset_name} 不一致"
        )

    print(
        f"已加载预处理文件: {preprocessed_path}, "
        f"bucket_by={config.get('bucket_by')}, batch_size={config.get('bucket_batch_size')}"
    )

    batch_plan = []
    if isinstance(raw_batch_plan, list):
        for item in raw_batch_plan:
            if not isinstance(item, dict):
                continue
            sample_ids_in_batch = item.get("sample_ids", [])
            if not isinstance(sample_ids_in_batch, list) or not sample_ids_in_batch:
                continue
            normalized_batch = [int(sample_id) for sample_id in sample_ids_in_batch]
            for sample_id in normalized_batch:
                if sample_id not in sample_to_bucket:
                    raise RuntimeError(
                        f"预处理 batch_plan 含未知 sample_id={sample_id}: {preprocessed_path}"
                    )
            batch_plan.append(normalized_batch)

    return {
        "path": preprocessed_path,
        "config": config,
        "sample_to_bucket": sample_to_bucket,
        "batch_plan": batch_plan,
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
    """从学生后端获取 image token id。"""
    model = unwrap_model(model)
    return get_student_image_token_id(
        model,
        backend=getattr(getattr(model, "config", None), "student_model_type", LLAVA_NEXT),
    )


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
    _add(getattr(model, "visual", None))

    get_base_model = getattr(model, "get_base_model", None)
    if callable(get_base_model):
        base_model = get_base_model()
        _add(base_model)
        _add(getattr(base_model, "model", None))
        _add(getattr(base_model, "language_model", None))
        _add(getattr(base_model, "vision_tower", None))
        _add(getattr(base_model, "visual", None))
        inner_model = getattr(base_model, "model", None)
        _add(getattr(inner_model, "language_model", None))
        _add(getattr(inner_model, "vision_tower", None))
        _add(getattr(inner_model, "visual", None))

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
                target.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
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
                target.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
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
    if np is not None:
        state["numpy_random_state"] = np.random.get_state()
    if torch.cuda.is_available():
        state["torch_cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: Optional[dict]):
    if not state:
        return

    python_random_state = state.get("python_random_state")
    if python_random_state is not None:
        random.setstate(python_random_state)

    numpy_random_state = state.get("numpy_random_state")
    if numpy_random_state is not None and np is not None:
        np.random.set_state(numpy_random_state)

    torch_rng_state = state.get("torch_rng_state")
    if torch_rng_state is not None:
        torch.set_rng_state(torch_rng_state)

    cuda_rng_state_all = state.get("torch_cuda_rng_state_all")
    if cuda_rng_state_all is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_rng_state_all)


def _get_trainable_parameter_state(model) -> dict:
    model = unwrap_model(model)
    trainable_state = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_state[name] = param.detach().cpu()
    return trainable_state


def _load_parameter_state(model, parameter_state: dict, state_name: str):
    model = unwrap_model(model)
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


def _to_cpu_obj(obj):
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _to_cpu_obj(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_cpu_obj(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_to_cpu_obj(v) for v in obj)
    return obj


def _move_optimizer_state_to_device_(optimizer, device: str):
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


def save_stage3_checkpoint(
    model,
    save_dir: str,
    remove_vq_codebook: bool = False,
    student_model_type: str = "llava_next",
):
    model = unwrap_model(model)
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir, safe_serialization=False)

    if not remove_vq_codebook:
        save_vq_codebook(model, os.path.join(save_dir, "vq_codebook.pt"))
    _save_projector_state(model, os.path.join(save_dir, "projector.pt"), student_model_type)
    torch.save(
        _get_trainable_parameter_state(model),
        os.path.join(save_dir, "trainable_model_state.pt")
    )
    print(f"Stage3 检查点已保存至 {save_dir}")


def setup_args():
    """设置训练参数"""
    parser = argparse.ArgumentParser(description="VQ-LoRD 训练脚本")
    
    # 模型参数
    parser.add_argument("--model_path", type=str, 
                       default="/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/luye/align_vq/downloads/models/llama3-llava-next-8b-hf",
                       help="LLaVA 模型路径")
    parser.add_argument("--student_model_type", type=str, default="llava_next",
                       choices=["llava_next", "qwen2_vl"],
                       help="学生模型后端类型")
    parser.add_argument("--victim_model", type=str,
                       default="gpt-4-vision-preview",
                       help="教师模型 (API)")
    parser.add_argument("--teacher_api_base", type=str, default="",
                       help="教师 API Base URL（OpenAI 兼容）")
    parser.add_argument("--teacher_api_key", type=str, default="",
                       help="教师 API Key（留空则回退到 OPENAI_API_KEY）")
    
    # VQ 参数
    parser.add_argument("--vq_codebook_size", type=int, default=1024,
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
    parser.add_argument("--remove_vq_codebook", type=int, default=0,
                       help="移除 VQ/codebook，使用学生模型原生视觉链路")
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
    parser.add_argument("--tau_delta", type=float, default=0.01,
                       help="LoRD 冷启动改进阈值: delta11 < tau_delta 时配合 tau1 触发冷启动")
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
    parser.add_argument("--stage3_grad_clip", type=float, default=1.0,
                       help="Stage3 梯度裁剪阈值；<=0 表示关闭")
    parser.add_argument("--stage3_train_projector", type=int, default=0,
                       help="Stage3 是否继续训练 projector，默认冻结以保护 Stage2 建立的视觉桥")
    parser.add_argument("--sub_stage_num", type=int, default=1,
                       help="Stage3 外层 sub-stage 数")
    parser.add_argument("--period_num", type=int, default=0,
                       help="Stage3 每个 sub-stage 的 period 数；<=0 时回退到 epochs")
    parser.add_argument("--sub_set_num", type=int, default=0,
                       help="Stage3 每个 sub-stage 的样本子集大小；<=0 表示全量")
    parser.add_argument("--stage3_eval_max_samples", type=int, default=0,
                       help="Stage3 每个 period 评估样本数；<=0 关闭 period 评估")
    parser.add_argument("--stage3_eval_every_period", type=int, default=1,
                       help="Stage3 每隔多少个 period 做一次评估")
    parser.add_argument("--stage3_eval_scienceqa_split", type=str, default="validation",
                       help="Stage3 独立评估使用的 ScienceQA split（默认 validation）")
    parser.add_argument("--stage3_eval_scienceqa_path", type=str, default="",
                       help="Stage3 独立评估使用的数据集路径（为空则复用 scienceqa_path）")
    parser.add_argument("--stage3_eval_train_num", type=int, default=0,
                       help="Stage3 独立评估样本数上限（0=该 split 全量）")
    parser.add_argument("--stage3_eval_answer_mode", type=str, default="logits",
                       choices=["generate", "logits"],
                       help="Stage3 内置评估口径：generate 或 logits；准确率优先时建议 logits")
    parser.add_argument("--stage3_field_weight_observed", type=float, default=1.0,
                       help="Stage3 教师正则中 observed_facts_visual 的 token 权重")
    parser.add_argument("--stage3_field_weight_context", type=float, default=1.0,
                       help="Stage3 教师正则中 context_textual 的 token 权重")
    parser.add_argument("--stage3_field_weight_reasoning", type=float, default=1.0,
                       help="Stage3 教师正则中 reasoning 的 token 权重")
    parser.add_argument("--stage3_field_weight_answer", type=float, default=1.0,
                       help="Stage3 教师正则中 answer 的 token 权重")
    parser.add_argument("--stage3_obj_weight", type=float, default=1.0,
                       help="Stage3 偏好目标 L_obj 的总权重")
    parser.add_argument("--stage3_reg_weight", type=float, default=1.0,
                       help="Stage3 教师正则 L_reg 的总权重")
    parser.add_argument("--stage3_answer_anchor_weight", type=float, default=1.0,
                       help="Stage3 answer token 直接锚定损失权重（基于 y_vic 的 Answer: X token NLL）")
    parser.add_argument("--stage3_mc_weight", type=float, default=1.0,
                       help="Stage3 选择题主损失权重（基于 Answer: 后下一 token logits 的多选 CE）")
    parser.add_argument("--stage3_pair_use_answer_correctness", type=int, default=1,
                       help="Stage3 Phase-A 是否优先按答案正确性选择 y+/y-，1=启用，0=关闭")
    parser.add_argument("--stage3_wrong_image_enable", type=int, default=0,
                       help="Stage3 是否启用错图负样本约束，默认关闭")
    parser.add_argument("--stage3_wrong_image_weight", type=float, default=0.2,
                       help="Stage3 错图负样本损失权重")
    parser.add_argument("--stage3_wrong_image_margin", type=float, default=0.0,
                       help="Stage3 错图负样本对比边际（token-level log-prob）")
    parser.add_argument("--stage3_force_cold_start_period0", type=int, default=1,
                       help="Stage3 是否在 period0 强制使用 y_vic 作为正样本，1=启用，0=关闭")
    
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
                       choices=["vq_lord", "scienceqa", "aokvqa", "textvqa"],
                       help="训练数据来源 (vq_lord / scienceqa / aokvqa / textvqa)")
    parser.add_argument("--scienceqa_split", type=str, default="train",
                       help="ScienceQA 数据集 split")
    parser.add_argument("--scienceqa_path", type=str, default="ScienceQA",
                       help="ScienceQA 数据集路径（本地路径或 HuggingFace 数据集名）")
    parser.add_argument("--scienceqa_seed", type=int, default=20240306,
                       help="ScienceQA 划分随机种子")
    parser.add_argument("--teacher_cache_path", type=str, default="",
                       help="ScienceQA 教师四字段缓存路径 (json, 建议 *_new.json)")
    parser.add_argument("--sample_only_cached_teacher", type=int, default=0,
                       help="仅从教师缓存中已具备四字段标注的样本抽样 (1=启用,0=关闭)")
    parser.add_argument("--collect_teacher_data", type=int, default=1,
                       help="是否自动采集缺失的教师四字段标注")
    parser.add_argument("--strict_teacher_distill", type=int, default=1,
                       help="兼容旧脚本参数（已废弃，无实际作用；vq_lord3 固定强门禁）")
    parser.add_argument("--teacher_lang", type=str, default="zh", choices=["zh", "en"],
                       help="教师回答统一语言：zh 或 en")
    parser.add_argument("--teacher_observed_max_tokens", type=int, default=256,
                       help="教师字段 observed_facts_visual 的最大 token 预算")
    parser.add_argument("--teacher_context_max_tokens", type=int, default=192,
                       help="教师字段 context_textual 的最大 token 预算")
    parser.add_argument("--teacher_reasoning_max_tokens", type=int, default=256,
                       help="教师字段 reasoning 的最大 token 预算")
    parser.add_argument("--teacher_answer_max_tokens", type=int, default=64,
                       help="教师字段 answer 的最大 token 预算")
    parser.add_argument("--teacher_max_new_tokens_total", type=int, default=768,
                       help="教师 API 采集最大新 token（总上限）")
    parser.add_argument("--stage3_vic_include_context", type=int, default=0,
                       help="Stage3 的 y_vic 是否包含 context_textual（默认0，仅 facts+reasoning+answer）")
    parser.add_argument("--reuse_vq_codebook", type=int, default=1,
                       help="若存在已保存的VQ codebook则复用并跳过Stage1")
    parser.add_argument("--vq_codebook_path", type=str, default="",
                       help="VQ codebook 保存/加载路径 (为空则使用save_path/stage1_vq)")
    parser.add_argument("--reuse_stage2", type=int, default=1,
                       help="若存在已保存的Stage2模型则复用并跳过Stage2")
    parser.add_argument("--stage2_ckpt_path", type=str, default="",
                       help="Stage2 checkpoint 保存/加载路径 (为空则使用save_path/stage2_vision)")
    parser.add_argument("--stage2_resume_path", type=str, default="",
                       help="Stage2 断点目录，非空时从该目录恢复训练")
    parser.add_argument("--stage2_resume_save_path", type=str, default="",
                       help="Stage2 断点保存目录，默认 save_path/stage2_resume_latest")
    parser.add_argument("--stage2_resume_save_optimizer", type=int, default=1,
                       help="Stage2 断点是否保存 optimizer state（1=保存,0=不保存）")
    parser.add_argument("--stage2_resume_save_interval", type=int, default=1,
                       help="Stage2 每隔多少个 epoch 保存一次断点")
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
    parser.add_argument("--stage3_resume_path", type=str, default="",
                       help="Stage3 断点目录，非空时从该目录恢复（包含模型可训练参数/optimizer/RNG/period状态）")
    parser.add_argument("--stage3_resume_save_path", type=str, default="",
                       help="Stage3 断点保存目录，默认 save_path/stage3_resume_latest")
    parser.add_argument("--stage3_resume_save_optimizer", type=int, default=1,
                       help="Stage3 断点是否保存 optimizer state（1=保存,0=不保存，文件更小）")
    parser.add_argument("--stage3_resume_save_interval", type=int, default=1,
                       help="Stage3 每隔多少个 period 保存一次断点")
    parser.add_argument("--stage3_sample_cache_path", type=str, default="",
                       help="Stage3SampleCache 落盘路径；命中且元信息匹配时直接复用")
    
    parser.add_argument("--device", type=str, default="cuda",
                       help="训练设备")
    
    return parser.parse_args()


def load_model_and_processor(args):
    """加载模型和处理器。"""
    args.student_model_type = normalize_student_model_type(
        getattr(args, "student_model_type", LLAVA_NEXT)
    )
    if is_main_process():
        print(f"加载模型: {args.model_path}")
        print(f"学生模型后端: {args.student_model_type}")
    model, processor = load_student_model_and_processor(args)
    processor.student_model_type = args.student_model_type
    return model, processor


def add_vq_to_model(model, args):
    """为学生模型添加 VQ 层。"""
    print("添加 VQ 离散化层...")
    model = add_vq_to_student_model(model, args)
    print(
        f"VQ 层已添加，codebook 大小: {args.vq_codebook_size}, "
        f"quantizer=VectorQuantizer2(legacy={bool(args.vq_legacy_loss)}), "
        f"dead_code_threshold={args.vq_dead_code_threshold}, "
        f"reset_interval={args.vq_dead_code_reset_interval}"
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
        target_modules=lora_target_modules(getattr(args, "student_model_type", LLAVA_NEXT)),
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model


def save_checkpoint(model, args, suffix=""):
    """保存检查点"""
    model = unwrap_model(model)
    save_path = os.path.join(args.save_path, suffix)
    os.makedirs(save_path, exist_ok=True)
    
    model.save_pretrained(save_path, safe_serialization=False)
    print(f"模型已保存至 {save_path}")


def _get_vq_state_path(codebook_path: str) -> str:
    return os.path.join(os.path.dirname(codebook_path), "vq_encoder_state.pt")


def save_vq_codebook(model, codebook_path: str):
    model = unwrap_model(model)
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
    model = unwrap_model(model)
    codebook = torch.load(codebook_path, map_location="cpu")
    model.vq_vision_encoder.vq.embedding.weight.data.copy_(codebook)
    vq_state_path = _get_vq_state_path(codebook_path)
    if os.path.exists(vq_state_path):
        vq_state = torch.load(vq_state_path, map_location="cpu")
        model.vq_vision_encoder.load_vq_state(vq_state)
    print(f"已加载 VQ codebook: {codebook_path}")


def _extract_projector_state(model, student_model_type: str) -> dict:
    model = unwrap_model(model)
    projector_state = {}
    for name, tensor in model.state_dict().items():
        if is_projector_param(name, student_model_type):
            projector_state[name] = tensor.detach().cpu()
    return projector_state


def _save_projector_state(model, projector_path: str, student_model_type: str) -> int:
    projector_state = _extract_projector_state(model, student_model_type)
    if not projector_state:
        print(f"[Warn] 未找到 projector 参数，跳过保存: {projector_path}")
        return 0
    torch.save(projector_state, projector_path)
    print(f"已保存 projector 权重: {projector_path} (params={len(projector_state)})")
    return len(projector_state)


def _load_projector_state(model, projector_path: str) -> Tuple[int, int]:
    model = unwrap_model(model)
    if not os.path.exists(projector_path):
        print(f"[Warn] 未找到 projector 权重文件，跳过加载: {projector_path}")
        return 0, 0

    state = torch.load(projector_path, map_location="cpu")
    model_keys = set(model.state_dict().keys())
    loadable_state = {k: v for k, v in state.items() if k in model_keys}
    loaded = len(loadable_state)
    skipped = max(0, len(state) - loaded)
    if loadable_state:
        model.load_state_dict(loadable_state, strict=False)
    print(f"已加载 projector 权重: {projector_path} (loaded={loaded}, skipped={skipped})")
    return loaded, skipped


def save_stage2_checkpoint(model, args, ckpt_path: str):
    """保存 Stage2 检查点（包括 VQ codebook 和 LoRA 权重）"""
    model = unwrap_model(model)
    os.makedirs(ckpt_path, exist_ok=True)

    remove_vq_codebook = bool(getattr(args, "remove_vq_codebook", 0))
    if not remove_vq_codebook:
        save_vq_codebook(model, os.path.join(ckpt_path, "vq_codebook.pt"))

    projector_path = os.path.join(ckpt_path, "projector.pt")
    projector_param_count = _save_projector_state(
        model,
        projector_path,
        str(getattr(args, "student_model_type", "llava_next")),
    )
    
    # 保存 LoRA 适配器
    model.save_pretrained(ckpt_path, safe_serialization=False)
    
    # 保存配置信息
    import json
    config_info = {
        "vq_codebook_size": args.vq_codebook_size,
        "remove_vq_codebook": int(remove_vq_codebook),
        "student_model_type": str(getattr(args, "student_model_type", "llava_next")),
        "model_path": str(getattr(args, "model_path", "")),
        "freeze_vision_tower": args.freeze_vision_tower,
        "beta": args.beta,
        "stage": 2,
        "stage2_answer_weight": float(args.stage2_answer_weight),
        "stage2_rationale_weight": float(args.stage2_rationale_weight),
        "stage2_train_lora": 1,
        "stage2_train_codebook": 0,
        "stage2_answer_format": "letter",
        "projector_state_path": "projector.pt",
        "projector_state_param_count": int(projector_param_count),
    }
    with open(os.path.join(ckpt_path, "stage2_config.json"), "w") as f:
        json.dump(config_info, f, indent=2)
    
    print(f"Stage2 检查点已保存至 {ckpt_path}")


def load_stage2_checkpoint(model, ckpt_path: str, remove_vq_codebook: bool = False):
    """加载 Stage2 检查点"""
    model = unwrap_model(model)
    vq_path = os.path.join(ckpt_path, "vq_codebook.pt")
    if not remove_vq_codebook and os.path.exists(vq_path):
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
        if state_dict is None:
            raise RuntimeError(
                f"检测到 adapter_config.json，但缺少 adapter 权重文件: {ckpt_path}"
            )
        # 直接覆盖现有 default adapter，避免额外 adapter name 干扰 Stage3 参数遍历。
        model.load_state_dict(state_dict, strict=False)
        print(f"已加载 LoRA 权重: {ckpt_path}")

    projector_path = os.path.join(ckpt_path, "projector.pt")
    _load_projector_state(model, projector_path)

    config_path = os.path.join(ckpt_path, "stage2_config.json")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            stage2_cfg = json.load(f)
        print(
            f"已读取 Stage2 配置: stage={stage2_cfg.get('stage')}, "
            f"answer_w={stage2_cfg.get('stage2_answer_weight')}, "
            f"rationale_w={stage2_cfg.get('stage2_rationale_weight')}"
        )
    else:
        print(f"[Warn] 未找到 stage2_config.json: {ckpt_path}")
    
    print(f"Stage2 检查点加载完成")
    return model


def _stage2_contract_missing_files(ckpt_path: str, require_vq: bool = True) -> List[str]:
    required_files = [
        "projector.pt",
        "stage2_config.json",
        "adapter_config.json",
    ]
    if require_vq:
        required_files.insert(0, "vq_codebook.pt")
    missing = []
    for rel in required_files:
        if not os.path.exists(os.path.join(ckpt_path, rel)):
            missing.append(rel)

    has_adapter_weights = (
        os.path.exists(os.path.join(ckpt_path, "adapter_model.safetensors"))
        or os.path.exists(os.path.join(ckpt_path, "adapter_model.bin"))
    )
    if not has_adapter_weights:
        missing.append("adapter_model.safetensors|adapter_model.bin")
    return missing


def wrap_model_for_ddp(model, args, find_unused_parameters: bool = False):
    if not getattr(args, "distributed", False):
        return model
    if isinstance(model, DDP):
        return model
    return DDP(
        model,
        device_ids=[args.local_rank],
        output_device=args.local_rank,
        # VQ EMA/usage 这类 buffer 不应依赖 DDP 广播来维持一致；
        # Stage2/Stage3 在 codebook 冻结时会显式禁止其更新。
        broadcast_buffers=False,
        find_unused_parameters=bool(find_unused_parameters),
    )


def wrap_model_for_stage2_ddp(model, args):
    return wrap_model_for_ddp(model, args, find_unused_parameters=False)


def wrap_model_for_stage3_ddp(model, args):
    return wrap_model_for_ddp(model, args, find_unused_parameters=True)



# Stage-specific training logic lives in separate modules for readability.
from vq_lord_stage1 import train_stage1_vq
from vq_lord_stage2 import (
    _get_stage2_vq_loss,
    _get_stage2_vq_stats,
    train_stage2_vision,
)
from vq_lord_stage3 import (
    PeriodState,
    PeriodTrainingDataset,
    PeriodTrainingItem,
    Stage3SampleCacheItem,
    _bootstrap_period_states,
    _build_pairs_and_next_states,
    _build_stage3_choice_token_map,
    _build_stage3_sample_cache,
    _build_stage3_static_batches,
    _build_vic_field_masks,
    _collate_period_training,
    _compute_choice_scores_from_prompt_batch,
    _compute_token_log_probs,
    _decode_generated_text_from_ids,
    _deserialize_period_states,
    _evaluate_stage3_answer_metrics,
    _extract_answer_letter,
    _extract_answer_letter_from_ids,
    _find_token_subsequence,
    _generate_candidate_batch,
    _generate_candidate_single,
    _get_letter_token_candidates,
    _get_stage3_phase_a_batch_size,
    _left_pad_and_stack_1d_tensors,
    _load_stage3_resume_state,
    _load_stage3_sample_cache,
    _pad_1d_tensor,
    _pad_and_stack_1d_tensors,
    _predict_choice_from_choice_scores,
    _resolve_stage3_eval_sample_cache_path,
    _resolve_stage3_sample_cache_path,
    _run_one_period_train,
    _save_stage3_resume_state,
    _save_stage3_sample_cache,
    _score_sequences_no_grad_batch,
    _serialize_period_states,
    _set_stage3_trainable_params,
    _stack_optional_image_sizes,
    _stack_padded_pixel_values,
    _stage3_static_sort_key,
    _strip_extra_image_tokens,
    log_clip,
    train_stage3_lord,
)

def main():
    args = setup_args()
    tb_writer = None
    init_distributed_mode(args)

    try:
        if (
            args.distributed
            and args.stage >= 1
            and not bool(args.reuse_vq_codebook)
            and not bool(args.remove_vq_codebook)
        ):
            raise RuntimeError("当前未实现 Stage1 多卡训练；请先单卡准备 Stage1 codebook，再用多卡执行 Stage2/Stage3。")

        if is_main_process():
            print("=" * 60)
            print("VQ-LoRD 训练")
            print("=" * 60)
            pprint(vars(args))
            print("=" * 60)
            if args.remove_vq_codebook:
                print("[Info] remove_vq_codebook=1，使用学生模型原生视觉链路并跳过 Stage1。")
            if args.reuse_vq_codebook:
                print("[Info] reuse_vq_codebook=1，若存在 stage1 codebook 将跳过 Stage1。")
            if args.reuse_stage2:
                print("[Info] reuse_stage2=1，若存在 stage2 ckpt 将跳过 Stage2。")

        os.makedirs(args.save_path, exist_ok=True)
        if not args.vq_codebook_path:
            args.vq_codebook_path = os.path.join(args.save_path, "stage1_vq", "vq_codebook.pt")
        if is_main_process():
            tb_writer = SummaryWriter(log_dir=os.path.join(args.save_path, "logs"))

        train_samples = None
        train_bucket_meta = None
        if _is_hf_multichoice_dataset(args.dataset_name):
            allowed_source_indices = None
            if bool(int(args.sample_only_cached_teacher)):
                allowed_source_indices = _load_cached_teacher_source_indices(
                    cache_path=args.teacher_cache_path,
                    dataset_name=args.dataset_name,
                    split=args.scienceqa_split,
                )
                if is_main_process():
                    print(
                        f"[SampleFilter] sample_only_cached_teacher=1, "
                        f"cached_valid_source_indices={len(allowed_source_indices)}"
                    )
                if len(allowed_source_indices) == 0:
                    raise RuntimeError(
                        "sample_only_cached_teacher=1，但缓存中没有可用四字段样本"
                    )
            if is_main_process():
                print(f"\n加载 {args.dataset_name} 样本（预加载阶段，不加载学生模型）...")
            train_samples = build_scienceqa_samples(
                scienceqa_path=args.scienceqa_path,
                split=args.scienceqa_split,
                train_num=args.train_num,
                seed=args.scienceqa_seed,
                dataset_name=args.dataset_name,
                allowed_source_indices=allowed_source_indices,
            )

            if int(args.stage) >= 2:
                if args.distributed:
                    if is_main_process():
                        print(f"加载 {args.dataset_name} 教师回答（stage>=2, rank0 负责补齐缓存）...")
                        train_samples = attach_gpt4v_teacher_responses(train_samples, args)
                    barrier_if_distributed()
                    if not is_main_process():
                        original_collect = args.collect_teacher_data
                        args.collect_teacher_data = 0
                        train_samples = attach_gpt4v_teacher_responses(train_samples, args)
                        args.collect_teacher_data = original_collect
                else:
                    if is_main_process():
                        print(f"加载 {args.dataset_name} 教师回答（stage>=2）...")
                    train_samples = attach_gpt4v_teacher_responses(train_samples, args)
            elif is_main_process():
                print("stage<2，跳过教师回答采集/补齐")

            train_bucket_meta = load_scienceqa_preprocessed_buckets(
                args.scienceqa_preprocessed_path,
                expected_len=len(train_samples),
                args=args,
            )

        model, processor = load_model_and_processor(args)
        model._runtime_args = args
        if not bool(args.remove_vq_codebook):
            model = add_vq_to_model(model, args)
        model._runtime_args = args
        if args.use_lora:
            model = apply_lora(model, args)

        if is_main_process():
            print("\n加载训练数据...")
        if _is_hf_multichoice_dataset(args.dataset_name):
            train_dataset = ScienceQADataset(
                processor=processor,
                scienceqa_path=args.scienceqa_path,
                dataset_name=args.dataset_name,
                max_length=args.max_length,
                samples=train_samples,
                seed=args.scienceqa_seed,
                teacher_lang=args.teacher_lang,
                teacher_observed_max_tokens=args.teacher_observed_max_tokens,
                teacher_context_max_tokens=args.teacher_context_max_tokens,
                teacher_reasoning_max_tokens=args.teacher_reasoning_max_tokens,
                teacher_answer_max_tokens=args.teacher_answer_max_tokens,
                stage3_vic_include_context=bool(int(args.stage3_vic_include_context)),
                require_teacher_annotation=bool(int(args.stage) >= 2),
            )
        else:
            collector = GPT4VDataCollector(save_dir=args.data_dir)
            visual_qa_data = collector.load_collected_data("visual_qa_data.json")
            image_descriptions = collector.load_collected_data("image_descriptions.json")
            if not visual_qa_data and not image_descriptions:
                print("警告：未找到预收集的数据，请先运行数据收集脚本")
                print("示例：python data_collector2.py --collect_data")
                return

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

            image_sizes_list = [b.get("image_sizes") for b in batch]
            for idx, size in enumerate(image_sizes_list):
                if size is None:
                    pixel_values = batch[idx].get("pixel_values")
                    if isinstance(pixel_values, torch.Tensor) and pixel_values.dim() >= 3:
                        height = int(pixel_values.shape[-2])
                        width = int(pixel_values.shape[-1])
                        image_sizes_list[idx] = torch.tensor([height, width], dtype=torch.long)

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
                _is_hf_multichoice_dataset(args.dataset_name)
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

                if args.distributed and stage_id == 2:
                    batch_sampler = DistributedScienceQABucketBatchSampler(
                        sample_to_bucket=train_bucket_meta["sample_to_bucket"],
                        batch_size=bucket_batch_size,
                        drop_last=bool(config.get("bucket_drop_last", False)),
                        shuffle=bool(config.get("shuffle", True)),
                        seed=int(config.get("seed", args.scienceqa_seed)),
                        rank=args.rank,
                        world_size=args.world_size,
                    )
                    if is_main_process():
                        print(
                            f"Stage{stage_id} 使用 sample_to_bucket 分布式分桶采样: "
                            f"path={train_bucket_meta['path']}, batch_size={batch_sampler.batch_size}, "
                            f"local_batches={len(batch_sampler)}"
                        )
                    return DataLoader(
                        train_dataset,
                        batch_sampler=batch_sampler,
                        num_workers=0,
                        collate_fn=vq_lord_collate,
                    )

                batch_sampler = ScienceQABucketBatchSampler(
                    sample_to_bucket=train_bucket_meta["sample_to_bucket"],
                    batch_size=bucket_batch_size,
                    drop_last=bool(config.get("bucket_drop_last", False)),
                    shuffle=bool(config.get("shuffle", True)),
                    seed=int(config.get("seed", args.scienceqa_seed)),
                )
                if is_main_process():
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

            if args.distributed and stage_id == 2:
                sampler = DistributedSampler(
                    train_dataset,
                    num_replicas=args.world_size,
                    rank=args.rank,
                    shuffle=True,
                    drop_last=False,
                )
                if is_main_process():
                    print(
                        f"Stage{stage_id} 使用常规 DistributedSampler: "
                        f"per_rank_batch_size={args.batch_size}, local_num_samples={len(sampler)}"
                    )
                return DataLoader(
                    train_dataset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    sampler=sampler,
                    num_workers=0,
                    collate_fn=vq_lord_collate,
                )

            if is_main_process():
                print(f"Stage{stage_id} 使用常规 DataLoader: batch_size={args.batch_size}, shuffle=True")
            return DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=0,
                collate_fn=vq_lord_collate,
            )

        if is_main_process():
            print(f"训练数据量: {len(train_dataset)}")

        if args.stage >= 1 and not bool(args.remove_vq_codebook):
            stage1_dataloader = build_train_dataloader(stage_id=1)
            if args.distributed and not os.path.exists(args.vq_codebook_path):
                raise RuntimeError(
                    "当前为 Stage2 多卡模式，但未找到可复用的 Stage1 codebook。"
                    f"请先单卡完成 Stage1，期望路径: {args.vq_codebook_path}"
                )
            if args.reuse_vq_codebook and os.path.exists(args.vq_codebook_path):
                if is_main_process():
                    print(f"检测到VQ codebook，跳过Stage1并加载: {args.vq_codebook_path}")
                load_vq_codebook(model, args.vq_codebook_path)
            else:
                model = train_stage1_vq(model, stage1_dataloader, args, tb_writer)
                if is_main_process():
                    save_stage1_checkpoint(model, args)
                barrier_if_distributed()

        if args.stage >= 2:
            stage2_dataloader = build_train_dataloader(stage_id=2)
            if not args.stage2_ckpt_path:
                args.stage2_ckpt_path = os.path.join(args.save_path, "stage2_vision")

            has_stage2_resume = bool(str(args.stage2_resume_path).strip())
            if has_stage2_resume:
                if is_main_process():
                    print(f"检测到Stage2 resume路径，优先从断点恢复训练: {args.stage2_resume_path}")
                model = wrap_model_for_stage2_ddp(model, args)
                model = train_stage2_vision(model, stage2_dataloader, args, tb_writer)
                model = unwrap_model(model)
                barrier_if_distributed()
                if is_main_process():
                    save_stage2_checkpoint(model, args, args.stage2_ckpt_path)
                    save_checkpoint(model, args, "stage2_vision")
                barrier_if_distributed()
            elif (
                args.reuse_stage2
                and os.path.exists(args.stage2_ckpt_path)
                and (
                    bool(args.remove_vq_codebook)
                    or os.path.exists(os.path.join(args.stage2_ckpt_path, "vq_codebook.pt"))
                )
            ):
                if is_main_process():
                    print(f"检测到Stage2检查点，跳过Stage2并加载: {args.stage2_ckpt_path}")
                model = load_stage2_checkpoint(
                    model,
                    args.stage2_ckpt_path,
                    remove_vq_codebook=bool(args.remove_vq_codebook),
                )
            else:
                model = wrap_model_for_stage2_ddp(model, args)
                model = train_stage2_vision(model, stage2_dataloader, args, tb_writer)
                model = unwrap_model(model)
                barrier_if_distributed()
                if is_main_process():
                    save_stage2_checkpoint(model, args, args.stage2_ckpt_path)
                    save_checkpoint(model, args, "stage2_vision")
                barrier_if_distributed()

        if args.stage >= 3:
            if not args.stage2_ckpt_path:
                args.stage2_ckpt_path = os.path.join(args.save_path, "stage2_vision")
            if not os.path.isdir(args.stage2_ckpt_path):
                raise RuntimeError(f"Stage3 需要可用的 Stage2 checkpoint 目录: {args.stage2_ckpt_path}")
            missing_stage2_files = _stage2_contract_missing_files(
                args.stage2_ckpt_path,
                require_vq=not bool(args.remove_vq_codebook),
            )
            if missing_stage2_files:
                raise RuntimeError(
                    "Stage3 启动前 Stage2 工件不完整，缺失: "
                    + ", ".join(missing_stage2_files)
                )
            model = wrap_model_for_stage3_ddp(model, args)
            model = train_stage3_lord(model, train_dataset, args, tb_writer, train_bucket_meta=train_bucket_meta)
            model = unwrap_model(model)
            barrier_if_distributed()
            if is_main_process():
                save_stage3_checkpoint(
                    model,
                    os.path.join(args.save_path, "stage3_lord_final"),
                    remove_vq_codebook=bool(args.remove_vq_codebook),
                    student_model_type=str(getattr(args, "student_model_type", "llava_next")),
                )
            barrier_if_distributed()

        if is_main_process():
            print("\n" + "=" * 60)
            print("VQ-LoRD 训练完成!")
            print("=" * 60)
    finally:
        if tb_writer is not None:
            tb_writer.close()
        cleanup_distributed()


if __name__ == "__main__":
    main()
