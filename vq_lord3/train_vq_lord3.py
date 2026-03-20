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
from dataclasses import dataclass
import torch
import argparse
from collections import defaultdict
from typing import Optional, List, Dict, Tuple
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
try:
    from .data_collector2 import (
        GPT4VDataCollector,
        VQLORDDataset,
        VisualQAItem,
        ImageDescriptionItem,
    )
except Exception:
    from data_collector2 import (
        GPT4VDataCollector,
        VQLORDDataset,
        VisualQAItem,
        ImageDescriptionItem,
    )

import torch.nn.functional as F

try:
    import numpy as np
except Exception:  # pragma: no cover - numpy 不是硬依赖
    np = None


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
        hint = item.get("hint", "")
        lecture = item.get("lecture", "")
        solution = item.get("solution", "")

        hint_block = f"Hint: {hint}\n" if hint else ""
        instruction = f"<image>\nQuestion: {question}\n{hint_block}Options:\n{choices_text}Answer:"
        if lecture:
            response = f"Explanation: {lecture}\nSolution: {solution}\nAnswer: {answer}"
        else:
            response = f"Solution: {solution}\nAnswer: {answer}"

        samples.append({
            "sample_id": sampled_pos,
            "source_index": dataset_idx,
            "split": split,
            "image": item.get("image"),
            "question": question,
            "hint": hint,
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
    source_index = sample.get("source_index")
    if source_index is not None:
        split_name = str(sample.get("split") or "unknown")
        return f"scienceqa::{split_name}::{int(source_index)}"

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


def _extract_json_payload(text: str) -> Optional[dict]:
    if not isinstance(text, str):
        return None
    raw = text.strip()
    if not raw:
        return None
    cleaned = raw
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = cleaned.strip()
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        frag = cleaned[start:end + 1]
        try:
            parsed = json.loads(frag)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return None
    return None


def _normalize_loaded_cache_keys(cache_data: dict, split: str) -> Dict[str, dict]:
    if not isinstance(cache_data, dict):
        return {}

    normalized = {}
    normalized_priority = {}
    stable_prefix = f"scienceqa::{split}::"

    for old_key, old_value in cache_data.items():
        if not isinstance(old_value, dict):
            continue

        if isinstance(old_key, str) and old_key.startswith(stable_prefix):
            new_key = old_key
            source_priority = 2
        else:
            meta = old_value.get("meta", {}) if isinstance(old_value.get("meta", {}), dict) else {}
            source_index = meta.get("source_index")
            if source_index is None:
                continue
            new_key = f"scienceqa::{split}::{int(source_index)}"
            source_priority = 1

        if new_key in normalized and source_priority <= normalized_priority.get(new_key, 0):
            continue
        normalized[new_key] = old_value
        normalized_priority[new_key] = source_priority

    return normalized


def _normalize_teacher_annotation(
    payload: Optional[dict],
    budget: dict,
) -> Optional[dict]:
    if not isinstance(payload, dict):
        return None

    canonical = {
        "format_version": TEACHER_SCHEMA_VERSION,
        "observed_facts_visual": payload.get("observed_facts_visual", payload.get("observed_facts", "")),
        "context_textual": payload.get("context_textual", payload.get("context", "")),
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
        "teacher_observed_max_tokens": int(getattr(args, "teacher_observed_max_tokens", 256)),
        "teacher_context_max_tokens": int(getattr(args, "teacher_context_max_tokens", 192)),
        "teacher_reasoning_max_tokens": int(getattr(args, "teacher_reasoning_max_tokens", 256)),
        "teacher_answer_max_tokens": int(getattr(args, "teacher_answer_max_tokens", 64)),
        "teacher_max_new_tokens_total": int(getattr(args, "teacher_max_new_tokens_total", 768)),
    }

    cache_path = args.teacher_cache_path
    if not cache_path:
        victim_tag = _safe_name(args.victim_model)
        cache_name = (
            f"scienceqa_teacher_{victim_tag}_{args.scienceqa_split}_"
            f"n{args.train_num}_seed{args.scienceqa_seed}_new.json"
        )
        cache_path = os.path.join(args.data_dir, cache_name)

    cache_data = {}
    teacher_lang = getattr(args, "teacher_lang", "zh")
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if isinstance(payload, dict):
                raw_cache_data = payload.get("samples", {}) or {}
                cache_data = _normalize_loaded_cache_keys(raw_cache_data, args.scienceqa_split)
                if payload.get("format_version") not in (None, TEACHER_SCHEMA_VERSION):
                    print(
                        f"警告: 教师缓存 format_version={payload.get('format_version')}，"
                        f"当前期望={TEACHER_SCHEMA_VERSION}。将按缺失样本重新采集。"
                    )
            print(f"已加载教师缓存(v2): {cache_path}, 条目数={len(cache_data)}")
        except Exception as e:
            print(f"警告: 教师缓存读取失败，将忽略缓存 ({e})")
            cache_data = {}

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
                    parsed = _extract_json_payload(teacher_response)
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
        teacher_observed_max_tokens: int = 256,
        teacher_context_max_tokens: int = 192,
        teacher_reasoning_max_tokens: int = 256,
        teacher_answer_max_tokens: int = 64,
        stage3_vic_include_context: bool = False,
        require_teacher_annotation: bool = False,
    ):
        self.processor = processor
        self.max_length = max_length
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
        answer_letter = item.get("answer_letter", "A")
        if not isinstance(answer_letter, str) or len(answer_letter) == 0:
            answer_letter = "A"
        answer_letter = answer_letter.strip().upper()[0]

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
            # Stage1 或未启用教师采集时，允许回退到旧 response（仅用于保持训练流程可跑）。
            fallback_response = _strip_image_tokens(item.get("response", ""))
            fallback_rationale = _extract_teacher_rationale(fallback_response)
            if fallback_rationale:
                rationale_target = f"Explanation: {fallback_rationale}\n{answer_target}"
            else:
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
        image = item["image"]
        image_sizes_default = torch.tensor([image.height, image.width], dtype=torch.long)

        instruction = item.get("instruction", "")
        instruction_text = instruction.replace("<image>", "").replace("< image >", "").replace("<Image>", "").strip()
        answer_target, rationale_target, _, has_rationale = self._build_targets(item, instruction_text)

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


def save_stage3_checkpoint(model, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir, safe_serialization=False)

    save_vq_codebook(model, os.path.join(save_dir, "vq_codebook.pt"))
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
    parser.add_argument("--stage3_field_weight_observed", type=float, default=1.0,
                       help="Stage3 教师正则中 observed_facts_visual 的 token 权重")
    parser.add_argument("--stage3_field_weight_context", type=float, default=1.0,
                       help="Stage3 教师正则中 context_textual 的 token 权重")
    parser.add_argument("--stage3_field_weight_reasoning", type=float, default=1.0,
                       help="Stage3 教师正则中 reasoning 的 token 权重")
    parser.add_argument("--stage3_field_weight_answer", type=float, default=1.0,
                       help="Stage3 教师正则中 answer 的 token 权重")
    parser.add_argument("--stage3_wrong_image_enable", type=int, default=0,
                       help="Stage3 是否启用错图负样本约束，默认关闭")
    parser.add_argument("--stage3_wrong_image_weight", type=float, default=0.2,
                       help="Stage3 错图负样本损失权重")
    parser.add_argument("--stage3_wrong_image_margin", type=float, default=0.0,
                       help="Stage3 错图负样本对比边际（token-level log-prob）")
    
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
                       help="ScienceQA 教师四字段缓存路径 (json, 建议 *_new.json)")
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


def _compute_token_log_probs(
    model,
    ids: torch.Tensor,
    mask: torch.Tensor,
    pixel_values: torch.Tensor,
    image_sizes: Optional[torch.Tensor],
    prompt_lens: torch.Tensor,
):
    """返回 token-level log-prob 与生成段 mask。"""
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

    return token_log_probs, token_mask


@dataclass
class Stage3SampleCacheItem:
    sample_idx: int
    prompt_ids: torch.Tensor
    prompt_mask: torch.Tensor
    y_vic_ids: torch.Tensor
    y_vic_mask: torch.Tensor
    pixel_values: torch.Tensor
    image_sizes: Optional[torch.Tensor]
    prompt_len: int
    observed_ids: Optional[torch.Tensor] = None
    context_ids: Optional[torch.Tensor] = None
    reasoning_ids: Optional[torch.Tensor] = None
    answer_ids: Optional[torch.Tensor] = None
    vic_observed_mask: Optional[torch.Tensor] = None
    vic_context_mask: Optional[torch.Tensor] = None
    vic_reasoning_mask: Optional[torch.Tensor] = None
    vic_answer_mask: Optional[torch.Tensor] = None
    vic_other_mask: Optional[torch.Tensor] = None


@dataclass
class PeriodState:
    sample_idx: int
    y11_ids: torch.Tensor
    y11_mask: torch.Tensor
    y12_ids: torch.Tensor
    y12_mask: torch.Tensor
    avg_lp_11: float
    avg_lp_12: float
    prob_11: float
    prob_12: float


@dataclass
class PeriodTrainingItem:
    sample_idx: int
    y_plus_ids: torch.Tensor
    y_plus_mask: torch.Tensor
    y_minus_ids: torch.Tensor
    y_minus_mask: torch.Tensor
    y_vic_ids: torch.Tensor
    y_vic_mask: torch.Tensor
    old_token_lp_plus: torch.Tensor
    old_token_mask_plus: torch.Tensor
    old_token_lp_minus: torch.Tensor
    old_token_mask_minus: torch.Tensor
    old_token_lp_vic: torch.Tensor
    old_token_mask_vic: torch.Tensor
    wrong_sample_idx: int


def _serialize_period_states(period_states: Optional[Dict[int, PeriodState]]) -> Optional[dict]:
    if period_states is None:
        return None
    payload = {}
    for sample_idx, state in period_states.items():
        payload[int(sample_idx)] = {
            "sample_idx": int(state.sample_idx),
            "y11_ids": state.y11_ids.detach().cpu(),
            "y11_mask": state.y11_mask.detach().cpu(),
            "y12_ids": state.y12_ids.detach().cpu(),
            "y12_mask": state.y12_mask.detach().cpu(),
            "avg_lp_11": float(state.avg_lp_11),
            "avg_lp_12": float(state.avg_lp_12),
            "prob_11": float(state.prob_11),
            "prob_12": float(state.prob_12),
        }
    return payload


def _deserialize_period_states(payload: Optional[dict]) -> Optional[Dict[int, PeriodState]]:
    if payload is None:
        return None
    states: Dict[int, PeriodState] = {}
    for key, item in payload.items():
        sample_idx = int(item.get("sample_idx", key))
        states[sample_idx] = PeriodState(
            sample_idx=sample_idx,
            y11_ids=item["y11_ids"].detach().cpu().long(),
            y11_mask=item["y11_mask"].detach().cpu().long(),
            y12_ids=item["y12_ids"].detach().cpu().long(),
            y12_mask=item["y12_mask"].detach().cpu().long(),
            avg_lp_11=float(item["avg_lp_11"]),
            avg_lp_12=float(item["avg_lp_12"]),
            prob_11=float(item["prob_11"]),
            prob_12=float(item["prob_12"]),
        )
    return states


def _save_stage3_resume_state(
    model,
    optimizer,
    resume_dir: str,
    progress: dict,
    include_optimizer_state: bool = True,
):
    os.makedirs(resume_dir, exist_ok=True)
    payload = {
        "trainable_model_state": _get_trainable_parameter_state(model),
        "optimizer_state": _to_cpu_obj(optimizer.state_dict()) if include_optimizer_state else None,
        "rng_state": _capture_rng_state(),
        "progress": progress,
    }
    torch.save(payload, os.path.join(resume_dir, "stage3_resume_state.pt"))
    meta = dict(progress)
    meta["has_period_states"] = bool(meta.get("period_states") is not None)
    meta["has_optimizer_state"] = bool(include_optimizer_state)
    meta.pop("period_states", None)
    with open(os.path.join(resume_dir, "stage3_resume_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[Stage3][Resume] 已保存断点: {resume_dir}")


def _load_stage3_resume_state(
    model,
    optimizer,
    resume_dir: str,
    device: str,
) -> Optional[dict]:
    state_path = os.path.join(resume_dir, "stage3_resume_state.pt")
    if not os.path.exists(state_path):
        print(f"[Stage3][Resume] 未找到断点文件: {state_path}")
        return None

    payload = torch.load(state_path, map_location="cpu")
    _load_parameter_state(model, payload.get("trainable_model_state", {}), "Stage3 resume trainable_model_state")

    optimizer_state = payload.get("optimizer_state")
    if optimizer_state:
        optimizer.load_state_dict(optimizer_state)
        _move_optimizer_state_to_device_(optimizer, device)
        print("[Stage3][Resume] 已加载 optimizer state")
    else:
        print("[Stage3][Resume] 断点中不含 optimizer state，将从当前优化器状态继续")

    _restore_rng_state(payload.get("rng_state"))
    progress = payload.get("progress", {})
    print(
        f"[Stage3][Resume] 已恢复进度: next_sub_stage={progress.get('next_sub_stage_idx', 0)}, "
        f"next_period={progress.get('next_period_idx', 0)}, global_step={progress.get('global_step', 0)}"
    )
    return progress


class PeriodTrainingDataset(torch.utils.data.Dataset):
    def __init__(self, items: List[PeriodTrainingItem], sample_cache: List[Stage3SampleCacheItem]):
        self.items = items
        self.sample_cache = sample_cache

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        pair = self.items[idx]
        sample = self.sample_cache[pair.sample_idx]
        wrong_sample = self.sample_cache[pair.wrong_sample_idx]
        return {
            "y_plus_ids": pair.y_plus_ids,
            "y_plus_mask": pair.y_plus_mask,
            "y_minus_ids": pair.y_minus_ids,
            "y_minus_mask": pair.y_minus_mask,
            "y_vic_ids": pair.y_vic_ids,
            "y_vic_mask": pair.y_vic_mask,
            "old_token_lp_plus": pair.old_token_lp_plus,
            "old_token_mask_plus": pair.old_token_mask_plus,
            "old_token_lp_minus": pair.old_token_lp_minus,
            "old_token_mask_minus": pair.old_token_mask_minus,
            "old_token_lp_vic": pair.old_token_lp_vic,
            "old_token_mask_vic": pair.old_token_mask_vic,
            "pixel_values": sample.pixel_values,
            "image_sizes": sample.image_sizes,
            "wrong_pixel_values": wrong_sample.pixel_values,
            "wrong_image_sizes": wrong_sample.image_sizes,
            "vic_observed_mask": sample.vic_observed_mask,
            "vic_context_mask": sample.vic_context_mask,
            "vic_reasoning_mask": sample.vic_reasoning_mask,
            "vic_answer_mask": sample.vic_answer_mask,
            "vic_other_mask": sample.vic_other_mask,
            "prompt_len": int(sample.prompt_len),
        }


def _pad_1d_tensor(tensor: torch.Tensor, target_len: int, pad_value: float) -> torch.Tensor:
    cur_len = int(tensor.shape[0])
    if cur_len == target_len:
        return tensor
    if cur_len > target_len:
        return tensor[:target_len]
    return F.pad(tensor, (0, target_len - cur_len), value=pad_value)


def _collate_period_training(batch: List[dict], pad_token_id: int) -> dict:
    if not batch:
        return None

    stream_to_old_key = {
        "y_plus": ("old_token_lp_plus", "old_token_mask_plus"),
        "y_minus": ("old_token_lp_minus", "old_token_mask_minus"),
        "y_vic": ("old_token_lp_vic", "old_token_mask_vic"),
    }

    def _collate_stream(prefix: str):
        ids_key = f"{prefix}_ids"
        mask_key = f"{prefix}_mask"
        old_lp_key, old_mask_key = stream_to_old_key[prefix]

        ids_list = [item[ids_key] for item in batch]
        mask_list = [item[mask_key] for item in batch]
        old_lp_list = [item[old_lp_key] for item in batch]
        old_mask_list = [item[old_mask_key] for item in batch]

        normalized_old_lp = []
        normalized_old_mask = []
        for ids, old_lp, old_mask in zip(ids_list, old_lp_list, old_mask_list):
            expected_len = max(0, int(ids.shape[0]) - 1)
            normalized_old_lp.append(_pad_1d_tensor(old_lp.float(), expected_len, 0.0))
            normalized_old_mask.append(_pad_1d_tensor(old_mask.float(), expected_len, 0.0))

        max_ids_len = max(int(x.shape[0]) for x in ids_list)
        target_old_len = max(0, max_ids_len - 1)

        ids = torch.stack([
            _pad_1d_tensor(x.long(), max_ids_len, pad_token_id) for x in ids_list
        ], dim=0)
        masks = torch.stack([
            _pad_1d_tensor(x.long(), max_ids_len, 0) for x in mask_list
        ], dim=0)
        old_lp = torch.stack([
            _pad_1d_tensor(x, target_old_len, 0.0) for x in normalized_old_lp
        ], dim=0)
        old_mask = torch.stack([
            _pad_1d_tensor(x, target_old_len, 0.0) for x in normalized_old_mask
        ], dim=0)
        return ids, masks, old_lp, old_mask

    plus_ids, plus_mask, old_lp_plus, old_mask_plus = _collate_stream("y_plus")
    minus_ids, minus_mask, old_lp_minus, old_mask_minus = _collate_stream("y_minus")
    vic_ids, vic_mask, old_lp_vic, old_mask_vic = _collate_stream("y_vic")

    def _stack_pixel_values(key: str) -> torch.Tensor:
        pv_list = []
        for item in batch:
            pv = item[key]
            if pv.dim() == 3:
                pv = pv.unsqueeze(0)
            pv_list.append(pv)

        pv_shapes = [tuple(pv.shape) for pv in pv_list]
        if len(set(pv_shapes)) > 1:
            max_patches = max(shape[0] for shape in pv_shapes)
            for i, pv in enumerate(pv_list):
                if pv.shape[0] < max_patches:
                    pad = torch.zeros(
                        (max_patches - pv.shape[0],) + pv.shape[1:],
                        dtype=pv.dtype,
                        device=pv.device,
                    )
                    pv_list[i] = torch.cat([pv, pad], dim=0)
        return torch.stack(pv_list, dim=0)

    def _stack_image_sizes(key: str) -> Optional[torch.Tensor]:
        image_sizes_list = [item.get(key) for item in batch]
        if not any(size is not None for size in image_sizes_list):
            return None
        return torch.stack([
            size if isinstance(size, torch.Tensor) else torch.tensor(size, dtype=torch.long)
            for size in image_sizes_list
        ], dim=0)

    def _collate_vic_field_mask(key: str) -> torch.Tensor:
        target_old_len = old_lp_vic.shape[1]
        masks = []
        for item in batch:
            field_mask = item.get(key)
            if field_mask is None:
                field_mask = torch.zeros((target_old_len,), dtype=torch.float32)
            masks.append(_pad_1d_tensor(field_mask.float(), target_old_len, 0.0))
        return torch.stack(masks, dim=0)

    pixel_values = _stack_pixel_values("pixel_values")
    wrong_pixel_values = _stack_pixel_values("wrong_pixel_values")
    image_sizes = _stack_image_sizes("image_sizes")
    wrong_image_sizes = _stack_image_sizes("wrong_image_sizes")
    vic_observed_mask = _collate_vic_field_mask("vic_observed_mask")
    vic_context_mask = _collate_vic_field_mask("vic_context_mask")
    vic_reasoning_mask = _collate_vic_field_mask("vic_reasoning_mask")
    vic_answer_mask = _collate_vic_field_mask("vic_answer_mask")
    vic_other_mask = _collate_vic_field_mask("vic_other_mask")

    prompt_lens = torch.tensor([int(item["prompt_len"]) for item in batch], dtype=torch.long)

    return {
        "y_plus_ids": plus_ids,
        "y_plus_mask": plus_mask,
        "y_minus_ids": minus_ids,
        "y_minus_mask": minus_mask,
        "y_vic_ids": vic_ids,
        "y_vic_mask": vic_mask,
        "old_token_lp_plus": old_lp_plus,
        "old_token_mask_plus": old_mask_plus,
        "old_token_lp_minus": old_lp_minus,
        "old_token_mask_minus": old_mask_minus,
        "old_token_lp_vic": old_lp_vic,
        "old_token_mask_vic": old_mask_vic,
        "pixel_values": pixel_values,
        "image_sizes": image_sizes,
        "wrong_pixel_values": wrong_pixel_values,
        "wrong_image_sizes": wrong_image_sizes,
        "vic_observed_mask": vic_observed_mask,
        "vic_context_mask": vic_context_mask,
        "vic_reasoning_mask": vic_reasoning_mask,
        "vic_answer_mask": vic_answer_mask,
        "vic_other_mask": vic_other_mask,
        "prompt_lens": prompt_lens,
    }


def _find_token_subsequence(haystack: List[int], needle: List[int], start_pos: int) -> int:
    if not needle:
        return -1
    max_start = len(haystack) - len(needle)
    for pos in range(max(0, start_pos), max_start + 1):
        if haystack[pos:pos + len(needle)] == needle:
            return pos
    return -1


def _build_vic_field_masks(
    y_vic_ids: torch.Tensor,
    prompt_len: int,
    tokenizer,
    vic_target: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    old_len = max(0, int(y_vic_ids.shape[0]) - 1)
    observed_mask = torch.zeros((old_len,), dtype=torch.float32)
    context_mask = torch.zeros((old_len,), dtype=torch.float32)
    reasoning_mask = torch.zeros((old_len,), dtype=torch.float32)
    answer_mask = torch.zeros((old_len,), dtype=torch.float32)

    if tokenizer is None or not vic_target or old_len == 0:
        other_mask = torch.ones((old_len,), dtype=torch.float32)
        return observed_mask, context_mask, reasoning_mask, answer_mask, other_mask

    generated_ids = y_vic_ids[int(prompt_len):].tolist()
    if not generated_ids:
        other_mask = torch.ones((old_len,), dtype=torch.float32)
        return observed_mask, context_mask, reasoning_mask, answer_mask, other_mask

    field_to_mask = {
        "Observed Facts:": observed_mask,
        "Context:": context_mask,
        "Reasoning:": reasoning_mask,
        "Answer:": answer_mask,
    }
    cursor = 0
    for raw_line in str(vic_target).splitlines():
        line = raw_line.strip()
        if not line:
            continue
        prefix = next((p for p in field_to_mask if line.startswith(p)), None)
        if prefix is None:
            continue
        line_tokens = tokenizer.encode(line, add_special_tokens=False)
        start = _find_token_subsequence(generated_ids, line_tokens, cursor)
        if start < 0:
            continue
        base = max(0, int(prompt_len) - 1 + start)
        end = min(old_len, base + len(line_tokens))
        if end > base:
            field_to_mask[prefix][base:end] = 1.0
        cursor = start + len(line_tokens)

    union_mask = torch.clamp(
        observed_mask + context_mask + reasoning_mask + answer_mask,
        min=0.0,
        max=1.0,
    )
    other_mask = torch.clamp(torch.ones((old_len,), dtype=torch.float32) - union_mask, min=0.0, max=1.0)
    return observed_mask, context_mask, reasoning_mask, answer_mask, other_mask


def _build_stage3_sample_cache(train_dataset) -> List[Stage3SampleCacheItem]:
    cache: List[Stage3SampleCacheItem] = []
    processor = getattr(train_dataset, "processor", None)
    tokenizer = getattr(getattr(train_dataset, "processor", None), "tokenizer", None)
    for idx in tqdm(range(len(train_dataset)), desc="构建 Stage3SampleCache"):
        sample = train_dataset[idx]
        if sample is None:
            continue

        raw_item = None
        if hasattr(train_dataset, "samples"):
            try:
                raw_item = train_dataset.samples[idx]
            except Exception:
                raw_item = None

        prompt_ids = sample.get("prompt_input_ids")
        prompt_mask = sample.get("prompt_attention_mask")
        y_vic_ids = sample.get("vic_input_ids")
        y_vic_mask = sample.get("vic_attention_mask")
        vic_target = None

        if prompt_ids is None:
            prompt_ids = sample["input_ids"]
        if prompt_mask is None:
            prompt_mask = sample.get("attention_mask", torch.ones_like(prompt_ids))

        if isinstance(raw_item, dict) and hasattr(train_dataset, "_build_targets"):
            instruction = raw_item.get("instruction", "")
            instruction_text = _strip_image_tokens(instruction)
            _, _, vic_target, _ = train_dataset._build_targets(raw_item, instruction_text)

        # Stage3 y_vic 默认走结构化组合（facts+reasoning+answer，context 可由开关控制）。
        if (
            y_vic_ids is None
            and processor is not None
            and isinstance(raw_item, dict)
            and vic_target is not None
        ):
            image = raw_item.get("image")
            if image is not None:
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
                    vic_conv = prompt_conv + [
                        {
                            "role": "assistant",
                            "content": [
                                {"type": "text", "text": vic_target},
                            ],
                        }
                    ]
                    vic_text = processor.apply_chat_template(vic_conv, add_generation_prompt=False)
                else:
                    prompt_text = f"<image>\n{instruction_text}"
                    vic_text = f"{prompt_text}\n{vic_target}"

                vic_inputs = processor(
                    text=vic_text,
                    images=image,
                    return_tensors="pt",
                    padding="longest",
                    truncation=False,
                )
                y_vic_ids = vic_inputs["input_ids"].squeeze(0)
                y_vic_mask = vic_inputs["attention_mask"].squeeze(0)

        if y_vic_ids is None:
            y_vic_ids = sample.get("full_input_ids", sample["input_ids"])
        if y_vic_mask is None:
            y_vic_mask = sample.get("full_attention_mask", sample.get("attention_mask", torch.ones_like(y_vic_ids)))

        prompt_len = int(prompt_ids.shape[0])
        pixel_values = sample["pixel_values"]
        if pixel_values.dim() == 3:
            pixel_values = pixel_values.unsqueeze(0)

        image_sizes = sample.get("image_sizes")
        if image_sizes is None:
            image_sizes = torch.tensor([pixel_values.shape[-2], pixel_values.shape[-1]], dtype=torch.long)
        if isinstance(image_sizes, torch.Tensor):
            image_sizes = image_sizes[:2].to(dtype=torch.long).cpu()
        else:
            image_sizes = torch.tensor(image_sizes, dtype=torch.long)[:2].cpu()

        observed_ids = None
        context_ids = None
        reasoning_ids = None
        answer_ids = None
        vic_observed_mask = None
        vic_context_mask = None
        vic_reasoning_mask = None
        vic_answer_mask = None
        vic_other_mask = torch.ones((max(0, int(y_vic_ids.shape[0]) - 1),), dtype=torch.float32)
        if tokenizer is not None and isinstance(raw_item, dict):
            ann = raw_item.get("teacher_annotation")
            if isinstance(ann, dict):
                observed = ann.get("observed_facts_visual", "")
                context = ann.get("context_textual", "")
                reasoning = ann.get("reasoning", "")
                answer = ann.get("answer", "")
                if isinstance(observed, str):
                    observed_ids = torch.tensor(
                        tokenizer.encode(observed, add_special_tokens=False),
                        dtype=torch.long,
                    )
                if isinstance(context, str):
                    context_ids = torch.tensor(
                        tokenizer.encode(context, add_special_tokens=False),
                        dtype=torch.long,
                    )
                if isinstance(reasoning, str):
                    reasoning_ids = torch.tensor(
                        tokenizer.encode(reasoning, add_special_tokens=False),
                        dtype=torch.long,
                    )
                if isinstance(answer, str):
                    answer_ids = torch.tensor(
                        tokenizer.encode(answer, add_special_tokens=False),
                        dtype=torch.long,
                    )
            if vic_target is not None:
                (
                    vic_observed_mask,
                    vic_context_mask,
                    vic_reasoning_mask,
                    vic_answer_mask,
                    vic_other_mask,
                ) = _build_vic_field_masks(
                    y_vic_ids=y_vic_ids.detach().cpu().long(),
                    prompt_len=prompt_len,
                    tokenizer=tokenizer,
                    vic_target=vic_target,
                )

        cache.append(Stage3SampleCacheItem(
            sample_idx=int(idx),
            prompt_ids=prompt_ids.detach().cpu().long(),
            prompt_mask=prompt_mask.detach().cpu().long(),
            y_vic_ids=y_vic_ids.detach().cpu().long(),
            y_vic_mask=y_vic_mask.detach().cpu().long(),
            pixel_values=pixel_values.detach().cpu(),
            image_sizes=image_sizes,
            prompt_len=prompt_len,
            observed_ids=observed_ids,
            context_ids=context_ids,
            reasoning_ids=reasoning_ids,
            answer_ids=answer_ids,
            vic_observed_mask=vic_observed_mask,
            vic_context_mask=vic_context_mask,
            vic_reasoning_mask=vic_reasoning_mask,
            vic_answer_mask=vic_answer_mask,
            vic_other_mask=vic_other_mask,
        ))
    return cache


def _set_stage3_trainable_params(model, args) -> List[torch.nn.Parameter]:
    for name, param in model.named_parameters():
        if not torch.is_floating_point(param):
            param.requires_grad = False
            continue
        name_l = name.lower()
        is_lora = "lora_" in name_l or "modules_to_save" in name_l
        is_projector = (
            ("projector" in name_l or "multi_modal_projector" in name_l)
            and bool(args.stage3_train_projector)
        )
        param.requires_grad = bool(is_lora or is_projector)
    return [p for p in model.parameters() if p.requires_grad]


def _strip_extra_image_tokens(
    ids: torch.Tensor,
    image_token_id: Optional[int],
    allowed_count: int,
    replacement_token_id: int,
) -> torch.Tensor:
    if image_token_id is None:
        return ids
    ids = ids.clone()
    pos = (ids[0] == int(image_token_id)).nonzero(as_tuple=False).view(-1)
    if pos.numel() > allowed_count:
        ids[0, pos[allowed_count:]] = int(replacement_token_id)
    return ids


def _generate_candidate_single(
    model,
    sample: Stage3SampleCacheItem,
    args,
    image_token_id: Optional[int],
    pad_token_id: int,
    do_sample: bool = True,
    temperature: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = args.device
    prompt_ids = sample.prompt_ids.unsqueeze(0).to(device)
    prompt_mask = sample.prompt_mask.unsqueeze(0).to(device)
    pixel_values = sample.pixel_values.unsqueeze(0).to(device)
    image_sizes = sanitize_image_sizes(sample.image_sizes.unsqueeze(0), batch_size=1)
    allowed_img_count = int((sample.prompt_ids == int(image_token_id)).sum().item()) if image_token_id is not None else 0

    gen_kwargs = dict(
        input_ids=prompt_ids,
        attention_mask=prompt_mask,
        pixel_values=pixel_values,
        image_sizes=image_sizes,
        do_sample=bool(do_sample),
        max_new_tokens=int(args.max_new_tokens),
        bad_words_ids=[[int(image_token_id)]] if image_token_id is not None else None,
        pad_token_id=int(pad_token_id),
    )
    if do_sample:
        gen_kwargs["temperature"] = float(args.temperature if temperature is None else temperature)
    with torch.no_grad():
        generated = model.generate(**gen_kwargs)
    eos_or_pad = model.config.eos_token_id or int(pad_token_id)
    generated = _strip_extra_image_tokens(
        generated,
        image_token_id=image_token_id,
        allowed_count=allowed_img_count,
        replacement_token_id=eos_or_pad,
    )
    generated = generated.squeeze(0).detach().cpu().long()
    generated_mask = (generated != int(pad_token_id)).long()
    return generated, generated_mask


def _extract_answer_letter(text: str) -> Optional[str]:
    if not text:
        return None
    text_u = str(text).strip()
    patterns = [
        r"(?:answer|答案)\s*[:：]\s*\(?\s*([A-D])\s*\)?",
        r"^\s*\(?\s*([A-D])\s*\)?\b",
    ]
    for pat in patterns:
        m = re.search(pat, text_u, flags=re.IGNORECASE)
        if m:
            return str(m.group(1)).upper()
    return None


def _evaluate_stage3_answer_metrics(
    model,
    sample_cache: List[Stage3SampleCacheItem],
    eval_indices: List[int],
    args,
    image_token_id: Optional[int],
    pad_token_id: int,
    tokenizer,
    answer_lookup: Dict[int, str],
) -> Tuple[float, float, int]:
    if not eval_indices or tokenizer is None:
        return 0.0, 0.0, 0

    model.eval()
    use_cache_states = _set_model_use_cache(model, True)
    gc_states = _set_model_gradient_checkpointing(model, False)
    try:
        total = 0
        fmt_hits = 0
        acc_hits = 0
        for idx in eval_indices:
            sample = sample_cache[idx]
            ids, _ = _generate_candidate_single(
                model=model,
                sample=sample,
                args=args,
                image_token_id=image_token_id,
                pad_token_id=pad_token_id,
                do_sample=False,
            )
            prompt_len = int(sample.prompt_len)
            gen_ids = ids[prompt_len:] if ids.shape[0] > prompt_len else ids
            text = tokenizer.decode(gen_ids.tolist(), skip_special_tokens=True)
            pred = _extract_answer_letter(text)
            gold = answer_lookup.get(int(idx))
            total += 1
            if pred is not None:
                fmt_hits += 1
            if pred is not None and gold is not None and pred == gold:
                acc_hits += 1
    finally:
        _restore_model_gradient_checkpointing(gc_states)
        _restore_model_use_cache(use_cache_states)

    denom = max(1, total)
    return float(acc_hits) / denom, float(fmt_hits) / denom, int(total)


def _score_sequence_no_grad(
    model,
    sample: Stage3SampleCacheItem,
    ids: torch.Tensor,
    mask: torch.Tensor,
    args,
) -> Tuple[float, float, torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        ids_b = ids.unsqueeze(0).to(args.device)
        mask_b = mask.unsqueeze(0).to(args.device)
        pixel_values = sample.pixel_values.unsqueeze(0).to(args.device)
        image_sizes = sanitize_image_sizes(sample.image_sizes.unsqueeze(0), batch_size=1)
        prompt_lens = torch.tensor([sample.prompt_len], device=ids_b.device, dtype=torch.long)

        token_lp, token_mask = _compute_token_log_probs(
            model=model,
            ids=ids_b,
            mask=mask_b,
            pixel_values=pixel_values,
            image_sizes=image_sizes,
            prompt_lens=prompt_lens,
        )
        avg_lp_tensor = (token_lp * token_mask).sum(dim=-1) / token_mask.sum(dim=-1).clamp(min=1.0)
        avg_lp = float(avg_lp_tensor.item())
        prob = float(math.exp(max(min(avg_lp, 50.0), -50.0)))
        return avg_lp, prob, token_lp.squeeze(0).detach().cpu(), token_mask.squeeze(0).detach().cpu()


def _bootstrap_period_states(
    model,
    sample_cache: List[Stage3SampleCacheItem],
    active_indices: List[int],
    args,
    image_token_id: Optional[int],
    pad_token_id: int,
) -> Dict[int, PeriodState]:
    states: Dict[int, PeriodState] = {}
    model.eval()
    use_cache_states = _set_model_use_cache(model, True)
    gc_states = _set_model_gradient_checkpointing(model, False)
    try:
        for idx in tqdm(active_indices, desc="Stage3 bootstrap"):
            sample = sample_cache[idx]
            cand1_ids, cand1_mask = _generate_candidate_single(model, sample, args, image_token_id, pad_token_id)
            cand2_ids, cand2_mask = _generate_candidate_single(model, sample, args, image_token_id, pad_token_id)
            cand1_lp, cand1_prob, _, _ = _score_sequence_no_grad(model, sample, cand1_ids, cand1_mask, args)
            cand2_lp, cand2_prob, _, _ = _score_sequence_no_grad(model, sample, cand2_ids, cand2_mask, args)

            # period0: y11 固定为 y_vic，y12 取低概率候选
            low_ids, low_mask, low_lp, low_prob = (
                (cand1_ids, cand1_mask, cand1_lp, cand1_prob)
                if cand1_prob <= cand2_prob
                else (cand2_ids, cand2_mask, cand2_lp, cand2_prob)
            )
            y_vic_lp, y_vic_prob, _, _ = _score_sequence_no_grad(
                model, sample, sample.y_vic_ids, sample.y_vic_mask, args
            )
            states[idx] = PeriodState(
                sample_idx=idx,
                y11_ids=sample.y_vic_ids.clone(),
                y11_mask=sample.y_vic_mask.clone(),
                y12_ids=low_ids.clone(),
                y12_mask=low_mask.clone(),
                avg_lp_11=float(y_vic_lp),
                avg_lp_12=float(low_lp),
                prob_11=float(y_vic_prob),
                prob_12=float(low_prob),
            )
    finally:
        _restore_model_gradient_checkpointing(gc_states)
        _restore_model_use_cache(use_cache_states)
    return states


def _build_pairs_and_next_states(
    model,
    sample_cache: List[Stage3SampleCacheItem],
    current_states: Dict[int, PeriodState],
    active_indices: List[int],
    args,
    image_token_id: Optional[int],
    pad_token_id: int,
    force_vic_positive: bool = False,
) -> Tuple[List[PeriodTrainingItem], Dict[int, PeriodState], int, int, float, float]:
    training_items: List[PeriodTrainingItem] = []
    next_states: Dict[int, PeriodState] = {}
    cold_start_count = 0
    swap_count = 0
    plus_lp_sum = 0.0
    minus_lp_sum = 0.0
    tau1 = float(args.tau1)
    tau_delta = float(getattr(args, "tau_delta", 0.01))

    model.eval()
    use_cache_states = _set_model_use_cache(model, True)
    gc_states = _set_model_gradient_checkpointing(model, False)
    try:
        num_active = len(active_indices)
        for rank, idx in enumerate(tqdm(active_indices, desc="Stage3 Phase-A")):
            state = current_states[idx]
            sample = sample_cache[idx]

            lp11, prob11, token11, mask11 = _score_sequence_no_grad(
                model, sample, state.y11_ids, state.y11_mask, args
            )
            lp12, prob12, token12, mask12 = _score_sequence_no_grad(
                model, sample, state.y12_ids, state.y12_mask, args
            )
            candidates = [
                {
                    "ids": state.y11_ids,
                    "mask": state.y11_mask,
                    "avg_lp": lp11,
                    "prob": prob11,
                    "old_token_lp": token11,
                    "old_token_mask": mask11,
                    "prev_prob": state.prob_11,
                },
                {
                    "ids": state.y12_ids,
                    "mask": state.y12_mask,
                    "avg_lp": lp12,
                    "prob": prob12,
                    "old_token_lp": token12,
                    "old_token_mask": mask12,
                    "prev_prob": state.prob_12,
                },
            ]
            candidates.sort(key=lambda x: x["prob"], reverse=True)
            y11 = candidates[0]
            y12 = candidates[1]
            if y11["ids"].data_ptr() != state.y11_ids.data_ptr():
                swap_count += 1

            delta11 = float(y11["prob"] - y11["prev_prob"])
            p_best = max(float(y11["prob"]), float(y12["prob"]))

            if torch.equal(state.y11_ids, sample.y_vic_ids) and torch.equal(state.y11_mask, sample.y_vic_mask):
                y_vic_lp, y_vic_prob, y_vic_token_lp, y_vic_token_mask = lp11, prob11, token11, mask11
            elif torch.equal(state.y12_ids, sample.y_vic_ids) and torch.equal(state.y12_mask, sample.y_vic_mask):
                y_vic_lp, y_vic_prob, y_vic_token_lp, y_vic_token_mask = lp12, prob12, token12, mask12
            else:
                y_vic_lp, y_vic_prob, y_vic_token_lp, y_vic_token_mask = _score_sequence_no_grad(
                    model, sample, sample.y_vic_ids, sample.y_vic_mask, args
                )

            use_cold_start = (p_best < tau1) and (delta11 < tau_delta)
            if force_vic_positive:
                y_plus_ids = sample.y_vic_ids
                y_plus_mask = sample.y_vic_mask
                old_lp_plus = y_vic_token_lp
                old_mask_plus = y_vic_token_mask
                y_plus_avg_lp = float(y_vic_lp)
                y_minus_ids = state.y12_ids
                y_minus_mask = state.y12_mask
                old_lp_minus = token12
                old_mask_minus = mask12
                y_minus_avg_lp = float(lp12)
            elif use_cold_start:
                y_plus_ids = sample.y_vic_ids
                y_plus_mask = sample.y_vic_mask
                old_lp_plus = y_vic_token_lp
                old_mask_plus = y_vic_token_mask
                y_plus_avg_lp = float(y_vic_lp)
                y_minus_ids = y12["ids"]
                y_minus_mask = y12["mask"]
                old_lp_minus = y12["old_token_lp"]
                old_mask_minus = y12["old_token_mask"]
                y_minus_avg_lp = float(y12["avg_lp"])
                cold_start_count += 1
            else:
                y_plus_ids = y11["ids"]
                y_plus_mask = y11["mask"]
                old_lp_plus = y11["old_token_lp"]
                old_mask_plus = y11["old_token_mask"]
                y_plus_avg_lp = float(y11["avg_lp"])
                y_minus_ids = y12["ids"]
                y_minus_mask = y12["mask"]
                old_lp_minus = y12["old_token_lp"]
                old_mask_minus = y12["old_token_mask"]
                y_minus_avg_lp = float(y12["avg_lp"])
            plus_lp_sum += y_plus_avg_lp
            minus_lp_sum += y_minus_avg_lp

            training_items.append(PeriodTrainingItem(
                sample_idx=int(idx),
                y_plus_ids=y_plus_ids.clone(),
                y_plus_mask=y_plus_mask.clone(),
                y_minus_ids=y_minus_ids.clone(),
                y_minus_mask=y_minus_mask.clone(),
                y_vic_ids=sample.y_vic_ids.clone(),
                y_vic_mask=sample.y_vic_mask.clone(),
                old_token_lp_plus=old_lp_plus.clone(),
                old_token_mask_plus=old_mask_plus.clone(),
                old_token_lp_minus=old_lp_minus.clone(),
                old_token_mask_minus=old_mask_minus.clone(),
                old_token_lp_vic=y_vic_token_lp.clone(),
                old_token_mask_vic=y_vic_token_mask.clone(),
                wrong_sample_idx=int(active_indices[(rank + 1) % num_active]) if num_active > 1 else int(idx),
            ))

            # Step6: 新候选完全替换下一 period 的 y11/y12
            next1_ids, next1_mask = _generate_candidate_single(model, sample, args, image_token_id, pad_token_id)
            next2_ids, next2_mask = _generate_candidate_single(model, sample, args, image_token_id, pad_token_id)
            next1_lp, next1_prob, _, _ = _score_sequence_no_grad(model, sample, next1_ids, next1_mask, args)
            next2_lp, next2_prob, _, _ = _score_sequence_no_grad(model, sample, next2_ids, next2_mask, args)
            next_sorted = [
                (next1_ids, next1_mask, next1_lp, next1_prob),
                (next2_ids, next2_mask, next2_lp, next2_prob),
            ]
            next_sorted.sort(key=lambda x: x[3], reverse=True)
            high = next_sorted[0]
            low = next_sorted[1]
            next_states[idx] = PeriodState(
                sample_idx=int(idx),
                y11_ids=high[0].clone(),
                y11_mask=high[1].clone(),
                y12_ids=low[0].clone(),
                y12_mask=low[1].clone(),
                avg_lp_11=float(high[2]),
                avg_lp_12=float(low[2]),
                prob_11=float(high[3]),
                prob_12=float(low[3]),
            )
    finally:
        _restore_model_gradient_checkpointing(gc_states)
        _restore_model_use_cache(use_cache_states)

    return training_items, next_states, cold_start_count, swap_count, plus_lp_sum, minus_lp_sum


def _run_one_period_train(
    model,
    optimizer,
    loader,
    args,
    tb_writer,
    global_step: int,
    sub_stage_idx: int,
    period_idx: int,
) -> Tuple[float, float, float, float, Dict[str, float], int]:
    model.train()
    grad_accum = max(1, int(getattr(args, "stage3_grad_accum", 0) or getattr(args, "grad_accum", 1)))
    grad_clip = float(getattr(args, "stage3_grad_clip", 1.0))
    wrong_image_enable = bool(int(getattr(args, "stage3_wrong_image_enable", 0)))
    wrong_image_weight = float(getattr(args, "stage3_wrong_image_weight", 0.2))
    wrong_image_margin = float(getattr(args, "stage3_wrong_image_margin", 0.0))
    field_weights = {
        "observed": float(getattr(args, "stage3_field_weight_observed", 1.0)),
        "context": float(getattr(args, "stage3_field_weight_context", 1.0)),
        "reasoning": float(getattr(args, "stage3_field_weight_reasoning", 1.0)),
        "answer": float(getattr(args, "stage3_field_weight_answer", 1.0)),
    }
    optimizer.zero_grad(set_to_none=True)

    total_loss = 0.0
    total_obj = 0.0
    total_reg = 0.0
    total_wrong = 0.0
    total_field = {
        "observed": 0.0,
        "context": 0.0,
        "reasoning": 0.0,
        "answer": 0.0,
    }
    steps = 0

    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return (values * mask).sum() / mask.sum().clamp(min=1.0)

    def _masked_mean_or_zero(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if float(mask.sum().item()) <= 0.0:
            return torch.zeros((), device=values.device, dtype=values.dtype)
        return _masked_mean(values, mask)

    for batch_idx, batch in enumerate(tqdm(loader, desc=f"Stage3 S{sub_stage_idx+1} P{period_idx+1}")):
        if batch is None:
            continue
        global_step += 1
        steps += 1

        y_plus_ids = batch["y_plus_ids"].to(args.device)
        y_plus_mask = batch["y_plus_mask"].to(args.device)
        y_minus_ids = batch["y_minus_ids"].to(args.device)
        y_minus_mask = batch["y_minus_mask"].to(args.device)
        y_vic_ids = batch["y_vic_ids"].to(args.device)
        y_vic_mask = batch["y_vic_mask"].to(args.device)
        pixel_values = batch["pixel_values"].to(args.device)
        image_sizes = sanitize_image_sizes(batch.get("image_sizes"), batch_size=y_plus_ids.size(0))
        wrong_pixel_values = batch["wrong_pixel_values"].to(args.device)
        wrong_image_sizes = sanitize_image_sizes(
            batch.get("wrong_image_sizes"),
            batch_size=y_plus_ids.size(0),
        )
        prompt_lens = batch["prompt_lens"].to(args.device)

        old_lp_plus = batch["old_token_lp_plus"].to(args.device)
        old_mask_plus = batch["old_token_mask_plus"].to(args.device)
        old_lp_minus = batch["old_token_lp_minus"].to(args.device)
        old_mask_minus = batch["old_token_mask_minus"].to(args.device)
        old_lp_vic = batch["old_token_lp_vic"].to(args.device)
        old_mask_vic = batch["old_token_mask_vic"].to(args.device)
        vic_observed_mask = batch["vic_observed_mask"].to(args.device)
        vic_context_mask = batch["vic_context_mask"].to(args.device)
        vic_reasoning_mask = batch["vic_reasoning_mask"].to(args.device)
        vic_answer_mask = batch["vic_answer_mask"].to(args.device)
        vic_other_mask = batch["vic_other_mask"].to(args.device)

        cur_lp_plus, cur_mask_plus = _compute_token_log_probs(
            model, y_plus_ids, y_plus_mask, pixel_values, image_sizes, prompt_lens
        )
        cur_lp_minus, cur_mask_minus = _compute_token_log_probs(
            model, y_minus_ids, y_minus_mask, pixel_values, image_sizes, prompt_lens
        )
        cur_lp_vic, cur_mask_vic = _compute_token_log_probs(
            model, y_vic_ids, y_vic_mask, pixel_values, image_sizes, prompt_lens
        )
        cur_lp_wrong_vic = None
        cur_mask_wrong_vic = None
        if wrong_image_enable:
            cur_lp_wrong_vic, cur_mask_wrong_vic = _compute_token_log_probs(
                model, y_vic_ids, y_vic_mask, wrong_pixel_values, wrong_image_sizes, prompt_lens
            )

        # collate 已对齐，这里额外防御
        if cur_lp_plus.shape[1] != old_lp_plus.shape[1]:
            target = min(cur_lp_plus.shape[1], old_lp_plus.shape[1])
            cur_lp_plus, cur_mask_plus = cur_lp_plus[:, :target], cur_mask_plus[:, :target]
            old_lp_plus, old_mask_plus = old_lp_plus[:, :target], old_mask_plus[:, :target]
        if cur_lp_minus.shape[1] != old_lp_minus.shape[1]:
            target = min(cur_lp_minus.shape[1], old_lp_minus.shape[1])
            cur_lp_minus, cur_mask_minus = cur_lp_minus[:, :target], cur_mask_minus[:, :target]
            old_lp_minus, old_mask_minus = old_lp_minus[:, :target], old_mask_minus[:, :target]
        if cur_lp_vic.shape[1] != old_lp_vic.shape[1]:
            target = min(cur_lp_vic.shape[1], old_lp_vic.shape[1])
            cur_lp_vic, cur_mask_vic = cur_lp_vic[:, :target], cur_mask_vic[:, :target]
            old_lp_vic, old_mask_vic = old_lp_vic[:, :target], old_mask_vic[:, :target]
            vic_observed_mask = vic_observed_mask[:, :target]
            vic_context_mask = vic_context_mask[:, :target]
            vic_reasoning_mask = vic_reasoning_mask[:, :target]
            vic_answer_mask = vic_answer_mask[:, :target]
            vic_other_mask = vic_other_mask[:, :target]
            if cur_lp_wrong_vic is not None and cur_mask_wrong_vic is not None:
                cur_lp_wrong_vic = cur_lp_wrong_vic[:, :target]
                cur_mask_wrong_vic = cur_mask_wrong_vic[:, :target]

        mask_plus = (old_mask_plus * cur_mask_plus).float()
        mask_minus = (old_mask_minus * cur_mask_minus).float()
        mask_vic = (old_mask_vic * cur_mask_vic).float()
        delta_vic = old_lp_vic - cur_lp_vic

        field_masks = {
            "observed": mask_vic * vic_observed_mask,
            "context": mask_vic * vic_context_mask,
            "reasoning": mask_vic * vic_reasoning_mask,
            "answer": mask_vic * vic_answer_mask,
        }
        weighted_mask_vic = (
            field_weights["observed"] * field_masks["observed"]
            + field_weights["context"] * field_masks["context"]
            + field_weights["reasoning"] * field_masks["reasoning"]
            + field_weights["answer"] * field_masks["answer"]
            + mask_vic * vic_other_mask
        )

        term1 = _masked_mean(log_clip(-old_lp_minus + cur_lp_minus), mask_minus)
        term2 = _masked_mean(log_clip(old_lp_plus - cur_lp_plus), mask_plus)
        term3 = _masked_mean_or_zero(delta_vic, weighted_mask_vic)
        loss_obj = term1 + term2
        loss_reg = term3
        loss_wrong = torch.zeros((), device=args.device, dtype=loss_reg.dtype)
        if cur_lp_wrong_vic is not None and cur_mask_wrong_vic is not None:
            wrong_mask = (mask_vic * cur_mask_wrong_vic).float()
            wrong_delta = F.relu(cur_lp_wrong_vic - cur_lp_vic + wrong_image_margin)
            loss_wrong = _masked_mean_or_zero(wrong_delta, wrong_mask)
        loss = loss_obj + loss_reg + wrong_image_weight * loss_wrong

        field_loss_vals = {}
        for name, field_mask in field_masks.items():
            field_loss = _masked_mean_or_zero(delta_vic, field_mask)
            field_loss_vals[name] = float(field_loss.item())
            total_field[name] += field_loss_vals[name]

        (loss / grad_accum).backward()

        if (batch_idx + 1) % grad_accum == 0 or batch_idx == len(loader) - 1:
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_norm=grad_clip,
                )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        loss_val = float(loss.item())
        obj_val = float(loss_obj.item())
        reg_val = float(loss_reg.item())
        wrong_val = float(loss_wrong.item())
        total_loss += loss_val
        total_obj += obj_val
        total_reg += reg_val
        total_wrong += wrong_val

        if global_step % args.log_step == 0:
            print(
                f"Step {global_step}, Total: {loss_val:.4f}, "
                f"L_obj: {obj_val:.4f}, L_reg: {reg_val:.4f}, L_wrong: {wrong_val:.4f}"
            )
            tb_writer.add_scalar("stage3/total_loss", loss_val, global_step)
            tb_writer.add_scalar("stage3/L_obj", obj_val, global_step)
            tb_writer.add_scalar("stage3/L_reg", reg_val, global_step)
            tb_writer.add_scalar("stage3/L_wrong_image", wrong_val, global_step)
            tb_writer.add_scalar("stage3/L_reg_observed", field_loss_vals["observed"], global_step)
            tb_writer.add_scalar("stage3/L_reg_context", field_loss_vals["context"], global_step)
            tb_writer.add_scalar("stage3/L_reg_reasoning", field_loss_vals["reasoning"], global_step)
            tb_writer.add_scalar("stage3/L_reg_answer", field_loss_vals["answer"], global_step)

    denom = max(1, steps)
    avg_field = {name: value / denom for name, value in total_field.items()}
    return (
        total_loss / denom,
        total_obj / denom,
        total_reg / denom,
        total_wrong / denom,
        avg_field,
        global_step,
    )


def train_stage3_lord(model, train_dataset, args, tb_writer):
    print("\n" + "=" * 50)
    print("阶段 3: LoRD-II 多 Period 训练")
    print("=" * 50)

    image_token_id = _get_image_token_id(model)
    pad_token_id = model.config.pad_token_id or model.config.eos_token_id or 0
    tau1 = float(args.tau1)
    tau_delta = float(getattr(args, "tau_delta", 0.01))
    sub_stage_num = max(1, int(getattr(args, "sub_stage_num", 1)))
    period_num = int(getattr(args, "period_num", 0))
    if period_num <= 0:
        period_num = max(1, int(args.epochs))
    sub_set_num = int(getattr(args, "sub_set_num", 0))
    stage3_lr = float(args.lr) * float(getattr(args, "stage3_lr_scale", 1.0))

    print(
        f"[Stage3] tau1={tau1}, tau_delta={tau_delta}, "
        f"sub_stage_num={sub_stage_num}, period_num={period_num}, sub_set_num={sub_set_num}, "
        f"lr={stage3_lr}, train_projector={int(bool(args.stage3_train_projector))}, "
        f"resume_opt={int(bool(getattr(args, 'stage3_resume_save_optimizer', 1)))}, "
        f"eval_max_samples={int(getattr(args, 'stage3_eval_max_samples', 0))}, "
        f"field_w=({float(getattr(args, 'stage3_field_weight_observed', 1.0)):.2f},"
        f"{float(getattr(args, 'stage3_field_weight_context', 1.0)):.2f},"
        f"{float(getattr(args, 'stage3_field_weight_reasoning', 1.0)):.2f},"
        f"{float(getattr(args, 'stage3_field_weight_answer', 1.0)):.2f}), "
        f"wrong_image={int(bool(getattr(args, 'stage3_wrong_image_enable', 0)))}"
    )

    trainable_params = _set_stage3_trainable_params(model, args)
    print(f"Stage3 可训练参数量: {sum(p.numel() for p in trainable_params):,}")
    if not trainable_params:
        raise RuntimeError("Stage3 没有可训练参数，请检查 LoRA 或 stage3_train_projector 配置")
    optimizer = torch.optim.AdamW(trainable_params, lr=stage3_lr)

    sample_cache = _build_stage3_sample_cache(train_dataset)
    if not sample_cache:
        raise RuntimeError("Stage3SampleCache 为空，无法开始 Stage3 训练")

    tokenizer = None
    answer_lookup: Dict[int, str] = {}
    if hasattr(train_dataset, "processor"):
        tokenizer = getattr(train_dataset.processor, "tokenizer", None)
    raw_samples = getattr(train_dataset, "samples", None)
    if isinstance(raw_samples, list):
        for idx, sample in enumerate(raw_samples):
            answer_letter = str(sample.get("answer_letter", "") or "").strip().upper()[:1]
            if answer_letter in {"A", "B", "C", "D"}:
                answer_lookup[int(idx)] = answer_letter

    stage3_eval_max_samples = int(getattr(args, "stage3_eval_max_samples", 0))
    eval_sample_cache: Optional[List[Stage3SampleCacheItem]] = None
    eval_answer_lookup: Dict[int, str] = {}
    eval_pool_indices: Optional[List[int]] = None
    stage3_eval_split = str(
        getattr(args, "stage3_eval_scienceqa_split", "validation") or ""
    ).strip()
    stage3_eval_path = str(
        getattr(args, "stage3_eval_scienceqa_path", "") or getattr(args, "scienceqa_path", "")
    ).strip()
    stage3_eval_train_num = int(getattr(args, "stage3_eval_train_num", 0))
    if (
        stage3_eval_max_samples > 0
        and tokenizer is not None
        and stage3_eval_split
        and stage3_eval_path
        and hasattr(train_dataset, "processor")
    ):
        same_eval_and_train = (
            stage3_eval_split == str(getattr(args, "scienceqa_split", "")).strip()
            and stage3_eval_path == str(getattr(args, "scienceqa_path", "")).strip()
        )
        if same_eval_and_train:
            print(
                "[Stage3][Warn] stage3_eval_scienceqa_split 与训练 split 相同，"
                "当前评估将与训练集重合。建议使用 validation。"
            )
        try:
            eval_samples = build_scienceqa_samples(
                scienceqa_path=stage3_eval_path,
                split=stage3_eval_split,
                train_num=stage3_eval_train_num,
                seed=int(getattr(args, "scienceqa_seed", 20240306)),
            )
            eval_dataset = ScienceQADataset(
                processor=train_dataset.processor,
                scienceqa_path=stage3_eval_path,
                split=stage3_eval_split,
                train_num=stage3_eval_train_num,
                max_length=args.max_length,
                samples=eval_samples,
                seed=int(getattr(args, "scienceqa_seed", 20240306)),
                teacher_lang=args.teacher_lang,
                teacher_observed_max_tokens=args.teacher_observed_max_tokens,
                teacher_context_max_tokens=args.teacher_context_max_tokens,
                teacher_reasoning_max_tokens=args.teacher_reasoning_max_tokens,
                teacher_answer_max_tokens=args.teacher_answer_max_tokens,
                stage3_vic_include_context=bool(int(args.stage3_vic_include_context)),
                require_teacher_annotation=False,
            )
            eval_sample_cache = _build_stage3_sample_cache(eval_dataset)
            eval_pool_indices = list(range(len(eval_sample_cache)))
            for idx, sample in enumerate(eval_samples):
                answer_letter = str(sample.get("answer_letter", "") or "").strip().upper()[:1]
                if answer_letter in {"A", "B", "C", "D"}:
                    eval_answer_lookup[int(idx)] = answer_letter
            print(
                f"[Stage3][Eval] 使用独立评估集: split={stage3_eval_split}, "
                f"path={stage3_eval_path}, samples={len(eval_sample_cache)}"
            )
        except Exception as exc:
            print(f"[Stage3][Warn] 构建独立评估集失败，将回退训练集评估: {exc}")
            eval_sample_cache = None
            eval_answer_lookup = {}
            eval_pool_indices = None

    all_indices = list(range(len(sample_cache)))
    global_step = 0
    base_seed = int(getattr(args, "scienceqa_seed", 20240306))
    period_counter = 0
    resume_save_optimizer = bool(int(getattr(args, "stage3_resume_save_optimizer", 1)))
    resume_save_interval = max(1, int(getattr(args, "stage3_resume_save_interval", 1)))
    stage3_eval_every_period = max(1, int(getattr(args, "stage3_eval_every_period", 1)))
    resume_save_dir = str(getattr(args, "stage3_resume_save_path", "") or "").strip()
    if not resume_save_dir:
        resume_save_dir = os.path.join(args.save_path, "stage3_resume_latest")

    resume_path = str(getattr(args, "stage3_resume_path", "") or "").strip()
    resume_progress = None
    if resume_path:
        resume_progress = _load_stage3_resume_state(model, optimizer, resume_path, args.device)

    resume_sub_stage_idx = 0
    resume_period_idx = 0
    resume_active_indices: Optional[List[int]] = None
    resume_period_states: Optional[Dict[int, PeriodState]] = None
    if resume_progress:
        resume_sub_stage_idx = int(resume_progress.get("next_sub_stage_idx", 0))
        resume_period_idx = int(resume_progress.get("next_period_idx", 0))
        global_step = int(resume_progress.get("global_step", 0))
        period_counter = int(resume_progress.get("period_counter", 0))
        resume_active_indices_raw = resume_progress.get("active_indices")
        if isinstance(resume_active_indices_raw, list):
            resume_active_indices = [int(v) for v in resume_active_indices_raw]
        resume_period_states = _deserialize_period_states(resume_progress.get("period_states"))
        if bool(resume_progress.get("completed", False)):
            print("[Stage3][Resume] 断点标记为 completed，跳过 Stage3 训练。")
            return model

    if resume_sub_stage_idx >= sub_stage_num:
        print("[Stage3][Resume] next_sub_stage_idx 超出配置，Stage3 无需继续训练。")
        return model

    for sub_stage_idx in range(resume_sub_stage_idx, sub_stage_num):
        is_resume_sub_stage = bool(
            resume_progress is not None and sub_stage_idx == resume_sub_stage_idx
        )

        if is_resume_sub_stage and resume_active_indices is not None:
            active_indices = list(resume_active_indices)
        elif sub_set_num > 0 and sub_set_num < len(all_indices):
            rng = random.Random(base_seed + sub_stage_idx)
            active_indices = sorted(rng.sample(all_indices, sub_set_num))
        else:
            active_indices = list(all_indices)

        print(
            f"[Stage3] sub_stage={sub_stage_idx+1}/{sub_stage_num}, "
            f"active_samples={len(active_indices)}"
        )

        if is_resume_sub_stage and resume_period_idx > 0 and resume_period_states is not None:
            period_states = resume_period_states
            print(
                f"[Stage3][Resume] 继续 sub_stage={sub_stage_idx+1}, period={resume_period_idx+1}/{period_num}"
            )
        else:
            period_states = _bootstrap_period_states(
                model=model,
                sample_cache=sample_cache,
                active_indices=active_indices,
                args=args,
                image_token_id=image_token_id,
                pad_token_id=pad_token_id,
            )
            resume_period_idx = 0

        for period_idx in range(resume_period_idx, period_num):
            print(f"[Stage3] 进入 period {period_idx+1}/{period_num}")
            phase_a_start = time.perf_counter()
            training_items, next_states, cold_start_count, swap_count, plus_lp_sum, minus_lp_sum = _build_pairs_and_next_states(
                model=model,
                sample_cache=sample_cache,
                current_states=period_states,
                active_indices=active_indices,
                args=args,
                image_token_id=image_token_id,
                pad_token_id=pad_token_id,
                force_vic_positive=bool(period_idx == 0),
            )
            phase_a_seconds = time.perf_counter() - phase_a_start
            period_states = next_states

            period_dataset = PeriodTrainingDataset(training_items, sample_cache)
            period_loader = DataLoader(
                period_dataset,
                batch_size=max(1, int(args.batch_size)),
                shuffle=True,
                num_workers=0,
                collate_fn=lambda b: _collate_period_training(b, pad_token_id),
            )
            avg_total, avg_obj, avg_reg, avg_wrong, avg_field_losses, global_step = _run_one_period_train(
                model=model,
                optimizer=optimizer,
                loader=period_loader,
                args=args,
                tb_writer=tb_writer,
                global_step=global_step,
                sub_stage_idx=sub_stage_idx,
                period_idx=period_idx,
            )

            cold_ratio = float(cold_start_count) / max(1, len(active_indices))
            swap_ratio = float(swap_count) / max(1, len(active_indices))
            avg_lp_plus = float(plus_lp_sum) / max(1, len(training_items))
            avg_lp_minus = float(minus_lp_sum) / max(1, len(training_items))
            period_counter += 1
            print(
                f"[Stage3][S{sub_stage_idx+1}P{period_idx+1}] "
                f"loss={avg_total:.4f}, L_obj={avg_obj:.4f}, L_reg={avg_reg:.4f}, "
                f"L_wrong={avg_wrong:.4f}, "
                f"L_obs={avg_field_losses['observed']:.4f}, "
                f"L_ctx={avg_field_losses['context']:.4f}, "
                f"L_reason={avg_field_losses['reasoning']:.4f}, "
                f"L_ans={avg_field_losses['answer']:.4f}, "
                f"cold_start_ratio={cold_ratio:.4f}, swap_ratio={swap_ratio:.4f}, "
                f"avg_lp_plus={avg_lp_plus:.4f}, avg_lp_minus={avg_lp_minus:.4f}, "
                f"phase_a_seconds={phase_a_seconds:.2f}"
            )
            tb_writer.add_scalar("stage3_epoch/total_loss", avg_total, period_counter)
            tb_writer.add_scalar("stage3_epoch/L_obj", avg_obj, period_counter)
            tb_writer.add_scalar("stage3_epoch/L_reg", avg_reg, period_counter)
            tb_writer.add_scalar("stage3_epoch/L_wrong_image", avg_wrong, period_counter)
            tb_writer.add_scalar("stage3_epoch/L_reg_observed", avg_field_losses["observed"], period_counter)
            tb_writer.add_scalar("stage3_epoch/L_reg_context", avg_field_losses["context"], period_counter)
            tb_writer.add_scalar("stage3_epoch/L_reg_reasoning", avg_field_losses["reasoning"], period_counter)
            tb_writer.add_scalar("stage3_epoch/L_reg_answer", avg_field_losses["answer"], period_counter)
            tb_writer.add_scalar("stage3/cold_start_ratio", cold_ratio, period_counter)
            tb_writer.add_scalar("stage3/swap_ratio", swap_ratio, period_counter)
            tb_writer.add_scalar("stage3/avg_lp_plus", avg_lp_plus, period_counter)
            tb_writer.add_scalar("stage3/avg_lp_minus", avg_lp_minus, period_counter)
            tb_writer.add_scalar("stage3/phase_a_seconds", phase_a_seconds, period_counter)
            tb_writer.add_scalar("stage3/period", period_idx + 1, period_counter)
            tb_writer.add_scalar("stage3/sub_stage", sub_stage_idx + 1, period_counter)

            if (
                stage3_eval_max_samples > 0
                and tokenizer is not None
                and (answer_lookup or eval_answer_lookup)
                and (period_counter % stage3_eval_every_period == 0)
            ):
                use_eval_cache = (
                    eval_sample_cache is not None
                    and eval_pool_indices is not None
                    and bool(eval_answer_lookup)
                )
                if use_eval_cache:
                    metrics_sample_cache = eval_sample_cache
                    metrics_indices_pool = eval_pool_indices
                    metrics_answer_lookup = eval_answer_lookup
                else:
                    metrics_sample_cache = sample_cache
                    metrics_indices_pool = active_indices
                    metrics_answer_lookup = answer_lookup

                eval_k = min(stage3_eval_max_samples, len(metrics_indices_pool))
                rng_eval = random.Random(base_seed + period_counter + 7919)
                if eval_k < len(metrics_indices_pool):
                    eval_indices = sorted(rng_eval.sample(metrics_indices_pool, eval_k))
                else:
                    eval_indices = list(metrics_indices_pool)
                val_acc, val_fmt, val_n = _evaluate_stage3_answer_metrics(
                    model=model,
                    sample_cache=metrics_sample_cache,
                    eval_indices=eval_indices,
                    args=args,
                    image_token_id=image_token_id,
                    pad_token_id=pad_token_id,
                    tokenizer=tokenizer,
                    answer_lookup=metrics_answer_lookup,
                )
                print(
                    f"[Stage3][Eval][S{sub_stage_idx+1}P{period_idx+1}] "
                    f"val_answer_acc={val_acc:.4f}, format_rate={val_fmt:.4f}, n={val_n}"
                )
                tb_writer.add_scalar("stage3/val_answer_acc", val_acc, period_counter)
                tb_writer.add_scalar("stage3/format_rate", val_fmt, period_counter)

            next_sub_stage_idx = sub_stage_idx
            next_period_idx = period_idx + 1
            next_period_states_to_save = period_states
            next_active_indices = active_indices
            if next_period_idx >= period_num:
                next_sub_stage_idx = sub_stage_idx + 1
                next_period_idx = 0
                next_period_states_to_save = None
                next_active_indices = None

            resume_progress_payload = {
                "version": 1,
                "completed": bool(next_sub_stage_idx >= sub_stage_num),
                "next_sub_stage_idx": int(next_sub_stage_idx),
                "next_period_idx": int(next_period_idx),
                "global_step": int(global_step),
                "period_counter": int(period_counter),
                "active_indices": next_active_indices,
                "period_states": _serialize_period_states(next_period_states_to_save),
                "sub_stage_num": int(sub_stage_num),
                "period_num": int(period_num),
                "sub_set_num": int(sub_set_num),
                "base_seed": int(base_seed),
            }
            should_save_resume = bool(
                resume_progress_payload["completed"]
                or (period_counter % resume_save_interval == 0)
            )
            if should_save_resume:
                _save_stage3_resume_state(
                    model=model,
                    optimizer=optimizer,
                    resume_dir=resume_save_dir,
                    progress=resume_progress_payload,
                    include_optimizer_state=resume_save_optimizer,
                )

            if int(getattr(args, "save_each_epoch", 0)) == 1:
                ckpt_dir = os.path.join(
                    args.save_path,
                    f"stage3_sub{sub_stage_idx+1}_period{period_idx+1}",
                )
                save_stage3_checkpoint(model, ckpt_dir)

        resume_period_idx = 0
        resume_period_states = None
        resume_active_indices = None

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


def _is_projector_param_name(name: str) -> bool:
    name_l = str(name).lower()
    return ("projector" in name_l) or ("multi_modal_projector" in name_l)


def _extract_projector_state(model) -> dict:
    projector_state = {}
    for name, tensor in model.state_dict().items():
        if _is_projector_param_name(name):
            projector_state[name] = tensor.detach().cpu()
    return projector_state


def _save_projector_state(model, projector_path: str) -> int:
    projector_state = _extract_projector_state(model)
    if not projector_state:
        print(f"[Warn] 未找到 projector 参数，跳过保存: {projector_path}")
        return 0
    torch.save(projector_state, projector_path)
    print(f"已保存 projector 权重: {projector_path} (params={len(projector_state)})")
    return len(projector_state)


def _load_projector_state(model, projector_path: str) -> Tuple[int, int]:
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
    os.makedirs(ckpt_path, exist_ok=True)
    
    save_vq_codebook(model, os.path.join(ckpt_path, "vq_codebook.pt"))

    projector_path = os.path.join(ckpt_path, "projector.pt")
    projector_param_count = _save_projector_state(model, projector_path)
    
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
        "projector_state_path": "projector.pt",
        "projector_state_param_count": int(projector_param_count),
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


def _stage2_contract_missing_files(ckpt_path: str) -> List[str]:
    required_files = [
        "vq_codebook.pt",
        "projector.pt",
        "stage2_config.json",
        "adapter_config.json",
    ]
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

    # 先准备 ScienceQA 数据；仅 stage>=2 时再准备教师回答，避免 Stage1-only 多余 API 调用。
    train_samples = None
    train_bucket_meta = None
    if args.dataset_name == "scienceqa":
        print("\n加载 ScienceQA 样本（预加载阶段，不加载学生模型）...")
        train_samples = build_scienceqa_samples(
            scienceqa_path=args.scienceqa_path,
            split=args.scienceqa_split,
            train_num=args.train_num,
            seed=args.scienceqa_seed,
        )

        if int(args.stage) >= 2:
            # 关键：仅在 Stage2/3 需要蒸馏时补齐教师回答
            print("加载 ScienceQA 教师回答（stage>=2）...")
            train_samples = attach_gpt4v_teacher_responses(train_samples, args)
        else:
            print("stage<2，跳过教师回答采集/补齐")

        train_bucket_meta = load_scienceqa_preprocessed_buckets(
            args.scienceqa_preprocessed_path,
            expected_len=len(train_samples),
            args=args,
        )
    
    # 加载模型
    model, processor = load_model_and_processor(args)
    
    # 添加 VQ 层
    model = add_vq_to_model(model, args)
    
    # 应用 LoRA
    if args.use_lora:
        model = apply_lora(model, args)
    
    # 加载数据
    print("\n加载训练数据...")
    if args.dataset_name == "scienceqa":
        train_dataset = ScienceQADataset(
            processor=processor,
            scienceqa_path=args.scienceqa_path,
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

        # 尝试加载已收集的数据
        visual_qa_data = collector.load_collected_data("visual_qa_data.json")
        image_descriptions = collector.load_collected_data("image_descriptions.json")

        if not visual_qa_data and not image_descriptions:
            print("警告：未找到预收集的数据，请先运行数据收集脚本")
            print("示例：python data_collector2.py --collect_data")
            return

        # 创建数据集
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
        if not args.stage2_ckpt_path:
            args.stage2_ckpt_path = os.path.join(args.save_path, "stage2_vision")
        if not os.path.isdir(args.stage2_ckpt_path):
            raise RuntimeError(f"Stage3 需要可用的 Stage2 checkpoint 目录: {args.stage2_ckpt_path}")
        missing_stage2_files = _stage2_contract_missing_files(args.stage2_ckpt_path)
        if missing_stage2_files:
            raise RuntimeError(
                "Stage3 启动前 Stage2 工件不完整，缺失: "
                + ", ".join(missing_stage2_files)
            )
        model = train_stage3_lord(model, train_dataset, args, tb_writer)
        save_stage3_checkpoint(model, os.path.join(args.save_path, "stage3_lord_final"))
    
    print("\n" + "=" * 60)
    print("VQ-LoRD 训练完成!")
    print("=" * 60)
    
    tb_writer.close()


if __name__ == "__main__":
    main()
