#!/usr/bin/env python3
"""Standalone common utilities for portable local teacher sampling."""

from __future__ import annotations

import json
import os
import random
import re
import threading
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import quote

import requests
import pyarrow.parquet as pq
from datasets import load_dataset
from openai import OpenAI
from PIL import Image
from huggingface_hub import HfApi, hf_hub_download

TEACHER_REQUIRED_FIELDS = (
    "observed_facts_visual",
    "context_textual",
    "reasoning",
    "answer",
)

DEFAULT_BUDGET = {
    "teacher_observed_max_tokens": 256,
    "teacher_context_max_tokens": 192,
    "teacher_reasoning_max_tokens": 256,
    "teacher_answer_max_tokens": 64,
    "teacher_max_new_tokens_total": 768,
}

DEFAULT_DATASET_SPECS = {
    "scienceqa": {
        "hf_path": "derek-thomas/ScienceQA",
        "split": "train",
        "count": 500,
    },
    "textvqa": {
        "hf_path": "lmms-lab/textvqa",
        "split": "train",
        "count": 150,
    },
    "aokvqa": {
        "hf_path": "HuggingFaceM4/A-OKVQA",
        "split": "train",
        "count": 150,
    },
}


@dataclass
class Sample:
    dataset: str
    sample_id: str
    image: Image.Image
    instruction: str
    question: str
    hint: str
    choices: List[str]
    reference_answer: str
    meta: Dict[str, Any]

    def to_training_sample(self) -> dict:
        return {
            "dataset": self.dataset,
            "sample_id": self.sample_id,
            "image": self.image,
            "instruction": self.instruction,
            "question": self.question,
            "hint": self.hint,
            "choices": self.choices,
        }


def parse_optional_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"invalid boolean value: {value}")


def sanitize_model_tag(model_name: str) -> str:
    text = str(model_name or "").strip()
    if not text:
        return "unknown-model"
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-.")
    return text or "unknown-model"


def set_hf_env_if_missing() -> None:
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")


def ensure_rgb(image: Image.Image) -> Image.Image:
    return image.convert("RGB") if image.mode != "RGB" else image


def resize_image_max_side(image: Image.Image, max_side: Optional[int]) -> Image.Image:
    if not max_side or max_side <= 0:
        return image
    width, height = image.size
    current_max = max(width, height)
    if current_max <= int(max_side):
        return image
    scale = float(max_side) / float(current_max)
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return image.resize(new_size, Image.LANCZOS)


def normalize_image(image_obj: Any) -> Image.Image:
    if isinstance(image_obj, (list, tuple)) and image_obj:
        return normalize_image(image_obj[0])
    if isinstance(image_obj, Image.Image):
        return ensure_rgb(image_obj)
    if isinstance(image_obj, dict):
        image_bytes = image_obj.get("bytes")
        image_path = image_obj.get("path")
        if image_bytes:
            return ensure_rgb(Image.open(BytesIO(image_bytes)))
        if image_path and os.path.exists(image_path):
            return ensure_rgb(Image.open(image_path))
    if isinstance(image_obj, str) and os.path.exists(image_obj):
        return ensure_rgb(Image.open(image_obj))
    raise TypeError(f"Unsupported image object type: {type(image_obj)!r}")


def random_select_indices(total: int, count: int, seed: int) -> List[int]:
    indices = list(range(total))
    rng = random.Random(seed)
    rng.shuffle(indices)
    if count > 0:
        indices = indices[: min(total, count)]
    return indices


def truncate_text_by_word_budget(text: str, max_tokens: int) -> str:
    text = str(text or "").strip()
    if not text:
        return ""
    if max_tokens <= 0:
        return text
    words = text.split()
    if len(words) >= max_tokens:
        return " ".join(words[:max_tokens]).strip()
    if len(words) <= 1 and len(text) > max_tokens:
        return text[:max_tokens].strip()
    return text


def normalize_match_text(text: str) -> str:
    text = str(text or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def build_canonical_context(sample: Optional[dict]) -> str:
    if not isinstance(sample, dict):
        return ""
    question = str(sample.get("question", "") or "").strip()
    hint = str(sample.get("hint", "") or "").strip()
    choices = sample.get("choices") or []
    parts: List[str] = []
    if question:
        parts.append(f"Question: {question}")
    if hint:
        parts.append(f"Hint: {hint}")
    if choices:
        option_lines = [f"({chr(65 + i)}) {str(choice).strip()}" for i, choice in enumerate(choices)]
        parts.append("Options:\n" + "\n".join(option_lines))
    return "\n".join(parts).strip()


def coerce_struct_field_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        return "\n".join(x for x in (coerce_struct_field_to_text(v) for v in value) if x).strip()
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value).strip()


def extract_json_payload(text: str) -> Optional[dict]:
    if not isinstance(text, str):
        return None
    cleaned = text.strip()
    if not cleaned:
        return None
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        frag = cleaned[start : end + 1]
        try:
            data = json.loads(frag)
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    field_names = "|".join(re.escape(name) for name in TEACHER_REQUIRED_FIELDS)
    kv_pattern = re.compile(rf"(?ms)^\s*({field_names})\s*:\s*(.*?)\s*(?=^\s*(?:{field_names})\s*:|\Z)")
    matches = kv_pattern.findall(cleaned)
    if matches:
        payload = {str(k).strip(): str(v).strip() for k, v in matches}
        if all(payload.get(field, "").strip() for field in TEACHER_REQUIRED_FIELDS):
            return payload
    return None


def extract_partial_struct_payload(text: str) -> Optional[dict]:
    if not isinstance(text, str):
        return None
    cleaned = text.strip()
    if not cleaned:
        return None
    payload: Dict[str, str] = {}
    patterns = {
        "answer": r'"answer"\s*:\s*"((?:\\.|[^"\\])*)"',
        "observed_facts_visual": r'"observed_facts_visual"\s*:\s*"((?:\\.|[^"\\])*)"',
        "context_textual": r'"context_textual"\s*:\s*"((?:\\.|[^"\\])*)"',
        "reasoning": r'"reasoning"\s*:\s*"((?:\\.|[^"\\])*)"',
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, cleaned, flags=re.DOTALL)
        if match:
            try:
                payload[key] = json.loads(f'"{match.group(1)}"')
            except Exception:
                payload[key] = match.group(1)
    if not all(payload.get(field, "").strip() for field in TEACHER_REQUIRED_FIELDS):
        return None
    return payload


def normalize_choice_answer(answer: str, sample: Optional[dict], max_tokens: int) -> str:
    answer = truncate_text_by_word_budget(answer, max_tokens)
    if not isinstance(sample, dict):
        return answer
    choices = sample.get("choices") or []
    if not choices:
        return answer
    max_letter = chr(65 + min(len(choices), 26) - 1)
    letter_match = re.search(rf"(?:option\s*)?\(?\s*([A-{max_letter}])\s*\)?", answer, flags=re.IGNORECASE)
    if letter_match:
        letter = letter_match.group(1).upper()
        idx = ord(letter) - ord("A")
        if 0 <= idx < len(choices):
            return f"({letter}) {str(choices[idx]).strip()}"
    norm_answer = normalize_match_text(answer)
    for idx, choice in enumerate(choices):
        choice_text = str(choice).strip()
        norm_choice = normalize_match_text(choice_text)
        if not norm_choice:
            continue
        if norm_answer == norm_choice or norm_answer in norm_choice or norm_choice in norm_answer:
            letter = chr(65 + idx)
            return f"({letter}) {choice_text}"
    return answer


def has_observed_leakage(text: str) -> bool:
    text_l = str(text or "").lower()
    banned_patterns = [
        r"\bbest fit\b",
        r"\bbest matches\b",
        r"\baligns best\b",
        r"\boption\s*[a-z]\b",
        r"\bthe answer\b",
        r"\bcorrect answer\b",
    ]
    return any(re.search(pattern, text_l) for pattern in banned_patterns)


def semantic_issue_flags(annotation: Optional[dict], sample: Optional[dict]) -> List[str]:
    if not isinstance(annotation, dict):
        return ["invalid_json_or_missing_fields"]
    issues: List[str] = []
    observed = str(annotation.get("observed_facts_visual", "")).strip()
    answer = str(annotation.get("answer", "")).strip()
    if has_observed_leakage(observed):
        issues.append("observed_leakage")
    if isinstance(sample, dict) and (sample.get("choices") or []):
        choices = sample.get("choices") or []
        max_letter = chr(65 + min(len(choices), 26) - 1)
        if not re.match(rf"^\(?[A-{max_letter}]\)?(?:\s+.*)?$", answer):
            issues.append("answer_not_choice_like")
    return issues


def normalize_struct_payload(payload: Optional[dict], budget: dict, sample: Optional[dict] = None) -> Optional[dict]:
    if not isinstance(payload, dict):
        return None
    normalized = {
        "format_version": "v2",
        "observed_facts_visual": coerce_struct_field_to_text(payload.get("observed_facts_visual", payload.get("observed_facts", ""))),
        "context_textual": coerce_struct_field_to_text(payload.get("context_textual", payload.get("context", ""))),
        "reasoning": coerce_struct_field_to_text(payload.get("reasoning", "")),
        "answer": coerce_struct_field_to_text(payload.get("answer", "")),
    }
    canonical_context = build_canonical_context(sample)
    if canonical_context:
        normalized["context_textual"] = canonical_context
    normalized["observed_facts_visual"] = truncate_text_by_word_budget(normalized["observed_facts_visual"], int(budget.get("teacher_observed_max_tokens", 0)))
    normalized["context_textual"] = truncate_text_by_word_budget(normalized["context_textual"], int(budget.get("teacher_context_max_tokens", 0)))
    normalized["reasoning"] = truncate_text_by_word_budget(normalized["reasoning"], int(budget.get("teacher_reasoning_max_tokens", 0)))
    normalized["answer"] = normalize_choice_answer(normalized["answer"], sample=sample, max_tokens=int(budget.get("teacher_answer_max_tokens", 0)))
    for field in TEACHER_REQUIRED_FIELDS:
        value = normalized.get(field)
        if not isinstance(value, str) or not value.strip():
            return None
    return normalized


def build_structured_teacher_prompt(sample: dict, lang: str = "en", extra_strict: bool = False) -> str:
    instruction = str(sample.get("instruction", "") or "").replace("<image>", "").strip()
    has_choices = bool(sample.get("choices"))
    if str(lang).lower() == "zh":
        if has_choices:
            base_prompt = (
                "你是严谨的视觉选择题助手。\n"
                "请仅输出严格 JSON，必须且只包含以下键，并按这个顺序输出：\n"
                "answer, observed_facts_visual, context_textual, reasoning。\n"
                "规则：\n"
                "0) 每个字段值都必须是纯字符串，不要输出数组、列表或对象。\n"
                "0.1) 输出必须是合法 JSON 对象，不要写成 field: value 文本。\n"
                "0.2) 如果不确定，也必须从给定选项中选出最可能正确的一项。\n"
                "0.3) 一旦犹豫，请直接把最可能的选项写进 answer，不要继续讨论不确定性。\n"
                "0.4) answer 必须最先输出，优先格式 '(A) option text'。\n"
                "1) observed_facts_visual 只写图像可见证据（可含 OCR），不要写推理、不要写选项匹配、不要写“对应某地区/某国家/某答案”之类判断。\n"
                "2) context_textual 必须完整重述题干、提示与选项文本条件。\n"
                "3) reasoning 写基于前两者的简短推理，只允许 1 句，可以引用选项，但不要把推理写进 observed_facts_visual。\n"
                "4) 不要输出 markdown，不要输出额外字段。\n\n"
            )
        else:
            base_prompt = (
                "你是严谨的视觉问答助手。\n"
                "请仅输出严格 JSON，必须且只包含以下键，并按这个顺序输出：\n"
                "answer, observed_facts_visual, context_textual, reasoning。\n"
                "规则：\n"
                "0) 每个字段值都必须是纯字符串，不要输出数组、列表或对象。\n"
                "0.1) 输出必须是合法 JSON 对象，不要写成 field: value 文本。\n"
                "1) answer 必须是简短直接答案。\n"
                "2) observed_facts_visual 只写图像可见证据（可含 OCR），不要写推理。\n"
                "3) context_textual 重述题目中的文字条件。\n"
                "4) reasoning 只写 1 句简短推理。\n"
                "5) 不要输出 markdown，不要输出额外字段。\n\n"
            )
        extra_prompt = (
            "额外强调：如果 observed_facts_visual 中出现 therefore、option、对应、最佳匹配 这类推理或选项判断，视为错误。\n\n"
            if extra_strict
            else ""
        )
        return base_prompt + extra_prompt + instruction

    if has_choices:
        base_prompt = (
            "You are a rigorous visual multiple-choice QA assistant.\n"
            "Return strict JSON only with exactly these keys in order:\n"
            "answer, observed_facts_visual, context_textual, reasoning.\n"
            "Rules:\n"
            "0) Every field value must be a plain string. Do not output arrays, lists, or objects.\n"
            "0.1) The output must be a valid JSON object, not plain 'field: value' text.\n"
            "0.2) If you are uncertain, you must still choose the option that is most likely to be correct from the provided choices.\n"
            "0.3) Once uncertain, write the most likely option in the answer field immediately and do not continue debating the uncertainty.\n"
            "0.4) The answer field must appear first, preferably as '(A) option text'.\n"
            "1) observed_facts_visual: only image-observable evidence (OCR allowed); no inference, no option matching, no 'corresponds to', no final identification beyond what is directly visible.\n"
            "2) context_textual: restate the full textual conditions from question, hint, and options.\n"
            "3) reasoning: one short sentence only; option comparison belongs here, not in observed_facts_visual.\n"
            "4) No markdown and no extra fields.\n\n"
        )
    else:
        base_prompt = (
            "You are a rigorous visual QA assistant.\n"
            "Return strict JSON only with exactly these keys in order:\n"
            "answer, observed_facts_visual, context_textual, reasoning.\n"
            "Rules:\n"
            "0) Every field value must be a plain string. Do not output arrays, lists, or objects.\n"
            "0.1) The output must be a valid JSON object, not plain 'field: value' text.\n"
            "1) answer must be a short direct answer.\n"
            "2) observed_facts_visual must contain only directly visible evidence, OCR allowed, no inference.\n"
            "3) context_textual must restate the textual conditions from the question.\n"
            "4) reasoning must be one short sentence.\n"
            "5) No markdown and no extra fields.\n\n"
        )
    extra_prompt = (
        "Extra reminder: if observed_facts_visual contains words like therefore, best fit, aligns best, option, corresponds to, the output is invalid.\n\n"
        if extra_strict
        else ""
    )
    return base_prompt + extra_prompt + instruction

class TeacherClient:
    def __init__(
        self,
        api_key: str,
        base_url: Optional[str],
        model: str,
        max_retries: int = 2,
        http_timeout: float = 300.0,
        enable_thinking: Optional[bool] = None,
        image_detail: str = "auto",
        image_max_side: Optional[int] = None,
    ):
        if not api_key:
            raise RuntimeError("Missing API key. Set OPENAI_API_KEY or pass --teacher_api_key.")
        self.client = OpenAI(api_key=api_key, base_url=base_url or None, timeout=http_timeout, max_retries=0)
        self.model = model
        self.max_retries = max(1, int(max_retries))
        self.enable_thinking = enable_thinking
        self.image_detail = image_detail
        self.image_max_side = int(image_max_side) if image_max_side else None

    def query_image(
        self,
        image: Image.Image,
        prompt: str,
        max_tokens: int,
        image_detail: Optional[str] = None,
        image_max_side: Optional[int] = None,
    ) -> str:
        import base64

        image = ensure_rgb(image)
        image = resize_image_max_side(image, image_max_side or self.image_max_side)
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        request_kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_b64}",
                                "detail": image_detail or self.image_detail,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "max_tokens": int(max_tokens),
        }
        if self.enable_thinking is not None:
            request_kwargs["extra_body"] = {"enable_thinking": self.enable_thinking}
        last_err = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(**request_kwargs)
                return response.choices[0].message.content or ""
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                if attempt < self.max_retries - 1:
                    time.sleep(2 * attempt + 1)
        raise RuntimeError(f"Teacher API failed after {self.max_retries} attempts: {last_err}")


def _extract_string_list(value: Any) -> List[str]:
    items: List[str] = []
    if value is None:
        return items
    if isinstance(value, list):
        for item in value:
            items.extend(_extract_string_list(item))
        return items
    if isinstance(value, dict):
        for key in ("answer", "text", "label", "value"):
            if key in value and value[key] is not None:
                text = str(value[key]).strip()
                if text:
                    items.append(text)
                    return items
        return items
    text = str(value).strip()
    if text:
        items.append(text)
    return items


def _majority_text(values: List[str]) -> str:
    cleaned = [str(v).strip() for v in values if str(v).strip()]
    if not cleaned:
        return ""
    counts = Counter(cleaned)
    return counts.most_common(1)[0][0]


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with open(temp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(temp, path)


def _upgrade_loaded_record(record: dict) -> dict:
    if not isinstance(record, dict):
        return {}
    upgraded = dict(record)
    ann = upgraded.get("teacher_annotation")
    if not isinstance(ann, dict):
        ann = {
            "format_version": upgraded.get("format_version", "v2"),
            "observed_facts_visual": upgraded.get("observed_facts_visual", ""),
            "context_textual": upgraded.get("context_textual", ""),
            "reasoning": upgraded.get("reasoning", ""),
            "answer": upgraded.get("answer", ""),
        }
    for field in TEACHER_REQUIRED_FIELDS:
        value = upgraded.get(field)
        if (not isinstance(value, str) or not value.strip()) and isinstance(ann.get(field), str):
            upgraded[field] = ann.get(field, "")
    upgraded["format_version"] = str(upgraded.get("format_version") or ann.get("format_version") or "v2")
    upgraded["teacher_annotation"] = {
        "format_version": str(ann.get("format_version") or upgraded.get("format_version") or "v2"),
        "observed_facts_visual": str(upgraded.get("observed_facts_visual", "") or ""),
        "context_textual": str(upgraded.get("context_textual", "") or ""),
        "reasoning": str(upgraded.get("reasoning", "") or ""),
        "answer": str(upgraded.get("answer", "") or ""),
    }
    if not isinstance(upgraded.get("meta"), dict):
        sample_meta = (((upgraded.get("sample") or {}).get("meta")) or {}) if isinstance(upgraded.get("sample"), dict) else {}
        upgraded["meta"] = sample_meta if isinstance(sample_meta, dict) else {}
    return upgraded


def record_is_training_ready(record: Optional[dict]) -> bool:
    upgraded = _upgrade_loaded_record(record or {})
    if not bool(upgraded.get("valid")):
        return False
    ann = upgraded.get("teacher_annotation")
    if not isinstance(ann, dict):
        return False
    for field in TEACHER_REQUIRED_FIELDS:
        value = upgraded.get(field)
        if not isinstance(value, str) or not value.strip():
            return False
        ann_value = ann.get(field)
        if not isinstance(ann_value, str) or not ann_value.strip():
            return False
    return True


def load_existing_sample_map(path: Path) -> Dict[str, dict]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    sample_map = payload.get("samples", {}) if isinstance(payload, dict) else {}
    if not isinstance(sample_map, dict):
        return {}
    return {str(k): _upgrade_loaded_record(v) for k, v in sample_map.items()}


def build_output_payload(dataset_name: str, dataset_path: str, split: str, teacher_model: str, budget: dict, sample_map: Dict[str, dict], extra_meta: Optional[dict] = None) -> dict:
    payload = {
        "format_version": "v2",
        "dataset": dataset_name,
        "dataset_path": dataset_path,
        "split": split,
        "teacher_model": teacher_model,
        "victim_model": teacher_model,
        "budget": budget,
        "sample_count": len(sample_map),
        "samples": {str(k): _upgrade_loaded_record(v) for k, v in sample_map.items()},
    }
    if extra_meta:
        payload["meta"] = extra_meta
    return payload

def _candidate_hf_api_bases() -> List[str]:
    candidates: List[str] = []
    explicit = os.environ.get("HF_API_BASE", "").strip()
    endpoint = os.environ.get("HF_ENDPOINT", "").strip()
    for base in (explicit, endpoint, "https://hf-mirror.com", "https://huggingface.co"):
        if not base:
            continue
        normalized = base.rstrip("/")
        if normalized not in candidates:
            candidates.append(normalized)
    return candidates


def _fetch_parquet_listing(repo_id: str, timeout: float = 30.0) -> dict:
    errors = []
    quoted_repo = quote(repo_id, safe="")
    headers = {"user-agent": "vq-lord-teacher-sampler/1.0"}
    for base in _candidate_hf_api_bases():
        url = f"{base}/api/datasets/{quoted_repo}/parquet"
        try:
            resp = requests.get(url, timeout=timeout, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, dict):
                return data
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{url} -> {exc}")
    raise RuntimeError(
        "Failed to query parquet listing for dataset "
        f"{repo_id}. Tried: {' | '.join(errors)}"
    )


def _extract_split_urls_from_listing(listing: dict, split: str) -> List[str]:
    urls: List[str] = []
    if "parquet_files" in listing and isinstance(listing["parquet_files"], list):
        for item in listing["parquet_files"]:
            if not isinstance(item, dict):
                continue
            if item.get("split") == split and item.get("url"):
                urls.append(str(item["url"]))
    if urls:
        return sorted(dict.fromkeys(urls))
    for _, cfg_value in listing.items():
        if isinstance(cfg_value, dict) and split in cfg_value and isinstance(cfg_value[split], list):
            urls.extend(str(x) for x in cfg_value[split] if x)
    return sorted(dict.fromkeys(urls))


def _direct_dataset_load(repo_id: str, split: str):
    return load_dataset(repo_id, split=split)


def _load_scienceqa_aligned_dataset(repo_id: str, split: str):
    """Load ScienceQA in a way that preserves legacy sample alignment by default.

    Default behavior is strict: require the original direct dataset loader so the
    filtered order matches the existing project collector as closely as possible.
    Parquet fallback is only allowed when the caller explicitly opts in because it
    may change row ordering and therefore the sampled subset.
    """
    try:
        ds = _direct_dataset_load(repo_id, split)
        return ds, "scienceqa_original_direct_v1"
    except Exception as exc:  # noqa: BLE001
        allow_fallback = parse_optional_bool(os.environ.get("SCIENCEQA_ALLOW_PARQUET_FALLBACK", ""))
        if not allow_fallback:
            raise RuntimeError(
                "ScienceQA direct load failed. To keep GPT ScienceQA sampling aligned with the existing project, "
                "the portable sampler now requires direct load by default. "
                "If you explicitly accept possible sample-set drift, set SCIENCEQA_ALLOW_PARQUET_FALLBACK=1 and rerun. "
                f"Original error: {exc}"
            ) from exc
    ds = load_remote_dataset_via_parquet(repo_id, split)
    return ds, "scienceqa_parquet_fallback_v1"


def _normalize_base_for_repo_files(base: str) -> str:
    normalized = base.rstrip("/")
    if normalized.endswith("/api"):
        normalized = normalized[:-4]
    return normalized


def _known_parquet_data_files(repo_id: str, split: str) -> List[str]:
    repo = str(repo_id or "").strip().lower()
    split_name = str(split or "").strip().lower()
    shard_counts = {
        "lmms-lab/textvqa": {
            "train": 20,
            "validation": 3,
            "test": 4,
        },
    }
    dataset_splits = shard_counts.get(repo)
    if not dataset_splits:
        return []
    num_shards = dataset_splits.get(split_name)
    if not num_shards:
        return []

    local_files: List[str] = []
    for idx in range(int(num_shards)):
        filename = f"data/{split_name}-{idx:05d}-of-{num_shards:05d}.parquet"
        local_path = hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=filename,
        )
        local_files.append(local_path)
    return local_files


def _iter_known_parquet_rows(repo_id: str, split: str, count: int, seed: int):
    local_files = _known_parquet_data_files(repo_id, split)
    if not local_files:
        raise RuntimeError(f"No known parquet files configured for dataset={repo_id}, split={split}")

    parquet_files = [pq.ParquetFile(path) for path in local_files]
    total_rows = sum(int(pf.metadata.num_rows) for pf in parquet_files)
    if total_rows <= 0:
        return

    selected_indices = None
    if int(count) > 0:
        selected_indices = set(random_select_indices(total_rows, count, seed))
    next_selected = min(selected_indices) if selected_indices else None
    global_index = 0

    for pf in parquet_files:
        for batch in pf.iter_batches(batch_size=32):
            rows = batch.to_pylist()
            for row in rows:
                if selected_indices is None:
                    yield row, global_index
                else:
                    if next_selected is None:
                        return
                    if global_index == next_selected:
                        yield row, global_index
                        selected_indices.remove(next_selected)
                        next_selected = min(selected_indices) if selected_indices else None
                global_index += 1

def _parquet_urls_from_repo_files(repo_id: str, split: str, timeout: float = 30.0) -> List[str]:
    errors = []
    for base in _candidate_hf_api_bases():
        endpoint = _normalize_base_for_repo_files(base)
        try:
            api = HfApi(endpoint=endpoint)
            files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
            parquet_files = []
            split_lower = split.lower()
            for path in files:
                if not str(path).endswith('.parquet'):
                    continue
                lowered = str(path).lower()
                if f'/{split_lower}/' in lowered or lowered.startswith(f'{split_lower}/') or f'-{split_lower}-' in lowered or f'_{split_lower}_' in lowered or lowered.endswith(f'/{split_lower}.parquet'):
                    parquet_files.append(str(path))
            if not parquet_files:
                parquet_files = [str(path) for path in files if str(path).endswith('.parquet')]
            if parquet_files:
                quoted_repo = '/'.join(quote(part, safe='') for part in repo_id.split('/'))
                urls = [f"{endpoint}/datasets/{quoted_repo}/resolve/main/{quote(path, safe='/')}" for path in sorted(dict.fromkeys(parquet_files))]
                return urls
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{endpoint} -> {exc}")
    raise RuntimeError(
        "Failed to list parquet files for dataset "
        f"{repo_id}. Tried: {' | '.join(errors)}"
    )


def load_remote_dataset_via_parquet(repo_id: str, split: str):
    errors = []

    try:
        data_files = _known_parquet_data_files(repo_id, split)
        if data_files:
            return load_dataset("parquet", data_files={split: data_files}, split=split)
        errors.append(f"known_files -> no known parquet files for split={split}")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"known_files -> {exc!r}")

    try:
        listing = _fetch_parquet_listing(repo_id)
        urls = _extract_split_urls_from_listing(listing, split)
        if urls:
            return load_dataset("parquet", data_files={split: urls}, split=split)
        errors.append(f"parquet_listing -> no parquet URLs found for split={split}")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"parquet_listing -> {exc}")

    try:
        urls = _parquet_urls_from_repo_files(repo_id, split)
        if urls:
            return load_dataset("parquet", data_files={split: urls}, split=split)
        errors.append(f"repo_files -> no parquet URLs found for split={split}")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"repo_files -> {exc}")

    try:
        return _direct_dataset_load(repo_id, split)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"direct_load -> {exc}")

    raise RuntimeError(
        f"Failed to load dataset={repo_id}, split={split}. Attempts: {' | '.join(errors)}"
    )


def should_retry_annotation(annotation: Optional[dict], issues: List[str], generation_attempt: int, attempts: int) -> bool:
    if generation_attempt >= max(1, attempts) - 1:
        return False
    if annotation is None:
        return True
    retryable = {"invalid_json_or_missing_fields", "observed_leakage"}
    return any(issue in retryable for issue in issues)


def collect_teacher_annotation(
    client: TeacherClient,
    sample: Sample,
    budget: dict,
    teacher_lang: str,
    attempts: int = 1,
    image_detail: Optional[str] = None,
    image_max_side: Optional[int] = None,
) -> Tuple[Optional[dict], str, List[str]]:
    raw_response = ""
    ann = None
    issues: List[str] = ["invalid_json_or_missing_fields"]
    sample_dict = sample.to_training_sample()
    for generation_attempt in range(max(1, attempts)):
        prompt = build_structured_teacher_prompt(sample_dict, teacher_lang, extra_strict=(generation_attempt > 0))
        raw_response = client.query_image(
            sample.image,
            prompt,
            int(budget["teacher_max_new_tokens_total"]),
            image_detail=image_detail,
            image_max_side=image_max_side,
        )
        parsed = extract_json_payload(raw_response)
        if parsed is None:
            parsed = extract_partial_struct_payload(raw_response)
        ann = normalize_struct_payload(parsed, budget, sample=sample_dict)
        issues = semantic_issue_flags(ann, sample_dict)
        if ann is not None and not issues:
            break
        if not should_retry_annotation(ann, issues, generation_attempt, attempts):
            break
    return ann, raw_response, issues


def _build_scienceqa_sample_from_item(
    item: dict,
    split: str,
    source_index: int,
    raw_train_index: int,
    sampling_scheme: str,
    legacy_sample_id: Optional[int] = None,
) -> Sample:
    question = str(item.get("question", "") or "").strip()
    hint = str(item.get("hint", "") or "").strip()
    choices = [str(x).strip() for x in (item.get("choices") or [])]
    answer_idx = int(item.get("answer", 0)) if choices else 0
    answer_text = choices[answer_idx] if 0 <= answer_idx < len(choices) else ""
    answer_letter = chr(65 + answer_idx) if 0 <= answer_idx < 26 else "A"
    hint_block = f"Hint: {hint}\n" if hint else ""
    choice_lines = "\n".join(f"({chr(65 + i)}) {choice}" for i, choice in enumerate(choices))
    instruction = f"<image>\nQuestion: {question}\n{hint_block}Options:\n{choice_lines}\nAnswer:"
    meta = {
        "split": split,
        "source_index": int(source_index),
        "filtered_source_index": int(source_index),
        "raw_train_index": int(raw_train_index),
        "answer_idx": answer_idx,
        "answer_letter": answer_letter,
        "sampling_scheme": sampling_scheme,
    }
    if legacy_sample_id is not None:
        meta["legacy_sample_id"] = int(legacy_sample_id)
    return Sample(
        dataset="ScienceQA",
        sample_id=f"scienceqa::{split}::{int(source_index)}",
        image=normalize_image(item["image"]),
        instruction=instruction,
        question=question,
        hint=hint,
        choices=choices,
        reference_answer=answer_text,
        meta=meta,
    )


def _load_scienceqa_alignment_targets(alignment_cache_path: str, split: str, count: int) -> List[dict]:
    with open(alignment_cache_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    sample_map = payload.get("samples", {}) if isinstance(payload, dict) else {}
    if not isinstance(sample_map, dict):
        raise RuntimeError(f"Invalid ScienceQA alignment cache format: {alignment_cache_path}")

    stable_prefix = f"scienceqa::{split}::"
    targets: List[dict] = []
    for key, record in sample_map.items():
        if not isinstance(key, str) or not key.startswith(stable_prefix) or not isinstance(record, dict):
            continue
        sample_meta = (((record.get("sample") or {}).get("meta")) or {}) if isinstance(record.get("sample"), dict) else {}
        direct_meta = record.get("meta") or {}
        meta = sample_meta if isinstance(sample_meta, dict) and sample_meta else direct_meta
        if not isinstance(meta, dict):
            continue
        source_index = meta.get("source_index")
        raw_train_index = meta.get("raw_train_index")
        legacy_sample_id = meta.get("sample_id")
        if source_index is None or raw_train_index is None:
            continue
        try:
            source_index = int(source_index)
            raw_train_index = int(raw_train_index)
            legacy_sample_id = int(legacy_sample_id) if legacy_sample_id is not None else source_index
        except (TypeError, ValueError):
            continue
        targets.append(
            {
                "source_index": source_index,
                "raw_train_index": raw_train_index,
                "legacy_sample_id": legacy_sample_id,
            }
        )

    targets.sort(key=lambda row: (int(row.get("legacy_sample_id", 10**12)), int(row["source_index"]), int(row["raw_train_index"])))
    if int(count) > 0 and len(targets) > int(count):
        targets = targets[: int(count)]
    return targets


def build_scienceqa_samples(path: str, split: str, count: int, seed: int) -> List[Sample]:
    """Build ScienceQA samples with deterministic legacy alignment support."""
    ds, sampling_scheme = _load_scienceqa_aligned_dataset(path, split)
    alignment_cache_path = str(os.environ.get("SCIENCEQA_ALIGNMENT_CACHE", "") or "").strip()
    if alignment_cache_path:
        targets = _load_scienceqa_alignment_targets(alignment_cache_path, split, count)
        if not targets:
            raise RuntimeError(
                f"ScienceQA alignment cache has no usable targets for split={split}: {alignment_cache_path}"
            )
        aligned_scheme = f"scienceqa_alignment_cache_v1::{Path(alignment_cache_path).name}"
        samples: List[Sample] = []
        for target in targets:
            raw_train_index = int(target["raw_train_index"])
            item = ds[raw_train_index]
            if item.get("image") is None:
                raise RuntimeError(
                    f"Aligned ScienceQA raw_train_index={raw_train_index} has no image in current dataset view."
                )
            samples.append(
                _build_scienceqa_sample_from_item(
                    item=item,
                    split=split,
                    source_index=int(target["source_index"]),
                    raw_train_index=raw_train_index,
                    sampling_scheme=aligned_scheme,
                    legacy_sample_id=int(target.get("legacy_sample_id", target["source_index"])),
                )
            )
        return samples

    dataset_with_images: List[Tuple[int, dict]] = []
    for raw_train_index, item in enumerate(ds):
        if item.get("image") is not None:
            dataset_with_images.append((raw_train_index, item))

    selected = random_select_indices(len(dataset_with_images), count, seed)
    samples: List[Sample] = []
    for filtered_source_index in selected:
        raw_train_index, item = dataset_with_images[filtered_source_index]
        samples.append(
            _build_scienceqa_sample_from_item(
                item=item,
                split=split,
                source_index=filtered_source_index,
                raw_train_index=raw_train_index,
                sampling_scheme=sampling_scheme,
            )
        )
    return samples


def _build_textvqa_sample(item: dict, split: str, idx: int) -> Sample:
    question = str(item.get("question", "") or "").strip()
    answers = _extract_string_list(item.get("answers"))
    majority = _majority_text(answers)
    return Sample(
        dataset="TextVQA",
        sample_id=f"textvqa::{split}::{item.get('question_id', idx)}",
        image=normalize_image(item["image"]),
        instruction=f"<image>\nQuestion: {question}\nAnswer:",
        question=question,
        hint="",
        choices=[],
        reference_answer=majority,
        meta={
            "split": split,
            "source_index": idx,
            "question_id": item.get("question_id"),
            "image_id": item.get("image_id"),
        },
    )


def build_textvqa_samples(path: str, split: str, count: int, seed: int) -> List[Sample]:
    return list(iter_textvqa_samples(path, split, count, seed))


def iter_textvqa_samples(path: str, split: str, count: int, seed: int):
    repo = str(path or "").strip()
    if repo.lower() == "lmms-lab/textvqa":
        for item, idx in _iter_known_parquet_rows(repo, split, count, seed):
            try:
                yield _build_textvqa_sample(item, split, idx)
            except Exception as exc:  # noqa: BLE001
                print(f"[TextVQA] Skip malformed sample idx={idx}: {exc}")
                continue
        return

    ds = load_remote_dataset_via_parquet(path, split)
    selected = random_select_indices(len(ds), count, seed)
    for idx in selected:
        try:
            yield _build_textvqa_sample(ds[idx], split, idx)
        except Exception as exc:  # noqa: BLE001
            print(f"[TextVQA] Skip malformed sample idx={idx}: {exc}")
            continue


def _build_aokvqa_sample(item: dict, split: str, idx: int) -> Sample:
    question = str(item.get("question", "") or "").strip()
    choices = [str(x).strip() for x in (item.get("choices") or [])]
    direct_answers = _extract_string_list(item.get("direct_answers"))

    raw_answer_idx = item.get("correct_choice_idx", 0)
    try:
        answer_idx = int(raw_answer_idx) if raw_answer_idx is not None and choices else 0
    except (TypeError, ValueError):
        answer_idx = 0

    answer_text = choices[answer_idx] if choices and 0 <= answer_idx < len(choices) else ""
    if not answer_text and direct_answers:
        answer_text = _majority_text(direct_answers)

    choice_lines = "\n".join(f"({chr(65 + i)}) {choice}" for i, choice in enumerate(choices))
    instruction = f"<image>\nQuestion: {question}\nOptions:\n{choice_lines}\nAnswer:"
    return Sample(
        dataset="AOKVQA",
        sample_id=f"aokvqa::{split}::{idx}",
        image=normalize_image(item["image"]),
        instruction=instruction,
        question=question,
        hint="",
        choices=choices,
        reference_answer=answer_text,
        meta={
            "split": split,
            "source_index": idx,
            "question_id": item.get("question_id"),
            "image_id": item.get("image_id"),
            "answer_idx": answer_idx,
            "direct_answers": direct_answers,
        },
    )


def build_aokvqa_samples(path: str, split: str, count: int, seed: int) -> List[Sample]:
    ds = load_remote_dataset_via_parquet(path, split)
    selected = random_select_indices(len(ds), count, seed)
    return [_build_aokvqa_sample(ds[idx], split, idx) for idx in selected]


def iter_aokvqa_samples(path: str, split: str, count: int, seed: int):
    ds = load_remote_dataset_via_parquet(path, split)
    selected = random_select_indices(len(ds), count, seed)
    for idx in selected:
        yield _build_aokvqa_sample(ds[idx], split, idx)


DATASET_DEFAULT_RENDER = {
    "scienceqa": {"image_detail": "low", "image_max_side": 896},
    "textvqa": {"image_detail": "high", "image_max_side": 1024},
    "aokvqa": {"image_detail": "low", "image_max_side": 896},
}


DATASET_BUILDERS = {
    "scienceqa": build_scienceqa_samples,
    "textvqa": build_textvqa_samples,
    "aokvqa": build_aokvqa_samples,
}


DATASET_STREAM_BUILDERS = {
    "textvqa": iter_textvqa_samples,
    "aokvqa": iter_aokvqa_samples,
}


def sample_to_serializable(sample: Sample) -> dict:
    return {
        "dataset": sample.dataset,
        "sample_id": sample.sample_id,
        "instruction": sample.instruction,
        "question": sample.question,
        "hint": sample.hint,
        "choices": sample.choices,
        "reference_answer": sample.reference_answer,
        "meta": sample.meta,
    }


def build_cache_record(sample: Sample, annotation: Optional[dict], raw_response: str, issues: List[str], teacher_model: str) -> dict:
    sample_payload = sample_to_serializable(sample)
    meta = dict(sample_payload.get("meta") or {})
    meta.setdefault("teacher_model", teacher_model)
    record = {
        "format_version": "v2",
        "dataset": sample.dataset,
        "sample": sample_payload,
        "meta": meta,
        "raw_teacher_response": raw_response,
        "issues": issues,
        "valid": bool(annotation is not None and not issues),
        "teacher_model": teacher_model,
        "collected_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if isinstance(annotation, dict):
        for field in TEACHER_REQUIRED_FIELDS:
            record[field] = str(annotation.get(field, "") or "")
        record["teacher_annotation"] = {
            "format_version": str(annotation.get("format_version") or "v2"),
            "observed_facts_visual": record["observed_facts_visual"],
            "context_textual": record["context_textual"],
            "reasoning": record["reasoning"],
            "answer": record["answer"],
        }
    else:
        record["teacher_annotation"] = None
    return record


def _validate_resume_compatibility(dataset_name: str, sample_map: Dict[str, dict], expected_sampling_scheme: Optional[str] = None) -> None:
    if not sample_map:
        return
    if str(dataset_name).lower() != "scienceqa":
        return
    first_record = next(iter(sample_map.values()))
    sample_meta = (((first_record or {}).get("sample") or {}).get("meta") or {})
    direct_meta = ((first_record or {}).get("meta") or {})
    meta = sample_meta if isinstance(sample_meta, dict) and sample_meta else direct_meta
    scheme = str((meta or {}).get("sampling_scheme", "") or "").strip()
    if not scheme:
        raise RuntimeError(
            "Existing ScienceQA cache has no sampling_scheme marker. "
            "Do not continue writing into this file. Start a new ScienceQA output file or migrate the old cache first."
        )
    if expected_sampling_scheme and scheme != str(expected_sampling_scheme).strip():
        raise RuntimeError(
            f"Existing ScienceQA cache uses sampling_scheme={scheme}, but current run uses {expected_sampling_scheme}. "
            "Do not continue writing into this file. Start a new ScienceQA output file for the new scheme."
        )


def collect_dataset_samples_streaming_parallel(
    client: TeacherClient,
    samples_iter: Iterable[Sample],
    dataset_name: str,
    dataset_path: str,
    split: str,
    output_path: Path,
    teacher_model: str,
    budget: dict,
    teacher_lang: str,
    max_workers: int,
    per_sample_attempts: int,
    save_every: int,
    image_detail: Optional[str] = None,
    image_max_side: Optional[int] = None,
    expected_sampling_scheme: Optional[str] = None,
) -> dict:
    sample_map = load_existing_sample_map(output_path)
    _validate_resume_compatibility(dataset_name, sample_map, expected_sampling_scheme=expected_sampling_scheme)
    runtime_meta: Dict[str, Any] = {}
    lock = threading.Lock()
    processed = 0
    inflight = {}

    def _worker(sample: Sample) -> Tuple[str, dict]:
        try:
            ann, raw_response, issues = collect_teacher_annotation(
                client=client,
                sample=sample,
                budget=budget,
                teacher_lang=teacher_lang,
                attempts=per_sample_attempts,
                image_detail=image_detail,
                image_max_side=image_max_side,
            )
        except Exception as exc:  # noqa: BLE001
            ann = None
            raw_response = ""
            issues = [f"worker_exception:{type(exc).__name__}", str(exc)]
        return sample.sample_id, build_cache_record(
            sample=sample,
            annotation=ann,
            raw_response=raw_response,
            issues=issues,
            teacher_model=teacher_model,
        )

    def _save_partial() -> None:
        payload = build_output_payload(dataset_name, dataset_path, split, teacher_model, budget, sample_map, extra_meta=runtime_meta or None)
        save_json(output_path, payload)

    def _flush_done(done_futures) -> None:
        nonlocal processed
        for future in done_futures:
            sample_id, record = future.result()
            inflight.pop(future, None)
            with lock:
                sample_map[sample_id] = record
                processed += 1
                if processed % max(1, int(save_every)) == 0:
                    _save_partial()

    _save_partial()
    with ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as executor:
        try:
            for sample in samples_iter:
                if record_is_training_ready(sample_map.get(sample.sample_id)):
                    continue
                future = executor.submit(_worker, sample)
                inflight[future] = sample.sample_id
                if len(inflight) >= max(1, int(max_workers)):
                    done, _ = wait(set(inflight.keys()), return_when=FIRST_COMPLETED)
                    _flush_done(done)
        except Exception as exc:  # noqa: BLE001
            runtime_meta["iterator_exception"] = f"{type(exc).__name__}: {exc}"
            print(f"[Collector] Iterator warning dataset={dataset_name}: {exc}")
        finally:
            while inflight:
                done, _ = wait(set(inflight.keys()), return_when=FIRST_COMPLETED)
                _flush_done(done)

    payload = build_output_payload(dataset_name, dataset_path, split, teacher_model, budget, sample_map, extra_meta=runtime_meta or None)
    save_json(output_path, payload)
    return payload


def collect_dataset_samples_parallel(
    client: TeacherClient,
    samples: List[Sample],
    dataset_name: str,
    dataset_path: str,
    split: str,
    output_path: Path,
    teacher_model: str,
    budget: dict,
    teacher_lang: str,
    max_workers: int,
    per_sample_attempts: int,
    save_every: int,
    image_detail: Optional[str] = None,
    image_max_side: Optional[int] = None,
    expected_sampling_scheme: Optional[str] = None,
) -> dict:
    sample_map = load_existing_sample_map(output_path)
    _validate_resume_compatibility(dataset_name, sample_map, expected_sampling_scheme=expected_sampling_scheme)
    lock = threading.Lock()
    pending = [sample for sample in samples if not record_is_training_ready(sample_map.get(sample.sample_id))]
    processed = 0

    def _worker(sample: Sample) -> Tuple[str, dict]:
        try:
            ann, raw_response, issues = collect_teacher_annotation(
                client=client,
                sample=sample,
                budget=budget,
                teacher_lang=teacher_lang,
                attempts=per_sample_attempts,
                image_detail=image_detail,
                image_max_side=image_max_side,
            )
        except Exception as exc:  # noqa: BLE001
            ann = None
            raw_response = ""
            issues = [f"worker_exception:{type(exc).__name__}", str(exc)]
        return sample.sample_id, build_cache_record(
            sample=sample,
            annotation=ann,
            raw_response=raw_response,
            issues=issues,
            teacher_model=teacher_model,
        )

    if pending:
        with ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as executor:
            futures = [executor.submit(_worker, sample) for sample in pending]
            for future in as_completed(futures):
                sample_id, record = future.result()
                with lock:
                    sample_map[sample_id] = record
                    processed += 1
                    if processed % max(1, int(save_every)) == 0 or processed == len(pending):
                        payload = build_output_payload(dataset_name, dataset_path, split, teacher_model, budget, sample_map)
                        save_json(output_path, payload)
    payload = build_output_payload(dataset_name, dataset_path, split, teacher_model, budget, sample_map)
    save_json(output_path, payload)
    return payload
