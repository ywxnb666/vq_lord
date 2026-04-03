import _eval_final_bootstrap  # noqa: F401

import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

from datasets import DownloadConfig, load_dataset
from PIL import Image
from tqdm import tqdm

from data_collector2 import GPT4VDataCollector
from mm_eval_suite_utils import (
    DEFAULT_SYSTEMLESS_OPEN_ENDED_SUFFIX,
    answers_match_open_ended,
    blur_image,
    clone_image,
    downsample_image,
    make_blank_image_like,
    shuffle_choices,
)
from sciqa_process import build_prompt as build_simple_scienceqa_prompt
from sciqa_process2 import build_legacy_instruction, extract_choice_from_output


@dataclass
class TeacherSample:
    sample_id: str
    dataset_name: str
    question: str
    task_type: str
    image_loader: Callable[[], Image.Image]
    choices: Optional[List[str]] = None
    answer_idx: Optional[int] = None
    answers: Optional[List[str]] = None
    hint: str = ""
    metadata: Optional[Dict] = None


def force_online_dataset_access() -> None:
    for key in ("HF_DATASETS_OFFLINE", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "DATASETS_OFFLINE"):
        os.environ[key] = "0"

    try:
        from datasets import config as datasets_config
        datasets_config.HF_DATASETS_OFFLINE = False
    except Exception:
        pass

    try:
        from huggingface_hub import constants as hub_constants
        hub_constants.HF_HUB_OFFLINE = False
    except Exception:
        pass


def _load_dataset_online(dataset_name: str, split: str):
    force_online_dataset_access()
    return load_dataset(
        dataset_name,
        split=split,
        download_config=DownloadConfig(local_files_only=False),
    )


def _load_dataset_image(dataset, idx: int) -> Image.Image:
    image = dataset[idx].get("image")
    if image is None:
        raise ValueError(f"sample at index {idx} has no image")
    cloned = clone_image(image)
    if hasattr(cloned, "convert"):
        return cloned.convert("RGB")
    return cloned


def _make_image_loader(dataset, idx: int) -> Callable[[], Image.Image]:
    return lambda dataset=dataset, idx=idx: _load_dataset_image(dataset, idx)


def load_teacher_ai2d_samples(dataset_name: str, split: str, max_samples: int = 0) -> List[TeacherSample]:
    dataset = _load_dataset_online(dataset_name, split=split)
    samples: List[TeacherSample] = []
    for idx, item in enumerate(dataset):
        image = item.get("image")
        question = item.get("question") or item.get("query") or ""
        options = item.get("options") or item.get("choices") or item.get("answer_texts")
        answer = item.get("answer")
        if image is None or not question or not options:
            continue
        if isinstance(answer, str) and len(answer.strip()) == 1 and answer.strip().isalpha():
            answer_idx = ord(answer.strip().upper()) - 65
        else:
            answer_idx = int(answer)
        samples.append(
            TeacherSample(
                sample_id=f"{split}_{idx}",
                dataset_name="ai2d",
                question=question,
                task_type="mcq",
                image_loader=_make_image_loader(dataset, idx),
                choices=list(options),
                answer_idx=answer_idx,
                metadata={"raw_answer": answer},
            )
        )
        if max_samples > 0 and len(samples) >= max_samples:
            break
    return samples


def load_teacher_chartqa_samples(dataset_name: str, split: str, max_samples: int = 0) -> List[TeacherSample]:
    dataset = _load_dataset_online(dataset_name, split=split)
    samples: List[TeacherSample] = []
    for idx, item in enumerate(dataset):
        image = item.get("image")
        question = item.get("query") or item.get("question") or ""
        label = item.get("label") if "label" in item else item.get("answer")
        if image is None or not question or label is None:
            continue
        if isinstance(label, list):
            answers = [str(x) for x in label if str(x).strip()]
        else:
            answers = [str(label)]
        samples.append(
            TeacherSample(
                sample_id=f"{split}_{idx}",
                dataset_name="chartqa",
                question=question,
                task_type="open",
                image_loader=_make_image_loader(dataset, idx),
                answers=answers,
                metadata={"human_or_machine": item.get("human_or_machine")},
            )
        )
        if max_samples > 0 and len(samples) >= max_samples:
            break
    return samples


def load_teacher_scienceqa_samples(scienceqa_path: str, split: str, max_samples: int = 0) -> List[TeacherSample]:
    dataset = _load_dataset_online(scienceqa_path, split=split)
    samples: List[TeacherSample] = []
    for idx, item in enumerate(dataset):
        image = item.get("image")
        if image is None:
            continue
        samples.append(
            TeacherSample(
                sample_id=f"{split}_{idx}",
                dataset_name="scienceqa",
                question=item.get("question", ""),
                task_type="mcq",
                image_loader=_make_image_loader(dataset, idx),
                choices=list(item.get("choices", [])),
                answer_idx=int(item.get("answer", 0)),
                hint=item.get("hint", "") or "",
                metadata={
                    "subject": item.get("subject"),
                    "grade": item.get("grade"),
                    "topic": item.get("topic"),
                    "lecture": item.get("lecture"),
                },
            )
        )
        if max_samples > 0 and len(samples) >= max_samples:
            break
    return samples


def build_teacher_control_variant(control_name: str, sample: TeacherSample, sample_idx: int, samples: Sequence[TeacherSample], shuffle_seed: int) -> TeacherSample:
    question = sample.question
    choices = list(sample.choices or [])
    answer_idx = int(sample.answer_idx)
    hint = sample.hint
    metadata = dict(sample.metadata or {})
    metadata["control_name"] = control_name

    if control_name == "baseline":
        image_loader = sample.image_loader
    elif control_name == "text_only_blank":
        image_loader = lambda base=sample.image_loader: make_blank_image_like(base())
    elif control_name == "hint_ablation":
        image_loader = sample.image_loader
        hint = ""
    elif control_name == "option_shuffle":
        choices, answer_idx, permutation = shuffle_choices(choices, answer_idx, seed=shuffle_seed + sample_idx)
        metadata["choice_permutation"] = permutation
        image_loader = sample.image_loader
    elif control_name == "random_image_swap":
        swap_idx = (sample_idx + 1) % len(samples)
        metadata["swap_sample_id"] = samples[swap_idx].sample_id
        image_loader = samples[swap_idx].image_loader
    elif control_name == "image_blur":
        image_loader = lambda base=sample.image_loader: blur_image(base())
    elif control_name == "image_downsample":
        image_loader = lambda base=sample.image_loader: downsample_image(base())
    else:
        raise ValueError(f"Unsupported control: {control_name}")

    return TeacherSample(
        sample_id=sample.sample_id,
        dataset_name=sample.dataset_name,
        question=question,
        task_type=sample.task_type,
        image_loader=image_loader,
        choices=choices,
        answer_idx=answer_idx,
        answers=(list(sample.answers) if sample.answers is not None else None),
        hint=hint,
        metadata=metadata,
    )


def build_teacher_mcq_prompt(
    question: str,
    choices: Sequence[str],
    hint: str = "",
    prompt_style: str = "legacy",
    dataset_name: str = "",
) -> str:
    if dataset_name == "scienceqa":
        if prompt_style == "simple":
            return build_simple_scienceqa_prompt(question, list(choices))
        return f"<image>\n{build_legacy_instruction(question, list(choices), hint)}"

    if prompt_style == "simple":
        prompt = build_simple_scienceqa_prompt(question, list(choices))
        if prompt.startswith("<image>\n"):
            prompt = prompt[len("<image>\n") :]
        return prompt

    instruction = build_legacy_instruction(question, list(choices), hint).rstrip()
    if instruction.endswith("Answer:"):
        instruction = instruction[: -len("Answer:")].rstrip()
    return instruction + "\nAnswer with only one option letter (A, B, C, ...). Do not explain.\nAnswer:"


def build_teacher_open_prompt(question: str) -> str:
    return question.strip() + DEFAULT_SYSTEMLESS_OPEN_ENDED_SUFFIX


def _normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def _extract_explicit_final_letter(text: str, num_choices: int) -> Optional[int]:
    if not text or num_choices <= 0:
        return None

    max_letter = chr(65 + num_choices - 1)
    letter_class = f"A-{max_letter}"
    patterns = [
        rf"(?:final answer|final choice|final option|correct answer|correct option|best answer|answer|答案|最终答案|正确答案|答案是|应选|应该选)\s*(?:is|:|：|为)?\s*(?:option|choice|选项)?\s*\(?\s*([{letter_class}])\s*\)?",
        rf"(?:i choose|i pick|i select|choose|pick|select|go with|my answer is|我选|我选择|选择)\s*(?:option|choice|选项)?\s*\(?\s*([{letter_class}])\s*\)?",
        rf"(?:option|choice|选项)\s*\(?\s*([{letter_class}])\s*\)?\s*(?:is correct|is the answer|正确|为正确答案)?",
    ]

    best_idx = None
    best_pos = -1
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            pos = match.start()
            idx = ord(match.group(1).upper()) - 65
            if 0 <= idx < num_choices and pos >= best_pos:
                best_idx = idx
                best_pos = pos
    return best_idx


def _extract_choice_text_from_tail(text: str, choices: Sequence[str]) -> Optional[int]:
    normalized_text = _normalize_text(text)
    hits: List[int] = []
    for idx, choice in enumerate(choices):
        norm_choice = _normalize_text(choice)
        if len(norm_choice) < 3:
            continue
        if norm_choice and norm_choice in normalized_text:
            hits.append(idx)
    unique_hits = sorted(set(hits))
    if len(unique_hits) == 1:
        return unique_hits[0]
    return None


def extract_choice_from_teacher_output(output_text: str, choices: Sequence[str]) -> Optional[int]:
    if not output_text:
        return None

    raw = output_text.strip()
    if not raw:
        return None

    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    last_lines = "\n".join(lines[-3:]) if lines else raw
    tail = raw[-400:]

    direct = extract_choice_from_output(raw, list(choices))
    direct_last = extract_choice_from_output(last_lines, list(choices))
    if direct_last is not None and len(raw) <= 160:
        return direct_last

    for segment in (last_lines, tail, raw):
        explicit = _extract_explicit_final_letter(segment, len(choices))
        if explicit is not None:
            return explicit

    for segment in (last_lines, tail):
        text_match = _extract_choice_text_from_tail(segment, choices)
        if text_match is not None:
            return text_match

    if direct_last is not None:
        return direct_last
    if len(raw) <= 80:
        return direct
    return None


def load_teacher_collector(
    victim_model: str,
    teacher_api_base: str,
    teacher_api_key: str,
    max_retries: int,
    enable_thinking: Optional[bool],
    save_dir: str,
) -> GPT4VDataCollector:
    collector = GPT4VDataCollector(
        api_key=(teacher_api_key if teacher_api_key else None),
        base_url=(teacher_api_base if teacher_api_base else None),
        model=victim_model,
        save_dir=save_dir,
        max_retries=max_retries,
        enable_thinking=enable_thinking,
    )
    if not collector.api_key:
        raise RuntimeError("缺少教师 API Key，请通过 --teacher_api_key 或环境变量提供。")
    return collector


def teacher_model_info(collector: GPT4VDataCollector) -> dict:
    return {
        "mode": "teacher_api",
        "victim_model": collector.model,
        "teacher_api_base": collector.base_url or "",
        "enable_thinking": collector.enable_thinking,
        "save_dir": os.path.abspath(collector.save_dir),
    }


def run_teacher_mcq_samples(
    collector: GPT4VDataCollector,
    samples: Sequence[TeacherSample],
    max_new_tokens: int,
    prompt_style: str,
    sleep_sec: float = 0.0,
    max_concurrency: int = 1,
    progress_desc: str = "teacher mcq",
) -> List[dict]:
    def _run_one(sample: TeacherSample) -> dict:
        prompt = build_teacher_mcq_prompt(
            question=sample.question,
            choices=sample.choices or [],
            hint=sample.hint,
            prompt_style=prompt_style,
            dataset_name=sample.dataset_name,
        )
        image = sample.image_loader()
        try:
            output_text = collector.query_gpt4v_image(
                image=image,
                prompt=prompt,
                max_tokens=max_new_tokens,
                image_format="PNG",
            )
        finally:
            try:
                image.close()
            except Exception:
                pass
        api_failed = output_text is None
        if api_failed:
            output_text = "[api_failed]"
        pred_idx = extract_choice_from_teacher_output(output_text, list(sample.choices or []))
        result = {
            "sample_id": sample.sample_id,
            "pred_idx": pred_idx,
            "output": output_text,
            "correct": pred_idx == sample.answer_idx,
            "api_failed": api_failed,
        }
        if sleep_sec > 0:
            import time
            time.sleep(sleep_sec)
        return result

    results: List[Optional[dict]] = [None] * len(samples)
    if max_concurrency <= 1 or len(samples) <= 1:
        for idx, sample in enumerate(tqdm(samples, desc=progress_desc, dynamic_ncols=True, leave=True)):
            results[idx] = _run_one(sample)
        return [row for row in results if row is not None]

    with tqdm(total=len(samples), desc=progress_desc, dynamic_ncols=True, leave=True) as pbar:
        with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
            future_to_idx = {executor.submit(_run_one, sample): idx for idx, sample in enumerate(samples)}
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                results[idx] = future.result()
                pbar.update(1)

    return [row for row in results if row is not None]


def run_teacher_open_samples(
    collector: GPT4VDataCollector,
    samples: Sequence[TeacherSample],
    max_new_tokens: int,
    sleep_sec: float = 0.0,
    max_concurrency: int = 1,
    progress_desc: str = "teacher open",
) -> List[dict]:
    def _run_one(sample: TeacherSample) -> dict:
        prompt = build_teacher_open_prompt(sample.question)
        image = sample.image_loader()
        try:
            output_text = collector.query_gpt4v_image(
                image=image,
                prompt=prompt,
                max_tokens=max_new_tokens,
                image_format="PNG",
            )
        finally:
            try:
                image.close()
            except Exception:
                pass
        api_failed = output_text is None
        if api_failed:
            output_text = "[api_failed]"
        result = {
            "sample_id": sample.sample_id,
            "output": output_text,
            "correct": answers_match_open_ended(output_text, sample.answers or []),
            "api_failed": api_failed,
        }
        if sleep_sec > 0:
            import time
            time.sleep(sleep_sec)
        return result

    results: List[Optional[dict]] = [None] * len(samples)
    if max_concurrency <= 1 or len(samples) <= 1:
        for idx, sample in enumerate(tqdm(samples, desc=progress_desc, dynamic_ncols=True, leave=True)):
            results[idx] = _run_one(sample)
        return [row for row in results if row is not None]

    with tqdm(total=len(samples), desc=progress_desc, dynamic_ncols=True, leave=True) as pbar:
        with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
            future_to_idx = {executor.submit(_run_one, sample): idx for idx, sample in enumerate(samples)}
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                results[idx] = future.result()
                pbar.update(1)

    return [row for row in results if row is not None]
