import time
from typing import Iterable, List, Sequence

import torch
from tqdm import tqdm

from mm_eval_suite_utils import (
    StandardSample,
    answers_match_open_ended,
    build_generic_mcq_prompt,
    build_open_ended_prompt,
)
from sciqa_process import (
    build_prompt as build_simple_scienceqa_prompt,
    extract_choice_from_output as extract_choice_from_stage2_output,
    predict_choice_with_next_token_logits as predict_stage2_choice_with_next_token_logits,
)
from sciqa_process2 import (
    build_prompt as build_legacy_scienceqa_prompt,
    extract_choice_from_output as extract_choice_from_stage3_output,
    predict_choice_with_next_token_logits as predict_stage3_choice_with_next_token_logits,
)
from sciqa_process2_parallel import predict_choices_with_next_token_logits_batch


def sort_samples_by_image_area(samples: Sequence[StandardSample]) -> List[StandardSample]:
    return sorted(samples, key=lambda sample: (sample.image.size[0] * sample.image.size[1], sample.image.size[0], sample.image.size[1]))


def iter_batches(items: Sequence, batch_size: int):
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def _set_left_padding(processor):
    if hasattr(processor, "tokenizer") and getattr(processor.tokenizer, "padding_side", None) != "left":
        processor.tokenizer.padding_side = "left"


def _prepare_batch_inputs(processor, model, prompts: Sequence[str], images: Sequence):
    inputs = processor(
        text=list(prompts),
        images=list(images),
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
    return input_ids, attention_mask, pixel_values, image_sizes


def _is_scienceqa_sample(sample: StandardSample) -> bool:
    return getattr(sample, "dataset_name", "") == "scienceqa"


def _build_mcq_prompt(processor, sample: StandardSample, prompt_style: str) -> str:
    # Keep ScienceQA prompts identical to the original stage2/stage3 eval scripts
    # so control/ablation accuracy matches the training-time test setup.
    if _is_scienceqa_sample(sample):
        choices = list(sample.choices or [])
        if prompt_style == "simple":
            return build_simple_scienceqa_prompt(sample.question, choices)
        return build_legacy_scienceqa_prompt(processor, sample.question, choices, sample.hint)

    return build_generic_mcq_prompt(
        processor=processor,
        question=sample.question,
        choices=sample.choices or [],
        hint=sample.hint,
        prompt_style=prompt_style,
    )


def run_scienceqa_mcq_samples_exact(
    model,
    processor,
    samples: Sequence[StandardSample],
    prompt_style: str,
    answer_mode: str,
    max_new_tokens: int,
    progress_desc: str = "ScienceQA exact",
    show_progress: bool = True,
) -> List[dict]:
    if prompt_style == "simple":
        extract_choice_fn = extract_choice_from_stage2_output
        predict_choice_fn = predict_stage2_choice_with_next_token_logits
    else:
        extract_choice_fn = extract_choice_from_stage3_output
        predict_choice_fn = predict_stage3_choice_with_next_token_logits

    rows: List[dict] = []
    sample_iter = samples
    if show_progress:
        sample_iter = tqdm(samples, desc=progress_desc, dynamic_ncols=True, leave=True)

    for sample in sample_iter:
        prompt = _build_mcq_prompt(processor=processor, sample=sample, prompt_style=prompt_style)
        inputs = processor(
            text=prompt,
            images=sample.image,
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
        choices = list(sample.choices or [])

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
            pred_idx = extract_choice_fn(output_text, choices)

        if pred_idx is None and answer_mode in ("logits", "hybrid"):
            pred_idx = predict_choice_fn(
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

        rows.append(
            {
                "sample_id": sample.sample_id,
                "pred_idx": pred_idx,
                "output": output_text,
                "correct": pred_idx == sample.answer_idx,
            }
        )

    return rows


def run_mcq_batches_logits(
    model,
    processor,
    samples: Sequence[StandardSample],
    batch_size: int,
    prompt_style: str,
    progress_desc: str = "MCQ logits",
    show_progress: bool = True,
) -> List[dict]:
    _set_left_padding(processor)
    ordered_samples = sort_samples_by_image_area(samples)
    results: List[dict] = []
    if not ordered_samples:
        return results

    total_samples = len(ordered_samples)
    total_batches = (total_samples + batch_size - 1) // batch_size
    batch_iter = iter_batches(ordered_samples, batch_size)
    if show_progress:
        batch_iter = tqdm(
            batch_iter,
            total=total_batches,
            desc=progress_desc,
            unit="batch",
            dynamic_ncols=True,
            leave=True,
        )

    processed = 0
    for batch_samples in batch_iter:
        prompts = [_build_mcq_prompt(processor=processor, sample=sample, prompt_style=prompt_style) for sample in batch_samples]
        images = [sample.image for sample in batch_samples]
        input_ids, attention_mask, pixel_values, image_sizes = _prepare_batch_inputs(
            processor=processor,
            model=model,
            prompts=prompts,
            images=images,
        )
        predictions = predict_choices_with_next_token_logits_batch(
            model=model,
            tokenizer=processor.tokenizer,
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_sizes=image_sizes,
            num_choices_list=[len(sample.choices or []) for sample in batch_samples],
        )
        for sample, pred_idx in zip(batch_samples, predictions):
            results.append(
                {
                    "sample_id": sample.sample_id,
                    "pred_idx": pred_idx,
                    "output": "[logits_batch_mode]",
                    "correct": pred_idx == sample.answer_idx,
                }
            )

        processed += len(batch_samples)
        if show_progress:
            batch_iter.set_postfix_str(f"samples={processed}/{total_samples}")

    return results


def run_open_batches_generate(
    model,
    processor,
    samples: Sequence[StandardSample],
    batch_size: int,
    max_new_tokens: int,
    prompt_style: str,
    progress_desc: str = "Open generate",
    show_progress: bool = True,
) -> List[dict]:
    _set_left_padding(processor)
    ordered_samples = sort_samples_by_image_area(samples)
    results: List[dict] = []
    if not ordered_samples:
        return results

    total_samples = len(ordered_samples)
    total_batches = (total_samples + batch_size - 1) // batch_size
    batch_iter = iter_batches(ordered_samples, batch_size)
    if show_progress:
        batch_iter = tqdm(
            batch_iter,
            total=total_batches,
            desc=progress_desc,
            unit="batch",
            dynamic_ncols=True,
            leave=True,
        )

    processed = 0
    for batch_samples in batch_iter:
        prompts = [
            build_open_ended_prompt(
                processor=processor,
                question=sample.question,
                prompt_style=prompt_style,
            )
            for sample in batch_samples
        ]
        images = [sample.image for sample in batch_samples]
        input_ids, attention_mask, pixel_values, image_sizes = _prepare_batch_inputs(
            processor=processor,
            model=model,
            prompts=prompts,
            images=images,
        )
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
        for row_idx, sample in enumerate(batch_samples):
            gen_tokens = generated[row_idx][prompt_len:]
            output_text = processor.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
            if not output_text:
                output_text = "[empty_generation]"
            results.append(
                {
                    "sample_id": sample.sample_id,
                    "output": output_text,
                    "correct": answers_match_open_ended(output_text, sample.answers or []),
                }
            )

        processed += len(batch_samples)
        if show_progress:
            batch_iter.set_postfix_str(f"samples={processed}/{total_samples}")

    return results


def measure_runtime(fn, *args, **kwargs):
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    duration_sec = time.perf_counter() - start
    return result, duration_sec


def accuracy_from_results(results: Iterable[dict]) -> dict:
    rows = list(results)
    total = len(rows)
    correct = sum(1 for row in rows if row.get("correct"))
    return {
        "correct": correct,
        "total": total,
        "accuracy": (correct / total if total else 0.0),
    }
