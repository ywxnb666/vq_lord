from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .common import TEACHER_REQUIRED_FIELDS, load_existing_sample_map, record_is_training_ready
from .datasets import dataset_count, dataset_name, dataset_path, dataset_split, detect_source_type, iter_samples


def _sample_summary(sample) -> Dict[str, Any]:
    return {
        "sample_id": sample.sample_id,
        "dataset": sample.dataset,
        "has_image": sample.image is not None,
        "question_nonempty": bool(str(sample.question or "").strip()),
        "choices_count": len(sample.choices or []),
        "reference_answer_nonempty": bool(str(sample.reference_answer or "").strip()),
        "meta_keys": sorted((sample.meta or {}).keys()),
    }


def validate_dataset_specs(specs: Iterable[Dict[str, Any]], default_seed: int, preview_count: int = 3) -> List[Dict[str, Any]]:
    reports: List[Dict[str, Any]] = []
    for spec in specs:
        name = dataset_name(spec)
        report: Dict[str, Any] = {
            "dataset": name,
            "path": dataset_path(spec, name),
            "split": dataset_split(spec, name),
            "count": dataset_count(spec, name),
            "source_type": detect_source_type(spec),
            "ok": False,
            "preview": [],
            "errors": [],
        }
        try:
            for idx, sample in enumerate(iter_samples(spec, default_seed=default_seed)):
                report["preview"].append(_sample_summary(sample))
                if idx + 1 >= max(1, int(preview_count)):
                    break
            if not report["preview"]:
                report["errors"].append("No samples were produced")
            report["ok"] = not report["errors"]
        except Exception as exc:  # noqa: BLE001
            report["errors"].append(f"{type(exc).__name__}: {exc}")
        reports.append(report)
    return reports


def validate_cache(path: str) -> Dict[str, Any]:
    cache_path = Path(path)
    with cache_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    samples = payload.get("samples", {}) if isinstance(payload, dict) else {}
    errors = []
    if not isinstance(payload, dict):
        errors.append("Top-level payload is not a dict")
    if not isinstance(samples, dict):
        errors.append("Top-level samples is not a dict")
        samples = {}
    ready = 0
    valid_flag = 0
    missing_required = 0
    for record in samples.values():
        if isinstance(record, dict) and record.get("valid"):
            valid_flag += 1
        if record_is_training_ready(record if isinstance(record, dict) else {}):
            ready += 1
        else:
            missing_required += 1
    return {
        "path": str(cache_path),
        "ok": not errors,
        "errors": errors,
        "format_version": payload.get("format_version") if isinstance(payload, dict) else None,
        "dataset": payload.get("dataset") if isinstance(payload, dict) else None,
        "teacher_model": payload.get("teacher_model") if isinstance(payload, dict) else None,
        "sample_count_field": payload.get("sample_count") if isinstance(payload, dict) else None,
        "sample_count_actual": len(samples),
        "valid_flag_count": valid_flag,
        "training_ready_count": ready,
        "not_training_ready_count": missing_required,
        "required_fields": list(TEACHER_REQUIRED_FIELDS),
    }


def existing_ready_count(path: str) -> int:
    return sum(1 for record in load_existing_sample_map(Path(path)).values() if record_is_training_ready(record))
