from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from .common import (
    DATASET_BUILDERS,
    DATASET_DEFAULT_RENDER,
    DATASET_STREAM_BUILDERS,
    DEFAULT_DATASET_SPECS,
    Sample,
)
from .local_parquet import iter_local_samples, parquet_files_from_path


SUPPORTED_DATASETS = tuple(sorted(set(DATASET_BUILDERS) | {"iconqa"}))


def dataset_name(spec: Dict[str, Any]) -> str:
    name = str(spec.get("name") or spec.get("dataset") or "").strip().lower()
    if not name:
        raise ValueError("Dataset spec requires name")
    if name not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported dataset: {name}. Supported: {', '.join(SUPPORTED_DATASETS)}")
    return name


def dataset_path(spec: Dict[str, Any], name: Optional[str] = None) -> str:
    ds_name = name or dataset_name(spec)
    default = DEFAULT_DATASET_SPECS.get(ds_name, {}).get("hf_path", "")
    return str(spec.get("path") or spec.get("dataset_path") or default).strip()


def dataset_split(spec: Dict[str, Any], name: Optional[str] = None) -> str:
    ds_name = name or dataset_name(spec)
    default = DEFAULT_DATASET_SPECS.get(ds_name, {}).get("split", "train")
    if ds_name == "iconqa" and not spec.get("split"):
        default = "val"
    return str(spec.get("split") or default).strip()


def dataset_count(spec: Dict[str, Any], name: Optional[str] = None) -> int:
    ds_name = name or dataset_name(spec)
    default = DEFAULT_DATASET_SPECS.get(ds_name, {}).get("count", 0)
    return int(spec.get("count", default))


def dataset_seed(spec: Dict[str, Any], default_seed: int) -> int:
    return int(spec.get("seed", default_seed))


def detect_source_type(spec: Dict[str, Any]) -> str:
    explicit = str(spec.get("source_type") or "").strip().lower()
    if explicit:
        return explicit
    path = dataset_path(spec)
    if path and Path(path).expanduser().exists():
        try:
            parquet_files_from_path(path, dataset_split(spec))
            return "local_parquet"
        except Exception:
            return "hf_or_parquet"
    if spec.get("alignment_path") and dataset_name(spec) == "iconqa":
        return "local_parquet"
    return "hf_or_parquet"


def iter_samples(spec: Dict[str, Any], default_seed: int) -> Iterable[Sample]:
    name = dataset_name(spec)
    path = dataset_path(spec, name)
    split = dataset_split(spec, name)
    count = dataset_count(spec, name)
    seed = dataset_seed(spec, default_seed)
    source_type = detect_source_type(spec)
    alignment_path = str(spec.get("alignment_path") or "").strip()
    start_offset = int(spec.get("start_offset", 0))

    if source_type in {"local", "local_parquet", "parquet"}:
        yield from iter_local_samples(
            dataset=name,
            path=path,
            split=split,
            count=count,
            seed=seed,
            alignment_path=alignment_path,
            start_offset=start_offset,
        )
        return

    if alignment_path:
        raise ValueError("alignment_path currently requires source_type=local_parquet")
    stream_builder = DATASET_STREAM_BUILDERS.get(name)
    if stream_builder is not None:
        yield from stream_builder(path, split, count, seed)
        return
    builder = DATASET_BUILDERS.get(name)
    if builder is None:
        raise ValueError(f"Dataset {name} requires source_type=local_parquet")
    yield from builder(path, split, count, seed)


def render_config(spec: Dict[str, Any]) -> Dict[str, Any]:
    name = dataset_name(spec)
    config = dict(DATASET_DEFAULT_RENDER.get(name, {}))
    if spec.get("image_detail") is not None:
        config["image_detail"] = spec.get("image_detail")
    if spec.get("image_max_side") is not None:
        config["image_max_side"] = int(spec.get("image_max_side"))
    return config
