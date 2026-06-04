from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .clients import build_teacher_client
from .common import (
    DEFAULT_BUDGET,
    build_output_payload,
    collect_dataset_samples_streaming_parallel,
    load_existing_sample_map,
    sanitize_model_tag,
    save_json,
    set_hf_env_if_missing,
)
from .config import deep_update, load_config, resolve_path
from .datasets import (
    dataset_count,
    dataset_name,
    dataset_path,
    dataset_seed,
    dataset_split,
    iter_samples,
    render_config,
)
from .validate import validate_cache, validate_dataset_specs


def _config_dir(config: Dict[str, Any]) -> Optional[str]:
    return config.get("_config_dir")


def _path_base(config: Dict[str, Any]) -> Optional[str]:
    config_dir = _config_dir(config)
    if not config_dir:
        return None
    path = Path(config_dir)
    if path.name == "configs":
        return str(path.parent)
    return str(path)


def _dataset_specs(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    specs = config.get("datasets", [])
    if isinstance(specs, dict):
        specs = [specs]
    if not isinstance(specs, list) or not specs:
        raise ValueError("Config must contain a non-empty datasets list")
    return [dict(item) for item in specs if isinstance(item, dict)]


def _budget(config: Dict[str, Any]) -> Dict[str, int]:
    budget = dict(DEFAULT_BUDGET)
    budget.update({k: int(v) for k, v in dict(config.get("budget", {})).items() if k in budget})
    return budget


def _provider(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    provider = dict(config.get("provider", {}))
    overrides = {
        "model": args.model,
        "base_url": args.base_url,
        "api_key": args.api_key,
        "type": args.provider_type,
    }
    for key, value in overrides.items():
        if value:
            provider[key] = value
    return provider


def _sampling(config: Dict[str, Any]) -> Dict[str, Any]:
    sampling = {
        "teacher_lang": "en",
        "max_workers": 8,
        "per_sample_attempts": 1,
        "save_every": 50,
        "seed": 20240306,
    }
    sampling.update(dict(config.get("sampling", {})))
    return sampling


def _default_output_path(config: Dict[str, Any], spec: Dict[str, Any], model: str) -> Path:
    name = dataset_name(spec)
    split = dataset_split(spec, name)
    count = dataset_count(spec, name)
    output_dir = str(config.get("output_dir") or "outputs")
    tag = sanitize_model_tag(model)
    filename = f"{name}_teacher_{tag}_{split}_n{count}.json"
    return resolve_path(str(Path(output_dir) / filename), _path_base(config))


def _output_path(config: Dict[str, Any], spec: Dict[str, Any], model: str, args_output: Optional[str]) -> Path:
    if args_output:
        return resolve_path(args_output, _path_base(config))
    if spec.get("output_path"):
        return resolve_path(str(spec["output_path"]), _path_base(config))
    return _default_output_path(config, spec, model)


def command_sample(args: argparse.Namespace) -> None:
    set_hf_env_if_missing()
    config = load_config(args.config)
    if args.dataset:
        requested = {x.strip().lower() for x in args.dataset.split(",") if x.strip()}
        specs = [spec for spec in _dataset_specs(config) if dataset_name(spec) in requested]
        if not specs:
            raise ValueError(f"No configured dataset matched --dataset={args.dataset}")
    else:
        specs = _dataset_specs(config)

    provider = _provider(config, args)
    client = build_teacher_client(provider)
    model = str(provider["model"])
    budget = _budget(config)
    sampling = _sampling(config)
    summary = {
        "format_version": "portable_sample_summary_v1",
        "teacher_model": model,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "datasets": [],
    }

    for spec in specs:
        name = dataset_name(spec)
        path = args.dataset_path or dataset_path(spec, name)
        if args.dataset_path:
            spec = deep_update(spec, {"path": path})
        split = dataset_split(spec, name)
        seed = dataset_seed(spec, int(sampling["seed"]))
        output_path = _output_path(config, spec, model, args.output_path if len(specs) == 1 else None)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        existing = load_existing_sample_map(output_path)
        save_json(
            output_path,
            build_output_payload(
                dataset_name=name,
                dataset_path=path,
                split=split,
                teacher_model=model,
                budget=budget,
                sample_map=existing,
                extra_meta={
                    "config_path": config.get("_config_path"),
                    "source_type": spec.get("source_type"),
                    "alignment_path": spec.get("alignment_path"),
                    "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
            ),
        )
        print(f"[sample] dataset={name} path={path} split={split} seed={seed} output={output_path}")
        render = render_config(spec)
        payload = collect_dataset_samples_streaming_parallel(
            client=client,
            samples_iter=iter_samples(spec, default_seed=seed),
            dataset_name=name,
            dataset_path=path,
            split=split,
            output_path=output_path,
            teacher_model=model,
            budget=budget,
            teacher_lang=str(sampling.get("teacher_lang", "en")),
            max_workers=int(sampling.get("max_workers", 8)),
            per_sample_attempts=int(sampling.get("per_sample_attempts", 1)),
            save_every=int(sampling.get("save_every", 50)),
            image_detail=render.get("image_detail"),
            image_max_side=render.get("image_max_side"),
            expected_sampling_scheme=None,
        )
        total = len(payload.get("samples", {}))
        valid = sum(1 for record in payload.get("samples", {}).values() if isinstance(record, dict) and record.get("valid"))
        summary["datasets"].append(
            {
                "dataset": name,
                "path": path,
                "split": split,
                "output_path": str(output_path),
                "sample_count": total,
                "valid_samples": valid,
                "valid_rate": (valid / total) if total else 0.0,
            }
        )
        print(f"[sample] finished dataset={name} saved={total} valid={valid}")

    summary["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    summary_path = resolve_path(
        str(config.get("summary_path") or f"outputs/collection_summary_{sanitize_model_tag(model)}.json"),
        _path_base(config),
    )
    save_json(summary_path, summary)
    print(f"[sample] summary={summary_path}")


def command_validate_dataset(args: argparse.Namespace) -> None:
    set_hf_env_if_missing()
    config = load_config(args.config)
    specs = _dataset_specs(config)
    if args.dataset:
        requested = {x.strip().lower() for x in args.dataset.split(",") if x.strip()}
        specs = [spec for spec in specs if dataset_name(spec) in requested]
    report = validate_dataset_specs(specs, default_seed=int(_sampling(config)["seed"]), preview_count=args.preview_count)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def command_validate_cache(args: argparse.Namespace) -> None:
    print(json.dumps(validate_cache(args.path), ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Portable teacher sampler")
    sub = parser.add_subparsers(dest="command", required=True)

    sample = sub.add_parser("sample", help="Collect teacher samples")
    sample.add_argument("--config", required=True)
    sample.add_argument("--dataset", default="")
    sample.add_argument("--dataset-path", default="")
    sample.add_argument("--output-path", default="")
    sample.add_argument("--provider-type", default="")
    sample.add_argument("--model", default="")
    sample.add_argument("--base-url", default="")
    sample.add_argument("--api-key", default="")
    sample.set_defaults(func=command_sample)

    validate_dataset = sub.add_parser("validate-dataset", help="Validate dataset loading and sample construction")
    validate_dataset.add_argument("--config", required=True)
    validate_dataset.add_argument("--dataset", default="")
    validate_dataset.add_argument("--preview-count", type=int, default=3)
    validate_dataset.set_defaults(func=command_validate_dataset)

    validate_cache_parser = sub.add_parser("validate-cache", help="Validate an output teacher cache")
    validate_cache_parser.add_argument("--path", required=True)
    validate_cache_parser.set_defaults(func=command_validate_cache)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
