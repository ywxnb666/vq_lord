#!/usr/bin/env python3
import argparse
import json
import os
import tempfile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trim a ScienceQA teacher cache JSON to the first N samples."
    )
    parser.add_argument("--input-path", required=True, help="Full teacher cache path")
    parser.add_argument("--output-path", required=True, help="Trimmed cache path")
    parser.add_argument("--sample-count", required=True, type=int, help="Samples to keep")
    parser.add_argument(
        "--train-num",
        type=int,
        default=None,
        help="Override train_num metadata; defaults to sample-count",
    )
    return parser.parse_args()


def atomic_dump_json(path: str, payload: dict) -> None:
    output_dir = os.path.dirname(path) or "."
    os.makedirs(output_dir, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=output_dir,
        delete=False,
    ) as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        temp_path = f.name
    os.replace(temp_path, path)


def _sample_sort_key(item_with_position):
    position, _, value = item_with_position
    meta = value.get("meta", {}) if isinstance(value, dict) else {}
    sample_id = meta.get("sample_id")
    if isinstance(sample_id, int):
        return (0, sample_id, position)
    return (1, position, position)


def main() -> None:
    args = parse_args()
    if args.sample_count <= 0:
        raise ValueError(f"sample-count must be positive, got {args.sample_count}")

    with open(args.input_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid teacher cache payload: {args.input_path}")

    samples = payload.get("samples")
    if not isinstance(samples, dict):
        raise RuntimeError(f"Invalid teacher cache samples field: {args.input_path}")

    total = len(samples)
    if total < args.sample_count:
        raise RuntimeError(
            f"Requested {args.sample_count} samples, but only found {total} in {args.input_path}"
        )

    enumerated_items = [
        (position, key, value) for position, (key, value) in enumerate(samples.items())
    ]
    ordered_items = sorted(enumerated_items, key=_sample_sort_key)
    trimmed_pairs = [(key, value) for _, key, value in ordered_items[: args.sample_count]]
    trimmed_samples = dict(trimmed_pairs)
    trimmed_payload = dict(payload)
    trimmed_payload["train_num"] = (
        int(args.train_num) if args.train_num is not None else int(args.sample_count)
    )
    trimmed_payload["samples"] = trimmed_samples

    atomic_dump_json(args.output_path, trimmed_payload)

    first_key = next(iter(trimmed_samples), None)
    last_key = next(reversed(trimmed_samples), None)
    first_sample_id = None
    last_sample_id = None
    if first_key is not None:
        first_sample_id = trimmed_samples[first_key].get("meta", {}).get("sample_id")
    if last_key is not None:
        last_sample_id = trimmed_samples[last_key].get("meta", {}).get("sample_id")
    print(
        "trim_teacher_cache:"
        f" input={args.input_path}"
        f" total={total}"
        f" output={args.output_path}"
        f" kept={len(trimmed_samples)}"
        f" train_num={trimmed_payload['train_num']}"
        f" first_key={first_key}"
        f" first_sample_id={first_sample_id}"
        f" last_key={last_key}"
        f" last_sample_id={last_sample_id}"
    )


if __name__ == "__main__":
    main()
