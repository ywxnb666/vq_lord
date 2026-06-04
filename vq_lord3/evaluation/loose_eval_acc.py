#!/usr/bin/env python3
"""Loose accuracy checker for VQ-LoRD evaluation result JSON files.

The script is dataset-agnostic as long as each result row contains:
`choices`, `answer_idx`, and generated text in `output` or `first_pass_output`.
It avoids counting echoed prompt options as answers by preferring the parsed
`readable_fields.answer` / final `Answer:` region, then stripping Context and
Options blocks before falling back to answer-text matching.
"""

import argparse
import json
import re
import string
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


NUMBER_WORDS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "thirteen": "13",
    "fourteen": "14",
    "fifteen": "15",
    "sixteen": "16",
    "seventeen": "17",
    "eighteen": "18",
    "nineteen": "19",
    "twenty": "20",
}


def load_rows(path: Path) -> Tuple[Any, List[Dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("results"), list):
        return payload, payload["results"]
    if isinstance(payload, list):
        return payload, payload
    raise ValueError(f"unsupported result JSON structure: {path}")


def normalize_text(value: Any) -> str:
    text = str(value).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip().strip(string.punctuation + " ")


def normalized_variants(value: Any) -> List[str]:
    base = normalize_text(value)
    variants = [base] if base else []
    for word, digit in NUMBER_WORDS.items():
        if base == digit:
            variants.append(word)
        elif base == word:
            variants.append(digit)
    return list(dict.fromkeys(v for v in variants if v))


def strip_context_options(text: str) -> str:
    text = re.sub(r"(?is)Context\s*:\s*Question\s*:.*?(?=Reasoning\s*:|Answer\s*:|$)", "", text)
    text = re.sub(r"(?im)^\s*Options\s*:\s*$", "", text)
    text = re.sub(r"(?im)^\s*\([A-Z]\)\s+.*$", "", text)
    return text


def answer_region(row: Dict[str, Any]) -> Tuple[str, str]:
    fields = row.get("readable_fields") or {}
    if isinstance(fields, dict):
        answer = fields.get("answer")
        if isinstance(answer, str) and answer.strip():
            return answer, "answer_field"

    output = row.get("output") or row.get("first_pass_output") or row.get("prediction") or ""
    output = str(output)
    matches = list(re.finditer(r"(?i)\banswer\s*:", output))
    if matches:
        return output[matches[-1].end() :], "after_answer_prefix"
    return strip_context_options(output), "no_answer_prefix_stripped"


def label_chars(num_choices: int) -> str:
    if num_choices <= 0 or num_choices > 26:
        raise ValueError(f"choices length must be 1..26, got {num_choices}")
    return string.ascii_uppercase[:num_choices]


def parse_choice_idx(text: str, choices: List[Any]) -> Optional[int]:
    if not isinstance(text, str) or not text.strip() or not choices:
        return None

    labels = label_chars(len(choices))
    label_class = re.escape(labels)
    head = text.strip()[:500]

    label_patterns = [
        rf"(?i)(?:^|[^\w])\(\s*([{label_class}])\s*\)",
        rf"(?i)^\s*([{label_class}])\s*[\).:]",
        rf"(?i)\b(?:answer|choice|option)\s*(?:is|:)?\s*\(?\s*([{label_class}])\s*\)?\b",
        rf"(?i)\bthe\s+(?:answer|choice|option)\s+is\s+([{label_class}])\b",
    ]
    for pattern in label_patterns:
        match = re.search(pattern, head)
        if match:
            return labels.index(match.group(1).upper())

    normalized_output = normalize_text(text)
    hits: List[int] = []
    for idx, choice in sorted(enumerate(choices), key=lambda item: len(str(item[1])), reverse=True):
        for variant in normalized_variants(choice):
            if re.search(r"(?<!\w)" + re.escape(variant) + r"(?!\w)", normalized_output):
                hits.append(idx)
                break

    unique_hits = list(dict.fromkeys(hits))
    if len(unique_hits) == 1:
        return unique_hits[0]
    if unique_hits:
        return unique_hits[0]
    return None


def safe_answer_idx(row: Dict[str, Any]) -> Optional[int]:
    value = row.get("answer_idx", row.get("correct_choice_idx"))
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def compute_metrics(rows: Iterable[Dict[str, Any]], annotate: bool = False) -> Dict[str, Any]:
    rows = list(rows)
    total = len(rows)
    strict_correct = sum(1 for row in rows if row.get("correct") is True)
    stored_pred_correct = 0
    parsed = 0
    loose_correct = 0
    source_stats: Dict[str, Dict[str, int]] = {}
    miss_examples = []

    for row in rows:
        answer_idx = safe_answer_idx(row)
        choices = row.get("choices") or []
        text, source = answer_region(row)
        pred_idx = parse_choice_idx(text, choices)

        if row.get("pred_idx") == answer_idx:
            stored_pred_correct += 1
        source_stats.setdefault(source, {"n": 0, "parsed": 0, "correct": 0})
        source_stats[source]["n"] += 1

        if pred_idx is not None:
            parsed += 1
            source_stats[source]["parsed"] += 1
        if pred_idx is not None and answer_idx is not None and pred_idx == answer_idx:
            loose_correct += 1
            source_stats[source]["correct"] += 1
        elif pred_idx is None and len(miss_examples) < 10:
            miss_examples.append(
                {
                    "sample_id": row.get("sample_id"),
                    "answer_idx": answer_idx,
                    "source": source,
                    "choices": choices,
                    "text_head": text[:200].replace("\n", " "),
                }
            )

        if annotate:
            row["loose_pred_idx"] = pred_idx
            row["loose_correct"] = pred_idx is not None and answer_idx is not None and pred_idx == answer_idx
            row["loose_parse_source"] = source

    return {
        "total": total,
        "strict_correct": strict_correct,
        "strict_acc": strict_correct / total if total else 0.0,
        "stored_pred_correct": stored_pred_correct,
        "stored_pred_acc": stored_pred_correct / total if total else 0.0,
        "loose_parsed": parsed,
        "loose_parse_rate": parsed / total if total else 0.0,
        "loose_correct": loose_correct,
        "loose_acc": loose_correct / total if total else 0.0,
        "loose_acc_on_parsed": loose_correct / parsed if parsed else 0.0,
        "source_stats": source_stats,
        "miss_examples": miss_examples,
    }


def print_text_report(path: Path, original_metrics: Any, metrics: Dict[str, Any]) -> None:
    total = metrics["total"]
    print(f"file: {path}")
    print(f"n: {total}")
    if original_metrics:
        print(f"json_metrics: {json.dumps(original_metrics, ensure_ascii=False)}")
    print(f"strict_correct: {metrics['strict_correct']}/{total} = {metrics['strict_acc']:.6f}")
    print(f"stored_pred_idx_correct: {metrics['stored_pred_correct']}/{total} = {metrics['stored_pred_acc']:.6f}")
    print(f"loose_parsed: {metrics['loose_parsed']}/{total} = {metrics['loose_parse_rate']:.6f}")
    print(f"loose_correct: {metrics['loose_correct']}/{total} = {metrics['loose_acc']:.6f}")
    print(f"loose_acc_on_parsed: {metrics['loose_acc_on_parsed']:.6f}")
    print("source_stats:")
    for source, stat in metrics["source_stats"].items():
        n = stat["n"]
        print(
            f"  {source}: n={n}, parsed={stat['parsed']} ({stat['parsed'] / n if n else 0.0:.6f}), "
            f"correct={stat['correct']} ({stat['correct'] / n if n else 0.0:.6f})"
        )
    if metrics["miss_examples"]:
        print("miss_examples:")
        for item in metrics["miss_examples"]:
            print(f"  {json.dumps(item, ensure_ascii=False)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute loose MCQ accuracy from VQ-LoRD result JSON.")
    parser.add_argument("result_json", type=Path, help="Path to evaluation result JSON.")
    parser.add_argument("--json", action="store_true", help="Print metrics as JSON.")
    parser.add_argument("--save-annotated", type=Path, default=None, help="Optional path for annotated result JSON.")
    args = parser.parse_args()

    payload, rows = load_rows(args.result_json)
    metrics = compute_metrics(rows, annotate=args.save_annotated is not None)
    original_metrics = payload.get("metrics", {}) if isinstance(payload, dict) else {}

    if args.save_annotated is not None:
        if isinstance(payload, dict):
            payload["loose_metrics"] = metrics
            out_payload = payload
        else:
            out_payload = {"loose_metrics": metrics, "results": rows}
        args.save_annotated.parent.mkdir(parents=True, exist_ok=True)
        args.save_annotated.write_text(json.dumps(out_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps({"file": str(args.result_json), "original_metrics": original_metrics, **metrics}, ensure_ascii=False, indent=2))
    else:
        print_text_report(args.result_json, original_metrics, metrics)


if __name__ == "__main__":
    main()
