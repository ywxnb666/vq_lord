from __future__ import annotations

import ast
import bisect
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pyarrow.parquet as pq

from .common import (
    Sample,
    _build_aokvqa_sample,
    _build_scienceqa_sample_from_item,
    _build_textvqa_sample,
    normalize_image,
    random_select_indices,
)


@dataclass
class LocalParquetDataset:
    files: List[Path]
    parquet_files: List[pq.ParquetFile]
    cumulative_rows: List[int]

    @classmethod
    def from_files(cls, files: Sequence[Path]) -> "LocalParquetDataset":
        parquet_files = [pq.ParquetFile(path) for path in files]
        cumulative_rows: List[int] = []
        total = 0
        for parquet_file in parquet_files:
            total += int(parquet_file.metadata.num_rows)
            cumulative_rows.append(total)
        return cls(files=list(files), parquet_files=parquet_files, cumulative_rows=cumulative_rows)

    def __len__(self) -> int:
        return self.cumulative_rows[-1] if self.cumulative_rows else 0

    def __getitem__(self, index: int) -> dict:
        global_index = int(index)
        file_idx = bisect.bisect_right(self.cumulative_rows, global_index)
        file_start = 0 if file_idx == 0 else self.cumulative_rows[file_idx - 1]
        local_index = global_index - file_start
        parquet_file = self.parquet_files[file_idx]
        row_start = 0
        for row_group_idx in range(parquet_file.num_row_groups):
            row_group_rows = int(parquet_file.metadata.row_group(row_group_idx).num_rows)
            if local_index < row_start + row_group_rows:
                table = parquet_file.read_row_group(row_group_idx)
                return table.slice(local_index - row_start, 1).to_pylist()[0]
            row_start += row_group_rows
        raise IndexError(global_index)


def parquet_files_from_path(path: str, split: str = "") -> List[Path]:
    root = Path(path).expanduser()
    if root.is_file():
        return [root]
    if not root.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {root}")
    split_text = str(split or "").lower()
    files = sorted(root.rglob("*.parquet"))
    if split_text:
        split_files = [p for p in files if split_text in p.name.lower() or f"/{split_text}" in str(p).lower()]
        if split_files:
            return split_files
    if files:
        return files
    raise FileNotFoundError(f"No parquet files found under: {root}")


def load_local_parquet_dataset(path: str, split: str = "") -> LocalParquetDataset:
    return LocalParquetDataset.from_files(parquet_files_from_path(path, split))


def normalize_text(value: Any) -> str:
    return str(value or "").strip()


def parse_choices(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if value is None:
        return []
    return [part.strip() for part in str(value).split(",") if part.strip()]


def parse_direct_answers(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    text = normalize_text(value)
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
    except Exception:
        return [text]
    if isinstance(parsed, list):
        return [str(x).strip() for x in parsed if str(x).strip()]
    return [text]


def option_letter(index: int) -> str:
    return chr(ord("A") + int(index))


def option_lines(choices: Sequence[str], style: str = "paren") -> str:
    if style == "dot":
        return "\n".join(f"{option_letter(i)}. {choice}" for i, choice in enumerate(choices))
    return "\n".join(f"({option_letter(i)}) {choice}" for i, choice in enumerate(choices))


def sample_source_index(sample_id: str, record: Dict[str, Any]) -> Optional[int]:
    containers = [
        record.get("meta") if isinstance(record.get("meta"), dict) else {},
        (record.get("sample") or {}).get("meta") if isinstance(record.get("sample"), dict) else {},
        record,
    ]
    for container in containers:
        if not isinstance(container, dict):
            continue
        value = container.get("raw_train_index")
        if value is None:
            value = container.get("source_index")
        if value is not None:
            return int(value)
    tail = str(sample_id).split("::")[-1]
    return int(tail) if tail.isdigit() else None


def load_alignment_targets(alignment_path: str, count: int = 0, start_offset: int = 0) -> List[Tuple[str, dict]]:
    with Path(alignment_path).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    samples = payload.get("samples", {}) if isinstance(payload, dict) else {}
    if not isinstance(samples, dict) or not samples:
        raise RuntimeError(f"Alignment cache has no dict samples: {alignment_path}")
    items = [(str(k), v) for k, v in samples.items() if isinstance(v, dict)]
    items.sort(key=lambda kv: kv[0])
    if start_offset > 0:
        items = items[int(start_offset) :]
    if count > 0:
        items = items[: int(count)]
    return items


def _build_iconqa_sample(item: dict, split: str, idx: int, sample_id: Optional[str] = None) -> Sample:
    choices = parse_choices(item.get("choices"))
    question = normalize_text(item.get("question")).replace("<image>", "").strip()
    answer_text = normalize_text(item.get("answer"))
    target_letter = ""
    answer_norm = re.sub(r"[^a-z0-9./:%+-]+", " ", answer_text.lower()).strip()
    for i, choice in enumerate(choices):
        choice_norm = re.sub(r"[^a-z0-9./:%+-]+", " ", normalize_text(choice).lower()).strip()
        if answer_norm and answer_norm == choice_norm:
            target_letter = option_letter(i)
            break
    qid = item.get("question_id", idx)
    return Sample(
        dataset="IconQA",
        sample_id=sample_id or f"iconqa::{split}::{qid}",
        image=normalize_image(item.get("query_image") or item.get("image")),
        instruction=(
            "<image>\n"
            "You are answering an IconQA visual reasoning multiple-choice question.\n"
            "Be careful: many IconQA items require exact counting, reading clocks, following an ordered picture sequence, or distinguishing slide/flip/turn.\n"
            "First solve the visual task explicitly, then map the final result to exactly one option.\n"
            "The answer field must match the final conclusion in reasoning and must use the exact option letter and option text.\n"
            "If the task asks for a position in a sequence, number the visible items from left to right before choosing.\n"
            "If the task asks for earlier/latest time, compare the shown times first, then choose the label paired with that time.\n"
            "If the task asks how many, count all relevant visible objects and do not count grid cells, frames, labels, or empty spaces unless the question asks for them.\n"
            "If the task asks slide/flip/turn, distinguish translation, mirror reflection, and rotation carefully.\n"
            f"Question: {question}\n"
            f"Options:\n{option_lines(choices, style='dot')}\n"
            "Answer:"
        ),
        question=question,
        hint="",
        choices=choices,
        reference_answer=answer_text,
        meta={
            "split": normalize_text(item.get("split")) or split,
            "source_index": idx,
            "question_id": qid,
            "answer_text": answer_text,
            "target_letter": target_letter,
            "ques_type": item.get("ques_type"),
            "grade": item.get("grade"),
            "skills": item.get("skills"),
            "label": item.get("label"),
            "alignment_sample_id": sample_id or f"iconqa::{split}::{qid}",
        },
    )


def build_local_sample(dataset: str, item: dict, split: str, idx: int, sample_id: Optional[str] = None) -> Sample:
    name = str(dataset).lower()
    if name == "scienceqa":
        sample = _build_scienceqa_sample_from_item(
            item=item,
            split=split,
            source_index=idx,
            raw_train_index=idx,
            sampling_scheme="local_parquet_v1",
        )
        if sample_id:
            sample.sample_id = sample_id
            sample.meta["alignment_sample_id"] = sample_id
        return sample
    if name == "textvqa":
        sample = _build_textvqa_sample(item, split, idx)
        if sample_id:
            sample.sample_id = sample_id
            sample.meta["alignment_sample_id"] = sample_id
        return sample
    if name == "aokvqa":
        sample = _build_aokvqa_sample(item, split, idx)
        if sample_id:
            sample.sample_id = sample_id
            sample.meta["alignment_sample_id"] = sample_id
        return sample
    if name == "iconqa":
        return _build_iconqa_sample(item, split, idx, sample_id=sample_id)
    raise ValueError(f"Unsupported local parquet dataset: {dataset}")


def iter_local_samples(
    dataset: str,
    path: str,
    split: str,
    count: int,
    seed: int,
    alignment_path: str = "",
    start_offset: int = 0,
) -> Iterable[Sample]:
    ds = load_local_parquet_dataset(path, split)
    if alignment_path:
        for sample_id, record in load_alignment_targets(alignment_path, count=count, start_offset=start_offset):
            idx = sample_source_index(sample_id, record)
            if idx is None:
                raise RuntimeError(f"Missing source index for aligned sample: {sample_id}")
            yield build_local_sample(dataset, ds[idx], split, idx, sample_id=sample_id)
        return
    selected = random_select_indices(len(ds), int(count), int(seed))
    for idx in selected:
        yield build_local_sample(dataset, ds[idx], split, idx)
