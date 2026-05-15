"""IconQA choose_txt local parquet adapter."""

from __future__ import annotations

import bisect
import glob
import os
import random
from io import BytesIO
from typing import Dict, List, Optional, Sequence, Set, Tuple

import pyarrow.parquet as pq
from PIL import Image


ICONQA_DATASET_NAME = "iconqa"
ICONQA_PARQUET_COLUMNS = [
    "question_id",
    "question",
    "choices",
    "answer",
    "query_image",
    "ques_type",
    "label",
    "grade",
    "skills",
]
ICONQA_NON_IMAGE_COLUMNS = [column for column in ICONQA_PARQUET_COLUMNS if column != "query_image"]
ICONQA_MIN_IMAGE_SIDE = 28


def _split_choices(raw_choices: object) -> List[str]:
    return [piece.strip() for piece in str(raw_choices or "").split(",") if piece.strip()]


def _resolve_answer_idx(answer: object, choices: Sequence[str]) -> int:
    answer_text = str(answer or "").strip()
    matches = [idx for idx, choice in enumerate(choices) if str(choice).strip() == answer_text]
    if len(matches) != 1:
        raise RuntimeError(f"IconQA choose_txt answer 无法唯一映射: answer={answer_text!r}, choices={choices!r}")
    return int(matches[0])


def _format_mcq_instruction(question: str, choices: Sequence[str]) -> str:
    choices_text = "".join(f"({chr(65 + idx)}) {choice}\n" for idx, choice in enumerate(choices))
    return (
        f"<image>\nQuestion: {question}\n"
        "Options:\n"
        f"{choices_text}"
        "\n"
        "Generate exactly four fields in this order:\n"
        "Observed Facts: describe only image-observable evidence.\n"
        "Context: restate relevant textual conditions from question, hint, and options.\n"
        "Reasoning: compare options briefly, but do not state the final answer here.\n"
        "Answer: give only the final option and answer, for example '(A) answer text'."
    )


def _ensure_min_image_side(image: Image.Image) -> Image.Image:
    width, height = image.size
    if width >= ICONQA_MIN_IMAGE_SIDE and height >= ICONQA_MIN_IMAGE_SIDE:
        return image
    new_width = max(int(width), ICONQA_MIN_IMAGE_SIDE)
    new_height = max(int(height), ICONQA_MIN_IMAGE_SIDE)
    canvas = Image.new("RGB", (new_width, new_height), (255, 255, 255))
    left = (new_width - int(width)) // 2
    top = (new_height - int(height)) // 2
    canvas.paste(image.convert("RGB"), (left, top))
    return canvas


class IconQAParquetReader:
    def __init__(self, dataset_path: str, split: str = "val") -> None:
        self.dataset_path = dataset_path
        self.split = split
        data_dir = os.path.join(dataset_path, "data")
        pattern = os.path.join(data_dir, f"{split}-*.parquet")
        self.files = sorted(glob.glob(pattern))
        if not self.files:
            raise FileNotFoundError(f"IconQA parquet files not found: {pattern}")

        self._parquet_files = [pq.ParquetFile(path) for path in self.files]
        self.row_counts = [int(pf.metadata.num_rows) for pf in self._parquet_files]
        self.starts: List[int] = []
        self.row_group_starts: List[List[int]] = []
        total = 0
        for file_idx, count in enumerate(self.row_counts):
            self.starts.append(total)
            local_total = 0
            group_starts: List[int] = []
            pf = self._parquet_files[file_idx]
            for row_group_idx in range(pf.metadata.num_row_groups):
                group_starts.append(local_total)
                local_total += int(pf.metadata.row_group(row_group_idx).num_rows)
            self.row_group_starts.append(group_starts)
            total += count
        self.num_rows = total

    def _file_and_local_index(self, source_index: int) -> Tuple[int, int]:
        idx = int(source_index)
        if idx < 0 or idx >= self.num_rows:
            raise IndexError(f"IconQA source_index out of range: {idx}, num_rows={self.num_rows}")
        file_idx = bisect.bisect_right(self.starts, idx) - 1
        if file_idx < 0:
            file_idx = 0
        return file_idx, idx - self.starts[file_idx]

    def image_ref(self, source_index: int) -> "LazyIconQAImage":
        file_idx, local_idx = self._file_and_local_index(int(source_index))
        group_starts = self.row_group_starts[file_idx]
        row_group_idx = bisect.bisect_right(group_starts, local_idx) - 1
        if row_group_idx < 0:
            row_group_idx = 0
        row_in_group = local_idx - group_starts[row_group_idx]
        return LazyIconQAImage(self.files[file_idx], row_group_idx, row_in_group)

    def read_rows(self, source_indices: Sequence[int], columns: Optional[Sequence[str]] = None) -> Dict[int, dict]:
        if columns is None:
            columns = ICONQA_PARQUET_COLUMNS
        grouped: Dict[int, List[Tuple[int, int]]] = {}
        for source_index in source_indices:
            file_idx, local_idx = self._file_and_local_index(int(source_index))
            grouped.setdefault(file_idx, []).append((int(source_index), local_idx))

        rows: Dict[int, dict] = {}
        for file_idx, pairs in grouped.items():
            table = pq.read_table(self.files[file_idx], columns=list(columns))
            py_rows = table.to_pylist()
            for source_index, local_idx in pairs:
                rows[int(source_index)] = py_rows[local_idx]
        return rows

    def choose_txt_source_indices(self) -> List[int]:
        indices: List[int] = []
        offset = 0
        for path in self.files:
            table = pq.read_table(path, columns=["ques_type"])
            for local_idx, row in enumerate(table.to_pylist()):
                if str(row.get("ques_type") or "") == "choose_txt":
                    indices.append(offset + local_idx)
            offset += table.num_rows
        return indices


class LazyIconQAImage:
    def __init__(self, file_path: str, row_group_idx: int, row_in_group: int) -> None:
        self.file_path = file_path
        self.row_group_idx = int(row_group_idx)
        self.row_in_group = int(row_in_group)
        self._size: Optional[Tuple[int, int]] = None

    @property
    def size(self) -> Tuple[int, int]:
        if self._size is None:
            self._size = self.load().size
        return self._size

    @property
    def width(self) -> int:
        return int(self.size[0])

    @property
    def height(self) -> int:
        return int(self.size[1])

    def load(self) -> Image.Image:
        pf = pq.ParquetFile(self.file_path)
        table = pf.read_row_group(self.row_group_idx, columns=["query_image"])
        row = table.slice(self.row_in_group, 1).to_pylist()[0]
        return decode_iconqa_image(row)


def decode_iconqa_image(row: dict) -> Image.Image:
    image = row.get("query_image")
    if isinstance(image, Image.Image):
        return _ensure_min_image_side(image.convert("RGB"))
    if isinstance(image, dict):
        image_bytes = image.get("bytes")
        image_path = image.get("path")
        if image_bytes:
            return _ensure_min_image_side(Image.open(BytesIO(image_bytes)).convert("RGB"))
        if image_path and os.path.exists(str(image_path)):
            return _ensure_min_image_side(Image.open(str(image_path)).convert("RGB"))
    raise RuntimeError("IconQA row does not contain decodable query_image bytes/path")


def materialize_iconqa_image(image):
    if isinstance(image, LazyIconQAImage):
        return image.load()
    return image


def select_iconqa_source_indices(
    choose_txt_indices: Sequence[int],
    train_num: int,
    seed: int,
    allowed_source_indices: Optional[Set[int]] = None,
) -> List[int]:
    allowed = set(int(idx) for idx in allowed_source_indices) if allowed_source_indices is not None else None
    indices = [int(idx) for idx in choose_txt_indices if allowed is None or int(idx) in allowed]
    rng = random.Random(int(seed))
    rng.shuffle(indices)
    if train_num > 0 and len(indices) > int(train_num):
        indices = indices[: int(train_num)]
    return indices


def build_iconqa_mcq_samples(
    dataset_path: str,
    split: str,
    train_num: int,
    seed: int,
    allowed_source_indices: Optional[Set[int]] = None,
) -> List[dict]:
    reader = IconQAParquetReader(dataset_path=dataset_path, split=split)
    source_indices = select_iconqa_source_indices(
        reader.choose_txt_source_indices(),
        train_num=train_num,
        seed=seed,
        allowed_source_indices=allowed_source_indices,
    )
    if not source_indices:
        return []

    rows = reader.read_rows(source_indices, columns=ICONQA_NON_IMAGE_COLUMNS)
    samples: List[dict] = []
    for sampled_pos, source_index in enumerate(source_indices):
        row = rows[int(source_index)]
        if str(row.get("ques_type") or "") != "choose_txt":
            raise RuntimeError(f"IconQA adapter 只支持 choose_txt: source_index={source_index}")
        question = str(row.get("question") or "").strip()
        choices = _split_choices(row.get("choices"))
        if not choices:
            raise RuntimeError(f"IconQA choose_txt choices 为空: source_index={source_index}")
        answer_idx = _resolve_answer_idx(row.get("answer"), choices)
        answer = choices[int(answer_idx)]
        answer_letter = chr(65 + int(answer_idx))
        question_id = str(row.get("question_id") or "").strip()
        if not question_id:
            raise RuntimeError(f"IconQA question_id 为空: source_index={source_index}")

        samples.append({
            "sample_id": int(sampled_pos),
            "source_index": int(source_index),
            "split": split,
            "dataset_name": ICONQA_DATASET_NAME,
            "question_id": question_id,
            "teacher_cache_key": f"{ICONQA_DATASET_NAME}::{split}::{question_id}",
            "image": reader.image_ref(int(source_index)),
            "question": question,
            "hint": "",
            "choices": choices,
            "answer_idx": int(answer_idx),
            "answer_letter": answer_letter,
            "answer_text": answer,
            "instruction": _format_mcq_instruction(question, choices),
            "response": f"Answer: {answer}",
            "ques_type": "choose_txt",
            "label": row.get("label"),
            "grade": row.get("grade"),
            "skills": row.get("skills"),
        })
    return samples
