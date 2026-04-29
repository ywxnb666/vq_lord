"""TextVQA local parquet to multiple-choice sample adapter.

TextVQA is open-ended. This adapter synthesizes MCQ options without changing
Stage2/Stage3 training losses. Distractors are hard negatives from other
questions, matched by answer/question shape, and never from the current
question's human answers.
"""

from __future__ import annotations

import bisect
import glob
import os
import random
import re
from collections import Counter, defaultdict
from io import BytesIO
from typing import Dict, List, Optional, Sequence, Set, Tuple

import pyarrow.parquet as pq
from PIL import Image


TEXTVQA_DATASET_NAME = "textvqa"
TEXTVQA_PARQUET_COLUMNS = [
    "image_id",
    "question_id",
    "question",
    "image",
    "image_width",
    "image_height",
    "answers",
    "ocr_tokens",
    "set_name",
]
TEXTVQA_NON_IMAGE_COLUMNS = [column for column in TEXTVQA_PARQUET_COLUMNS if column != "image"]
_ARTICLES = {"a", "an", "the"}
_NUMBER_MAP = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10",
}


def normalize_textvqa_answer(answer: object) -> str:
    text = str(answer or "").strip().lower()
    text = text.replace("\n", " ").replace("\t", " ")
    text = re.sub(r"[\"'`]+", "", text)
    text = re.sub(r"(?<!\d)[,.:;!?](?!\d)", " ", text)
    tokens = []
    for token in re.sub(r"\s+", " ", text).split():
        token = _NUMBER_MAP.get(token, token)
        if token not in _ARTICLES:
            tokens.append(token)
    return " ".join(tokens)


def canonical_textvqa_answer(answers: Sequence[object]) -> str:
    candidates = [str(answer).strip() for answer in answers if str(answer).strip()]
    if not candidates:
        return "unanswerable"
    counts = Counter(normalize_textvqa_answer(answer) for answer in candidates)
    best_norm, _ = sorted(counts.items(), key=lambda item: (-item[1], len(item[0]), item[0]))[0]
    matching = [answer for answer in candidates if normalize_textvqa_answer(answer) == best_norm]
    return sorted(matching, key=lambda text: (len(text), text.lower()))[0]


def textvqa_soft_score(pred_answer: object, human_answers: Sequence[object]) -> float:
    pred_norm = normalize_textvqa_answer(pred_answer)
    if not pred_norm:
        return 0.0
    match_count = sum(1 for answer in human_answers if normalize_textvqa_answer(answer) == pred_norm)
    return min(float(match_count) / 3.0, 1.0)


def _is_number(text: str) -> bool:
    return bool(re.fullmatch(r"[-+]?\d+(?:[.,:]\d+)?", text))


def _is_time(text: str) -> bool:
    return bool(re.fullmatch(r"\d{1,2}:\d{2}(?:\s?(?:am|pm))?", text))


def _is_date(text: str) -> bool:
    return bool(re.search(r"\b(?:19|20)\d{2}\b", text)) or bool(re.search(r"\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b", text))


def _is_ocr_like(text: str) -> bool:
    compact = text.replace(" ", "")
    if len(compact) <= 1:
        return False
    if re.fullmatch(r"[a-z0-9-]+", compact) and (any(ch.isdigit() for ch in compact) or "-" in compact):
        return True
    return compact.isupper() and len(compact) <= 12


def answer_type(answer: object) -> str:
    norm = normalize_textvqa_answer(answer)
    if norm in {"yes", "no"}:
        return "yesno"
    if norm == "unanswerable":
        return "unanswerable"
    if _is_time(norm):
        return "time"
    if _is_date(norm):
        return "date"
    if _is_number(norm):
        return "number"
    if _is_ocr_like(str(answer).strip()):
        return "ocr_like"
    token_count = len(norm.split())
    if token_count <= 1:
        return "single_token"
    if token_count <= 3:
        return "multi_token"
    return "phrase"


def question_type(question: object) -> str:
    q = normalize_textvqa_answer(question)
    if q.startswith("is ") or q.startswith("are ") or q.startswith("does ") or q.startswith("do ") or q.startswith("can "):
        return "yesno"
    if "how many" in q or "number" in q:
        return "number"
    if q.startswith("when") or "time" in q or "date" in q or "year" in q:
        return "time_date"
    if q.startswith("who"):
        return "who"
    if "brand" in q or "company" in q or "make" in q or "manufacturer" in q:
        return "brand"
    if "say" in q or "written" in q or "text" in q or "word" in q or "letter" in q:
        return "ocr_text"
    if q.startswith("where"):
        return "where"
    return "other"


def answer_meta(answer: str, question: str = "") -> dict:
    norm = normalize_textvqa_answer(answer)
    return {
        "answer": answer,
        "norm": norm,
        "answer_type": answer_type(answer),
        "question_type": question_type(question),
        "token_count": len(norm.split()),
        "char_length": len(norm),
        "ocr_like": _is_ocr_like(answer),
    }


class TextVQAParquetReader:
    def __init__(self, dataset_path: str, split: str = "train") -> None:
        self.dataset_path = dataset_path
        self.split = split
        data_dir = os.path.join(dataset_path, "data")
        pattern = os.path.join(data_dir, f"{split}-*.parquet")
        self.files = sorted(glob.glob(pattern))
        if not self.files:
            raise FileNotFoundError(f"TextVQA parquet files not found: {pattern}")

        self.row_counts = [pq.ParquetFile(path).metadata.num_rows for path in self.files]
        self._parquet_files = [pq.ParquetFile(path) for path in self.files]
        self.starts: List[int] = []
        self.row_group_starts: List[List[int]] = []
        total = 0
        for file_idx, count in enumerate(self.row_counts):
            self.starts.append(total)
            group_starts: List[int] = []
            local_total = 0
            pf = self._parquet_files[file_idx]
            for row_group_idx in range(pf.metadata.num_row_groups):
                group_starts.append(local_total)
                local_total += int(pf.metadata.row_group(row_group_idx).num_rows)
            self.row_group_starts.append(group_starts)
            total += int(count)
        self.num_rows = total

    def _file_and_local_index(self, source_index: int) -> Tuple[int, int]:
        idx = int(source_index)
        if idx < 0 or idx >= self.num_rows:
            raise IndexError(f"TextVQA source_index out of range: {idx}, num_rows={self.num_rows}")
        file_idx = bisect.bisect_right(self.starts, idx) - 1
        if file_idx < 0:
            file_idx = 0
        return file_idx, idx - self.starts[file_idx]

    def image_ref(self, source_index: int, width: Optional[int] = None, height: Optional[int] = None) -> "LazyTextVQAImage":
        file_idx, local_idx = self._file_and_local_index(int(source_index))
        group_starts = self.row_group_starts[file_idx]
        row_group_idx = bisect.bisect_right(group_starts, local_idx) - 1
        if row_group_idx < 0:
            row_group_idx = 0
        row_in_group = local_idx - group_starts[row_group_idx]
        return LazyTextVQAImage(self.files[file_idx], row_group_idx, row_in_group, width=width, height=height)

    def read_rows(self, source_indices: Sequence[int], columns: Optional[Sequence[str]] = None) -> Dict[int, dict]:
        if columns is None:
            columns = TEXTVQA_PARQUET_COLUMNS
        grouped: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
        for source_index in source_indices:
            file_idx, local_idx = self._file_and_local_index(int(source_index))
            grouped[file_idx].append((int(source_index), local_idx))

        rows: Dict[int, dict] = {}
        for file_idx, pairs in grouped.items():
            table = pq.read_table(self.files[file_idx], columns=list(columns))
            py_rows = table.to_pylist()
            for source_index, local_idx in pairs:
                rows[source_index] = py_rows[local_idx]
        return rows

    def read_answer_entries(self) -> Tuple[Dict[int, str], List[dict]]:
        answer_by_index: Dict[int, str] = {}
        entries: List[dict] = []
        offset = 0
        for path in self.files:
            table = pq.read_table(path, columns=["question", "answers", "ocr_tokens"])
            for local_idx, row in enumerate(table.to_pylist()):
                source_index = offset + local_idx
                answer = canonical_textvqa_answer(row.get("answers") or [])
                meta = answer_meta(answer, row.get("question") or "")
                meta.update({
                    "source_index": source_index,
                    "frequency": 1,
                    "ocr_density": len(row.get("ocr_tokens") or []),
                })
                answer_by_index[source_index] = answer
                entries.append(meta)
            offset += table.num_rows
        freq = Counter(entry["norm"] for entry in entries)
        for entry in entries:
            entry["frequency"] = int(freq[entry["norm"]])
        return answer_by_index, entries


class LazyTextVQAImage:
    def __init__(self, file_path: str, row_group_idx: int, row_in_group: int, width: Optional[int] = None, height: Optional[int] = None) -> None:
        self.file_path = file_path
        self.row_group_idx = int(row_group_idx)
        self.row_in_group = int(row_in_group)
        self._width = int(width) if width else None
        self._height = int(height) if height else None

    @property
    def size(self) -> Tuple[int, int]:
        if self._width is not None and self._height is not None:
            return (self._width, self._height)
        image = self.load()
        self._width, self._height = image.size
        return image.size

    @property
    def width(self) -> int:
        return int(self.size[0])

    @property
    def height(self) -> int:
        return int(self.size[1])

    def load(self) -> Image.Image:
        pf = pq.ParquetFile(self.file_path)
        table = pf.read_row_group(self.row_group_idx, columns=["image"])
        row = table.slice(self.row_in_group, 1).to_pylist()[0]
        return decode_textvqa_image(row)


def materialize_textvqa_image(image):
    if isinstance(image, LazyTextVQAImage):
        return image.load()
    return image


def decode_textvqa_image(row: dict) -> Image.Image:
    image = row.get("image")
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, dict):
        image_bytes = image.get("bytes")
        image_path = image.get("path")
        if image_bytes:
            return Image.open(BytesIO(image_bytes)).convert("RGB")
        if image_path and os.path.exists(str(image_path)):
            return Image.open(str(image_path)).convert("RGB")
    raise RuntimeError("TextVQA row does not contain decodable image bytes/path")


def build_distractor_index(answer_entries: Sequence[dict]) -> dict:
    index = {
        "qtype_atype_token": defaultdict(list),
        "atype_token": defaultdict(list),
        "shape": defaultdict(list),
        "global": list(answer_entries),
    }
    for entry in answer_entries:
        token_bucket = int(entry["token_count"])
        char_bucket = int(entry["char_length"]) // 4
        index["qtype_atype_token"][(entry["question_type"], entry["answer_type"], token_bucket)].append(entry)
        index["atype_token"][(entry["answer_type"], token_bucket)].append(entry)
        index["shape"][(entry["ocr_like"], char_bucket)].append(entry)
    return index


def _extend_from_pool(
    pool: Sequence[dict],
    target: dict,
    excluded: Set[str],
    seen: Set[str],
    source_index: int,
    rng: random.Random,
    stage: str,
    candidates: List[str],
    stage_counts: Counter,
    max_needed: int,
) -> None:
    working = list(pool)
    rng.shuffle(working)
    working.sort(key=lambda item: (-int(item["frequency"]), abs(int(item["char_length"]) - target["char_length"])))
    for entry in working:
        if int(entry["source_index"]) == int(source_index):
            continue
        if entry["norm"] in excluded or entry["norm"] in seen:
            continue
        candidates.append(entry["answer"])
        seen.add(entry["norm"])
        stage_counts[stage] += 1
        if len(candidates) >= max_needed:
            return


def build_textvqa_choices(
    gold_answer: str,
    human_answers: Sequence[object],
    question: str,
    distractor_index: dict,
    seed: int,
    source_index: int,
    num_choices: int = 4,
) -> Tuple[List[str], int, dict]:
    gold_answer = str(gold_answer).strip() or "unanswerable"
    target = answer_meta(gold_answer, question)
    excluded = {normalize_textvqa_answer(answer) for answer in human_answers if normalize_textvqa_answer(answer)}
    excluded.add(target["norm"])

    rng = random.Random((int(seed) * 1000003) + int(source_index))
    candidates: List[str] = []
    seen = {target["norm"]}
    stage_counts = Counter()
    max_needed = num_choices - 1
    token_bucket = int(target["token_count"])
    char_bucket = int(target["char_length"]) // 4

    stages = [
        ("qtype_atype_token", distractor_index["qtype_atype_token"].get((target["question_type"], target["answer_type"], token_bucket), [])),
        ("atype_token", distractor_index["atype_token"].get((target["answer_type"], token_bucket), [])),
        ("shape", distractor_index["shape"].get((target["ocr_like"], char_bucket), [])),
        ("global", distractor_index["global"]),
    ]
    for stage, pool in stages:
        _extend_from_pool(pool, target, excluded, seen, int(source_index), rng, stage, candidates, stage_counts, max_needed)
        if len(candidates) >= max_needed:
            break

    if len(candidates) < max_needed:
        raise RuntimeError(f"TextVQA hard distractor 不足: source_index={source_index}")

    choices = [gold_answer] + candidates[:max_needed]
    rng.shuffle(choices)
    answer_idx = choices.index(gold_answer)
    choice_soft_scores = [textvqa_soft_score(choice, human_answers) for choice in choices]
    diagnostics = {
        "choice_soft_scores": choice_soft_scores,
        "gold_soft_score": choice_soft_scores[answer_idx],
        "choice_oracle_soft_score": max(choice_soft_scores),
        "distractor_soft_scores": [score for idx, score in enumerate(choice_soft_scores) if idx != answer_idx],
        "distractor_stage_counts": dict(stage_counts),
    }
    return choices, answer_idx, diagnostics


def format_mcq_instruction(question: str, choices: Sequence[str]) -> str:
    choices_text = "".join(f"({chr(65 + idx)}) {choice}\n" for idx, choice in enumerate(choices))
    return f"<image>\nQuestion: {question}\nOptions:\n{choices_text}Answer:"


def select_textvqa_source_indices(num_rows: int, train_num: int, seed: int, allowed_source_indices: Optional[Set[int]] = None) -> List[int]:
    if allowed_source_indices is not None:
        indices = [int(idx) for idx in sorted(allowed_source_indices) if 0 <= int(idx) < int(num_rows)]
    else:
        indices = list(range(int(num_rows)))
    rng = random.Random(int(seed))
    rng.shuffle(indices)
    if train_num > 0 and len(indices) > int(train_num):
        indices = indices[: int(train_num)]
    return indices


def build_textvqa_mcq_samples(
    dataset_path: str,
    split: str,
    train_num: int,
    seed: int,
    allowed_source_indices: Optional[Set[int]] = None,
) -> List[dict]:
    reader = TextVQAParquetReader(dataset_path=dataset_path, split=split)
    source_indices = select_textvqa_source_indices(reader.num_rows, train_num, seed, allowed_source_indices)
    if not source_indices:
        return []

    answer_by_index, answer_entries = reader.read_answer_entries()
    distractor_index = build_distractor_index(answer_entries)
    rows = reader.read_rows(source_indices, columns=TEXTVQA_NON_IMAGE_COLUMNS)

    samples: List[dict] = []
    for sampled_pos, source_index in enumerate(source_indices):
        row = rows[int(source_index)]
        question = str(row.get("question") or "").strip()
        human_answers = list(row.get("answers") or [])
        gold_answer = answer_by_index[int(source_index)]
        choices, answer_idx, diagnostics = build_textvqa_choices(
            gold_answer=gold_answer,
            human_answers=human_answers,
            question=question,
            distractor_index=distractor_index,
            seed=seed,
            source_index=int(source_index),
        )
        answer_letter = chr(65 + int(answer_idx))
        instruction = format_mcq_instruction(question, choices)
        answer = choices[int(answer_idx)]
        image = reader.image_ref(int(source_index), width=row.get("image_width"), height=row.get("image_height"))

        samples.append({
            "sample_id": sampled_pos,
            "source_index": int(source_index),
            "split": split,
            "dataset_name": TEXTVQA_DATASET_NAME,
            "image": image,
            "question": question,
            "hint": "",
            "choices": choices,
            "answer_idx": int(answer_idx),
            "answer_letter": answer_letter,
            "answer_text": answer,
            "textvqa_answers": human_answers,
            "textvqa_question_id": row.get("question_id"),
            "textvqa_image_id": row.get("image_id"),
            "ocr_tokens": list(row.get("ocr_tokens") or []),
            "textvqa_choice_soft_scores": diagnostics["choice_soft_scores"],
            "textvqa_gold_soft_score": diagnostics["gold_soft_score"],
            "textvqa_choice_oracle_soft_score": diagnostics["choice_oracle_soft_score"],
            "textvqa_distractor_soft_scores": diagnostics["distractor_soft_scores"],
            "textvqa_distractor_stage_counts": diagnostics["distractor_stage_counts"],
            "instruction": instruction,
            "response": f"Answer: {answer}",
        })
    return samples
