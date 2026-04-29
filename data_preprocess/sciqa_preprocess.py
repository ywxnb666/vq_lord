#!/usr/bin/env python3
"""ScienceQA 分桶预处理脚本。

功能：
1. 加载 ScienceQA 指定 split，并按 train_vq_lord.py 的采样逻辑过滤有图像样本。
2. 使用 LlavaNextProcessor 计算每个样本的 patch 数。
3. 按 patches / size / none 构建 bucket。
4. 在桶内打乱、切 batch、再进行桶间打乱，生成可复现的批次计划。
5. 输出统计摘要，并可保存为 JSON 供后续训练接入。

示例：
python data_preprocess/sciqa_preprocess.py \
  --dataset-path /root/autodl-tmp/datasets/ScienceQA \
	--model-path /root/autodl-tmp/models/llama3-llava-next-8b-hf \
  --split train \
  --train-num 500 \
  --bucket-by patches \
  --bucket-batch-size 4 \
  --save-json /root/workspace/align_vq/vq_lord_ckpts/sciqa_bucket_plan.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import defaultdict
from typing import Dict, Hashable, List, Optional, Sequence, Set, Tuple

from datasets import load_dataset
from transformers import AutoProcessor, LlavaNextProcessor

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VQ_LORD3_DIR = os.path.join(REPO_ROOT, "vq_lord3")
if VQ_LORD3_DIR not in sys.path:
	sys.path.insert(0, VQ_LORD3_DIR)

from textvqa_mcq import build_textvqa_mcq_samples, materialize_textvqa_image


# 与 train_vq_lord.py 保持一致，避免旧版配置里的 image_token 触发报错。
_original_init = LlavaNextProcessor.__init__


def _new_init(self, *args, **kwargs):
	kwargs.pop("image_token", None)
	_original_init(self, *args, **kwargs)


LlavaNextProcessor.__init__ = _new_init


DEFAULT_MODEL_CANDIDATES = [
	"/root/autodl-tmp/models/llama3-llava-next-8b-hf",
	"/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/luye/align_vq/downloads/models/llama3-llava-next-8b-hf",
	"llava-hf/llama3-llava-next-8b-hf",
]


def normalize_dataset_name(dataset_name: str) -> str:
	name = str(dataset_name or "").strip().lower()
	if not name:
		return "scienceqa"
	return name


def resolve_mc_answer_idx(item: dict, dataset_name: str) -> int:
	dataset_name = normalize_dataset_name(dataset_name)
	if dataset_name == "aokvqa":
		field_candidates = ("correct_choice_idx", "answer")
	else:
		field_candidates = ("answer", "correct_choice_idx")

	for field in field_candidates:
		raw_value = item.get(field)
		if raw_value is None:
			continue
		try:
			return int(raw_value)
		except (TypeError, ValueError):
			continue

	available = ",".join(sorted(item.keys()))
	raise RuntimeError(
		f"样本缺少有效答案字段。dataset_name={dataset_name}, "
		f"expected_fields={field_candidates}, available_fields={available}"
	)


def build_scienceqa_samples(
	dataset_path: str,
	split: str,
	train_num: int,
	seed: int,
	dataset_name: str = "scienceqa",
	allowed_source_indices: Optional[Set[int]] = None,
) -> List[dict]:
	"""复用 train_vq_lord.py 的采样语义，只保留有图像样本。"""
	dataset_name = normalize_dataset_name(dataset_name)
	if dataset_name == "textvqa":
		return build_textvqa_mcq_samples(
			dataset_path=dataset_path,
			split=split,
			train_num=train_num,
			seed=seed,
			allowed_source_indices=allowed_source_indices,
		)

	try:
		dataset = load_dataset(dataset_path, split=split)
	except Exception as exc:
		offline = os.environ.get("HF_DATASETS_OFFLINE", "0") == "1"
		if offline and not os.path.exists(dataset_path):
			raise RuntimeError(
				f"无法加载 ScienceQA: dataset_path={dataset_path}。当前处于离线模式，且该路径不是本地目录。"
			) from exc
		raise RuntimeError(
			f"无法加载 ScienceQA: dataset_path={dataset_path}, split={split}"
		) from exc
	dataset_with_images = [item for item in dataset if item.get("image") is not None]

	if allowed_source_indices is not None:
		all_indices = [
			idx for idx in sorted(allowed_source_indices)
			if 0 <= int(idx) < len(dataset_with_images)
		]
	else:
		all_indices = list(range(len(dataset_with_images)))
	random.seed(seed)
	random.shuffle(all_indices)
	if train_num > 0 and len(all_indices) > train_num:
		all_indices = all_indices[:train_num]

	samples: List[dict] = []
	for sampled_pos, dataset_idx in enumerate(all_indices):
		item = dataset_with_images[dataset_idx]
		question = item.get("question", "")
		choices = item.get("choices", [])
		if not isinstance(choices, list):
			raise RuntimeError(
				f"样本 choices 不是 list。dataset_name={dataset_name}, split={split}, source_index={dataset_idx}"
			)
		if not choices:
			raise RuntimeError(
				f"样本 choices 为空。dataset_name={dataset_name}, split={split}, source_index={dataset_idx}"
			)
		choices_text = ""
		for choice_idx, choice in enumerate(choices):
			choices_text += f"({chr(65 + choice_idx)}) {choice}\n"

		answer_idx = resolve_mc_answer_idx(item, dataset_name=dataset_name)
		if answer_idx < 0 or answer_idx >= len(choices):
			raise RuntimeError(
				f"样本 answer_idx 越界。dataset_name={dataset_name}, split={split}, "
				f"source_index={dataset_idx}, answer_idx={answer_idx}, num_choices={len(choices)}"
			)
		answer = choices[answer_idx] if choices and answer_idx < len(choices) else ""
		lecture = item.get("lecture", "")
		solution = item.get("solution", "")

		instruction = f"<image>\nQuestion: {question}\nOptions:\n{choices_text}Answer:"
		if lecture:
			response = f"Explanation: {lecture}\nSolution: {solution}\nAnswer: {answer}"
		else:
			response = f"Solution: {solution}\nAnswer: {answer}"

		samples.append({
			"sample_id": sampled_pos,
			"source_index": dataset_idx,
			"question": question,
			"choices": choices,
			"answer": answer,
			"image": item.get("image"),
			"instruction": instruction,
			"response": response,
		})

	return samples


def load_cached_teacher_source_indices(
	cache_path: str,
	dataset_name: str,
	split: str,
) -> Set[int]:
	if not cache_path:
		raise RuntimeError("sample-only-cached-teacher=1 时必须提供 --teacher-cache-path")
	if not os.path.exists(cache_path):
		raise RuntimeError(f"teacher cache 不存在: {cache_path}")

	with open(cache_path, "r", encoding="utf-8") as f:
		payload = json.load(f)
	cache_samples = payload.get("samples")
	if not isinstance(cache_samples, dict):
		raise RuntimeError(f"教师缓存 samples 字段无效: {cache_path}")

	required_fields = (
		"observed_facts_visual",
		"context_textual",
		"reasoning",
		"answer",
	)
	prefix = f"{normalize_dataset_name(dataset_name)}::{split}::"
	allowed: Set[int] = set()
	for cache_key, teacher_payload in cache_samples.items():
		if not isinstance(cache_key, str) or not cache_key.startswith(prefix):
			continue
		suffix = cache_key[len(prefix):]
		try:
			source_index = int(suffix)
		except (TypeError, ValueError):
			continue
		if not isinstance(teacher_payload, dict):
			continue
		valid = True
		for field in required_fields:
			value = teacher_payload.get(field)
			if not isinstance(value, str) or len(value.strip()) == 0:
				valid = False
				break
		if valid:
			allowed.add(source_index)
	return allowed


def extract_image_hw(image) -> Tuple[int, int]:
	if image is None:
		raise ValueError("image is None")
	if hasattr(image, "size") and image.size is not None:
		width, height = image.size
		return int(width), int(height)
	if isinstance(image, dict) and image.get("width") is not None and image.get("height") is not None:
		return int(image["width"]), int(image["height"])
	raise ValueError("unable to extract image size")


def estimate_patch_count(processor, image) -> int:
	image = materialize_textvqa_image(image)
	image_processor = getattr(processor, "image_processor", None)
	if image_processor is not None:
		inputs = image_processor(images=image, return_tensors="pt")
	else:
		inputs = processor(text="", images=image, return_tensors="pt")
	pixel_values = inputs["pixel_values"]
	if pixel_values.dim() == 2:
		if "image_grid_thw" not in inputs:
			raise RuntimeError("Qwen2-VL pixel_values 为二维，但 processor 输出缺少 image_grid_thw")
		return int(pixel_values.shape[0])
	if pixel_values.dim() == 5:
		return int(pixel_values.shape[1])
	if pixel_values.dim() == 4:
		return 1
	raise RuntimeError(f"unexpected pixel_values shape: {tuple(pixel_values.shape)}")


def resolve_model_path(requested_model_path: str) -> str:
	if requested_model_path and (os.path.exists(requested_model_path) or "/" not in requested_model_path):
		return requested_model_path

	for candidate in DEFAULT_MODEL_CANDIDATES:
		if os.path.exists(candidate) or "/" not in candidate:
			print(f"[Info] model_path 不可用，回退到: {candidate}")
			return candidate

	raise FileNotFoundError(
		"未找到可用的 processor 路径，请显式传入 --model-path。"
	)


def build_sample_records(
	samples: Sequence[dict],
	processor,
	bucket_by: str,
) -> List[dict]:
	records: List[dict] = []
	for sample in samples:
		image = sample["image"]
		width, height = extract_image_hw(image)
		patch_count = estimate_patch_count(processor, image)

		if bucket_by == "patches":
			bucket_key: Hashable = patch_count
		elif bucket_by == "size":
			bucket_key = f"{width}x{height}"
		else:
			bucket_key = "all"

		records.append({
			"sample_id": int(sample["sample_id"]),
			"source_index": int(sample["source_index"]),
			"question": sample.get("question", ""),
			"image_size": [width, height],
			"patch_count": int(patch_count),
			"bucket_key": str(bucket_key),
		})
	return records


def build_bucket_map(records: Sequence[dict]) -> Dict[str, List[int]]:
	buckets: Dict[str, List[int]] = defaultdict(list)
	for record in records:
		buckets[str(record["bucket_key"])].append(int(record["sample_id"]))
	return dict(buckets)


def build_batch_plan(
	bucket_map: Dict[str, List[int]],
	batch_size: int,
	seed: int,
	shuffle: bool,
	drop_last: bool,
) -> List[dict]:
	rng = random.Random(seed)
	all_batches: List[dict] = []

	for bucket_key, indices in bucket_map.items():
		working = list(indices)
		if shuffle:
			rng.shuffle(working)

		for start in range(0, len(working), batch_size):
			batch_indices = working[start:start + batch_size]
			if drop_last and len(batch_indices) < batch_size:
				continue
			all_batches.append({
				"bucket_key": bucket_key,
				"sample_ids": batch_indices,
				"batch_size": len(batch_indices),
			})

	if shuffle:
		rng.shuffle(all_batches)

	for batch_id, batch in enumerate(all_batches):
		batch["batch_id"] = batch_id

	return all_batches


def summarize_buckets(bucket_map: Dict[str, List[int]], batch_size: int, drop_last: bool) -> List[dict]:
	summary: List[dict] = []
	for bucket_key, indices in sorted(bucket_map.items(), key=lambda item: (-len(item[1]), item[0])):
		count = len(indices)
		full_batches = count // batch_size
		tail_size = count % batch_size
		if drop_last:
			planned_batches = full_batches
		else:
			planned_batches = math.ceil(count / batch_size) if count > 0 else 0
		summary.append({
			"bucket_key": bucket_key,
			"sample_count": count,
			"full_batches": full_batches,
			"tail_size": tail_size,
			"planned_batches": planned_batches,
		})
	return summary


def enrich_batches(batch_plan: Sequence[dict], records: Sequence[dict]) -> List[dict]:
	record_map = {int(record["sample_id"]): record for record in records}
	enriched: List[dict] = []
	for batch in batch_plan:
		sample_ids = [int(sample_id) for sample_id in batch["sample_ids"]]
		samples = [record_map[sample_id] for sample_id in sample_ids]
		enriched.append({
			"batch_id": int(batch["batch_id"]),
			"bucket_key": str(batch["bucket_key"]),
			"batch_size": int(batch["batch_size"]),
			"sample_ids": sample_ids,
			"patch_counts": [int(sample["patch_count"]) for sample in samples],
			"image_sizes": [sample["image_size"] for sample in samples],
		})
	return enriched


def print_summary(records: Sequence[dict], bucket_summary: Sequence[dict], batch_plan: Sequence[dict], args) -> None:
	patch_values = [int(record["patch_count"]) for record in records]
	print("=" * 72)
	print("ScienceQA 预处理完成")
	print(f"split: {args.split}")
	print(f"样本数: {len(records)}")
	print(f"bucket_by: {args.bucket_by}")
	print(f"batch_size: {args.bucket_batch_size}")
	print(f"drop_last: {bool(args.bucket_drop_last)}")
	if patch_values:
		print(
			f"patch_count: min={min(patch_values)}, avg={sum(patch_values)/len(patch_values):.2f}, max={max(patch_values)}"
		)
	print(f"bucket 数: {len(bucket_summary)}")
	print(f"计划 batch 数: {len(batch_plan)}")
	print("Top bucket:")
	for rank, bucket in enumerate(bucket_summary[: args.preview_buckets], start=1):
		print(
			f"  {rank}. key={bucket['bucket_key']} samples={bucket['sample_count']} "
			f"batches={bucket['planned_batches']} tail={bucket['tail_size']}"
		)
	print("Preview batches:")
	for batch in batch_plan[: args.preview_batches]:
		print(
			f"  batch_id={batch['batch_id']} key={batch['bucket_key']} "
			f"size={batch['batch_size']} sample_ids={batch['sample_ids']}"
		)


def build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="ScienceQA patch 分桶预处理")
	parser.add_argument(
		"--dataset-name",
		type=str,
		default="scienceqa",
		choices=["scienceqa", "aokvqa", "textvqa"],
		help="数据集名称（用于字段映射）",
	)
	parser.add_argument(
		"--dataset-path",
		type=str,
		default="/root/autodl-tmp/datasets/ScienceQA",
		help="ScienceQA 数据集路径（本地路径或 HF 数据集名）",
	)
	parser.add_argument(
		"--model-path",
		type=str,
		default="/root/autodl-tmp/models/llama3-llava-next-8b-hf",
		help="LLaVA-Next processor 路径，用于估算 patch_count",
	)
	parser.add_argument(
		"--split",
		type=str,
		default="train",
		help="ScienceQA split，建议 train",
	)
	parser.add_argument(
		"--train-num",
		type=int,
		default=500,
		help="采样样本数，0 表示使用该 split 全量有图像样本",
	)
	parser.add_argument(
		"--seed",
		type=int,
		default=20240306,
		help="样本采样随机种子",
	)
	parser.add_argument(
		"--sample-only-cached-teacher",
		type=int,
		default=0,
		help="是否仅从教师缓存可用样本中抽样，1=是，0=否",
	)
	parser.add_argument(
		"--teacher-cache-path",
		type=str,
		default="",
		help="教师缓存路径（启用 sample-only-cached-teacher 时必填）",
	)
	parser.add_argument(
		"--bucket-by",
		type=str,
		default="patches",
		choices=["patches", "size", "none"],
		help="分桶依据：patches / size / none",
	)
	parser.add_argument(
		"--bucket-batch-size",
		type=int,
		default=4,
		help="分桶后目标 batch size",
	)
	parser.add_argument(
		"--bucket-drop-last",
		type=int,
		default=0,
		help="是否丢弃不足 batch_size 的尾包，0=保留，1=丢弃",
	)
	parser.add_argument(
		"--shuffle",
		type=int,
		default=1,
		help="是否执行桶内/桶间打乱，0=否，1=是",
	)
	parser.add_argument(
		"--preview-buckets",
		type=int,
		default=10,
		help="打印前 N 个 bucket 摘要",
	)
	parser.add_argument(
		"--preview-batches",
		type=int,
		default=10,
		help="打印前 N 个 batch 预览",
	)
	parser.add_argument(
		"--save-json",
		type=str,
		default="",
		help="保存预处理结果 JSON 的路径",
	)
	return parser


def main() -> None:
	args = build_arg_parser().parse_args()

	if args.bucket_batch_size <= 0:
		raise ValueError("--bucket-batch-size 必须 > 0")

	args.model_path = resolve_model_path(args.model_path)
	print(f"加载 processor: {args.model_path}")
	processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)

	print(f"加载 ScienceQA split={args.split}, train_num={args.train_num}, seed={args.seed}")
	allowed_source_indices = None
	if int(args.sample_only_cached_teacher) == 1:
		allowed_source_indices = load_cached_teacher_source_indices(
			cache_path=args.teacher_cache_path,
			dataset_name=args.dataset_name,
			split=args.split,
		)
		print(
			f"[SampleFilter] sample-only-cached-teacher=1, "
			f"cached_valid_source_indices={len(allowed_source_indices)}"
		)
		if len(allowed_source_indices) == 0:
			raise RuntimeError("sample-only-cached-teacher=1 但缓存中没有可用四字段样本")
	samples = build_scienceqa_samples(
		dataset_path=args.dataset_path,
		dataset_name=args.dataset_name,
		split=args.split,
		train_num=args.train_num,
		seed=args.seed,
		allowed_source_indices=allowed_source_indices,
	)

	print(f"开始构建样本记录，bucket_by={args.bucket_by}")
	records = build_sample_records(samples, processor, args.bucket_by)
	bucket_map = build_bucket_map(records)
	bucket_summary = summarize_buckets(
		bucket_map=bucket_map,
		batch_size=args.bucket_batch_size,
		drop_last=bool(args.bucket_drop_last),
	)
	batch_plan = build_batch_plan(
		bucket_map=bucket_map,
		batch_size=args.bucket_batch_size,
		seed=args.seed,
		shuffle=bool(args.shuffle),
		drop_last=bool(args.bucket_drop_last),
	)
	enriched_batches = enrich_batches(batch_plan, records)

	print_summary(records, bucket_summary, enriched_batches, args)

	if args.save_json:
		save_dir = os.path.dirname(args.save_json)
		if save_dir:
			os.makedirs(save_dir, exist_ok=True)
		payload = {
			"config": {
				"dataset_name": args.dataset_name,
				"dataset_path": args.dataset_path,
				"model_path": args.model_path,
				"split": args.split,
				"train_num": args.train_num,
				"seed": args.seed,
				"sample_only_cached_teacher": bool(args.sample_only_cached_teacher),
				"teacher_cache_path": args.teacher_cache_path,
				"bucket_by": args.bucket_by,
				"bucket_batch_size": args.bucket_batch_size,
				"bucket_drop_last": bool(args.bucket_drop_last),
				"shuffle": bool(args.shuffle),
			},
			"summary": {
				"sample_count": len(records),
				"bucket_count": len(bucket_summary),
				"batch_count": len(enriched_batches),
			},
			"bucket_summary": bucket_summary,
			"samples": records,
			"batch_plan": enriched_batches,
		}
		with open(args.save_json, "w", encoding="utf-8") as f:
			json.dump(payload, f, ensure_ascii=False, indent=2)
		print(f"JSON 已保存: {args.save_json}")


if __name__ == "__main__":
	main()
