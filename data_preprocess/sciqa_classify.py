#!/usr/bin/env python3
"""ScienceQA 数据统计脚本。

功能：
1. 统计无图像问题数量。
2. 按图像尺寸动态分类并统计数量（如 512x512, 640x480 等）。
3. 支持按 split 分别统计与整体汇总。

示例：
python data_preprocess/sciqa_classify.py \
  --dataset-path /root/autodl-tmp/datasets/ScienceQA \
  --save-json /root/workspace/align_vq/vq_lord_ckpts/sciqa_size_stats.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from typing import Dict, Iterable, List, Optional, Tuple

from datasets import get_dataset_split_names, load_dataset


def _resolve_splits(dataset_path: str, requested_split: str) -> List[str]:
	"""解析需要统计的 split 列表。"""
	if requested_split != "all":
		return [requested_split]

	try:
		splits = list(get_dataset_split_names(dataset_path))
		if splits:
			return splits
	except Exception:
		pass

	# 回退到 ScienceQA 常见 split 名称
	fallback = ["train", "validation", "test"]
	valid: List[str] = []
	for split in fallback:
		try:
			_ = load_dataset(dataset_path, split=split)
			valid.append(split)
		except Exception:
			continue

	if not valid:
		raise RuntimeError(f"未能识别可用 split: dataset_path={dataset_path}")
	return valid


def _extract_size(item: dict) -> Optional[Tuple[int, int]]:
	"""从样本中提取图片尺寸 (width, height)。无图像返回 None。"""
	image = item.get("image")
	if image is None:
		return None

	# 常见情况：PIL.Image
	if hasattr(image, "size") and image.size is not None:
		try:
			width, height = image.size
			return int(width), int(height)
		except Exception:
			pass

	# 少数场景：image 可能是字典结构
	if isinstance(image, dict):
		width = image.get("width")
		height = image.get("height")
		if width is not None and height is not None:
			return int(width), int(height)

	return None


def classify_split(dataset_path: str, split: str) -> Dict[str, object]:
	"""对单个 split 做分类统计。"""
	dataset = load_dataset(dataset_path, split=split)

	total = 0
	no_image = 0
	size_counter: Counter[str] = Counter()

	for item in dataset:
		total += 1
		size = _extract_size(item)
		if size is None:
			no_image += 1
		else:
			width, height = size
			size_counter[f"{width}x{height}"] += 1

	with_image = total - no_image
	sorted_sizes = sorted(size_counter.items(), key=lambda x: (-x[1], x[0]))

	return {
		"split": split,
		"total": total,
		"no_image": no_image,
		"with_image": with_image,
		"size_buckets": [{"size": size, "count": count} for size, count in sorted_sizes],
	}


def _merge_size_buckets(reports: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
	"""汇总多个 split 的尺寸统计。"""
	total_counter: Counter[str] = Counter()
	for report in reports:
		for bucket in report["size_buckets"]:
			total_counter[bucket["size"]] += int(bucket["count"])

	return [
		{"size": size, "count": count}
		for size, count in sorted(total_counter.items(), key=lambda x: (-x[1], x[0]))
	]


def print_report(report: Dict[str, object]) -> None:
	"""打印单个 split 统计结果。"""
	print("=" * 72)
	print(f"Split: {report['split']}")
	print(f"总样本数: {report['total']}")
	print(f"无图像问题: {report['no_image']}")
	print(f"有图像问题: {report['with_image']}")
	print("图像尺寸分类（按数量降序）:")

	buckets = report["size_buckets"]
	if not buckets:
		print("  (无图像尺寸数据)")
		return

	for idx, bucket in enumerate(buckets, start=1):
		print(f"  {idx}. 尺寸 {bucket['size']} -> {bucket['count']} 条")


def build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="ScienceQA 图像尺寸动态分类统计")
	parser.add_argument(
		"--dataset-path",
		type=str,
		default="/root/autodl-tmp/datasets/ScienceQA",
		help="ScienceQA 数据集路径（本地路径或 HF 数据集名）",
	)
	parser.add_argument(
		"--split",
		type=str,
		default="train",
		help="要统计的 split（train/validation/test/all），默认 train",
	)
	parser.add_argument(
		"--save-json",
		type=str,
		default="",
		help="可选：将统计结果保存为 JSON 文件",
	)
	return parser


def main() -> None:
	args = build_arg_parser().parse_args()
	splits = _resolve_splits(args.dataset_path, args.split)

	reports: List[Dict[str, object]] = []
	for split in splits:
		report = classify_split(args.dataset_path, split)
		reports.append(report)
		print_report(report)

	summary = {
		"dataset_path": args.dataset_path,
		"splits": splits,
		"total": sum(int(r["total"]) for r in reports),
		"no_image": sum(int(r["no_image"]) for r in reports),
		"with_image": sum(int(r["with_image"]) for r in reports),
		"size_buckets": _merge_size_buckets(reports),
	}

	print("=" * 72)
	print("Overall Summary")
	print(f"总样本数: {summary['total']}")
	print(f"无图像问题: {summary['no_image']}")
	print(f"有图像问题: {summary['with_image']}")
	print(f"尺寸类别数: {len(summary['size_buckets'])}")

	if args.save_json:
		payload = {
			"summary": summary,
			"reports": reports,
		}
		with open(args.save_json, "w", encoding="utf-8") as f:
			json.dump(payload, f, ensure_ascii=False, indent=2)
		print(f"JSON 已保存: {args.save_json}")


if __name__ == "__main__":
	main()
