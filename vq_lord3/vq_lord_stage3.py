"""Stage3 training logic for VQ-LoRD3."""

import contextlib
import json
import math
import os
import random
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from train_vq_lord3 import (
    ScienceQADataset,
    _capture_rng_state,
    _get_image_token_id,
    _is_hf_multichoice_dataset,
    _get_trainable_parameter_state,
    _load_parameter_state,
    _move_optimizer_state_to_device_,
    _restore_model_gradient_checkpointing,
    _restore_model_use_cache,
    _restore_rng_state,
    _set_model_gradient_checkpointing,
    _set_model_use_cache,
    _strip_image_tokens,
    _to_cpu_obj,
    build_scienceqa_samples,
    barrier_if_distributed,
    get_rank,
    get_world_size,
    is_distributed,
    is_main_process,
    is_projector_param,
    maybe_set_dataloader_epoch,
    reduce_numeric_dict,
    sanitize_image_sizes,
    save_stage3_checkpoint,
    student_forward,
    student_generate,
    unwrap_model,
)

def log_clip(tnsr, epsilon=1.0):
    """对数裁剪函数，防止数值不稳定。
    将输入裁剪到 [log(1-epsilon), log(1+epsilon)] 范围内。
    epsilon=1.0 对应 [-inf, log(2)] ≈ [-inf, 0.693]，
    实际 max 侧裁剪为 log(1+epsilon)=0.693，min 侧裁剪为 -10（防 -inf）。
    """
    upper = torch.log(torch.tensor(1.0 + epsilon, device=tnsr.device, dtype=tnsr.dtype))
    lower = torch.tensor(-10.0, device=tnsr.device, dtype=tnsr.dtype)  # 下界不用 log(1-1)=-inf
    return torch.clamp(tnsr, min=lower.item(), max=upper.item())


def _reduce_sum_tensor(tensor: torch.Tensor) -> torch.Tensor:
    reduced = tensor.detach().clone()
    if is_distributed():
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    return reduced


def _stage3_global_mean_from_local_sum(
    local_loss_sum: torch.Tensor,
    local_weight: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    global_weight = _reduce_sum_tensor(local_weight.to(device=local_loss_sum.device, dtype=torch.float32))
    if float(global_weight.item()) <= 0.0:
        return local_loss_sum * 0.0, 0.0

    global_loss_sum = _reduce_sum_tensor(local_loss_sum.to(device=local_loss_sum.device, dtype=torch.float32))
    # DDP backward 会把梯度再除以 world_size，这里先乘回 world_size / global_weight，
    # 使最终梯度等价于单卡上的 global masked mean。
    scale = float(get_world_size()) / float(global_weight.item())
    backward_loss = local_loss_sum * scale
    log_loss = float((global_loss_sum / global_weight).item())
    return backward_loss, log_loss


def _stage3_masked_mean_ddp(
    values: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    local_mask = mask.to(dtype=values.dtype)
    local_sum = (values * local_mask).sum()
    local_weight = local_mask.sum().to(dtype=torch.float32)
    return _stage3_global_mean_from_local_sum(local_sum, local_weight)


def _compute_token_log_probs(
    model,
    args,
    ids: torch.Tensor,
    mask: torch.Tensor,
    pixel_values: torch.Tensor,
    image_sizes: Optional[torch.Tensor],
    prompt_lens: torch.Tensor,
    image_grid_thw: Optional[torch.Tensor] = None,
):
    """返回 token-level log-prob 与生成段 mask。"""
    out = student_forward(
        model=model,
        args=args,
        input_ids=ids,
        attention_mask=mask,
        pixel_values=pixel_values,
        image_sizes=image_sizes,
        image_grid_thw=image_grid_thw,
    )
    logits = out.logits[:, :-1, :]
    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(dim=-1, index=ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    token_mask = mask[:, 1:].float()

    for batch_idx in range(ids.shape[0]):
        gen_start = max(int(prompt_lens[batch_idx].item()) - 1, 0)
        if gen_start > 0:
            token_mask[batch_idx, :gen_start] = 0.0

    return token_log_probs, token_mask


@dataclass
class Stage3SampleCacheItem:
    sample_idx: int
    prompt_ids: torch.Tensor
    prompt_mask: torch.Tensor
    y_vic_ids: torch.Tensor
    y_vic_mask: torch.Tensor
    pixel_values: torch.Tensor
    image_sizes: Optional[torch.Tensor]
    image_grid_thw: Optional[torch.Tensor]
    prompt_len: int
    observed_ids: Optional[torch.Tensor] = None
    context_ids: Optional[torch.Tensor] = None
    reasoning_ids: Optional[torch.Tensor] = None
    answer_ids: Optional[torch.Tensor] = None
    vic_observed_mask: Optional[torch.Tensor] = None
    vic_context_mask: Optional[torch.Tensor] = None
    vic_reasoning_mask: Optional[torch.Tensor] = None
    vic_answer_mask: Optional[torch.Tensor] = None
    vic_other_mask: Optional[torch.Tensor] = None
    answer_letter: str = ""
    answer_idx: int = -1
    num_choices: int = 0


@dataclass
class PeriodState:
    sample_idx: int
    y11_ids: torch.Tensor
    y11_mask: torch.Tensor
    y12_ids: torch.Tensor
    y12_mask: torch.Tensor
    avg_lp_11: float
    avg_lp_12: float
    prob_11: float
    prob_12: float


@dataclass
class PeriodTrainingItem:
    sample_idx: int
    y_plus_ids: torch.Tensor
    y_plus_mask: torch.Tensor
    y_minus_ids: torch.Tensor
    y_minus_mask: torch.Tensor
    y_vic_ids: torch.Tensor
    y_vic_mask: torch.Tensor
    old_token_lp_plus: torch.Tensor
    old_token_mask_plus: torch.Tensor
    old_token_lp_minus: torch.Tensor
    old_token_mask_minus: torch.Tensor
    old_token_lp_vic: torch.Tensor
    old_token_mask_vic: torch.Tensor
    wrong_sample_idx: int


def _stage3_resume_state_path(resume_dir: str) -> str:
    return os.path.join(resume_dir, "stage3_resume_state.pt")


def _stage3_resume_meta_path(resume_dir: str) -> str:
    return os.path.join(resume_dir, "stage3_resume_meta.json")


def _stage3_resume_config(args) -> dict:
    return {
        "model_path": str(args.model_path),
        "student_model_type": str(getattr(args, "student_model_type", "llava_next")),
        "use_lora": int(args.use_lora),
        "lora_rank": int(args.lora_rank),
        "lora_alpha": int(args.lora_alpha),
        "use_4bit": int(args.use_4bit),
        "model_dtype": str(args.model_dtype),
        "dataset_name": str(args.dataset_name),
        "scienceqa_path": str(getattr(args, "scienceqa_path", "")),
        "scienceqa_split": str(getattr(args, "scienceqa_split", "")),
        "scienceqa_seed": int(getattr(args, "scienceqa_seed", 0)),
        "train_num": int(getattr(args, "train_num", 0)),
        "max_length": int(getattr(args, "max_length", 0)),
        "teacher_lang": str(getattr(args, "teacher_lang", "")),
        "teacher_observed_max_tokens": int(getattr(args, "teacher_observed_max_tokens", 0)),
        "teacher_context_max_tokens": int(getattr(args, "teacher_context_max_tokens", 0)),
        "teacher_reasoning_max_tokens": int(getattr(args, "teacher_reasoning_max_tokens", 0)),
        "teacher_answer_max_tokens": int(getattr(args, "teacher_answer_max_tokens", 0)),
        "stage3_vic_include_context": int(getattr(args, "stage3_vic_include_context", 0)),
        "lr": float(args.lr),
        "stage3_lr_scale": float(getattr(args, "stage3_lr_scale", 0.0)),
        "stage3_grad_clip": float(getattr(args, "stage3_grad_clip", 0.0)),
        "stage3_grad_accum": int(getattr(args, "stage3_grad_accum", 0)),
        "grad_accum": int(getattr(args, "grad_accum", 1)),
        "batch_size": int(getattr(args, "batch_size", 1)),
        "bucket_batch_size": int(getattr(args, "bucket_batch_size", 0)),
        "stage3_bucket_batch_size": int(getattr(args, "stage3_bucket_batch_size", 0)),
        "disable_bucket_for_stage3": int(getattr(args, "disable_bucket_for_stage3", 0)),
        "sub_stage_num": int(getattr(args, "sub_stage_num", 0)),
        "period_num": int(getattr(args, "period_num", 0)),
        "sub_set_num": int(getattr(args, "sub_set_num", 0)),
        "tau1": float(getattr(args, "tau1", 0.0)),
        "tau_delta": float(getattr(args, "tau_delta", 0.0)),
        "temperature": float(getattr(args, "temperature", 0.0)),
        "max_new_tokens": int(getattr(args, "max_new_tokens", 0)),
        "stage3_train_projector": int(getattr(args, "stage3_train_projector", 0)),
        "stage3_field_weight_observed": float(getattr(args, "stage3_field_weight_observed", 0.0)),
        "stage3_field_weight_context": float(getattr(args, "stage3_field_weight_context", 0.0)),
        "stage3_field_weight_reasoning": float(getattr(args, "stage3_field_weight_reasoning", 0.0)),
        "stage3_field_weight_answer": float(getattr(args, "stage3_field_weight_answer", 0.0)),
        "stage3_obj_weight": float(getattr(args, "stage3_obj_weight", 0.0)),
        "stage3_reg_weight": float(getattr(args, "stage3_reg_weight", 0.0)),
        "stage3_answer_anchor_weight": float(getattr(args, "stage3_answer_anchor_weight", 0.0)),
        "stage3_mc_weight": float(getattr(args, "stage3_mc_weight", 0.0)),
        "stage3_pair_use_answer_correctness": int(getattr(args, "stage3_pair_use_answer_correctness", 0)),
        "stage3_wrong_image_enable": int(getattr(args, "stage3_wrong_image_enable", 0)),
        "stage3_wrong_image_weight": float(getattr(args, "stage3_wrong_image_weight", 0.0)),
        "stage3_wrong_image_margin": float(getattr(args, "stage3_wrong_image_margin", 0.0)),
        "stage3_force_cold_start_period0": int(getattr(args, "stage3_force_cold_start_period0", 0)),
        "scienceqa_preprocessed_path": str(getattr(args, "scienceqa_preprocessed_path", "")),
        "stage3_sample_cache_path": str(getattr(args, "stage3_sample_cache_path", "")),
        "vq_codebook_path": str(getattr(args, "vq_codebook_path", "")),
        "remove_vq_codebook": int(getattr(args, "remove_vq_codebook", 0)),
        "stage2_ckpt_path": str(getattr(args, "stage2_ckpt_path", "")),
        "world_size": int(getattr(args, "world_size", 1)),
    }


def _validate_stage3_resume_config(resume_config: dict, args):
    if not isinstance(resume_config, dict):
        return

    current = _stage3_resume_config(args)
    strict_keys = [
        "model_path",
        "student_model_type",
        "use_lora",
        "lora_rank",
        "lora_alpha",
        "use_4bit",
        "model_dtype",
        "dataset_name",
        "scienceqa_path",
        "scienceqa_split",
        "scienceqa_seed",
        "train_num",
        "max_length",
        "teacher_lang",
        "teacher_observed_max_tokens",
        "teacher_context_max_tokens",
        "teacher_reasoning_max_tokens",
        "teacher_answer_max_tokens",
        "stage3_vic_include_context",
        "bucket_batch_size",
        "stage3_bucket_batch_size",
        "disable_bucket_for_stage3",
        "sub_set_num",
        "tau1",
        "tau_delta",
        "temperature",
        "max_new_tokens",
        "stage3_train_projector",
        "stage3_field_weight_observed",
        "stage3_field_weight_context",
        "stage3_field_weight_reasoning",
        "stage3_field_weight_answer",
        "stage3_obj_weight",
        "stage3_reg_weight",
        "stage3_answer_anchor_weight",
        "stage3_mc_weight",
        "stage3_pair_use_answer_correctness",
        "stage3_wrong_image_enable",
        "stage3_wrong_image_weight",
        "stage3_wrong_image_margin",
        "stage3_force_cold_start_period0",
        "scienceqa_preprocessed_path",
        "remove_vq_codebook",
        "stage2_ckpt_path",
    ]
    if not int(getattr(args, "remove_vq_codebook", 0)):
        strict_keys.append("vq_codebook_path")
    for key in strict_keys:
        if key in resume_config and resume_config[key] != current[key]:
            raise RuntimeError(
                f"Stage3 resume 配置不一致: key={key}, "
                f"resume={resume_config[key]!r}, current={current[key]!r}"
            )

    warning_keys = [
        "lr",
        "stage3_lr_scale",
        "stage3_grad_clip",
        "stage3_grad_accum",
        "grad_accum",
        "batch_size",
        "sub_stage_num",
        "period_num",
        "stage3_sample_cache_path",
        "world_size",
    ]
    mismatched = []
    for key in warning_keys:
        if key in resume_config and resume_config[key] != current[key]:
            mismatched.append(
                f"{key}: resume={resume_config[key]!r}, current={current[key]!r}"
            )
    if mismatched and is_main_process():
        print("[Stage3][Resume][Warn] 检测到以下配置与断点不一致：")
        for item in mismatched:
            print(f"  - {item}")


def _serialize_period_states(period_states: Optional[Dict[int, PeriodState]]) -> Optional[dict]:
    if period_states is None:
        return None
    payload = {}
    for sample_idx, state in period_states.items():
        payload[int(sample_idx)] = {
            "sample_idx": int(state.sample_idx),
            "y11_ids": state.y11_ids.detach().cpu(),
            "y11_mask": state.y11_mask.detach().cpu(),
            "y12_ids": state.y12_ids.detach().cpu(),
            "y12_mask": state.y12_mask.detach().cpu(),
            "avg_lp_11": float(state.avg_lp_11),
            "avg_lp_12": float(state.avg_lp_12),
            "prob_11": float(state.prob_11),
            "prob_12": float(state.prob_12),
        }
    return payload


def _deserialize_period_states(payload: Optional[dict]) -> Optional[Dict[int, PeriodState]]:
    if payload is None:
        return None
    states: Dict[int, PeriodState] = {}
    for key, item in payload.items():
        sample_idx = int(item.get("sample_idx", key))
        states[sample_idx] = PeriodState(
            sample_idx=sample_idx,
            y11_ids=item["y11_ids"].detach().cpu().long(),
            y11_mask=item["y11_mask"].detach().cpu().long(),
            y12_ids=item["y12_ids"].detach().cpu().long(),
            y12_mask=item["y12_mask"].detach().cpu().long(),
            avg_lp_11=float(item["avg_lp_11"]),
            avg_lp_12=float(item["avg_lp_12"]),
            prob_11=float(item["prob_11"]),
            prob_12=float(item["prob_12"]),
        )
    return states


def _save_stage3_resume_state(
    model,
    optimizer,
    resume_dir: str,
    progress: dict,
    args,
    include_optimizer_state: bool = True,
):
    model = unwrap_model(model)
    if is_main_process():
        os.makedirs(resume_dir, exist_ok=True)

    local_rng_state = _capture_rng_state()
    rng_state_by_rank = {int(get_rank()): local_rng_state}
    if is_distributed():
        world_size = get_world_size()
        gathered_rng_states = [None for _ in range(world_size)] if is_main_process() else None
        dist.gather_object(local_rng_state, gathered_rng_states, dst=0)
        if not is_main_process():
            return
        rng_state_by_rank = {
            int(rank_idx): gathered_rng_states[rank_idx]
            for rank_idx in range(world_size)
        }

    payload = {
        "version": 1,
        "trainable_model_state": _get_trainable_parameter_state(model),
        "optimizer_state": _to_cpu_obj(optimizer.state_dict()) if include_optimizer_state else None,
        "rng_state": local_rng_state,
        "rng_state_by_rank": rng_state_by_rank,
        "progress": dict(progress),
        "config": _stage3_resume_config(args),
    }
    torch.save(payload, _stage3_resume_state_path(resume_dir))
    meta = dict(progress)
    meta["config"] = _stage3_resume_config(args)
    meta["has_period_states"] = bool(meta.get("period_states") is not None)
    meta["has_optimizer_state"] = bool(include_optimizer_state)
    meta.pop("period_states", None)
    with open(_stage3_resume_meta_path(resume_dir), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    if is_main_process():
        print(f"[Stage3][Resume] 已保存断点: {resume_dir}")


def _load_stage3_resume_state(
    model,
    optimizer,
    resume_dir: str,
    device: str,
    args,
) -> Optional[dict]:
    state_path = _stage3_resume_state_path(resume_dir)
    if not os.path.exists(state_path):
        print(f"[Stage3][Resume] 未找到断点文件: {state_path}")
        return None

    payload = torch.load(state_path, map_location="cpu")
    _validate_stage3_resume_config(payload.get("config", {}), args)
    _load_parameter_state(model, payload.get("trainable_model_state", {}), "Stage3 resume trainable_model_state")

    optimizer_state = payload.get("optimizer_state")
    if optimizer_state:
        optimizer.load_state_dict(optimizer_state)
        _move_optimizer_state_to_device_(optimizer, device)
        if is_main_process():
            print("[Stage3][Resume] 已加载 optimizer state")
    else:
        if is_main_process():
            print("[Stage3][Resume] 断点中不含 optimizer state，将从当前优化器状态继续")

    rng_state_by_rank = payload.get("rng_state_by_rank")
    local_rank_key = int(get_rank())
    local_rng_state = None
    if isinstance(rng_state_by_rank, dict):
        local_rng_state = rng_state_by_rank.get(local_rank_key)
        if local_rng_state is None:
            local_rng_state = rng_state_by_rank.get(str(local_rank_key))
    if local_rng_state is None:
        if isinstance(rng_state_by_rank, dict) and is_main_process():
            print(
                f"[Stage3][Resume][Warn] 断点中未找到 rank={local_rank_key} 的专属 RNG 状态，"
                "将回退到共享 RNG 状态。"
            )
        local_rng_state = payload.get("rng_state")
    _restore_rng_state(local_rng_state)
    progress = payload.get("progress", {})
    if is_main_process():
        print(
            f"[Stage3][Resume] 已恢复进度: next_sub_stage={progress.get('next_sub_stage_idx', 0)}, "
            f"next_period={progress.get('next_period_idx', 0)}, global_step={progress.get('global_step', 0)}, "
            f"completed={bool(progress.get('completed', False))}"
        )
    return progress


def _stage3_phase_name_offset(phase_name: str) -> int:
    text = str(phase_name or "")
    total = 0
    for idx, ch in enumerate(text):
        total = (total + (idx + 1) * ord(ch)) % 1_000_003
    return total


def _make_stage3_phase_seed(
    base_seed: int,
    sub_stage_idx: int,
    period_idx: int,
    phase_name: str,
    global_batch_idx: int,
) -> int:
    phase_offset = _stage3_phase_name_offset(phase_name)
    seed = (
        int(base_seed) * 1_000_003
        + int(sub_stage_idx) * 100_003
        + int(period_idx) * 10_007
        + int(global_batch_idx) * 1_009
        + phase_offset
    )
    return int(seed % (2**31 - 1))


def _make_stage3_torch_generator(device: str, seed: int) -> torch.Generator:
    generator_device = "cuda" if str(device).startswith("cuda") and torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device=generator_device)
    generator.manual_seed(int(seed))
    return generator


def _shard_stage3_global_batches(global_batches: List[List[int]]) -> List[Tuple[int, List[int]]]:
    return [
        (global_batch_idx, list(batch_indices))
        for global_batch_idx, batch_indices in enumerate(global_batches)
        if (global_batch_idx % get_world_size()) == get_rank()
    ]


def _stage3_gather_object_to_main(local_obj):
    if not is_distributed():
        return [local_obj]
    gathered = [None for _ in range(get_world_size())] if is_main_process() else None
    dist.gather_object(local_obj, gathered, dst=0)
    return gathered


def _stage3_broadcast_object_from_main(obj):
    if not is_distributed():
        return obj
    object_list = [obj if is_main_process() else None]
    dist.broadcast_object_list(object_list, src=0)
    return object_list[0]


def _build_stage3_period_dataloader(
    period_dataset: torch.utils.data.Dataset,
    args,
    pad_token_id: int,
    seed: int,
):
    if is_distributed():
        try:
            sampler = DistributedSampler(
                period_dataset,
                num_replicas=get_world_size(),
                rank=get_rank(),
                shuffle=True,
                seed=int(seed),
                drop_last=False,
            )
        except TypeError:
            sampler = DistributedSampler(
                period_dataset,
                num_replicas=get_world_size(),
                rank=get_rank(),
                shuffle=True,
                drop_last=False,
            )
            if hasattr(sampler, "seed"):
                sampler.seed = int(seed)
        loader = DataLoader(
            period_dataset,
            batch_size=max(1, int(args.batch_size)),
            shuffle=False,
            sampler=sampler,
            num_workers=0,
            collate_fn=lambda b: _collate_period_training(b, pad_token_id),
        )
        maybe_set_dataloader_epoch(loader, 0)
        return loader

    return DataLoader(
        period_dataset,
        batch_size=max(1, int(args.batch_size)),
        shuffle=True,
        num_workers=0,
        collate_fn=lambda b: _collate_period_training(b, pad_token_id),
    )


def _validate_stage3_chunk_merge(
    merged_sample_indices: List[int],
    active_indices: List[int],
    desc: str,
):
    merged_sorted = sorted(int(idx) for idx in merged_sample_indices)
    active_sorted = sorted(int(idx) for idx in active_indices)
    if merged_sorted != active_sorted:
        raise RuntimeError(
            f"[Stage3][DDP] {desc} merge 后 sample_idx 覆盖不完整: "
            f"merged={len(merged_sorted)}, active={len(active_sorted)}"
        )


def _normalize_image_size_key(image_sizes: Optional[torch.Tensor]) -> Tuple[int, int]:
    if image_sizes is None:
        return (-1, -1)
    if isinstance(image_sizes, torch.Tensor):
        flat = image_sizes.detach().view(-1)
        if flat.numel() >= 2:
            return (int(flat[0].item()), int(flat[1].item()))
    if isinstance(image_sizes, (list, tuple)) and len(image_sizes) >= 2:
        return (int(image_sizes[0]), int(image_sizes[1]))
    return (-1, -1)


def _count_image_tokens(ids: torch.Tensor, image_token_id: Optional[int]) -> int:
    if image_token_id is None:
        return -1
    return int((ids == int(image_token_id)).sum().item())


def _build_rotating_next_lookup(groups: Dict[tuple, List[int]]) -> Dict[int, int]:
    lookup: Dict[int, int] = {}
    for _, members in groups.items():
        if len(members) <= 1:
            continue
        for idx, sample_idx in enumerate(members):
            lookup[int(sample_idx)] = int(members[(idx + 1) % len(members)])
    return lookup


def _build_wrong_image_lookup(
    active_indices: List[int],
    sample_cache: List[Stage3SampleCacheItem],
    image_token_id: Optional[int],
) -> Tuple[Dict[int, int], Dict[str, int]]:
    ordered_indices = [int(idx) for idx in active_indices]
    if not ordered_indices:
        return {}, {"primary": 0, "fallback_patch": 0, "fallback_token": 0, "self": 0}

    primary_groups: Dict[tuple, List[int]] = defaultdict(list)
    fallback_patch_groups: Dict[tuple, List[int]] = defaultdict(list)
    fallback_token_groups: Dict[tuple, List[int]] = defaultdict(list)

    for idx in ordered_indices:
        sample = sample_cache[idx]
        image_token_count = _count_image_tokens(sample.y_vic_ids, image_token_id)
        patch_count = int(sample.pixel_values.shape[0]) if sample.pixel_values.dim() >= 1 else 0
        image_size_key = _normalize_image_size_key(sample.image_sizes)
        primary_groups[(image_token_count, patch_count, image_size_key)].append(idx)
        fallback_patch_groups[(image_token_count, patch_count)].append(idx)
        fallback_token_groups[(image_token_count,)].append(idx)

    primary_next = _build_rotating_next_lookup(primary_groups)
    fallback_patch_next = _build_rotating_next_lookup(fallback_patch_groups)
    fallback_token_next = _build_rotating_next_lookup(fallback_token_groups)

    wrong_lookup: Dict[int, int] = {}
    stats = {"primary": 0, "fallback_patch": 0, "fallback_token": 0, "self": 0}
    for idx in ordered_indices:
        if idx in primary_next:
            wrong_lookup[idx] = int(primary_next[idx])
            stats["primary"] += 1
            continue
        if idx in fallback_patch_next:
            wrong_lookup[idx] = int(fallback_patch_next[idx])
            stats["fallback_patch"] += 1
            continue
        if idx in fallback_token_next:
            wrong_lookup[idx] = int(fallback_token_next[idx])
            stats["fallback_token"] += 1
            continue
        wrong_lookup[idx] = int(idx)
        stats["self"] += 1
    return wrong_lookup, stats


class PeriodTrainingDataset(torch.utils.data.Dataset):
    def __init__(self, items: List[PeriodTrainingItem], sample_cache: List[Stage3SampleCacheItem]):
        self.items = items
        self.sample_cache = sample_cache

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        pair = self.items[idx]
        sample = self.sample_cache[pair.sample_idx]
        wrong_sample = self.sample_cache[pair.wrong_sample_idx]
        return {
            "prompt_ids": sample.prompt_ids,
            "prompt_mask": sample.prompt_mask,
            "y_plus_ids": pair.y_plus_ids,
            "y_plus_mask": pair.y_plus_mask,
            "y_minus_ids": pair.y_minus_ids,
            "y_minus_mask": pair.y_minus_mask,
            "y_vic_ids": pair.y_vic_ids,
            "y_vic_mask": pair.y_vic_mask,
            "old_token_lp_plus": pair.old_token_lp_plus,
            "old_token_mask_plus": pair.old_token_mask_plus,
            "old_token_lp_minus": pair.old_token_lp_minus,
            "old_token_mask_minus": pair.old_token_mask_minus,
            "old_token_lp_vic": pair.old_token_lp_vic,
            "old_token_mask_vic": pair.old_token_mask_vic,
            "pixel_values": sample.pixel_values,
            "image_sizes": sample.image_sizes,
            "image_grid_thw": sample.image_grid_thw,
            "wrong_pixel_values": wrong_sample.pixel_values,
            "wrong_image_sizes": wrong_sample.image_sizes,
            "wrong_image_grid_thw": wrong_sample.image_grid_thw,
            "vic_observed_mask": sample.vic_observed_mask,
            "vic_context_mask": sample.vic_context_mask,
            "vic_reasoning_mask": sample.vic_reasoning_mask,
            "vic_answer_mask": sample.vic_answer_mask,
            "vic_other_mask": sample.vic_other_mask,
            "prompt_len": int(sample.prompt_len),
            "answer_idx": int(sample.answer_idx),
            "num_choices": int(sample.num_choices),
        }


def _pad_1d_tensor(tensor: torch.Tensor, target_len: int, pad_value: float) -> torch.Tensor:
    cur_len = int(tensor.shape[0])
    if cur_len == target_len:
        return tensor
    if cur_len > target_len:
        return tensor[:target_len]
    return F.pad(tensor, (0, target_len - cur_len), value=pad_value)


def _pad_and_stack_1d_tensors(
    tensors: List[torch.Tensor],
    pad_value: float,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    if not tensors:
        raise ValueError("tensors 不能为空")
    target_len = max(int(t.shape[0]) for t in tensors)
    out_dtype = dtype or tensors[0].dtype
    return torch.stack(
        [_pad_1d_tensor(t.to(dtype=out_dtype), target_len, pad_value) for t in tensors],
        dim=0,
    )


def _left_pad_and_stack_1d_tensors(
    tensors: List[torch.Tensor],
    pad_value: float,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    if not tensors:
        raise ValueError("tensors 不能为空")
    target_len = max(int(t.shape[0]) for t in tensors)
    out_dtype = dtype or tensors[0].dtype
    padded = []
    for tensor in tensors:
        tensor = tensor.to(dtype=out_dtype)
        cur_len = int(tensor.shape[0])
        if cur_len >= target_len:
            padded.append(tensor[-target_len:])
            continue
        padded.append(F.pad(tensor, (target_len - cur_len, 0), value=pad_value))
    return torch.stack(padded, dim=0)


def _stack_padded_pixel_values(pv_list: List[torch.Tensor]) -> torch.Tensor:
    normalized = []
    for pv in pv_list:
        if pv.dim() == 3:
            pv = pv.unsqueeze(0)
        normalized.append(pv)

    pv_shapes = [tuple(pv.shape) for pv in normalized]
    if len(set(pv_shapes)) > 1:
        max_patches = max(shape[0] for shape in pv_shapes)
        for i, pv in enumerate(normalized):
            if pv.shape[0] < max_patches:
                pad = torch.zeros(
                    (max_patches - pv.shape[0],) + pv.shape[1:],
                    dtype=pv.dtype,
                    device=pv.device,
                )
                normalized[i] = torch.cat([pv, pad], dim=0)
    return torch.stack(normalized, dim=0)


def _stack_optional_image_sizes(image_sizes_list: List[Optional[torch.Tensor]]) -> Optional[torch.Tensor]:
    if not any(size is not None for size in image_sizes_list):
        return None
    return torch.stack(
        [
            size if isinstance(size, torch.Tensor) else torch.tensor(size, dtype=torch.long)
            for size in image_sizes_list
        ],
        dim=0,
    )


def _stack_optional_image_grid_thw(grid_list: List[Optional[torch.Tensor]]) -> Optional[torch.Tensor]:
    if not any(grid is not None for grid in grid_list):
        return None
    normalized = []
    for grid in grid_list:
        if grid is None:
            raise RuntimeError("同一 batch 中 Qwen image_grid_thw 不能部分缺失")
        if not isinstance(grid, torch.Tensor):
            grid = torch.tensor(grid, dtype=torch.long)
        if grid.dim() == 1:
            grid = grid.unsqueeze(0)
        normalized.append(grid.to(dtype=torch.long))
    return torch.cat(normalized, dim=0)


def _collate_period_training(batch: List[dict], pad_token_id: int) -> dict:
    if not batch:
        return None

    prompt_ids = torch.stack([
        _pad_1d_tensor(item["prompt_ids"].long(), max(int(x["prompt_ids"].shape[0]) for x in batch), pad_token_id)
        for item in batch
    ], dim=0)
    prompt_mask = torch.stack([
        _pad_1d_tensor(item["prompt_mask"].long(), prompt_ids.shape[1], 0)
        for item in batch
    ], dim=0)

    stream_to_old_key = {
        "y_plus": ("old_token_lp_plus", "old_token_mask_plus"),
        "y_minus": ("old_token_lp_minus", "old_token_mask_minus"),
        "y_vic": ("old_token_lp_vic", "old_token_mask_vic"),
    }

    def _collate_stream(prefix: str):
        ids_key = f"{prefix}_ids"
        mask_key = f"{prefix}_mask"
        old_lp_key, old_mask_key = stream_to_old_key[prefix]

        ids_list = [item[ids_key] for item in batch]
        mask_list = [item[mask_key] for item in batch]
        old_lp_list = [item[old_lp_key] for item in batch]
        old_mask_list = [item[old_mask_key] for item in batch]

        normalized_old_lp = []
        normalized_old_mask = []
        for ids, old_lp, old_mask in zip(ids_list, old_lp_list, old_mask_list):
            expected_len = max(0, int(ids.shape[0]) - 1)
            normalized_old_lp.append(_pad_1d_tensor(old_lp.float(), expected_len, 0.0))
            normalized_old_mask.append(_pad_1d_tensor(old_mask.float(), expected_len, 0.0))

        max_ids_len = max(int(x.shape[0]) for x in ids_list)
        target_old_len = max(0, max_ids_len - 1)

        ids = torch.stack([
            _pad_1d_tensor(x.long(), max_ids_len, pad_token_id) for x in ids_list
        ], dim=0)
        masks = torch.stack([
            _pad_1d_tensor(x.long(), max_ids_len, 0) for x in mask_list
        ], dim=0)
        old_lp = torch.stack([
            _pad_1d_tensor(x, target_old_len, 0.0) for x in normalized_old_lp
        ], dim=0)
        old_mask = torch.stack([
            _pad_1d_tensor(x, target_old_len, 0.0) for x in normalized_old_mask
        ], dim=0)
        return ids, masks, old_lp, old_mask

    plus_ids, plus_mask, old_lp_plus, old_mask_plus = _collate_stream("y_plus")
    minus_ids, minus_mask, old_lp_minus, old_mask_minus = _collate_stream("y_minus")
    vic_ids, vic_mask, old_lp_vic, old_mask_vic = _collate_stream("y_vic")

    def _stack_pixel_values(key: str) -> torch.Tensor:
        pv_list = []
        for item in batch:
            pv = item[key]
            if pv.dim() == 3:
                pv = pv.unsqueeze(0)
            pv_list.append(pv)

        pv_shapes = [tuple(pv.shape) for pv in pv_list]
        if len(set(pv_shapes)) > 1:
            max_patches = max(shape[0] for shape in pv_shapes)
            for i, pv in enumerate(pv_list):
                if pv.shape[0] < max_patches:
                    pad = torch.zeros(
                        (max_patches - pv.shape[0],) + pv.shape[1:],
                        dtype=pv.dtype,
                        device=pv.device,
                    )
                    pv_list[i] = torch.cat([pv, pad], dim=0)
        return torch.stack(pv_list, dim=0)

    def _stack_image_sizes(key: str) -> Optional[torch.Tensor]:
        image_sizes_list = [item.get(key) for item in batch]
        if not any(size is not None for size in image_sizes_list):
            return None
        return torch.stack([
            size if isinstance(size, torch.Tensor) else torch.tensor(size, dtype=torch.long)
            for size in image_sizes_list
        ], dim=0)

    def _collate_vic_field_mask(key: str) -> torch.Tensor:
        target_old_len = old_lp_vic.shape[1]
        masks = []
        for item in batch:
            field_mask = item.get(key)
            if field_mask is None:
                field_mask = torch.zeros((target_old_len,), dtype=torch.float32)
            masks.append(_pad_1d_tensor(field_mask.float(), target_old_len, 0.0))
        return torch.stack(masks, dim=0)

    pixel_values = _stack_pixel_values("pixel_values")
    wrong_pixel_values = _stack_pixel_values("wrong_pixel_values")
    image_sizes = _stack_image_sizes("image_sizes")
    wrong_image_sizes = _stack_image_sizes("wrong_image_sizes")
    image_grid_thw = _stack_optional_image_grid_thw([item.get("image_grid_thw") for item in batch])
    wrong_image_grid_thw = _stack_optional_image_grid_thw([item.get("wrong_image_grid_thw") for item in batch])
    vic_observed_mask = _collate_vic_field_mask("vic_observed_mask")
    vic_context_mask = _collate_vic_field_mask("vic_context_mask")
    vic_reasoning_mask = _collate_vic_field_mask("vic_reasoning_mask")
    vic_answer_mask = _collate_vic_field_mask("vic_answer_mask")
    vic_other_mask = _collate_vic_field_mask("vic_other_mask")

    prompt_lens = torch.tensor([int(item["prompt_len"]) for item in batch], dtype=torch.long)

    return {
        "prompt_ids": prompt_ids,
        "prompt_mask": prompt_mask,
        "y_plus_ids": plus_ids,
        "y_plus_mask": plus_mask,
        "y_minus_ids": minus_ids,
        "y_minus_mask": minus_mask,
        "y_vic_ids": vic_ids,
        "y_vic_mask": vic_mask,
        "old_token_lp_plus": old_lp_plus,
        "old_token_mask_plus": old_mask_plus,
        "old_token_lp_minus": old_lp_minus,
        "old_token_mask_minus": old_mask_minus,
        "old_token_lp_vic": old_lp_vic,
        "old_token_mask_vic": old_mask_vic,
        "pixel_values": pixel_values,
        "image_sizes": image_sizes,
        "image_grid_thw": image_grid_thw,
        "wrong_pixel_values": wrong_pixel_values,
        "wrong_image_sizes": wrong_image_sizes,
        "wrong_image_grid_thw": wrong_image_grid_thw,
        "vic_observed_mask": vic_observed_mask,
        "vic_context_mask": vic_context_mask,
        "vic_reasoning_mask": vic_reasoning_mask,
        "vic_answer_mask": vic_answer_mask,
        "vic_other_mask": vic_other_mask,
        "prompt_lens": prompt_lens,
        "answer_idx": torch.tensor([int(item["answer_idx"]) for item in batch], dtype=torch.long),
        "num_choices": torch.tensor([int(item["num_choices"]) for item in batch], dtype=torch.long),
    }


def _find_token_subsequence(haystack: List[int], needle: List[int], start_pos: int) -> int:
    if not needle:
        return -1
    max_start = len(haystack) - len(needle)
    for pos in range(max(0, start_pos), max_start + 1):
        if haystack[pos:pos + len(needle)] == needle:
            return pos
    return -1


def _build_vic_field_masks(
    y_vic_ids: torch.Tensor,
    prompt_len: int,
    tokenizer,
    vic_target: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    old_len = max(0, int(y_vic_ids.shape[0]) - 1)
    observed_mask = torch.zeros((old_len,), dtype=torch.float32)
    context_mask = torch.zeros((old_len,), dtype=torch.float32)
    reasoning_mask = torch.zeros((old_len,), dtype=torch.float32)
    answer_mask = torch.zeros((old_len,), dtype=torch.float32)

    if tokenizer is None or not vic_target or old_len == 0:
        other_mask = torch.ones((old_len,), dtype=torch.float32)
        return observed_mask, context_mask, reasoning_mask, answer_mask, other_mask

    generated_ids = y_vic_ids[int(prompt_len):].tolist()
    if not generated_ids:
        other_mask = torch.ones((old_len,), dtype=torch.float32)
        return observed_mask, context_mask, reasoning_mask, answer_mask, other_mask

    field_to_mask = {
        "Observed Facts:": observed_mask,
        "Context:": context_mask,
        "Reasoning:": reasoning_mask,
        "Answer:": answer_mask,
    }
    cursor = 0
    for raw_line in str(vic_target).splitlines():
        line = raw_line.strip()
        if not line:
            continue
        prefix = next((p for p in field_to_mask if line.startswith(p)), None)
        if prefix is None:
            continue
        line_tokens = tokenizer.encode(line, add_special_tokens=False)
        start = _find_token_subsequence(generated_ids, line_tokens, cursor)
        if start < 0:
            continue
        base = max(0, int(prompt_len) - 1 + start)
        end = min(old_len, base + len(line_tokens))
        if end > base:
            field_to_mask[prefix][base:end] = 1.0
        cursor = start + len(line_tokens)

    union_mask = torch.clamp(
        observed_mask + context_mask + reasoning_mask + answer_mask,
        min=0.0,
        max=1.0,
    )
    other_mask = torch.clamp(torch.ones((old_len,), dtype=torch.float32) - union_mask, min=0.0, max=1.0)
    return observed_mask, context_mask, reasoning_mask, answer_mask, other_mask


def _build_stage3_sample_cache(train_dataset) -> List[Stage3SampleCacheItem]:
    cache: List[Stage3SampleCacheItem] = []
    processor = train_dataset.processor
    tokenizer = train_dataset.processor.tokenizer
    for idx in tqdm(range(len(train_dataset)), desc="构建 Stage3SampleCache"):
        sample = train_dataset[idx]
        raw_item = train_dataset.samples[idx]

        prompt_ids = sample["prompt_input_ids"]
        prompt_mask = sample["prompt_attention_mask"]

        instruction = raw_item["instruction"]
        instruction_text = _strip_image_tokens(instruction)
        _, _, vic_target, _ = train_dataset._build_targets(raw_item, instruction_text)

        prompt_conv = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction_text},
                    {"type": "image"},
                ],
            }
        ]
        vic_conv = prompt_conv + [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": vic_target},
                ],
            }
        ]
        vic_text = processor.apply_chat_template(vic_conv, add_generation_prompt=False)

        vic_inputs = processor(
            text=vic_text,
            images=raw_item["image"],
            return_tensors="pt",
            padding="longest",
            truncation=False,
        )
        y_vic_ids = vic_inputs["input_ids"].squeeze(0)
        y_vic_mask = vic_inputs["attention_mask"].squeeze(0)

        prompt_len = int(prompt_ids.shape[0])
        pixel_values = sample["pixel_values"]
        if pixel_values.dim() == 3:
            pixel_values = pixel_values.unsqueeze(0)

        image_sizes = sample["image_sizes"][:2].to(dtype=torch.long).cpu()

        observed_ids = None
        context_ids = None
        reasoning_ids = None
        answer_ids = None
        vic_observed_mask = None
        vic_context_mask = None
        vic_reasoning_mask = None
        vic_answer_mask = None
        vic_other_mask = torch.ones((max(0, int(y_vic_ids.shape[0]) - 1),), dtype=torch.float32)
        ann = raw_item.get("teacher_annotation")
        if isinstance(ann, dict):
            observed = ann.get("observed_facts_visual", "")
            context = ann.get("context_textual", "")
            reasoning = ann.get("reasoning", "")
            answer = ann.get("answer", "")
            if isinstance(observed, str):
                observed_ids = torch.tensor(
                    tokenizer.encode(observed, add_special_tokens=False),
                    dtype=torch.long,
                )
            if isinstance(context, str):
                context_ids = torch.tensor(
                    tokenizer.encode(context, add_special_tokens=False),
                    dtype=torch.long,
                )
            if isinstance(reasoning, str):
                reasoning_ids = torch.tensor(
                    tokenizer.encode(reasoning, add_special_tokens=False),
                    dtype=torch.long,
                )
            if isinstance(answer, str):
                answer_ids = torch.tensor(
                    tokenizer.encode(answer, add_special_tokens=False),
                    dtype=torch.long,
                )
        (
            vic_observed_mask,
            vic_context_mask,
            vic_reasoning_mask,
            vic_answer_mask,
            vic_other_mask,
        ) = _build_vic_field_masks(
            y_vic_ids=y_vic_ids.detach().cpu().long(),
            prompt_len=prompt_len,
            tokenizer=tokenizer,
            vic_target=vic_target,
        )

        cache.append(Stage3SampleCacheItem(
            sample_idx=int(idx),
            prompt_ids=prompt_ids.detach().cpu().long(),
            prompt_mask=prompt_mask.detach().cpu().long(),
            y_vic_ids=y_vic_ids.detach().cpu().long(),
            y_vic_mask=y_vic_mask.detach().cpu().long(),
            pixel_values=pixel_values.detach().cpu(),
            image_sizes=image_sizes,
            image_grid_thw=sample.get("image_grid_thw"),
            prompt_len=prompt_len,
            observed_ids=observed_ids,
            context_ids=context_ids,
            reasoning_ids=reasoning_ids,
            answer_ids=answer_ids,
            vic_observed_mask=vic_observed_mask,
            vic_context_mask=vic_context_mask,
            vic_reasoning_mask=vic_reasoning_mask,
            vic_answer_mask=vic_answer_mask,
            vic_other_mask=vic_other_mask,
            answer_letter=str(sample.get("answer_letter", "") or "").strip().upper()[:1],
            answer_idx=int(sample["answer_idx"]),
            num_choices=len(raw_item["choices"]),
        ))
    return cache


def _resolve_stage3_sample_cache_path(args) -> str:
    cache_path = str(args.stage3_sample_cache_path).strip()
    if not cache_path:
        raise RuntimeError("stage3_sample_cache_path 不能为空")
    return cache_path


def _resolve_stage3_eval_sample_cache_path(
    base_cache_path: str,
    eval_split: str,
    eval_path: str,
    eval_train_num: int,
) -> str:
    base_cache_path = str(base_cache_path or "").strip()
    if not base_cache_path:
        return ""
    root, ext = os.path.splitext(base_cache_path)
    if not ext:
        ext = ".pt"
    eval_split = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(eval_split or "eval"))
    eval_name = os.path.basename(str(eval_path or "").rstrip("/")) or "scienceqa"
    eval_name = re.sub(r"[^a-zA-Z0-9_.-]+", "_", eval_name)
    return f"{root}.eval_{eval_split}_{eval_name}_n{int(eval_train_num)}{ext}"


def _load_stage3_sample_cache(cache_path: str) -> Optional[List[Stage3SampleCacheItem]]:
    if not cache_path or not os.path.exists(cache_path):
        return None
    payload = torch.load(cache_path, map_location="cpu")

    if isinstance(payload, list):
        cache_items = payload
    elif isinstance(payload, dict) and isinstance(payload.get("samples"), list):
        cache_items = payload["samples"]
    else:
        raise RuntimeError(f"[Stage3][Cache] sample cache 格式无效: {cache_path}")
    if cache_items:
        probe = cache_items[0]
        for field_name in ("prompt_ids", "prompt_mask", "answer_letter", "answer_idx", "num_choices"):
            if not hasattr(probe, field_name):
                raise RuntimeError(
                    f"[Stage3][Cache] sample cache 缺少字段 {field_name}: {cache_path}"
                )
    print(f"[Stage3][Cache] 已复用 Stage3SampleCache: {cache_path}, samples={len(cache_items)}")
    return cache_items


def _save_stage3_sample_cache(cache_path: str, cache_items: List[Stage3SampleCacheItem]):
    cache_dir = os.path.dirname(cache_path) or "."
    os.makedirs(cache_dir, exist_ok=True)
    torch.save(cache_items, cache_path)
    print(f"[Stage3][Cache] 已保存 Stage3SampleCache: {cache_path}, samples={len(cache_items)}")


def _set_stage3_trainable_params(model, args) -> List[torch.nn.Parameter]:
    base_model = unwrap_model(model)
    for name, param in base_model.named_parameters():
        if not torch.is_floating_point(param):
            param.requires_grad = False
            continue
        name_l = name.lower()
        is_lora = "lora_" in name_l or "modules_to_save" in name_l
        is_projector = (
            is_projector_param(name, getattr(args, "student_model_type", "llava_next"))
            and bool(args.stage3_train_projector)
        )
        param.requires_grad = bool(is_lora or is_projector)
    return [p for p in base_model.parameters() if p.requires_grad]


def _strip_extra_image_tokens(
    ids: torch.Tensor,
    image_token_id: Optional[int],
    allowed_count: int,
    replacement_token_id: int,
) -> torch.Tensor:
    if image_token_id is None:
        return ids
    ids = ids.clone()
    pos = (ids[0] == int(image_token_id)).nonzero(as_tuple=False).view(-1)
    if pos.numel() > allowed_count:
        ids[0, pos[allowed_count:]] = int(replacement_token_id)
    return ids


def _generate_candidate_single(
    model,
    sample: Stage3SampleCacheItem,
    args,
    image_token_id: Optional[int],
    pad_token_id: int,
    do_sample: bool = True,
    temperature: Optional[float] = None,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = args.device
    prompt_ids = sample.prompt_ids.unsqueeze(0).to(device)
    prompt_mask = sample.prompt_mask.unsqueeze(0).to(device)
    pixel_values = sample.pixel_values.unsqueeze(0).to(device)
    image_sizes = sanitize_image_sizes(sample.image_sizes.unsqueeze(0), batch_size=1)
    image_grid_thw = _stack_optional_image_grid_thw([sample.image_grid_thw])
    allowed_img_count = int((sample.prompt_ids == int(image_token_id)).sum().item()) if image_token_id is not None else 0

    gen_kwargs = dict(
        input_ids=prompt_ids,
        attention_mask=prompt_mask,
        pixel_values=pixel_values,
        image_sizes=image_sizes,
        image_grid_thw=image_grid_thw,
        do_sample=bool(do_sample),
        max_new_tokens=int(args.max_new_tokens),
        bad_words_ids=[[int(image_token_id)]] if image_token_id is not None else None,
        pad_token_id=int(pad_token_id),
    )
    if do_sample:
        gen_kwargs["temperature"] = float(args.temperature if temperature is None else temperature)
    rng_state = None
    if do_sample and generator is not None:
        rng_state = _capture_rng_state()
        seed = int(generator.initial_seed())
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    try:
        with torch.no_grad():
            generated = student_generate(model, args, **gen_kwargs)
    finally:
        if rng_state is not None:
            _restore_rng_state(rng_state)
    eos_or_pad = model.config.eos_token_id or int(pad_token_id)
    generated = _strip_extra_image_tokens(
        generated,
        image_token_id=image_token_id,
        allowed_count=allowed_img_count,
        replacement_token_id=eos_or_pad,
    )
    generated = generated.squeeze(0).detach().cpu().long()
    generated_mask = (generated != int(pad_token_id)).long()
    return generated, generated_mask


def _generate_candidate_batch(
    model,
    samples: List[Stage3SampleCacheItem],
    args,
    image_token_id: Optional[int],
    pad_token_id: int,
    do_sample: bool = True,
    temperature: Optional[float] = None,
    generator: Optional[torch.Generator] = None,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    if not samples:
        return []

    device = args.device
    prompt_ids = _left_pad_and_stack_1d_tensors(
        [sample.prompt_ids for sample in samples],
        pad_value=float(pad_token_id),
        dtype=torch.long,
    ).to(device)
    prompt_mask = _left_pad_and_stack_1d_tensors(
        [sample.prompt_mask for sample in samples],
        pad_value=0.0,
        dtype=torch.long,
    ).to(device)
    pixel_values = _stack_padded_pixel_values([sample.pixel_values for sample in samples]).to(device)
    image_sizes = sanitize_image_sizes(
        _stack_optional_image_sizes([sample.image_sizes for sample in samples]),
        batch_size=len(samples),
    )
    image_grid_thw = _stack_optional_image_grid_thw([sample.image_grid_thw for sample in samples])
    allowed_img_counts = [
        int((sample.prompt_ids == int(image_token_id)).sum().item()) if image_token_id is not None else 0
        for sample in samples
    ]

    gen_kwargs = dict(
        input_ids=prompt_ids,
        attention_mask=prompt_mask,
        pixel_values=pixel_values,
        image_sizes=image_sizes,
        image_grid_thw=image_grid_thw,
        do_sample=bool(do_sample),
        max_new_tokens=int(args.max_new_tokens),
        bad_words_ids=[[int(image_token_id)]] if image_token_id is not None else None,
        pad_token_id=int(pad_token_id),
    )
    if do_sample:
        gen_kwargs["temperature"] = float(args.temperature if temperature is None else temperature)
    rng_state = None
    if do_sample and generator is not None:
        rng_state = _capture_rng_state()
        seed = int(generator.initial_seed())
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    try:
        with torch.no_grad():
            generated = student_generate(model, args, **gen_kwargs)
    finally:
        if rng_state is not None:
            _restore_rng_state(rng_state)

    eos_or_pad = model.config.eos_token_id or int(pad_token_id)
    outputs: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for row_idx in range(generated.shape[0]):
        row = _strip_extra_image_tokens(
            generated[row_idx:row_idx + 1],
            image_token_id=image_token_id,
            allowed_count=allowed_img_counts[row_idx],
            replacement_token_id=eos_or_pad,
        ).squeeze(0).detach().cpu().long()
        non_pad = (row != int(pad_token_id)).nonzero(as_tuple=False).view(-1)
        keep_len = int(non_pad[-1].item()) + 1 if non_pad.numel() > 0 else 1
        row = row[:keep_len]
        row_mask = (row != int(pad_token_id)).long()
        outputs.append((row, row_mask))
    return outputs


def _extract_answer_letter(text: str) -> Optional[str]:
    if not text:
        return None
    text_u = str(text).strip()
    patterns = [
        r"(?:answer|答案)\s*[:：]\s*\(?\s*([A-Z])\s*\)?",
        r"^\s*\(?\s*([A-Z])\s*\)?\b",
    ]
    for pat in patterns:
        m = re.search(pat, text_u, flags=re.IGNORECASE)
        if m:
            return str(m.group(1)).upper()
    return None


def _decode_generated_text_from_ids(
    ids: torch.Tensor,
    prompt_len: int,
    tokenizer,
) -> str:
    if tokenizer is None or ids is None:
        return ""
    gen_ids = ids[prompt_len:] if ids.shape[0] > prompt_len else ids
    return tokenizer.decode(gen_ids.tolist(), skip_special_tokens=True)


def _extract_answer_letter_from_ids(
    ids: torch.Tensor,
    prompt_len: int,
    tokenizer,
) -> Optional[str]:
    return _extract_answer_letter(_decode_generated_text_from_ids(ids, prompt_len, tokenizer))


def _get_letter_token_candidates(tokenizer, letter: str) -> List[int]:
    variants = [
        letter,
        f" {letter}",
        f"({letter})",
        f" ({letter})",
        f"{letter}.",
        f" {letter}.",
        f"答案:{letter}",
        f"Answer: {letter}",
    ]
    candidate_ids = set()
    for text in variants:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) == 1:
            candidate_ids.add(int(ids[0]))
    if not candidate_ids:
        ids = tokenizer.encode(letter, add_special_tokens=False)
        if ids:
            candidate_ids.add(int(ids[0]))
    return sorted(candidate_ids)


def _build_stage3_choice_token_map(tokenizer, max_choices: int = 8) -> Dict[int, List[int]]:
    choice_token_map: Dict[int, List[int]] = {}
    for idx in range(max(0, int(max_choices))):
        choice_token_map[idx] = _get_letter_token_candidates(tokenizer, chr(65 + idx))
    return choice_token_map


def _compute_choice_scores_from_prompt_batch(
    model,
    args,
    prompt_ids: torch.Tensor,
    prompt_mask: torch.Tensor,
    pixel_values: torch.Tensor,
    image_sizes: Optional[torch.Tensor],
    num_choices: torch.Tensor,
    choice_token_map: Dict[int, List[int]],
    image_grid_thw: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    outputs = student_forward(
        model=model,
        args=args,
        input_ids=prompt_ids,
        attention_mask=prompt_mask,
        pixel_values=pixel_values,
        image_sizes=image_sizes,
        image_grid_thw=image_grid_thw,
        use_cache=False,
    )
    last_positions = prompt_mask.long().sum(dim=-1).clamp(min=1) - 1
    row_indices = torch.arange(prompt_ids.shape[0], device=prompt_ids.device)
    next_token_logits = outputs.logits[row_indices, last_positions, :]

    max_num_choices = max(1, int(num_choices.max().item())) if num_choices.numel() > 0 else 1
    choice_scores = torch.full(
        (prompt_ids.shape[0], max_num_choices),
        fill_value=-1e9,
        device=prompt_ids.device,
        dtype=next_token_logits.dtype,
    )
    valid_rows = torch.zeros((prompt_ids.shape[0],), device=prompt_ids.device, dtype=torch.bool)
    for row_idx in range(prompt_ids.shape[0]):
        cur_num_choices = min(max_num_choices, int(num_choices[row_idx].item()))
        if cur_num_choices <= 0:
            continue
        row_valid = False
        for choice_idx in range(cur_num_choices):
            cand_ids = choice_token_map.get(choice_idx, [])
            if not cand_ids:
                continue
            choice_scores[row_idx, choice_idx] = next_token_logits[row_idx, cand_ids].max()
            row_valid = True
        valid_rows[row_idx] = row_valid
    return choice_scores, valid_rows


def _predict_choice_from_choice_scores(
    choice_scores: torch.Tensor,
    num_choices: torch.Tensor,
    valid_rows: torch.Tensor,
) -> List[Optional[int]]:
    predictions: List[Optional[int]] = []
    for row_idx in range(choice_scores.shape[0]):
        cur_num_choices = int(num_choices[row_idx].item())
        if cur_num_choices <= 0 or not bool(valid_rows[row_idx].item()):
            predictions.append(None)
            continue
        predictions.append(int(choice_scores[row_idx, :cur_num_choices].argmax(dim=-1).item()))
    return predictions


def _evaluate_stage3_answer_metrics(
    model,
    sample_cache: List[Stage3SampleCacheItem],
    eval_indices: List[int],
    args,
    image_token_id: Optional[int],
    pad_token_id: int,
    tokenizer,
    answer_lookup: Dict[int, str],
    choice_token_map: Optional[Dict[int, List[int]]] = None,
) -> Tuple[float, float, int]:
    if not eval_indices or tokenizer is None:
        return 0.0, 0.0, 0

    model = unwrap_model(model)
    eval_mode = str(args.stage3_eval_answer_mode).strip().lower()
    model.eval()
    use_cache_states = _set_model_use_cache(model, True)
    gc_states = _set_model_gradient_checkpointing(model, False)
    try:
        total = 0
        fmt_hits = 0
        acc_hits = 0
        if eval_mode == "logits" and choice_token_map is not None:
            batch_size = max(1, int(args.batch_size))
            sorted_indices = sorted(eval_indices, key=lambda idx: _stage3_static_sort_key(sample_cache[idx]))
            for start in range(0, len(sorted_indices), batch_size):
                batch_indices = sorted_indices[start:start + batch_size]
                batch_samples = [sample_cache[idx] for idx in batch_indices]
                prompt_ids = torch.stack([
                    _pad_1d_tensor(sample.prompt_ids.long(), max(int(x.prompt_ids.shape[0]) for x in batch_samples), pad_token_id)
                    for sample in batch_samples
                ], dim=0).to(args.device)
                prompt_mask = torch.stack([
                    _pad_1d_tensor(sample.prompt_mask.long(), prompt_ids.shape[1], 0)
                    for sample in batch_samples
                ], dim=0).to(args.device)
                pixel_values = _stack_padded_pixel_values([sample.pixel_values for sample in batch_samples]).to(args.device)
                image_sizes = sanitize_image_sizes(
                    _stack_optional_image_sizes([sample.image_sizes for sample in batch_samples]),
                    batch_size=len(batch_samples),
                )
                image_grid_thw = _stack_optional_image_grid_thw([sample.image_grid_thw for sample in batch_samples])
                num_choices = torch.tensor(
                    [int(sample.num_choices) for sample in batch_samples],
                    device=args.device,
                    dtype=torch.long,
                )
                choice_scores, valid_rows = _compute_choice_scores_from_prompt_batch(
                    model=model,
                    args=args,
                    prompt_ids=prompt_ids,
                    prompt_mask=prompt_mask,
                    pixel_values=pixel_values,
                    image_sizes=image_sizes,
                    num_choices=num_choices,
                    choice_token_map=choice_token_map,
                    image_grid_thw=image_grid_thw,
                )
                pred_indices = _predict_choice_from_choice_scores(choice_scores, num_choices, valid_rows)
                for local_idx, sample_idx in enumerate(batch_indices):
                    pred_idx = pred_indices[local_idx]
                    gold = answer_lookup.get(int(sample_idx))
                    total += 1
                    if pred_idx is not None:
                        fmt_hits += 1
                    if pred_idx is not None and gold is not None and pred_idx == (ord(gold) - 65):
                        acc_hits += 1
        else:
            for idx in eval_indices:
                sample = sample_cache[idx]
                ids, _ = _generate_candidate_single(
                    model=model,
                    sample=sample,
                    args=args,
                    image_token_id=image_token_id,
                    pad_token_id=pad_token_id,
                    do_sample=False,
                )
                prompt_len = int(sample.prompt_len)
                gen_ids = ids[prompt_len:] if ids.shape[0] > prompt_len else ids
                text = tokenizer.decode(gen_ids.tolist(), skip_special_tokens=True)
                pred = _extract_answer_letter(text)
                gold = answer_lookup.get(int(idx))
                total += 1
                if pred is not None:
                    fmt_hits += 1
                if pred is not None and gold is not None and pred == gold:
                    acc_hits += 1
    finally:
        _restore_model_gradient_checkpointing(gc_states)
        _restore_model_use_cache(use_cache_states)

    denom = max(1, total)
    return float(acc_hits) / denom, float(fmt_hits) / denom, int(total)


def _score_sequences_no_grad_batch(
    model,
    samples: List[Stage3SampleCacheItem],
    ids_list: List[torch.Tensor],
    mask_list: List[torch.Tensor],
    args,
    pad_token_id: int,
) -> List[Tuple[float, float, torch.Tensor, torch.Tensor]]:
    if not samples:
        return []
    if not (len(samples) == len(ids_list) == len(mask_list)):
        raise ValueError("samples, ids_list, mask_list 长度必须一致")

    with torch.no_grad():
        ids_b = _pad_and_stack_1d_tensors(ids_list, pad_value=float(pad_token_id), dtype=torch.long).to(args.device)
        mask_b = _pad_and_stack_1d_tensors(mask_list, pad_value=0.0, dtype=torch.long).to(args.device)
        pixel_values = _stack_padded_pixel_values([sample.pixel_values for sample in samples]).to(args.device)
        image_sizes = sanitize_image_sizes(
            _stack_optional_image_sizes([sample.image_sizes for sample in samples]),
            batch_size=len(samples),
        )
        image_grid_thw = _stack_optional_image_grid_thw([sample.image_grid_thw for sample in samples])
        prompt_lens = torch.tensor(
            [sample.prompt_len for sample in samples],
            device=ids_b.device,
            dtype=torch.long,
        )

        token_lp, token_mask = _compute_token_log_probs(
            model=model,
            args=args,
            ids=ids_b,
            mask=mask_b,
            pixel_values=pixel_values,
            image_sizes=image_sizes,
            prompt_lens=prompt_lens,
            image_grid_thw=image_grid_thw,
        )
        avg_lp_tensor = (token_lp * token_mask).sum(dim=-1) / token_mask.sum(dim=-1).clamp(min=1.0)

    outputs: List[Tuple[float, float, torch.Tensor, torch.Tensor]] = []
    for row_idx, ids in enumerate(ids_list):
        old_len = max(0, int(ids.shape[0]) - 1)
        avg_lp = float(avg_lp_tensor[row_idx].item())
        prob = float(math.exp(max(min(avg_lp, 50.0), -50.0)))
        outputs.append(
            (
                avg_lp,
                prob,
                token_lp[row_idx, :old_len].detach().cpu(),
                token_mask[row_idx, :old_len].detach().cpu(),
            )
        )
    return outputs


def _get_stage3_phase_a_batch_size(args, train_bucket_meta: Optional[dict] = None) -> int:
    batch_size = max(1, int(args.batch_size))
    if train_bucket_meta is not None:
        config = train_bucket_meta.get("config", {})
        batch_size = max(1, int(config.get("bucket_batch_size", batch_size)))
    if int(args.bucket_batch_size) > 0:
        batch_size = int(args.bucket_batch_size)
    if int(args.stage3_bucket_batch_size) > 0:
        batch_size = int(args.stage3_bucket_batch_size)
    return max(1, batch_size)


def _stage3_static_sort_key(sample: Stage3SampleCacheItem) -> Tuple[int, int, int, int]:
    patch_count = int(sample.pixel_values.shape[0]) if sample.pixel_values.dim() >= 1 else 0
    prompt_len = int(sample.prompt_len)
    vic_len = int(sample.y_vic_ids.shape[0])
    return (patch_count, prompt_len, vic_len, int(sample.sample_idx))


def _build_stage3_static_batches(
    active_indices: List[int],
    sample_cache: List[Stage3SampleCacheItem],
    args,
    train_bucket_meta: Optional[dict] = None,
) -> List[List[int]]:
    if not active_indices:
        return []

    batch_size = _get_stage3_phase_a_batch_size(args, train_bucket_meta)
    if batch_size <= 1:
        return [[int(idx)] for idx in active_indices]

    use_bucket = _is_hf_multichoice_dataset(args.dataset_name)

    batches: List[List[int]] = []
    if use_bucket:
        if train_bucket_meta is None:
            raise RuntimeError("Stage3 需要有效的 ScienceQA 分桶预处理结果")
        sample_to_bucket = train_bucket_meta.get("sample_to_bucket", {})
        bucket_to_indices = defaultdict(list)
        for idx in active_indices:
            if int(idx) not in sample_to_bucket:
                raise RuntimeError(f"Stage3 缺少 sample_id={int(idx)} 的 bucket 映射")
            bucket_key = sample_to_bucket[int(idx)]
            bucket_to_indices[str(bucket_key)].append(int(idx))

        for bucket_key in sorted(bucket_to_indices.keys()):
            bucket_indices = sorted(
                bucket_to_indices[bucket_key],
                key=lambda i: _stage3_static_sort_key(sample_cache[i]),
            )
            for start in range(0, len(bucket_indices), batch_size):
                batches.append(bucket_indices[start:start + batch_size])
        return batches

    sorted_indices = sorted(active_indices, key=lambda i: _stage3_static_sort_key(sample_cache[i]))
    for start in range(0, len(sorted_indices), batch_size):
        batches.append(sorted_indices[start:start + batch_size])
    return batches


def _bootstrap_period_states_distributed(
    model,
    sample_cache: List[Stage3SampleCacheItem],
    active_indices: List[int],
    args,
    image_token_id: Optional[int],
    pad_token_id: int,
    train_bucket_meta: Optional[dict] = None,
    sub_stage_idx: int = 0,
) -> Dict[int, PeriodState]:
    if not is_distributed():
        return _bootstrap_period_states(
            model=model,
            sample_cache=sample_cache,
            active_indices=active_indices,
            args=args,
            image_token_id=image_token_id,
            pad_token_id=pad_token_id,
            train_bucket_meta=train_bucket_meta,
        )

    base_model = unwrap_model(model)
    global_batches = _build_stage3_static_batches(
        active_indices=active_indices,
        sample_cache=sample_cache,
        args=args,
        train_bucket_meta=train_bucket_meta,
    )
    local_batch_plan = _shard_stage3_global_batches(global_batches)
    local_chunks = []

    base_model.eval()
    use_cache_states = _set_model_use_cache(base_model, True)
    gc_states = _set_model_gradient_checkpointing(base_model, False)
    try:
        for global_batch_idx, batch_indices in tqdm(
            local_batch_plan,
            desc=f"Stage3 bootstrap rank{get_rank()}",
            disable=not is_main_process(),
        ):
            batch_samples = [sample_cache[idx] for idx in batch_indices]
            cand1_outputs = _generate_candidate_batch(
                base_model,
                batch_samples,
                args,
                image_token_id,
                pad_token_id,
                generator=_make_stage3_torch_generator(
                    args.device,
                    _make_stage3_phase_seed(
                        int(args.scienceqa_seed),
                        sub_stage_idx,
                        -1,
                        "bootstrap_cand1",
                        global_batch_idx,
                    ),
                ),
            )
            cand2_outputs = _generate_candidate_batch(
                base_model,
                batch_samples,
                args,
                image_token_id,
                pad_token_id,
                generator=_make_stage3_torch_generator(
                    args.device,
                    _make_stage3_phase_seed(
                        int(args.scienceqa_seed),
                        sub_stage_idx,
                        -1,
                        "bootstrap_cand2",
                        global_batch_idx,
                    ),
                ),
            )
            cand1_scores = _score_sequences_no_grad_batch(
                model=base_model,
                samples=batch_samples,
                ids_list=[ids for ids, _ in cand1_outputs],
                mask_list=[mask for _, mask in cand1_outputs],
                args=args,
                pad_token_id=pad_token_id,
            )
            cand2_scores = _score_sequences_no_grad_batch(
                model=base_model,
                samples=batch_samples,
                ids_list=[ids for ids, _ in cand2_outputs],
                mask_list=[mask for _, mask in cand2_outputs],
                args=args,
                pad_token_id=pad_token_id,
            )
            y_vic_scores = _score_sequences_no_grad_batch(
                model=base_model,
                samples=batch_samples,
                ids_list=[sample.y_vic_ids for sample in batch_samples],
                mask_list=[sample.y_vic_mask for sample in batch_samples],
                args=args,
                pad_token_id=pad_token_id,
            )

            batch_states: Dict[int, PeriodState] = {}
            for row_idx, idx in enumerate(batch_indices):
                sample = batch_samples[row_idx]
                cand1_ids, cand1_mask = cand1_outputs[row_idx]
                cand2_ids, cand2_mask = cand2_outputs[row_idx]
                cand1_lp, cand1_prob, _, _ = cand1_scores[row_idx]
                cand2_lp, cand2_prob, _, _ = cand2_scores[row_idx]
                y_vic_lp, y_vic_prob, _, _ = y_vic_scores[row_idx]

                low_ids, low_mask, low_lp, low_prob = (
                    (cand1_ids, cand1_mask, cand1_lp, cand1_prob)
                    if cand1_prob <= cand2_prob
                    else (cand2_ids, cand2_mask, cand2_lp, cand2_prob)
                )
                batch_states[int(idx)] = PeriodState(
                    sample_idx=int(idx),
                    y11_ids=sample.y_vic_ids.clone(),
                    y11_mask=sample.y_vic_mask.clone(),
                    y12_ids=low_ids.clone(),
                    y12_mask=low_mask.clone(),
                    avg_lp_11=float(y_vic_lp),
                    avg_lp_12=float(low_lp),
                    prob_11=float(y_vic_prob),
                    prob_12=float(low_prob),
                )
            local_chunks.append((int(global_batch_idx), batch_states))
    finally:
        _restore_model_gradient_checkpointing(gc_states)
        _restore_model_use_cache(use_cache_states)

    gathered = _stage3_gather_object_to_main(local_chunks)
    merged_states = None
    if is_main_process():
        merged_states = {}
        ordered = []
        for rank_chunks in gathered:
            if not rank_chunks:
                continue
            ordered.extend(rank_chunks)
        ordered.sort(key=lambda item: int(item[0]))
        merged_sample_indices = []
        for _, chunk_states in ordered:
            for sample_idx, state in chunk_states.items():
                merged_states[int(sample_idx)] = state
                merged_sample_indices.append(int(sample_idx))
        _validate_stage3_chunk_merge(merged_sample_indices, active_indices, "bootstrap")
    merged_states = _stage3_broadcast_object_from_main(merged_states)
    return merged_states


def _build_pairs_and_next_states_distributed(
    model,
    sample_cache: List[Stage3SampleCacheItem],
    current_states: Dict[int, PeriodState],
    active_indices: List[int],
    args,
    image_token_id: Optional[int],
    pad_token_id: int,
    force_vic_positive: bool = False,
    train_bucket_meta: Optional[dict] = None,
    tokenizer=None,
    sub_stage_idx: int = 0,
    period_idx: int = 0,
) -> Tuple[List[PeriodTrainingItem], Dict[int, PeriodState], int, int, float, float]:
    if not is_distributed():
        return _build_pairs_and_next_states(
            model=model,
            sample_cache=sample_cache,
            current_states=current_states,
            active_indices=active_indices,
            args=args,
            image_token_id=image_token_id,
            pad_token_id=pad_token_id,
            force_vic_positive=force_vic_positive,
            train_bucket_meta=train_bucket_meta,
            tokenizer=tokenizer,
        )

    base_model = unwrap_model(model)
    wrong_sample_lookup, wrong_lookup_stats = _build_wrong_image_lookup(
        active_indices=active_indices,
        sample_cache=sample_cache,
        image_token_id=image_token_id,
    )
    if is_main_process():
        print(
            "[Stage3] wrong-image pairing stats: "
            f"primary={wrong_lookup_stats['primary']}, "
            f"fallback_patch={wrong_lookup_stats['fallback_patch']}, "
            f"fallback_token={wrong_lookup_stats['fallback_token']}, "
            f"self={wrong_lookup_stats['self']}"
        )
    global_batches = _build_stage3_static_batches(
        active_indices=active_indices,
        sample_cache=sample_cache,
        args=args,
        train_bucket_meta=train_bucket_meta,
    )
    local_batch_plan = _shard_stage3_global_batches(global_batches)
    local_training_chunks = []
    local_state_chunks = []
    local_stats = {
        "cold_start_count": 0,
        "swap_count": 0,
        "plus_lp_sum": 0.0,
        "minus_lp_sum": 0.0,
    }
    tau1 = float(args.tau1)
    tau_delta = float(args.tau_delta)

    base_model.eval()
    use_cache_states = _set_model_use_cache(base_model, True)
    gc_states = _set_model_gradient_checkpointing(base_model, False)
    try:
        for global_batch_idx, batch_indices in tqdm(
            local_batch_plan,
            desc=f"Stage3 Phase-A rank{get_rank()}",
            disable=not is_main_process(),
        ):
            batch_samples = [sample_cache[idx] for idx in batch_indices]
            batch_states = [current_states[idx] for idx in batch_indices]

            score11 = _score_sequences_no_grad_batch(
                model=base_model,
                samples=batch_samples,
                ids_list=[state.y11_ids for state in batch_states],
                mask_list=[state.y11_mask for state in batch_states],
                args=args,
                pad_token_id=pad_token_id,
            )
            score12 = _score_sequences_no_grad_batch(
                model=base_model,
                samples=batch_samples,
                ids_list=[state.y12_ids for state in batch_states],
                mask_list=[state.y12_mask for state in batch_states],
                args=args,
                pad_token_id=pad_token_id,
            )
            vic_scores = _score_sequences_no_grad_batch(
                model=base_model,
                samples=batch_samples,
                ids_list=[sample.y_vic_ids for sample in batch_samples],
                mask_list=[sample.y_vic_mask for sample in batch_samples],
                args=args,
                pad_token_id=pad_token_id,
            )
            next1_outputs = _generate_candidate_batch(
                base_model,
                batch_samples,
                args,
                image_token_id,
                pad_token_id,
                generator=_make_stage3_torch_generator(
                    args.device,
                    _make_stage3_phase_seed(
                        int(args.scienceqa_seed),
                        sub_stage_idx,
                        period_idx,
                        "phasea_next1",
                        global_batch_idx,
                    ),
                ),
            )
            next2_outputs = _generate_candidate_batch(
                base_model,
                batch_samples,
                args,
                image_token_id,
                pad_token_id,
                generator=_make_stage3_torch_generator(
                    args.device,
                    _make_stage3_phase_seed(
                        int(args.scienceqa_seed),
                        sub_stage_idx,
                        period_idx,
                        "phasea_next2",
                        global_batch_idx,
                    ),
                ),
            )
            next1_scores = _score_sequences_no_grad_batch(
                model=base_model,
                samples=batch_samples,
                ids_list=[ids for ids, _ in next1_outputs],
                mask_list=[mask for _, mask in next1_outputs],
                args=args,
                pad_token_id=pad_token_id,
            )
            next2_scores = _score_sequences_no_grad_batch(
                model=base_model,
                samples=batch_samples,
                ids_list=[ids for ids, _ in next2_outputs],
                mask_list=[mask for _, mask in next2_outputs],
                args=args,
                pad_token_id=pad_token_id,
            )

            batch_training_items: List[PeriodTrainingItem] = []
            batch_next_states: Dict[int, PeriodState] = {}
            for row_idx, idx in enumerate(batch_indices):
                state = batch_states[row_idx]
                sample = batch_samples[row_idx]
                lp11, prob11, token11, mask11 = score11[row_idx]
                lp12, prob12, token12, mask12 = score12[row_idx]
                y_vic_lp, y_vic_prob, y_vic_token_lp, y_vic_token_mask = vic_scores[row_idx]

                candidates = [
                    {
                        "ids": state.y11_ids,
                        "mask": state.y11_mask,
                        "avg_lp": lp11,
                        "prob": prob11,
                        "old_token_lp": token11,
                        "old_token_mask": mask11,
                        "prev_prob": state.prob_11,
                    },
                    {
                        "ids": state.y12_ids,
                        "mask": state.y12_mask,
                        "avg_lp": lp12,
                        "prob": prob12,
                        "old_token_lp": token12,
                        "old_token_mask": mask12,
                        "prev_prob": state.prob_12,
                    },
                ]
                candidates.sort(key=lambda x: x["prob"], reverse=True)
                y11 = candidates[0]
                y12 = candidates[1]
                if prob12 > prob11:
                    local_stats["swap_count"] += 1

                if bool(int(args.stage3_pair_use_answer_correctness)) and tokenizer is not None:
                    gold_letter = str(getattr(sample, "answer_letter", "") or "").strip().upper()[:1]
                    if len(gold_letter) == 1 and "A" <= gold_letter <= "Z":
                        cand_scored = []
                        for cand in candidates:
                            pred_letter = _extract_answer_letter_from_ids(
                                cand["ids"],
                                int(sample.prompt_len),
                                tokenizer,
                            )
                            cand_scored.append({
                                **cand,
                                "pred_letter": pred_letter,
                                "is_correct": 1 if pred_letter == gold_letter else 0,
                            })
                        cand_scored.sort(
                            key=lambda x: (x["is_correct"], float(x["prob"])),
                            reverse=True,
                        )
                        y11 = cand_scored[0]
                        y12 = cand_scored[1]

                delta11 = float(y11["prob"] - y11["prev_prob"])
                p_best = max(float(y11["prob"]), float(y12["prob"]))
                use_cold_start = (p_best < tau1) and (delta11 < tau_delta)
                if force_vic_positive:
                    y_plus_ids = sample.y_vic_ids
                    y_plus_mask = sample.y_vic_mask
                    old_lp_plus = y_vic_token_lp
                    old_mask_plus = y_vic_token_mask
                    y_plus_avg_lp = float(y_vic_lp)
                    y_minus_ids = state.y12_ids
                    y_minus_mask = state.y12_mask
                    old_lp_minus = token12
                    old_mask_minus = mask12
                    y_minus_avg_lp = float(lp12)
                elif use_cold_start:
                    y_plus_ids = sample.y_vic_ids
                    y_plus_mask = sample.y_vic_mask
                    old_lp_plus = y_vic_token_lp
                    old_mask_plus = y_vic_token_mask
                    y_plus_avg_lp = float(y_vic_lp)
                    y_minus_ids = y12["ids"]
                    y_minus_mask = y12["mask"]
                    old_lp_minus = y12["old_token_lp"]
                    old_mask_minus = y12["old_token_mask"]
                    y_minus_avg_lp = float(y12["avg_lp"])
                    local_stats["cold_start_count"] += 1
                else:
                    y_plus_ids = y11["ids"]
                    y_plus_mask = y11["mask"]
                    old_lp_plus = y11["old_token_lp"]
                    old_mask_plus = y11["old_token_mask"]
                    y_plus_avg_lp = float(y11["avg_lp"])
                    y_minus_ids = y12["ids"]
                    y_minus_mask = y12["mask"]
                    old_lp_minus = y12["old_token_lp"]
                    old_mask_minus = y12["old_token_mask"]
                    y_minus_avg_lp = float(y12["avg_lp"])
                local_stats["plus_lp_sum"] += y_plus_avg_lp
                local_stats["minus_lp_sum"] += y_minus_avg_lp

                batch_training_items.append(PeriodTrainingItem(
                    sample_idx=int(idx),
                    y_plus_ids=y_plus_ids.clone(),
                    y_plus_mask=y_plus_mask.clone(),
                    y_minus_ids=y_minus_ids.clone(),
                    y_minus_mask=y_minus_mask.clone(),
                    y_vic_ids=sample.y_vic_ids.clone(),
                    y_vic_mask=sample.y_vic_mask.clone(),
                    old_token_lp_plus=old_lp_plus.clone(),
                    old_token_mask_plus=old_mask_plus.clone(),
                    old_token_lp_minus=old_lp_minus.clone(),
                    old_token_mask_minus=old_mask_minus.clone(),
                    old_token_lp_vic=y_vic_token_lp.clone(),
                    old_token_mask_vic=y_vic_token_mask.clone(),
                    wrong_sample_idx=wrong_sample_lookup[int(idx)],
                ))

                next1_ids, next1_mask = next1_outputs[row_idx]
                next2_ids, next2_mask = next2_outputs[row_idx]
                next1_lp, next1_prob, _, _ = next1_scores[row_idx]
                next2_lp, next2_prob, _, _ = next2_scores[row_idx]
                next_sorted = [
                    (next1_ids, next1_mask, next1_lp, next1_prob),
                    (next2_ids, next2_mask, next2_lp, next2_prob),
                ]
                next_sorted.sort(key=lambda x: x[3], reverse=True)
                high = next_sorted[0]
                low = next_sorted[1]
                batch_next_states[int(idx)] = PeriodState(
                    sample_idx=int(idx),
                    y11_ids=high[0].clone(),
                    y11_mask=high[1].clone(),
                    y12_ids=low[0].clone(),
                    y12_mask=low[1].clone(),
                    avg_lp_11=float(high[2]),
                    avg_lp_12=float(low[2]),
                    prob_11=float(high[3]),
                    prob_12=float(low[3]),
                )

            local_training_chunks.append((int(global_batch_idx), batch_training_items))
            local_state_chunks.append((int(global_batch_idx), batch_next_states))
    finally:
        _restore_model_gradient_checkpointing(gc_states)
        _restore_model_use_cache(use_cache_states)

    gathered_training = _stage3_gather_object_to_main(local_training_chunks)
    gathered_states = _stage3_gather_object_to_main(local_state_chunks)
    gathered_stats = _stage3_gather_object_to_main(local_stats)

    merged_training_items = None
    merged_next_states = None
    merged_stats = None
    if is_main_process():
        merged_training_items = []
        merged_next_states = {}
        merged_sample_indices = []

        ordered_training = []
        for rank_chunks in gathered_training:
            if not rank_chunks:
                continue
            ordered_training.extend(rank_chunks)
        ordered_training.sort(key=lambda item: int(item[0]))
        for _, chunk_items in ordered_training:
            merged_training_items.extend(chunk_items)
            merged_sample_indices.extend(int(item.sample_idx) for item in chunk_items)

        ordered_states = []
        for rank_chunks in gathered_states:
            if not rank_chunks:
                continue
            ordered_states.extend(rank_chunks)
        ordered_states.sort(key=lambda item: int(item[0]))
        merged_state_indices = []
        for _, chunk_states in ordered_states:
            for sample_idx, state in chunk_states.items():
                merged_next_states[int(sample_idx)] = state
                merged_state_indices.append(int(sample_idx))

        _validate_stage3_chunk_merge(merged_sample_indices, active_indices, "training_items")
        _validate_stage3_chunk_merge(merged_state_indices, active_indices, "next_states")
        if len(merged_training_items) != len(active_indices):
            raise RuntimeError(
                f"[Stage3][DDP] training_items 长度异常: "
                f"got={len(merged_training_items)}, expected={len(active_indices)}"
            )

        merged_stats = {
            "cold_start_count": 0,
            "swap_count": 0,
            "plus_lp_sum": 0.0,
            "minus_lp_sum": 0.0,
        }
        for stat in gathered_stats:
            if not stat:
                continue
            merged_stats["cold_start_count"] += int(stat.get("cold_start_count", 0))
            merged_stats["swap_count"] += int(stat.get("swap_count", 0))
            merged_stats["plus_lp_sum"] += float(stat.get("plus_lp_sum", 0.0))
            merged_stats["minus_lp_sum"] += float(stat.get("minus_lp_sum", 0.0))

    merged_training_items = _stage3_broadcast_object_from_main(merged_training_items)
    merged_next_states = _stage3_broadcast_object_from_main(merged_next_states)
    merged_stats = _stage3_broadcast_object_from_main(merged_stats)
    return (
        merged_training_items,
        merged_next_states,
        int(merged_stats["cold_start_count"]),
        int(merged_stats["swap_count"]),
        float(merged_stats["plus_lp_sum"]),
        float(merged_stats["minus_lp_sum"]),
    )


def _bootstrap_period_states(
    model,
    sample_cache: List[Stage3SampleCacheItem],
    active_indices: List[int],
    args,
    image_token_id: Optional[int],
    pad_token_id: int,
    train_bucket_meta: Optional[dict] = None,
) -> Dict[int, PeriodState]:
    model = unwrap_model(model)
    states: Dict[int, PeriodState] = {}
    model.eval()
    use_cache_states = _set_model_use_cache(model, True)
    gc_states = _set_model_gradient_checkpointing(model, False)
    try:
        stage3_batches = _build_stage3_static_batches(
            active_indices=active_indices,
            sample_cache=sample_cache,
            args=args,
            train_bucket_meta=train_bucket_meta,
        )
        for batch_indices in tqdm(stage3_batches, desc="Stage3 bootstrap"):
            batch_samples = [sample_cache[idx] for idx in batch_indices]
            cand1_outputs = _generate_candidate_batch(
                model, batch_samples, args, image_token_id, pad_token_id
            )
            cand2_outputs = _generate_candidate_batch(
                model, batch_samples, args, image_token_id, pad_token_id
            )
            cand1_scores = _score_sequences_no_grad_batch(
                model=model,
                samples=batch_samples,
                ids_list=[ids for ids, _ in cand1_outputs],
                mask_list=[mask for _, mask in cand1_outputs],
                args=args,
                pad_token_id=pad_token_id,
            )
            cand2_scores = _score_sequences_no_grad_batch(
                model=model,
                samples=batch_samples,
                ids_list=[ids for ids, _ in cand2_outputs],
                mask_list=[mask for _, mask in cand2_outputs],
                args=args,
                pad_token_id=pad_token_id,
            )
            y_vic_scores = _score_sequences_no_grad_batch(
                model=model,
                samples=batch_samples,
                ids_list=[sample.y_vic_ids for sample in batch_samples],
                mask_list=[sample.y_vic_mask for sample in batch_samples],
                args=args,
                pad_token_id=pad_token_id,
            )

            for row_idx, idx in enumerate(batch_indices):
                sample = batch_samples[row_idx]
                cand1_ids, cand1_mask = cand1_outputs[row_idx]
                cand2_ids, cand2_mask = cand2_outputs[row_idx]
                cand1_lp, cand1_prob, _, _ = cand1_scores[row_idx]
                cand2_lp, cand2_prob, _, _ = cand2_scores[row_idx]
                y_vic_lp, y_vic_prob, _, _ = y_vic_scores[row_idx]

                # period0: y11 固定为 y_vic，y12 取低概率候选
                low_ids, low_mask, low_lp, low_prob = (
                    (cand1_ids, cand1_mask, cand1_lp, cand1_prob)
                    if cand1_prob <= cand2_prob
                    else (cand2_ids, cand2_mask, cand2_lp, cand2_prob)
                )
                states[idx] = PeriodState(
                    sample_idx=idx,
                    y11_ids=sample.y_vic_ids.clone(),
                    y11_mask=sample.y_vic_mask.clone(),
                    y12_ids=low_ids.clone(),
                    y12_mask=low_mask.clone(),
                    avg_lp_11=float(y_vic_lp),
                    avg_lp_12=float(low_lp),
                    prob_11=float(y_vic_prob),
                    prob_12=float(low_prob),
                )
    finally:
        _restore_model_gradient_checkpointing(gc_states)
        _restore_model_use_cache(use_cache_states)
    return states


def _build_pairs_and_next_states(
    model,
    sample_cache: List[Stage3SampleCacheItem],
    current_states: Dict[int, PeriodState],
    active_indices: List[int],
    args,
    image_token_id: Optional[int],
    pad_token_id: int,
    force_vic_positive: bool = False,
    train_bucket_meta: Optional[dict] = None,
    tokenizer=None,
) -> Tuple[List[PeriodTrainingItem], Dict[int, PeriodState], int, int, float, float]:
    model = unwrap_model(model)
    training_items: List[PeriodTrainingItem] = []
    next_states: Dict[int, PeriodState] = {}
    cold_start_count = 0
    swap_count = 0
    plus_lp_sum = 0.0
    minus_lp_sum = 0.0
    tau1 = float(args.tau1)
    tau_delta = float(args.tau_delta)

    model.eval()
    use_cache_states = _set_model_use_cache(model, True)
    gc_states = _set_model_gradient_checkpointing(model, False)
    try:
        wrong_sample_lookup, wrong_lookup_stats = _build_wrong_image_lookup(
            active_indices=active_indices,
            sample_cache=sample_cache,
            image_token_id=image_token_id,
        )
        if is_main_process():
            print(
                "[Stage3] wrong-image pairing stats: "
                f"primary={wrong_lookup_stats['primary']}, "
                f"fallback_patch={wrong_lookup_stats['fallback_patch']}, "
                f"fallback_token={wrong_lookup_stats['fallback_token']}, "
                f"self={wrong_lookup_stats['self']}"
            )
        stage3_batches = _build_stage3_static_batches(
            active_indices=active_indices,
            sample_cache=sample_cache,
            args=args,
            train_bucket_meta=train_bucket_meta,
        )
        for batch_indices in tqdm(stage3_batches, desc="Stage3 Phase-A"):
            batch_samples = [sample_cache[idx] for idx in batch_indices]
            batch_states = [current_states[idx] for idx in batch_indices]

            score11 = _score_sequences_no_grad_batch(
                model=model,
                samples=batch_samples,
                ids_list=[state.y11_ids for state in batch_states],
                mask_list=[state.y11_mask for state in batch_states],
                args=args,
                pad_token_id=pad_token_id,
            )
            score12 = _score_sequences_no_grad_batch(
                model=model,
                samples=batch_samples,
                ids_list=[state.y12_ids for state in batch_states],
                mask_list=[state.y12_mask for state in batch_states],
                args=args,
                pad_token_id=pad_token_id,
            )
            vic_scores = _score_sequences_no_grad_batch(
                model=model,
                samples=batch_samples,
                ids_list=[sample.y_vic_ids for sample in batch_samples],
                mask_list=[sample.y_vic_mask for sample in batch_samples],
                args=args,
                pad_token_id=pad_token_id,
            )
            next1_outputs = _generate_candidate_batch(
                model, batch_samples, args, image_token_id, pad_token_id
            )
            next2_outputs = _generate_candidate_batch(
                model, batch_samples, args, image_token_id, pad_token_id
            )
            next1_scores = _score_sequences_no_grad_batch(
                model=model,
                samples=batch_samples,
                ids_list=[ids for ids, _ in next1_outputs],
                mask_list=[mask for _, mask in next1_outputs],
                args=args,
                pad_token_id=pad_token_id,
            )
            next2_scores = _score_sequences_no_grad_batch(
                model=model,
                samples=batch_samples,
                ids_list=[ids for ids, _ in next2_outputs],
                mask_list=[mask for _, mask in next2_outputs],
                args=args,
                pad_token_id=pad_token_id,
            )

            for row_idx, idx in enumerate(batch_indices):
                state = batch_states[row_idx]
                sample = batch_samples[row_idx]
                lp11, prob11, token11, mask11 = score11[row_idx]
                lp12, prob12, token12, mask12 = score12[row_idx]
                y_vic_lp, y_vic_prob, y_vic_token_lp, y_vic_token_mask = vic_scores[row_idx]

                candidates = [
                    {
                        "ids": state.y11_ids,
                        "mask": state.y11_mask,
                        "avg_lp": lp11,
                        "prob": prob11,
                        "old_token_lp": token11,
                        "old_token_mask": mask11,
                        "prev_prob": state.prob_11,
                    },
                    {
                        "ids": state.y12_ids,
                        "mask": state.y12_mask,
                        "avg_lp": lp12,
                        "prob": prob12,
                        "old_token_lp": token12,
                        "old_token_mask": mask12,
                        "prev_prob": state.prob_12,
                    },
                ]
                candidates.sort(key=lambda x: x["prob"], reverse=True)
                y11 = candidates[0]
                y12 = candidates[1]
                if prob12 > prob11:
                    swap_count += 1

                if bool(int(args.stage3_pair_use_answer_correctness)) and tokenizer is not None:
                    gold_letter = str(getattr(sample, "answer_letter", "") or "").strip().upper()[:1]
                    if len(gold_letter) == 1 and "A" <= gold_letter <= "Z":
                        cand_scored = []
                        for cand in candidates:
                            pred_letter = _extract_answer_letter_from_ids(
                                cand["ids"],
                                int(sample.prompt_len),
                                tokenizer,
                            )
                            cand_scored.append({
                                **cand,
                                "pred_letter": pred_letter,
                                "is_correct": 1 if pred_letter == gold_letter else 0,
                            })
                        cand_scored.sort(
                            key=lambda x: (x["is_correct"], float(x["prob"])),
                            reverse=True,
                        )
                        y11 = cand_scored[0]
                        y12 = cand_scored[1]

                delta11 = float(y11["prob"] - y11["prev_prob"])
                p_best = max(float(y11["prob"]), float(y12["prob"]))

                use_cold_start = (p_best < tau1) and (delta11 < tau_delta)
                if force_vic_positive:
                    y_plus_ids = sample.y_vic_ids
                    y_plus_mask = sample.y_vic_mask
                    old_lp_plus = y_vic_token_lp
                    old_mask_plus = y_vic_token_mask
                    y_plus_avg_lp = float(y_vic_lp)
                    y_minus_ids = state.y12_ids
                    y_minus_mask = state.y12_mask
                    old_lp_minus = token12
                    old_mask_minus = mask12
                    y_minus_avg_lp = float(lp12)
                elif use_cold_start:
                    y_plus_ids = sample.y_vic_ids
                    y_plus_mask = sample.y_vic_mask
                    old_lp_plus = y_vic_token_lp
                    old_mask_plus = y_vic_token_mask
                    y_plus_avg_lp = float(y_vic_lp)
                    y_minus_ids = y12["ids"]
                    y_minus_mask = y12["mask"]
                    old_lp_minus = y12["old_token_lp"]
                    old_mask_minus = y12["old_token_mask"]
                    y_minus_avg_lp = float(y12["avg_lp"])
                    cold_start_count += 1
                else:
                    y_plus_ids = y11["ids"]
                    y_plus_mask = y11["mask"]
                    old_lp_plus = y11["old_token_lp"]
                    old_mask_plus = y11["old_token_mask"]
                    y_plus_avg_lp = float(y11["avg_lp"])
                    y_minus_ids = y12["ids"]
                    y_minus_mask = y12["mask"]
                    old_lp_minus = y12["old_token_lp"]
                    old_mask_minus = y12["old_token_mask"]
                    y_minus_avg_lp = float(y12["avg_lp"])
                plus_lp_sum += y_plus_avg_lp
                minus_lp_sum += y_minus_avg_lp

                training_items.append(PeriodTrainingItem(
                    sample_idx=int(idx),
                    y_plus_ids=y_plus_ids.clone(),
                    y_plus_mask=y_plus_mask.clone(),
                    y_minus_ids=y_minus_ids.clone(),
                    y_minus_mask=y_minus_mask.clone(),
                    y_vic_ids=sample.y_vic_ids.clone(),
                    y_vic_mask=sample.y_vic_mask.clone(),
                    old_token_lp_plus=old_lp_plus.clone(),
                    old_token_mask_plus=old_mask_plus.clone(),
                    old_token_lp_minus=old_lp_minus.clone(),
                    old_token_mask_minus=old_mask_minus.clone(),
                    old_token_lp_vic=y_vic_token_lp.clone(),
                    old_token_mask_vic=y_vic_token_mask.clone(),
                    wrong_sample_idx=wrong_sample_lookup[int(idx)],
                ))

                next1_ids, next1_mask = next1_outputs[row_idx]
                next2_ids, next2_mask = next2_outputs[row_idx]
                next1_lp, next1_prob, _, _ = next1_scores[row_idx]
                next2_lp, next2_prob, _, _ = next2_scores[row_idx]
                next_sorted = [
                    (next1_ids, next1_mask, next1_lp, next1_prob),
                    (next2_ids, next2_mask, next2_lp, next2_prob),
                ]
                next_sorted.sort(key=lambda x: x[3], reverse=True)
                high = next_sorted[0]
                low = next_sorted[1]
                next_states[idx] = PeriodState(
                    sample_idx=int(idx),
                    y11_ids=high[0].clone(),
                    y11_mask=high[1].clone(),
                    y12_ids=low[0].clone(),
                    y12_mask=low[1].clone(),
                    avg_lp_11=float(high[2]),
                    avg_lp_12=float(low[2]),
                    prob_11=float(high[3]),
                    prob_12=float(low[3]),
                )
    finally:
        _restore_model_gradient_checkpointing(gc_states)
        _restore_model_use_cache(use_cache_states)

    return training_items, next_states, cold_start_count, swap_count, plus_lp_sum, minus_lp_sum


def _run_one_period_train(
    model,
    optimizer,
    loader,
    args,
    tb_writer,
    global_step: int,
    sub_stage_idx: int,
    period_idx: int,
    choice_token_map: Optional[Dict[int, List[int]]] = None,
) -> Tuple[float, float, float, float, float, float, float, Dict[str, float], int]:
    model.train()
    grad_accum = max(1, int(args.stage3_grad_accum or args.grad_accum))
    grad_clip = float(args.stage3_grad_clip)
    wrong_image_enable = bool(int(args.stage3_wrong_image_enable))
    wrong_image_weight = float(args.stage3_wrong_image_weight)
    wrong_image_margin = float(args.stage3_wrong_image_margin)
    obj_weight = float(args.stage3_obj_weight)
    reg_weight = float(args.stage3_reg_weight)
    answer_anchor_weight = float(args.stage3_answer_anchor_weight)
    mc_weight = float(args.stage3_mc_weight)
    field_weights = {
        "observed": float(args.stage3_field_weight_observed),
        "context": float(args.stage3_field_weight_context),
        "reasoning": float(args.stage3_field_weight_reasoning),
        "answer": float(args.stage3_field_weight_answer),
    }
    optimizer.zero_grad(set_to_none=True)

    total_loss = 0.0
    total_obj = 0.0
    total_reg = 0.0
    total_answer_anchor = 0.0
    total_wrong = 0.0
    total_mc = 0.0
    total_mc_acc = 0.0
    wrong_mismatch_count = 0
    total_field = {
        "observed": 0.0,
        "context": 0.0,
        "reasoning": 0.0,
        "answer": 0.0,
    }
    steps = 0

    def _sync_skip(flag: bool, message: str) -> bool:
        reduced = reduce_numeric_dict(
            {"skip": 1.0 if flag else 0.0},
            device=args.device,
        )
        if reduced["skip"] > 0.0:
            if is_main_process():
                print(message)
            optimizer.zero_grad(set_to_none=True)
            return True
        return False

    progress_bar = tqdm(
        loader,
        desc=f"Stage3 S{sub_stage_idx+1} P{period_idx+1}",
        disable=not is_main_process(),
    )
    for batch_idx, batch in enumerate(progress_bar):
        if batch is None:
            continue
        global_step += 1

        prompt_ids = batch["prompt_ids"].to(args.device)
        prompt_mask = batch["prompt_mask"].to(args.device)
        y_plus_ids = batch["y_plus_ids"].to(args.device)
        y_plus_mask = batch["y_plus_mask"].to(args.device)
        y_minus_ids = batch["y_minus_ids"].to(args.device)
        y_minus_mask = batch["y_minus_mask"].to(args.device)
        y_vic_ids = batch["y_vic_ids"].to(args.device)
        y_vic_mask = batch["y_vic_mask"].to(args.device)
        pixel_values = batch["pixel_values"].to(args.device)
        image_sizes = sanitize_image_sizes(batch.get("image_sizes"), batch_size=y_plus_ids.size(0))
        image_grid_thw = batch.get("image_grid_thw")
        wrong_pixel_values = batch["wrong_pixel_values"].to(args.device)
        wrong_image_sizes = sanitize_image_sizes(
            batch.get("wrong_image_sizes"),
            batch_size=y_plus_ids.size(0),
        )
        wrong_image_grid_thw = batch.get("wrong_image_grid_thw")
        prompt_lens = batch["prompt_lens"].to(args.device)

        old_lp_plus = batch["old_token_lp_plus"].to(args.device)
        old_mask_plus = batch["old_token_mask_plus"].to(args.device)
        old_lp_minus = batch["old_token_lp_minus"].to(args.device)
        old_mask_minus = batch["old_token_mask_minus"].to(args.device)
        old_lp_vic = batch["old_token_lp_vic"].to(args.device)
        old_mask_vic = batch["old_token_mask_vic"].to(args.device)
        vic_observed_mask = batch["vic_observed_mask"].to(args.device)
        vic_context_mask = batch["vic_context_mask"].to(args.device)
        vic_reasoning_mask = batch["vic_reasoning_mask"].to(args.device)
        vic_answer_mask = batch["vic_answer_mask"].to(args.device)
        vic_other_mask = batch["vic_other_mask"].to(args.device)
        answer_idx = batch["answer_idx"].to(args.device)
        num_choices = batch["num_choices"].to(args.device)

        # 分路前向 + 分路 backward，避免同时保留 plus/minus/vic 三路完整计算图。
        is_last_micro_step = (
            ((batch_idx + 1) % grad_accum == 0)
            or (batch_idx == len(loader) - 1)
        )
        sync_ctx = (
            contextlib.nullcontext()
            if (is_last_micro_step or not is_distributed() or not hasattr(model, "no_sync"))
            else model.no_sync()
        )

        cur_lp_plus, cur_mask_plus = _compute_token_log_probs(
            model, args, y_plus_ids, y_plus_mask, pixel_values, image_sizes, prompt_lens, image_grid_thw=image_grid_thw
        )
        if cur_lp_plus.shape[1] != old_lp_plus.shape[1]:
            target = min(cur_lp_plus.shape[1], old_lp_plus.shape[1])
            cur_lp_plus, cur_mask_plus = cur_lp_plus[:, :target], cur_mask_plus[:, :target]
            old_lp_plus, old_mask_plus = old_lp_plus[:, :target], old_mask_plus[:, :target]
        mask_plus = (old_mask_plus * cur_mask_plus).float()
        loss_plus, loss_plus_val = _stage3_masked_mean_ddp(
            log_clip(old_lp_plus - cur_lp_plus),
            mask_plus,
        )
        if _sync_skip(
            not torch.isfinite(loss_plus).item(),
            f"[Stage3][Warn] step={global_step} 检测到非有限 plus loss，所有 rank 同步跳过该 batch。",
        ):
            continue
        with sync_ctx:
            ((obj_weight * loss_plus) / grad_accum).backward()
        del cur_lp_plus, cur_mask_plus, mask_plus, loss_plus

        cur_lp_minus, cur_mask_minus = _compute_token_log_probs(
            model, args, y_minus_ids, y_minus_mask, pixel_values, image_sizes, prompt_lens, image_grid_thw=image_grid_thw
        )
        if cur_lp_minus.shape[1] != old_lp_minus.shape[1]:
            target = min(cur_lp_minus.shape[1], old_lp_minus.shape[1])
            cur_lp_minus, cur_mask_minus = cur_lp_minus[:, :target], cur_mask_minus[:, :target]
            old_lp_minus, old_mask_minus = old_lp_minus[:, :target], old_mask_minus[:, :target]
        mask_minus = (old_mask_minus * cur_mask_minus).float()
        loss_minus, loss_minus_val = _stage3_masked_mean_ddp(
            log_clip(-old_lp_minus + cur_lp_minus),
            mask_minus,
        )
        if _sync_skip(
            not torch.isfinite(loss_minus).item(),
            f"[Stage3][Warn] step={global_step} 检测到非有限 minus loss，所有 rank 同步跳过该 batch。",
        ):
            continue
        with sync_ctx:
            ((obj_weight * loss_minus) / grad_accum).backward()
        del cur_lp_minus, cur_mask_minus, mask_minus, loss_minus

        cur_lp_vic, cur_mask_vic = _compute_token_log_probs(
            model, args, y_vic_ids, y_vic_mask, pixel_values, image_sizes, prompt_lens, image_grid_thw=image_grid_thw
        )
        if cur_lp_vic.shape[1] != old_lp_vic.shape[1]:
            target = min(cur_lp_vic.shape[1], old_lp_vic.shape[1])
            cur_lp_vic, cur_mask_vic = cur_lp_vic[:, :target], cur_mask_vic[:, :target]
            old_lp_vic, old_mask_vic = old_lp_vic[:, :target], old_mask_vic[:, :target]
            vic_observed_mask = vic_observed_mask[:, :target]
            vic_context_mask = vic_context_mask[:, :target]
            vic_reasoning_mask = vic_reasoning_mask[:, :target]
            vic_answer_mask = vic_answer_mask[:, :target]
            vic_other_mask = vic_other_mask[:, :target]
        mask_vic = (old_mask_vic * cur_mask_vic).float()
        delta_vic = old_lp_vic - cur_lp_vic

        field_masks = {
            "observed": mask_vic * vic_observed_mask,
            "context": mask_vic * vic_context_mask,
            "reasoning": mask_vic * vic_reasoning_mask,
            "answer": mask_vic * vic_answer_mask,
        }
        weighted_mask_vic = (
            field_weights["observed"] * field_masks["observed"]
            + field_weights["context"] * field_masks["context"]
            + field_weights["reasoning"] * field_masks["reasoning"]
            + field_weights["answer"] * field_masks["answer"]
            + mask_vic * vic_other_mask
        )

        loss_reg, loss_reg_val = _stage3_masked_mean_ddp(delta_vic, weighted_mask_vic)
        field_loss_vals = {}
        for name, field_mask in field_masks.items():
            _, field_loss_val = _stage3_masked_mean_ddp(delta_vic, field_mask)
            field_loss_vals[name] = field_loss_val
            total_field[name] += field_loss_vals[name]
        answer_anchor_mask = field_masks["answer"]
        loss_answer_anchor, loss_answer_anchor_val = _stage3_masked_mean_ddp(
            -cur_lp_vic,
            answer_anchor_mask,
        )

        loss_wrong = torch.zeros((), device=args.device, dtype=loss_reg.dtype)
        loss_wrong_val = 0.0
        cur_lp_wrong_vic = None
        cur_mask_wrong_vic = None
        wrong_mask = None
        wrong_delta = None
        if wrong_image_enable:
            try:
                cur_lp_wrong_vic, cur_mask_wrong_vic = _compute_token_log_probs(
                    model, args, y_vic_ids, y_vic_mask, wrong_pixel_values, wrong_image_sizes, prompt_lens, image_grid_thw=wrong_image_grid_thw
                )
            except ValueError as exc:
                if "Image features and image tokens do not match" not in str(exc):
                    raise
                wrong_mismatch_count += 1
                if is_main_process():
                    print(
                        f"[Stage3][Warn] step={global_step} wrong-image token/feature mismatch，"
                        f"本 batch 跳过 wrong-image loss。detail={exc}"
                    )
            if cur_lp_wrong_vic is not None and cur_mask_wrong_vic is not None:
                if cur_lp_wrong_vic.shape[1] != cur_lp_vic.shape[1]:
                    target = min(cur_lp_wrong_vic.shape[1], cur_lp_vic.shape[1])
                    cur_lp_wrong_vic = cur_lp_wrong_vic[:, :target]
                    cur_mask_wrong_vic = cur_mask_wrong_vic[:, :target]
                    cur_lp_vic = cur_lp_vic[:, :target]
                    mask_vic = mask_vic[:, :target]
                wrong_mask = (mask_vic * cur_mask_wrong_vic).float()
                wrong_delta = F.relu(cur_lp_wrong_vic - cur_lp_vic + wrong_image_margin)
                loss_wrong, loss_wrong_val = _stage3_masked_mean_ddp(wrong_delta, wrong_mask)

        loss_vic_total = reg_weight * loss_reg + answer_anchor_weight * loss_answer_anchor + wrong_image_weight * loss_wrong
        if _sync_skip(
            not torch.isfinite(loss_vic_total).item(),
            f"[Stage3][Warn] step={global_step} 检测到非有限 vic loss，所有 rank 同步跳过该 batch。",
        ):
            continue
        has_mc_backward = False
        valid_mc_rows_exist = False
        choice_scores = None
        valid_mc_rows = None
        if mc_weight > 0.0 and choice_token_map:
            choice_scores, valid_mc_rows = _compute_choice_scores_from_prompt_batch(
                model=model,
                args=args,
                prompt_ids=prompt_ids,
                prompt_mask=prompt_mask,
                pixel_values=pixel_values,
                image_sizes=image_sizes,
                num_choices=num_choices,
                choice_token_map=choice_token_map,
                image_grid_thw=image_grid_thw,
            )
            valid_mc_rows = valid_mc_rows & (num_choices > 0) & (answer_idx >= 0) & (answer_idx < num_choices)
            valid_mc_rows_exist = bool(valid_mc_rows.any().item())
            has_mc_backward = valid_mc_rows_exist

        if has_mc_backward and is_distributed() and hasattr(model, "no_sync"):
            with model.no_sync() if is_last_micro_step else sync_ctx:
                (loss_vic_total / grad_accum).backward()
        else:
            with sync_ctx:
                (loss_vic_total / grad_accum).backward()
        del cur_lp_vic, cur_mask_vic, mask_vic, delta_vic, loss_reg, loss_answer_anchor, loss_vic_total
        if cur_lp_wrong_vic is not None and cur_mask_wrong_vic is not None:
            del cur_lp_wrong_vic, cur_mask_wrong_vic, wrong_mask, wrong_delta

        loss_mc = torch.zeros((), device=args.device, dtype=old_lp_vic.dtype)
        mc_acc_proxy = 0.0
        if has_mc_backward and choice_scores is not None and valid_mc_rows is not None:
            if valid_mc_rows_exist:
                loss_mc = F.cross_entropy(choice_scores[valid_mc_rows], answer_idx[valid_mc_rows])
                if _sync_skip(
                    not torch.isfinite(loss_mc).item(),
                    f"[Stage3][Warn] step={global_step} 检测到非有限多选 loss，所有 rank 同步跳过该 batch。",
                ):
                    del choice_scores, valid_mc_rows
                    continue
                pred_idx = choice_scores[valid_mc_rows].argmax(dim=-1)
                mc_acc_proxy = float((pred_idx == answer_idx[valid_mc_rows]).float().mean().item())
                ((mc_weight * loss_mc) / grad_accum).backward()
                del pred_idx
            del choice_scores, valid_mc_rows

        obj_val = loss_plus_val + loss_minus_val
        reg_val = loss_reg_val
        answer_anchor_val = loss_answer_anchor_val
        wrong_val = loss_wrong_val
        mc_val = float(loss_mc.item())
        loss_val = (
            mc_weight * mc_val
            + obj_weight * obj_val
            + reg_weight * reg_val
            + answer_anchor_weight * answer_anchor_val
            + wrong_image_weight * wrong_val
        )

        if (batch_idx + 1) % grad_accum == 0 or batch_idx == len(loader) - 1:
            invalid_grad = 0.0
            if grad_clip > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_norm=grad_clip,
                )
                if isinstance(grad_norm, torch.Tensor) and not torch.isfinite(grad_norm):
                    invalid_grad = 1.0
            reduced_grad = reduce_numeric_dict({"invalid_grad": invalid_grad}, device=args.device)
            if reduced_grad["invalid_grad"] > 0.0:
                if is_main_process():
                    print(
                        f"[Stage3][Warn] step={global_step} 检测到非有限梯度范数，"
                        "所有 rank 同步跳过 optimizer.step()。"
                    )
                optimizer.zero_grad(set_to_none=True)
                continue
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        steps += 1
        total_loss += loss_val
        total_obj += obj_val
        total_reg += reg_val
        total_answer_anchor += answer_anchor_val
        total_wrong += wrong_val
        total_mc += mc_val
        total_mc_acc += mc_acc_proxy

        if global_step % args.log_step == 0 and is_main_process():
            print(
                f"Step {global_step}, Total: {loss_val:.4f}, "
                f"L_mc: {mc_val:.4f}, MCAcc: {mc_acc_proxy:.4f}, "
                f"L_obj: {obj_val:.4f}, L_reg: {reg_val:.4f}, "
                f"L_ans_anchor: {answer_anchor_val:.4f}, L_wrong: {wrong_val:.4f}"
            )
            if tb_writer is not None:
                tb_writer.add_scalar("stage3/total_loss", loss_val, global_step)
                tb_writer.add_scalar("stage3/L_mc", mc_val, global_step)
                tb_writer.add_scalar("stage3/L_mc_weighted", mc_weight * mc_val, global_step)
                tb_writer.add_scalar("stage3/mc_acc_proxy", mc_acc_proxy, global_step)
                tb_writer.add_scalar("stage3/L_obj", obj_val, global_step)
                tb_writer.add_scalar("stage3/L_reg", reg_val, global_step)
                tb_writer.add_scalar("stage3/L_obj_weighted", obj_weight * obj_val, global_step)
                tb_writer.add_scalar("stage3/L_reg_weighted", reg_weight * reg_val, global_step)
                tb_writer.add_scalar("stage3/L_answer_anchor", answer_anchor_val, global_step)
                tb_writer.add_scalar("stage3/L_wrong_image", wrong_val, global_step)
                tb_writer.add_scalar("stage3/L_reg_observed", field_loss_vals["observed"], global_step)
                tb_writer.add_scalar("stage3/L_reg_context", field_loss_vals["context"], global_step)
                tb_writer.add_scalar("stage3/L_reg_reasoning", field_loss_vals["reasoning"], global_step)
                tb_writer.add_scalar("stage3/L_reg_answer", field_loss_vals["answer"], global_step)

    reduced_epoch = reduce_numeric_dict(
        {
            "total_loss": total_loss,
            "total_obj": total_obj,
            "total_reg": total_reg,
            "total_answer_anchor": total_answer_anchor,
            "total_wrong": total_wrong,
            "total_mc": total_mc,
            "total_mc_acc": total_mc_acc,
            "wrong_mismatch_count": float(wrong_mismatch_count),
            "total_observed": total_field["observed"],
            "total_context": total_field["context"],
            "total_reasoning": total_field["reasoning"],
            "total_answer": total_field["answer"],
            "steps": float(steps),
        },
        device=args.device,
    )
    denom = max(1.0, reduced_epoch["steps"])
    avg_field = {
        "observed": reduced_epoch["total_observed"] / denom,
        "context": reduced_epoch["total_context"] / denom,
        "reasoning": reduced_epoch["total_reasoning"] / denom,
        "answer": reduced_epoch["total_answer"] / denom,
    }
    if is_main_process() and reduced_epoch.get("wrong_mismatch_count", 0.0) > 0.0:
        print(
            f"[Stage3][Warn] 当前 period 共有 "
            f"{int(reduced_epoch['wrong_mismatch_count'])} 个 batch 触发 wrong-image mismatch 兜底。"
        )
    return (
        reduced_epoch["total_loss"] / denom,
        reduced_epoch["total_mc"] / denom,
        reduced_epoch["total_mc_acc"] / denom,
        reduced_epoch["total_obj"] / denom,
        reduced_epoch["total_reg"] / denom,
        reduced_epoch["total_answer_anchor"] / denom,
        reduced_epoch["total_wrong"] / denom,
        avg_field,
        global_step,
    )


def train_stage3_lord(model, train_dataset, args, tb_writer, train_bucket_meta: Optional[dict] = None):
    if is_main_process():
        print("\n" + "=" * 50)
        print("阶段 3: LoRD-II 多 Period 训练")
        print("=" * 50)

    base_model = unwrap_model(model)
    image_token_id = _get_image_token_id(base_model)
    pad_token_id = base_model.config.pad_token_id or base_model.config.eos_token_id or 0
    tau1 = float(args.tau1)
    tau_delta = float(args.tau_delta)
    sub_stage_num = max(1, int(args.sub_stage_num))
    period_num = int(args.period_num)
    if period_num <= 0:
        period_num = max(1, int(args.epochs))
    sub_set_num = int(args.sub_set_num)
    stage3_lr = float(args.lr) * float(args.stage3_lr_scale)
    stage3_phase_a_batch_size = _get_stage3_phase_a_batch_size(args, train_bucket_meta)
    stage3_use_static_bucket = bool(_is_hf_multichoice_dataset(args.dataset_name) and train_bucket_meta is not None)
    stage3_eval_answer_mode = str(args.stage3_eval_answer_mode).strip().lower()

    if is_main_process():
        print(
            f"[Stage3] tau1={tau1}, tau_delta={tau_delta}, "
            f"sub_stage_num={sub_stage_num}, period_num={period_num}, sub_set_num={sub_set_num}, "
            f"lr={stage3_lr}, train_projector={int(bool(args.stage3_train_projector))}, "
            f"resume_opt={int(bool(args.stage3_resume_save_optimizer))}, "
            f"eval_max_samples={int(args.stage3_eval_max_samples)}, "
            f"field_w=({float(args.stage3_field_weight_observed):.2f},"
            f"{float(args.stage3_field_weight_context):.2f},"
            f"{float(args.stage3_field_weight_reasoning):.2f},"
            f"{float(args.stage3_field_weight_answer):.2f}), "
            f"mc_w={float(args.stage3_mc_weight):.2f}, "
            f"obj_w={float(args.stage3_obj_weight):.2f}, "
            f"reg_w={float(args.stage3_reg_weight):.2f}, "
            f"ans_anchor_w={float(args.stage3_answer_anchor_weight):.2f}, "
            f"pair_by_answer={int(bool(args.stage3_pair_use_answer_correctness))}, "
            f"wrong_image={int(bool(args.stage3_wrong_image_enable))}, "
            f"force_cold_start_p0={int(bool(args.stage3_force_cold_start_period0))}, "
            f"phaseA_batch_size={stage3_phase_a_batch_size}, static_bucket={int(stage3_use_static_bucket)}, "
            f"eval_mode={stage3_eval_answer_mode}, distributed={int(is_distributed())}, "
            f"world_size={get_world_size()}"
        )

    trainable_params = _set_stage3_trainable_params(model, args)
    if is_main_process():
        print(f"Stage3 可训练参数量: {sum(p.numel() for p in trainable_params):,}")
    if not trainable_params:
        raise RuntimeError("Stage3 没有可训练参数，请检查 LoRA 或 stage3_train_projector 配置")
    optimizer = torch.optim.AdamW(trainable_params, lr=stage3_lr)

    stage3_sample_cache_path = _resolve_stage3_sample_cache_path(args)
    sample_cache = _load_stage3_sample_cache(stage3_sample_cache_path)
    if sample_cache is None:
        if is_distributed():
            if is_main_process():
                sample_cache = _build_stage3_sample_cache(train_dataset)
                _save_stage3_sample_cache(stage3_sample_cache_path, sample_cache)
            barrier_if_distributed()
            if not is_main_process():
                sample_cache = _load_stage3_sample_cache(stage3_sample_cache_path)
        else:
            sample_cache = _build_stage3_sample_cache(train_dataset)
            _save_stage3_sample_cache(stage3_sample_cache_path, sample_cache)
    if not sample_cache:
        raise RuntimeError("Stage3SampleCache 为空，无法开始 Stage3 训练")

    tokenizer = train_dataset.processor.tokenizer
    choice_token_map: Optional[Dict[int, List[int]]] = None
    answer_lookup: Dict[int, str] = {}
    choice_token_map = _build_stage3_choice_token_map(tokenizer, max_choices=26)
    raw_samples = train_dataset.samples
    for idx, sample in enumerate(raw_samples):
        answer_letter = str(sample.get("answer_letter", "") or "").strip().upper()[:1]
        if len(answer_letter) == 1 and "A" <= answer_letter <= "Z":
            answer_lookup[int(idx)] = answer_letter

    stage3_eval_max_samples = int(args.stage3_eval_max_samples)
    eval_sample_cache: Optional[List[Stage3SampleCacheItem]] = None
    eval_answer_lookup: Dict[int, str] = {}
    eval_pool_indices: Optional[List[int]] = None
    stage3_eval_split = str(args.stage3_eval_scienceqa_split).strip()
    stage3_eval_path = str(args.stage3_eval_scienceqa_path or args.scienceqa_path).strip()
    stage3_eval_train_num = int(args.stage3_eval_train_num)
    stage3_eval_sample_cache_path = _resolve_stage3_eval_sample_cache_path(
        stage3_sample_cache_path,
        stage3_eval_split,
        stage3_eval_path,
        stage3_eval_train_num,
    )
    if (
        stage3_eval_max_samples > 0
        and tokenizer is not None
        and stage3_eval_split
        and stage3_eval_path
    ):
        eval_samples = build_scienceqa_samples(
            scienceqa_path=stage3_eval_path,
            dataset_name=args.dataset_name,
            split=stage3_eval_split,
            train_num=stage3_eval_train_num,
            seed=int(args.scienceqa_seed),
            include_dataset_answer=True,
        )
        eval_dataset = ScienceQADataset(
            processor=train_dataset.processor,
            scienceqa_path=stage3_eval_path,
            dataset_name=args.dataset_name,
            split=stage3_eval_split,
            train_num=stage3_eval_train_num,
            max_length=args.max_length,
            samples=eval_samples,
            seed=int(args.scienceqa_seed),
            teacher_lang=args.teacher_lang,
            teacher_observed_max_tokens=args.teacher_observed_max_tokens,
            teacher_context_max_tokens=args.teacher_context_max_tokens,
            teacher_reasoning_max_tokens=args.teacher_reasoning_max_tokens,
            teacher_answer_max_tokens=args.teacher_answer_max_tokens,
            stage3_vic_include_context=bool(int(args.stage3_vic_include_context)),
            require_teacher_annotation=False,
        )
        eval_sample_cache = _load_stage3_sample_cache(stage3_eval_sample_cache_path)
        if eval_sample_cache is None:
            if is_distributed():
                if is_main_process():
                    eval_sample_cache = _build_stage3_sample_cache(eval_dataset)
                    _save_stage3_sample_cache(stage3_eval_sample_cache_path, eval_sample_cache)
                barrier_if_distributed()
                if not is_main_process():
                    eval_sample_cache = _load_stage3_sample_cache(stage3_eval_sample_cache_path)
            else:
                eval_sample_cache = _build_stage3_sample_cache(eval_dataset)
                _save_stage3_sample_cache(stage3_eval_sample_cache_path, eval_sample_cache)
        eval_pool_indices = list(range(len(eval_sample_cache)))
        for idx, sample in enumerate(eval_samples):
            answer_letter = str(sample.get("answer_letter", "") or "").strip().upper()[:1]
            if len(answer_letter) == 1 and "A" <= answer_letter <= "Z":
                eval_answer_lookup[int(idx)] = answer_letter
        if is_main_process():
            print(
                f"[Stage3][Eval] 使用独立评估集: split={stage3_eval_split}, "
                f"path={stage3_eval_path}, samples={len(eval_sample_cache)}"
            )

    all_indices = list(range(len(sample_cache)))
    global_step = 0
    base_seed = int(args.scienceqa_seed)
    period_counter = 0
    resume_save_optimizer = bool(int(args.stage3_resume_save_optimizer))
    resume_save_interval = max(1, int(args.stage3_resume_save_interval))
    stage3_eval_every_period = max(1, int(args.stage3_eval_every_period))
    resume_save_dir = str(args.stage3_resume_save_path).strip()
    if not resume_save_dir:
        resume_save_dir = os.path.join(args.save_path, "stage3_resume_latest")

    resume_path = str(args.stage3_resume_path).strip()
    resume_progress = None
    if resume_path:
        resume_progress = _load_stage3_resume_state(model, optimizer, resume_path, args.device, args)

    resume_sub_stage_idx = 0
    resume_period_idx = 0
    resume_active_indices: Optional[List[int]] = None
    resume_period_states: Optional[Dict[int, PeriodState]] = None
    if resume_progress:
        resume_sub_stage_idx = int(resume_progress.get("next_sub_stage_idx", 0))
        resume_period_idx = int(resume_progress.get("next_period_idx", 0))
        global_step = int(resume_progress.get("global_step", 0))
        period_counter = int(resume_progress.get("period_counter", 0))
        resume_active_indices_raw = resume_progress.get("active_indices")
        if isinstance(resume_active_indices_raw, list):
            resume_active_indices = [int(v) for v in resume_active_indices_raw]
        resume_period_states = _deserialize_period_states(resume_progress.get("period_states"))
        if bool(resume_progress.get("completed", False)):
            if resume_sub_stage_idx >= sub_stage_num:
                if is_main_process():
                    print("[Stage3][Resume] 断点标记为 completed，跳过 Stage3 训练。")
                return model
            if is_main_process():
                print(
                    "[Stage3][Resume] 断点来自已完成训练，但当前 sub_stage_num 更大；"
                    f"将从 sub_stage={resume_sub_stage_idx + 1} 继续。"
                )

    if resume_sub_stage_idx >= sub_stage_num:
        if is_main_process():
            print("[Stage3][Resume] next_sub_stage_idx 超出配置，Stage3 无需继续训练。")
        return model

    for sub_stage_idx in range(resume_sub_stage_idx, sub_stage_num):
        is_resume_sub_stage = bool(
            resume_progress is not None and sub_stage_idx == resume_sub_stage_idx
        )

        if is_resume_sub_stage and resume_active_indices is not None:
            active_indices = list(resume_active_indices)
        elif sub_set_num > 0 and sub_set_num < len(all_indices):
            rng = random.Random(base_seed + sub_stage_idx)
            active_indices = sorted(rng.sample(all_indices, sub_set_num))
        else:
            active_indices = list(all_indices)

        if is_main_process():
            print(
                f"[Stage3] sub_stage={sub_stage_idx+1}/{sub_stage_num}, "
                f"active_samples={len(active_indices)}"
            )

        if is_resume_sub_stage and resume_period_idx > 0 and resume_period_states is not None:
            period_states = resume_period_states
            if is_main_process():
                print(
                    f"[Stage3][Resume] 继续 sub_stage={sub_stage_idx+1}, "
                    f"period={resume_period_idx+1}/{period_num}"
                )
        else:
            if is_distributed():
                period_states = _bootstrap_period_states_distributed(
                    model=model,
                    sample_cache=sample_cache,
                    active_indices=active_indices,
                    args=args,
                    image_token_id=image_token_id,
                    pad_token_id=pad_token_id,
                    train_bucket_meta=train_bucket_meta,
                    sub_stage_idx=sub_stage_idx,
                )
            else:
                period_states = _bootstrap_period_states(
                    model=base_model,
                    sample_cache=sample_cache,
                    active_indices=active_indices,
                    args=args,
                    image_token_id=image_token_id,
                    pad_token_id=pad_token_id,
                    train_bucket_meta=train_bucket_meta,
                )
            resume_period_idx = 0

        for period_idx in range(resume_period_idx, period_num):
            if is_main_process():
                print(f"[Stage3] 进入 period {period_idx+1}/{period_num}")
            if is_distributed():
                barrier_if_distributed()
            phase_a_start = time.perf_counter()
            if is_distributed():
                training_items, next_states, cold_start_count, swap_count, plus_lp_sum, minus_lp_sum = _build_pairs_and_next_states_distributed(
                    model=model,
                    sample_cache=sample_cache,
                    current_states=period_states,
                    active_indices=active_indices,
                    args=args,
                    image_token_id=image_token_id,
                    pad_token_id=pad_token_id,
                    force_vic_positive=bool(
                        period_idx == 0 and int(args.stage3_force_cold_start_period0)
                    ),
                    train_bucket_meta=train_bucket_meta,
                    tokenizer=tokenizer,
                    sub_stage_idx=sub_stage_idx,
                    period_idx=period_idx,
                )
            else:
                training_items, next_states, cold_start_count, swap_count, plus_lp_sum, minus_lp_sum = _build_pairs_and_next_states(
                    model=base_model,
                    sample_cache=sample_cache,
                    current_states=period_states,
                    active_indices=active_indices,
                    args=args,
                    image_token_id=image_token_id,
                    pad_token_id=pad_token_id,
                    force_vic_positive=bool(
                        period_idx == 0 and int(args.stage3_force_cold_start_period0)
                    ),
                    train_bucket_meta=train_bucket_meta,
                    tokenizer=tokenizer,
                )
            phase_a_seconds = time.perf_counter() - phase_a_start
            period_states = next_states

            period_dataset = PeriodTrainingDataset(training_items, sample_cache)
            period_loader = _build_stage3_period_dataloader(
                period_dataset=period_dataset,
                args=args,
                pad_token_id=pad_token_id,
                seed=base_seed + sub_stage_idx * 100_003 + period_idx * 1_009 + 17,
            )
            avg_total, avg_mc, avg_mc_acc, avg_obj, avg_reg, avg_answer_anchor, avg_wrong, avg_field_losses, global_step = _run_one_period_train(
                model=model,
                optimizer=optimizer,
                loader=period_loader,
                args=args,
                tb_writer=tb_writer,
                global_step=global_step,
                sub_stage_idx=sub_stage_idx,
                period_idx=period_idx,
                choice_token_map=choice_token_map,
            )

            cold_ratio = float(cold_start_count) / max(1, len(active_indices))
            swap_ratio = float(swap_count) / max(1, len(active_indices))
            avg_lp_plus = float(plus_lp_sum) / max(1, len(training_items))
            avg_lp_minus = float(minus_lp_sum) / max(1, len(training_items))
            avg_mc_weighted = float(args.stage3_mc_weight) * avg_mc
            avg_obj_weighted = float(args.stage3_obj_weight) * avg_obj
            avg_reg_weighted = float(args.stage3_reg_weight) * avg_reg
            avg_answer_anchor_weighted = (
                float(args.stage3_answer_anchor_weight) * avg_answer_anchor
            )
            avg_wrong_weighted = float(args.stage3_wrong_image_weight) * avg_wrong
            period_counter += 1
            if is_main_process():
                print(
                    f"[Stage3][S{sub_stage_idx+1}P{period_idx+1}] "
                    f"loss={avg_total:.4f}, L_mc={avg_mc:.4f}, MCAcc={avg_mc_acc:.4f}, "
                    f"L_obj={avg_obj:.4f}, L_reg={avg_reg:.4f}, "
                    f"L_mc_w={avg_mc_weighted:.4f}, "
                    f"L_obj_w={avg_obj_weighted:.4f}, L_reg_w={avg_reg_weighted:.4f}, "
                    f"L_ans_anchor={avg_answer_anchor:.4f}, "
                    f"L_ans_anchor_w={avg_answer_anchor_weighted:.4f}, "
                    f"L_wrong={avg_wrong:.4f}, L_wrong_w={avg_wrong_weighted:.4f}, "
                    f"L_obs={avg_field_losses['observed']:.4f}, "
                    f"L_ctx={avg_field_losses['context']:.4f}, "
                    f"L_reason={avg_field_losses['reasoning']:.4f}, "
                    f"L_ans={avg_field_losses['answer']:.4f}, "
                    f"cold_start_ratio={cold_ratio:.4f}, swap_ratio={swap_ratio:.4f}, "
                    f"avg_lp_plus={avg_lp_plus:.4f}, avg_lp_minus={avg_lp_minus:.4f}, "
                    f"phase_a_seconds={phase_a_seconds:.2f}"
                )
                if tb_writer is not None:
                    tb_writer.add_scalar("stage3_epoch/total_loss", avg_total, period_counter)
                    tb_writer.add_scalar("stage3_epoch/L_mc", avg_mc, period_counter)
                    tb_writer.add_scalar("stage3_epoch/L_mc_weighted", avg_mc_weighted, period_counter)
                    tb_writer.add_scalar("stage3_epoch/mc_acc_proxy", avg_mc_acc, period_counter)
                    tb_writer.add_scalar("stage3_epoch/L_obj", avg_obj, period_counter)
                    tb_writer.add_scalar("stage3_epoch/L_reg", avg_reg, period_counter)
                    tb_writer.add_scalar("stage3_epoch/L_obj_weighted", avg_obj_weighted, period_counter)
                    tb_writer.add_scalar("stage3_epoch/L_reg_weighted", avg_reg_weighted, period_counter)
                    tb_writer.add_scalar("stage3_epoch/L_answer_anchor", avg_answer_anchor, period_counter)
                    tb_writer.add_scalar("stage3_epoch/L_answer_anchor_weighted", avg_answer_anchor_weighted, period_counter)
                    tb_writer.add_scalar("stage3_epoch/L_wrong_image", avg_wrong, period_counter)
                    tb_writer.add_scalar("stage3_epoch/L_wrong_image_weighted", avg_wrong_weighted, period_counter)
                    tb_writer.add_scalar("stage3_epoch/L_reg_observed", avg_field_losses["observed"], period_counter)
                    tb_writer.add_scalar("stage3_epoch/L_reg_context", avg_field_losses["context"], period_counter)
                    tb_writer.add_scalar("stage3_epoch/L_reg_reasoning", avg_field_losses["reasoning"], period_counter)
                    tb_writer.add_scalar("stage3_epoch/L_reg_answer", avg_field_losses["answer"], period_counter)
                    tb_writer.add_scalar("stage3/cold_start_ratio", cold_ratio, period_counter)
                    tb_writer.add_scalar("stage3/swap_ratio", swap_ratio, period_counter)
                    tb_writer.add_scalar("stage3/avg_lp_plus", avg_lp_plus, period_counter)
                    tb_writer.add_scalar("stage3/avg_lp_minus", avg_lp_minus, period_counter)
                    tb_writer.add_scalar("stage3/phase_a_seconds", phase_a_seconds, period_counter)
                    tb_writer.add_scalar("stage3/period", period_idx + 1, period_counter)
                    tb_writer.add_scalar("stage3/sub_stage", sub_stage_idx + 1, period_counter)

            if (
                stage3_eval_max_samples > 0
                and tokenizer is not None
                and (answer_lookup or eval_answer_lookup)
                and (period_counter % stage3_eval_every_period == 0)
            ):
                if is_distributed():
                    barrier_if_distributed()
                use_eval_cache = (
                    eval_sample_cache is not None
                    and eval_pool_indices is not None
                    and bool(eval_answer_lookup)
                )
                if use_eval_cache:
                    metrics_sample_cache = eval_sample_cache
                    metrics_indices_pool = eval_pool_indices
                    metrics_answer_lookup = eval_answer_lookup
                else:
                    metrics_sample_cache = sample_cache
                    metrics_indices_pool = active_indices
                    metrics_answer_lookup = answer_lookup

                eval_k = min(stage3_eval_max_samples, len(metrics_indices_pool))
                rng_eval = random.Random(base_seed + period_counter + 7919)
                if eval_k < len(metrics_indices_pool):
                    eval_indices = sorted(rng_eval.sample(metrics_indices_pool, eval_k))
                else:
                    eval_indices = list(metrics_indices_pool)
                if is_main_process():
                    val_acc, val_fmt, val_n = _evaluate_stage3_answer_metrics(
                        model=base_model,
                        sample_cache=metrics_sample_cache,
                        eval_indices=eval_indices,
                        args=args,
                        image_token_id=image_token_id,
                        pad_token_id=pad_token_id,
                        tokenizer=tokenizer,
                        answer_lookup=metrics_answer_lookup,
                        choice_token_map=choice_token_map,
                    )
                    print(
                        f"[Stage3][Eval][S{sub_stage_idx+1}P{period_idx+1}] "
                        f"val_answer_acc={val_acc:.4f}, format_rate={val_fmt:.4f}, "
                        f"mode={stage3_eval_answer_mode}, n={val_n}"
                    )
                    if tb_writer is not None:
                        tb_writer.add_scalar("stage3/val_answer_acc", val_acc, period_counter)
                        tb_writer.add_scalar("stage3/format_rate", val_fmt, period_counter)
                        tb_writer.add_scalar("stage3_eval/val_answer_acc", val_acc, period_counter)
                        tb_writer.add_scalar("stage3_eval/format_rate", val_fmt, period_counter)
                        tb_writer.add_scalar("stage3_eval/val_samples", float(val_n), period_counter)
                if is_distributed():
                    barrier_if_distributed()

            next_sub_stage_idx = sub_stage_idx
            next_period_idx = period_idx + 1
            next_period_states_to_save = period_states
            next_active_indices = active_indices
            if next_period_idx >= period_num:
                next_sub_stage_idx = sub_stage_idx + 1
                next_period_idx = 0
                next_period_states_to_save = None
                next_active_indices = None

            resume_progress_payload = {
                "version": 1,
                "completed": bool(next_sub_stage_idx >= sub_stage_num),
                "next_sub_stage_idx": int(next_sub_stage_idx),
                "next_period_idx": int(next_period_idx),
                "global_step": int(global_step),
                "period_counter": int(period_counter),
                "active_indices": next_active_indices,
                "period_states": _serialize_period_states(next_period_states_to_save),
                "sub_stage_num": int(sub_stage_num),
                "period_num": int(period_num),
                "sub_set_num": int(sub_set_num),
                "base_seed": int(base_seed),
            }
            should_save_resume = bool(
                resume_progress_payload["completed"]
                or (period_counter % resume_save_interval == 0)
            )
            barrier_if_distributed()
            if should_save_resume:
                _save_stage3_resume_state(
                    model=model,
                    optimizer=optimizer,
                    resume_dir=resume_save_dir,
                    progress=resume_progress_payload,
                    args=args,
                    include_optimizer_state=resume_save_optimizer,
                )
            barrier_if_distributed()

            if int(args.save_each_epoch) == 1:
                barrier_if_distributed()
                if is_main_process():
                    ckpt_dir = os.path.join(
                        args.save_path,
                        f"stage3_sub{sub_stage_idx+1}_period{period_idx+1}",
                    )
                    save_stage3_checkpoint(
                        model,
                        ckpt_dir,
                        remove_vq_codebook=bool(args.remove_vq_codebook),
                        student_model_type=str(args.student_model_type),
                    )
                barrier_if_distributed()

        resume_period_idx = 0
        resume_period_states = None
        resume_active_indices = None

    return model
