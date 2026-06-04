"""Stage1 training logic for VQ-LoRD3."""

import contextlib
import json
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
from tqdm import tqdm

from vq_lord3.training.train_vq_lord import (
    _capture_rng_state,
    _get_image_token_id,
    _get_trainable_parameter_state,
    _load_parameter_state,
    _move_optimizer_state_to_device_,
    _restore_rng_state,
    _to_cpu_obj,
    barrier_if_distributed,
    get_rank,
    get_world_size,
    is_distributed,
    is_main_process,
    is_projector_param,
    is_vision_param,
    maybe_set_dataloader_epoch,
    reduce_numeric_dict,
    sanitize_image_sizes,
    save_stage1_checkpoint,
    student_forward,
    unwrap_model,
)


def _get_stage1_vq_loss(model, device: str, dtype: torch.dtype) -> torch.Tensor:
    model = unwrap_model(model)
    container = getattr(model, "_vq_loss_container", None)
    if not isinstance(container, dict):
        return torch.zeros((), device=device, dtype=dtype)
    vq_loss = container.get("loss", None)
    if isinstance(vq_loss, torch.Tensor):
        return vq_loss
    return torch.zeros((), device=device, dtype=dtype)


def _get_stage1_vq_stats(model) -> tuple[float, int, int]:
    model = unwrap_model(model)
    container = getattr(model, "_vq_loss_container", None)
    if not isinstance(container, dict):
        return 0.0, 0, 0
    perplexity = container.get("perplexity", None)
    if isinstance(perplexity, torch.Tensor):
        perplexity_val = float(perplexity.detach().item())
    else:
        perplexity_val = 0.0
    dead_code_resets = int(container.get("dead_code_resets", 0))
    dead_code_count = int(container.get("dead_code_count", 0))
    return perplexity_val, dead_code_resets, dead_code_count


def _reduce_sum_tensor(tensor: torch.Tensor) -> torch.Tensor:
    reduced = tensor.detach().clone()
    if is_distributed():
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    return reduced


def _compute_stage1_token_loss_sum(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> tuple[torch.Tensor, torch.Tensor]:
    if logits is None or labels is None:
        raise ValueError("Stage1 token loss 计算需要 logits 和 labels")
    if logits.dim() != 3 or labels.dim() != 2:
        raise ValueError(
            f"Stage1 token loss 输入维度错误: logits.shape={tuple(logits.shape)}, "
            f"labels.shape={tuple(labels.shape)}"
        )

    shift_logits = logits[..., :-1, :].contiguous().float()
    shift_labels = labels[..., 1:].contiguous()
    valid_count = shift_labels.ne(ignore_index).sum().to(device=logits.device, dtype=torch.float32)
    if float(valid_count.item()) <= 0.0:
        return logits.sum() * 0.0, valid_count

    loss_sum = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=ignore_index,
        reduction="sum",
    )
    return loss_sum, valid_count


def _compute_stage1_weighted_token_loss_sum(
    logits: torch.Tensor,
    labels: torch.Tensor,
    token_weights: torch.Tensor,
    ignore_index: int = -100,
) -> tuple[torch.Tensor, torch.Tensor]:
    if logits is None or labels is None or token_weights is None:
        raise ValueError("Stage1 weighted token loss 计算需要 logits、labels 和 token_weights")
    if logits.dim() != 3 or labels.dim() != 2 or token_weights.dim() != 2:
        raise ValueError(
            f"Stage1 weighted loss 输入维度错误: logits={tuple(logits.shape)}, "
            f"labels={tuple(labels.shape)}, weights={tuple(token_weights.shape)}"
        )
    shift_logits = logits[..., :-1, :].contiguous().float()
    shift_labels = labels[..., 1:].contiguous()
    shift_weights = token_weights[..., 1:].contiguous().to(device=logits.device, dtype=torch.float32)
    valid_mask = shift_labels.ne(ignore_index)
    weight = shift_weights * valid_mask.float()
    weight_sum = weight.sum()
    if float(weight_sum.detach().item()) <= 0.0:
        return logits.sum() * 0.0, weight_sum

    vocab = shift_logits.size(-1)
    flat_labels = shift_labels.view(-1)
    ce = F.cross_entropy(
        shift_logits.view(-1, vocab),
        flat_labels,
        ignore_index=ignore_index,
        reduction="none",
    ).view_as(shift_labels)
    loss_sum = (ce * weight).sum()
    return loss_sum, weight_sum


def _stage1_global_mean_from_local_sum(
    local_loss_sum: torch.Tensor,
    local_weight: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    global_weight = _reduce_sum_tensor(local_weight.to(device=local_loss_sum.device, dtype=torch.float32))
    if float(global_weight.item()) <= 0.0:
        return local_loss_sum * 0.0, 0.0

    global_loss_sum = _reduce_sum_tensor(local_loss_sum.to(device=local_loss_sum.device, dtype=torch.float32))
    # DDP 在 backward 时会把梯度再除以 world_size。
    # 这里先乘回 world_size / global_weight，使最终得到的就是
    # d(global_loss_sum / global_weight) / d(params)，与单卡全局 mean 等价。
    scale = float(get_world_size()) / float(global_weight.item())
    backward_loss = local_loss_sum * scale
    log_loss = float((global_loss_sum / global_weight).item())
    return backward_loss, log_loss


def _stage1_resume_state_path(resume_dir: str) -> str:
    return os.path.join(resume_dir, "stage1_resume_state.pt")


def _stage1_resume_meta_path(resume_dir: str) -> str:
    return os.path.join(resume_dir, "stage1_resume_meta.json")


def _stage1_resume_rng_state_path(resume_dir: str, rank: int) -> str:
    return os.path.join(resume_dir, f"stage1_resume_rng_rank{int(rank)}.pt")


def _stage1_resume_config(args) -> dict:
    return {
        "model_path": str(args.model_path),
        "student_model_type": str(getattr(args, "student_model_type", "llava_next")),
        "use_lora": int(args.use_lora),
        "lora_rank": int(args.lora_rank),
        "lora_alpha": int(args.lora_alpha),
        "use_4bit": int(args.use_4bit),
        "model_dtype": str(args.model_dtype),
        "lr": float(args.lr),
        "beta": float(args.beta),
        "stage1_answer_weight": float(args.stage1_answer_weight),
        "stage1_rationale_weight": float(args.stage1_rationale_weight),
        "stage1_field_weight_observed": float(getattr(args, "stage1_field_weight_observed", 1.0)),
        "stage1_field_weight_context": float(getattr(args, "stage1_field_weight_context", 1.0)),
        "stage1_field_weight_reasoning": float(getattr(args, "stage1_field_weight_reasoning", 0.5)),
        "stage1_field_weight_answer": float(getattr(args, "stage1_field_weight_answer", 6.0)),
        "stage1_answer_prefix_weight": float(getattr(args, "stage1_answer_prefix_weight", 12.0)),
        "stage1_prepost_lr_scale": float(args.stage1_prepost_lr_scale),
        "stage1_vision_lr_scale": float(args.stage1_vision_lr_scale),
        "stage1_grad_clip": float(args.stage1_grad_clip),
        "stage1_grad_accum": int(getattr(args, "stage1_grad_accum", 0)),
        "grad_accum": int(getattr(args, "grad_accum", 1)),
        "batch_size": int(args.batch_size),
        "bucket_batch_size": int(getattr(args, "bucket_batch_size", 0)),
        "freeze_vision_tower": int(args.freeze_vision_tower),
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
        "vq_codebook_path": str(getattr(args, "vq_codebook_path", "")),
        "remove_vq_codebook": int(getattr(args, "remove_vq_codebook", 0)),
        "scienceqa_preprocessed_path": str(getattr(args, "scienceqa_preprocessed_path", "")),
        "world_size": int(getattr(args, "world_size", 1)),
    }


def _validate_stage1_resume_config(resume_config: dict, args):
    if not isinstance(resume_config, dict):
        return

    current = _stage1_resume_config(args)
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
        "remove_vq_codebook",
        "freeze_vision_tower",
    ]
    if not int(getattr(args, "remove_vq_codebook", 0)):
        strict_keys.append("vq_codebook_path")
    for key in strict_keys:
        if key in resume_config and resume_config[key] != current[key]:
            raise RuntimeError(
                f"Stage1 resume 配置不一致: key={key}, "
                f"resume={resume_config[key]!r}, current={current[key]!r}"
            )

    warning_keys = [
        "lr",
        "beta",
        "stage1_answer_weight",
        "stage1_rationale_weight",
        "stage1_field_weight_observed",
        "stage1_field_weight_context",
        "stage1_field_weight_reasoning",
        "stage1_field_weight_answer",
        "stage1_answer_prefix_weight",
        "stage1_prepost_lr_scale",
        "stage1_vision_lr_scale",
        "stage1_grad_clip",
        "batch_size",
        "bucket_batch_size",
        "stage1_grad_accum",
        "grad_accum",
        "scienceqa_preprocessed_path",
        "world_size",
    ]
    mismatched = []
    for key in warning_keys:
        if key in resume_config and resume_config[key] != current[key]:
            mismatched.append(
                f"{key}: resume={resume_config[key]!r}, current={current[key]!r}"
            )
    if mismatched and is_main_process():
        print("[Stage1][Resume][Warn] 检测到以下配置与断点不一致：")
        for item in mismatched:
            print(f"  - {item}")
        if "world_size" in resume_config:
            old_ws = int(resume_config.get("world_size", 1))
            new_ws = int(current.get("world_size", 1))
            if old_ws != new_ws:
                old_eff_bs = int(resume_config.get("batch_size", current["batch_size"])) * max(
                    1, int(resume_config.get("stage1_grad_accum", resume_config.get("grad_accum", 1)))
                ) * max(1, old_ws)
                new_eff_bs = int(current["batch_size"]) * max(
                    1, int(current.get("stage1_grad_accum", current.get("grad_accum", 1)))
                ) * max(1, new_ws)
                print(
                    f"  - [严重警告] world_size 变化: {old_ws} -> {new_ws}, "
                    f"effective_batch_size: {old_eff_bs} -> {new_eff_bs}"
                )
                print("    允许恢复，但训练轨迹不会与固定卡数运行完全一致。")


def _save_stage1_resume_state(
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
    barrier_if_distributed()

    local_rank = int(get_rank())
    local_rng_state = _capture_rng_state()
    torch.save(
        {"rank": local_rank, "rng_state": local_rng_state},
        _stage1_resume_rng_state_path(resume_dir, local_rank),
    )

    rng_state_by_rank = {local_rank: local_rng_state}
    if is_distributed():
        world_size = get_world_size()
        barrier_if_distributed()
        if not is_main_process():
            return
        rng_state_by_rank = {
            int(rank_idx): {
                "path": os.path.basename(_stage1_resume_rng_state_path(resume_dir, rank_idx))
            }
            for rank_idx in range(world_size)
        }

    payload = {
        "version": 1,
        "trainable_model_state": _get_trainable_parameter_state(model),
        "optimizer_state": _to_cpu_obj(optimizer.state_dict()) if include_optimizer_state else None,
        "rng_state": local_rng_state,
        "rng_state_by_rank": rng_state_by_rank,
        "progress": dict(progress),
        "config": _stage1_resume_config(args),
    }
    torch.save(payload, _stage1_resume_state_path(resume_dir))

    meta = dict(progress)
    meta["config"] = _stage1_resume_config(args)
    meta["has_optimizer_state"] = bool(include_optimizer_state)
    with open(_stage1_resume_meta_path(resume_dir), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    if is_main_process():
        print(f"[Stage1][Resume] 已保存断点: {resume_dir}")


def _load_stage1_resume_state(
    model,
    optimizer,
    resume_dir: str,
    device: str,
    args,
):
    state_path = _stage1_resume_state_path(resume_dir)
    if not os.path.exists(state_path):
        raise FileNotFoundError(f"未找到 Stage1 断点文件: {state_path}")

    payload = torch.load(state_path, map_location="cpu")
    _validate_stage1_resume_config(payload.get("config", {}), args)
    _load_parameter_state(model, payload.get("trainable_model_state", {}), "Stage1 resume trainable_model_state")

    optimizer_state = payload.get("optimizer_state")
    if optimizer_state:
        optimizer.load_state_dict(optimizer_state)
        _move_optimizer_state_to_device_(optimizer, device)
        if is_main_process():
            print("[Stage1][Resume] 已加载 optimizer state")
    else:
        if is_main_process():
            print("[Stage1][Resume] 断点中不含 optimizer state，将从当前优化器状态继续")

    rng_state_by_rank = payload.get("rng_state_by_rank")
    local_rank_key = int(get_rank())
    local_rng_state = None
    if isinstance(rng_state_by_rank, dict):
        local_rng_state = rng_state_by_rank.get(local_rank_key)
        if local_rng_state is None:
            local_rng_state = rng_state_by_rank.get(str(local_rank_key))
        if isinstance(local_rng_state, dict) and "path" in local_rng_state:
            shard_path = os.path.join(resume_dir, str(local_rng_state["path"]))
            if os.path.exists(shard_path):
                shard_payload = torch.load(shard_path, map_location="cpu")
                local_rng_state = shard_payload.get("rng_state")
            else:
                local_rng_state = None
        elif isinstance(local_rng_state, str):
            shard_path = os.path.join(resume_dir, local_rng_state)
            if os.path.exists(shard_path):
                shard_payload = torch.load(shard_path, map_location="cpu")
                local_rng_state = shard_payload.get("rng_state")
            else:
                local_rng_state = None
    if local_rng_state is None and is_distributed():
        shard_path = _stage1_resume_rng_state_path(resume_dir, local_rank_key)
        if os.path.exists(shard_path):
            shard_payload = torch.load(shard_path, map_location="cpu")
            local_rng_state = shard_payload.get("rng_state")
    if local_rng_state is None:
        if isinstance(rng_state_by_rank, dict) and is_main_process():
            print(
                f"[Stage1][Resume][Warn] 断点中未找到 rank={local_rank_key} 的专属 RNG 状态，"
                "将回退到共享 RNG 状态。"
            )
        local_rng_state = payload.get("rng_state")
    _restore_rng_state(local_rng_state)
    progress = payload.get("progress", {})
    if is_main_process():
        print(
            f"[Stage1][Resume] 已恢复进度: next_epoch={progress.get('next_epoch', 0)}, "
            f"global_step={progress.get('global_step', 0)}, completed={bool(progress.get('completed', False))}"
        )
    return progress


def train_stage1_vision(model, dataloader, args, tb_writer):
    """
    阶段 2: 视觉能力蒸馏

    目标：在固定 Stage0 codebook 的前提下，利用答案强监督 + 解释弱监督
    将视觉能力迁移到可被 Stage2 继续优化的生成策略。
    """
    if is_main_process():
        print("\n" + "=" * 50)
        print("Stage1")
        print("=" * 50)

    image_token_id = _get_image_token_id(model)
    model.train()
    base_model = unwrap_model(model)

    main_params = []
    prepost_params = []
    vision_params = []

    for name, param in base_model.named_parameters():
        if not torch.is_floating_point(param):
            param.requires_grad = False
            continue

        name_l = name.lower()
        is_lora = "lora_" in name_l or "modules_to_save" in name_l
        is_projector = is_projector_param(name, getattr(args, "student_model_type", "llava_next"))
        is_prepost = "pre_quant" in name_l or "post_quant" in name_l
        is_vq_embedding = "vq.embedding.weight" in name_l
        is_vision = is_vision_param(name, getattr(args, "student_model_type", "llava_next")) and not args.freeze_vision_tower

        if is_vq_embedding:
            param.requires_grad = False
            continue

        if is_lora or is_projector:
            param.requires_grad = True
            main_params.append(param)
        elif is_prepost:
            param.requires_grad = True
            prepost_params.append(param)
        elif is_vision:
            param.requires_grad = True
            vision_params.append(param)
        else:
            param.requires_grad = False

    trainable_param_list = [p for p in base_model.parameters() if p.requires_grad]
    trainable_params = sum(p.numel() for p in trainable_param_list)
    if is_main_process():
        print(f"可训练参数量: {trainable_params:,}")

    param_groups = []
    if main_params:
        param_groups.append({"params": main_params, "lr": args.lr})
    if prepost_params:
        param_groups.append({"params": prepost_params, "lr": args.lr * float(args.stage1_prepost_lr_scale)})
    if vision_params:
        param_groups.append({"params": vision_params, "lr": args.lr * float(args.stage1_vision_lr_scale)})

    if not param_groups:
        raise RuntimeError("Stage1 没有可训练参数，请检查冻结策略")

    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    grad_accum = max(1, int(getattr(args, "stage1_grad_accum", 0) or getattr(args, "grad_accum", 1)))
    if is_main_process():
        print(
            f"[Stage1] grad_accum={grad_accum}, "
            f"field_w=({float(getattr(args, 'stage1_field_weight_observed', 1.0)):.2f},"
            f"{float(getattr(args, 'stage1_field_weight_context', 1.0)):.2f},"
            f"{float(getattr(args, 'stage1_field_weight_reasoning', 0.5)):.2f},"
            f"{float(getattr(args, 'stage1_field_weight_answer', 6.0)):.2f}), "
            f"answer_prefix_w={float(getattr(args, 'stage1_answer_prefix_weight', 12.0)):.2f}, "
            f"prepost_lr_scale={args.stage1_prepost_lr_scale}, "
            f"vision_lr_scale={args.stage1_vision_lr_scale}, grad_clip={args.stage1_grad_clip}"
        )

    resume_path = str(getattr(args, "stage1_resume_path", "")).strip()
    start_epoch = 0
    global_step = 0
    if resume_path:
        progress = _load_stage1_resume_state(
            model=model,
            optimizer=optimizer,
            resume_dir=resume_path,
            device=args.device,
            args=args,
        )
        resume_next_epoch = int(progress.get("next_epoch", 0))
        resume_global_step = int(progress.get("global_step", 0))
        resume_completed = bool(progress.get("completed", False))
        resume_saved_epochs = int(progress.get("epochs", 0))
        if resume_completed and resume_next_epoch >= int(args.epochs):
            if is_main_process():
                print("[Stage1][Resume] 断点标记为 completed，跳过 Stage1 训练。")
            return model
        if resume_completed and resume_next_epoch < int(args.epochs) and is_main_process():
            print(
                "[Stage1][Resume] 断点来自已完成训练，但当前 args.epochs 更大；"
                f"将从 epoch={resume_next_epoch} 继续训练到 epoch={int(args.epochs)} "
                f"(saved_epochs={resume_saved_epochs})."
            )
        start_epoch = resume_next_epoch
        global_step = resume_global_step

    resume_save_dir = str(getattr(args, "stage1_resume_save_path", "")).strip()
    if not resume_save_dir:
        resume_save_dir = os.path.join(args.save_path, "stage1_resume_latest")
    resume_save_optimizer = bool(int(getattr(args, "stage1_resume_save_optimizer", 1)))
    resume_save_interval = max(1, int(getattr(args, "stage1_resume_save_interval", 1)))

    for epoch in range(start_epoch, args.epochs):
        maybe_set_dataloader_epoch(dataloader, epoch)
        epoch_total = 0.0
        epoch_answer = 0.0
        epoch_rationale = 0.0
        epoch_vq = 0.0
        epoch_vq_ratio = 0.0
        epoch_vq_perplexity = 0.0
        epoch_count = 0.0

        optimizer.zero_grad(set_to_none=True)
        progress_bar = tqdm(
            dataloader,
            desc=f"Stage1 Epoch {epoch+1}",
            disable=not is_main_process(),
        )
        for batch_idx, batch in enumerate(progress_bar):
            global_step += 1

            pixel_values = batch["pixel_values"].to(args.device)
            image_sizes = sanitize_image_sizes(batch.get("image_sizes"), batch_size=pixel_values.shape[0])

            full_input_ids = batch["full_input_ids"].to(args.device)
            full_attention_mask = batch["full_attention_mask"].to(args.device)
            full_labels = batch["full_labels"].to(args.device)
            field_masks = {
                "observed": batch["field_mask_observed"].to(args.device).float(),
                "context": batch["field_mask_context"].to(args.device).float(),
                "reasoning": batch["field_mask_reasoning"].to(args.device).float(),
                "answer": batch["field_mask_answer"].to(args.device).float(),
            }
            answer_prefix_mask = batch["field_mask_answer_prefix"].to(args.device).float()
            field_weights = {
                "observed": float(getattr(args, "stage1_field_weight_observed", 1.0)),
                "context": float(getattr(args, "stage1_field_weight_context", 1.0)),
                "reasoning": float(getattr(args, "stage1_field_weight_reasoning", 0.5)),
                "answer": float(getattr(args, "stage1_field_weight_answer", 6.0)),
            }
            answer_prefix_weight = float(getattr(args, "stage1_answer_prefix_weight", field_weights["answer"]))
            token_weights = (
                field_weights["observed"] * field_masks["observed"]
                + field_weights["context"] * field_masks["context"]
                + field_weights["reasoning"] * field_masks["reasoning"]
                + field_weights["answer"] * field_masks["answer"]
                + max(0.0, answer_prefix_weight - field_weights["answer"]) * answer_prefix_mask
            ).masked_fill(full_labels.eq(-100), 0.0)

            if global_step == 1:
                n_img = (full_input_ids == image_token_id).sum(dim=1)
                if (n_img == 0).any():
                    raise RuntimeError(
                        f"首个 batch 缺少 image token: counts={n_img.tolist()}, id={image_token_id}"
                    )

            outputs_full = student_forward(
                model=model,
                args=args,
                input_ids=full_input_ids,
                attention_mask=full_attention_mask,
                pixel_values=pixel_values,
                image_sizes=image_sizes,
                image_grid_thw=batch.get("full_image_grid_thw", batch.get("image_grid_thw")),
                labels=full_labels,
            )
            readable_loss_sum, readable_weight = _compute_stage1_weighted_token_loss_sum(
                outputs_full.logits,
                full_labels,
                token_weights,
            )
            readable_loss, readable_loss_val = _stage1_global_mean_from_local_sum(
                readable_loss_sum,
                readable_weight,
            )
            vq_loss_raw = _get_stage1_vq_loss(model, args.device, outputs_full.logits.dtype)
            field_loss_vals = {}
            for field_name, field_mask in field_masks.items():
                field_sum, field_count = _compute_stage1_weighted_token_loss_sum(
                    outputs_full.logits,
                    full_labels,
                    field_mask,
                )
                _, field_loss_val = _stage1_global_mean_from_local_sum(field_sum, field_count)
                field_loss_vals[field_name] = field_loss_val

            local_batch_weight = torch.tensor(
                float(pixel_values.shape[0]),
                device=args.device,
                dtype=torch.float32,
            )
            vq_loss, vq_loss_val = _stage1_global_mean_from_local_sum(
                vq_loss_raw.float() * local_batch_weight,
                local_batch_weight,
            )

            total_loss = readable_loss + args.beta * vq_loss

            skip_tensor = torch.tensor(
                0.0 if torch.isfinite(total_loss) else 1.0,
                device=total_loss.device,
                dtype=torch.float32,
            )
            reduced_skip = reduce_numeric_dict({"skip": float(skip_tensor.item())}, device=args.device)
            if reduced_skip["skip"] > 0.0:
                if is_main_process():
                    print(f"[Stage1][Warn] step={global_step} 检测到非有限损失，所有 rank 同步跳过该 batch。")
                optimizer.zero_grad(set_to_none=True)
                continue

            should_sync = (
                ((batch_idx + 1) % grad_accum == 0)
                or (batch_idx == len(dataloader) - 1)
                or not is_distributed()
                or not hasattr(model, "no_sync")
            )
            sync_ctx = contextlib.nullcontext() if should_sync else model.no_sync()
            with sync_ctx:
                (total_loss / grad_accum).backward()

            if (batch_idx + 1) % grad_accum == 0 or batch_idx == len(dataloader) - 1:
                invalid_grad = 0.0
                if args.stage1_grad_clip > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        trainable_param_list,
                        max_norm=float(args.stage1_grad_clip),
                    )
                    if isinstance(grad_norm, torch.Tensor) and not torch.isfinite(grad_norm):
                        invalid_grad = 1.0
                reduced_grad = reduce_numeric_dict({"invalid_grad": invalid_grad}, device=args.device)
                if reduced_grad["invalid_grad"] > 0.0:
                    if is_main_process():
                        print(f"[Stage1][Warn] step={global_step} 检测到非有限梯度范数，所有 rank 同步跳过 optimizer.step()。")
                    optimizer.zero_grad(set_to_none=True)
                    continue

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            total_loss_val = readable_loss_val + float(args.beta) * vq_loss_val
            weighted_vq_ratio = float(args.beta) * vq_loss_val / max(readable_loss_val, 1e-8)
            perplexity_val, dead_code_resets, dead_code_count = _get_stage1_vq_stats(model)

            epoch_total += total_loss_val
            epoch_answer += field_loss_vals["answer"]
            epoch_rationale += readable_loss_val
            epoch_vq += vq_loss_val
            epoch_vq_ratio += weighted_vq_ratio
            epoch_vq_perplexity += perplexity_val
            epoch_count += 1.0

            if global_step % args.log_step == 0 and is_main_process():
                print(
                    f"Step {global_step}, Total: {total_loss_val:.4f}, "
                    f"ReadableLoss: {readable_loss_val:.4f}, "
                    f"ObservedLoss: {field_loss_vals['observed']:.4f}, "
                    f"ContextLoss: {field_loss_vals['context']:.4f}, "
                    f"ReasoningLoss: {field_loss_vals['reasoning']:.4f}, "
                    f"AnswerLoss: {field_loss_vals['answer']:.4f}, "
                    f"AnswerTokenWeight: {field_weights['answer']:.2f}, "
                    f"VQ: {vq_loss_val:.4f}, VQ/Readable: {weighted_vq_ratio:.4f}, "
                    f"PPL: {perplexity_val:.2f}, Dead: {dead_code_count}"
                )
                if tb_writer is not None:
                    tb_writer.add_scalar("stage1/total_loss", total_loss_val, global_step)
                    tb_writer.add_scalar("stage1/readable_loss", readable_loss_val, global_step)
                    tb_writer.add_scalar("stage1/observed_loss", field_loss_vals["observed"], global_step)
                    tb_writer.add_scalar("stage1/context_loss", field_loss_vals["context"], global_step)
                    tb_writer.add_scalar("stage1/reasoning_loss", field_loss_vals["reasoning"], global_step)
                    tb_writer.add_scalar("stage1/answer_loss", field_loss_vals["answer"], global_step)
                    tb_writer.add_scalar("stage1/vq_loss", vq_loss_val, global_step)
                    tb_writer.add_scalar("stage1/vq_readable_ratio", weighted_vq_ratio, global_step)
                    tb_writer.add_scalar("stage1/vq_perplexity", perplexity_val, global_step)
                    tb_writer.add_scalar("stage1/dead_code_resets", dead_code_resets, global_step)
                    tb_writer.add_scalar("stage1/dead_code_count", dead_code_count, global_step)

        reduced_epoch = reduce_numeric_dict(
            {
                "epoch_total": epoch_total,
                "epoch_answer": epoch_answer,
                "epoch_rationale": epoch_rationale,
                "epoch_vq": epoch_vq,
                "epoch_vq_ratio": epoch_vq_ratio,
                "epoch_vq_perplexity": epoch_vq_perplexity,
                "epoch_count": epoch_count,
            },
            device=args.device,
        )
        denom = max(1.0, reduced_epoch["epoch_count"])
        avg_total = reduced_epoch["epoch_total"] / denom
        avg_answer = reduced_epoch["epoch_answer"] / denom
        avg_readable = reduced_epoch["epoch_rationale"] / denom
        avg_vq = reduced_epoch["epoch_vq"] / denom
        avg_vq_ratio = reduced_epoch["epoch_vq_ratio"] / denom
        avg_vq_perplexity = reduced_epoch["epoch_vq_perplexity"] / denom

        if is_main_process():
            print(
                f"Epoch {epoch+1} 平均损失: Total={avg_total:.4f}, "
                f"Readable={avg_readable:.4f}, Answer={avg_answer:.4f}, "
                f"VQ={avg_vq:.4f}, VQ/Readable={avg_vq_ratio:.4f}, "
                f"PPL={avg_vq_perplexity:.2f}"
            )
            if tb_writer is not None:
                tb_writer.add_scalar("stage1_epoch/total_loss", avg_total, epoch + 1)
                tb_writer.add_scalar("stage1_epoch/readable_loss", avg_readable, epoch + 1)
                tb_writer.add_scalar("stage1_epoch/answer_loss", avg_answer, epoch + 1)
                tb_writer.add_scalar("stage1_epoch/vq_loss", avg_vq, epoch + 1)
                tb_writer.add_scalar("stage1_epoch/vq_readable_ratio", avg_vq_ratio, epoch + 1)
                tb_writer.add_scalar("stage1_epoch/vq_perplexity", avg_vq_perplexity, epoch + 1)

        progress_payload = {
            "completed": bool(epoch + 1 >= args.epochs),
            "next_epoch": int(epoch + 1),
            "global_step": int(global_step),
            "epochs": int(args.epochs),
        }
        should_save_resume = bool(
            progress_payload["completed"]
            or ((epoch + 1) % resume_save_interval == 0)
        )
        barrier_if_distributed()
        if should_save_resume:
            _save_stage1_resume_state(
                model=model,
                optimizer=optimizer,
                resume_dir=resume_save_dir,
                progress=progress_payload,
                args=args,
                include_optimizer_state=resume_save_optimizer,
            )
        barrier_if_distributed()

        if int(getattr(args, "save_each_epoch", 0)) == 1:
            barrier_if_distributed()
            if is_main_process():
                epoch_ckpt_path = os.path.join(args.save_path, f"stage1_vision_epoch{epoch+1}")
                save_stage1_checkpoint(model, args, epoch_ckpt_path)
            barrier_if_distributed()

    return model
