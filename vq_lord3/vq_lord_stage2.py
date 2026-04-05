"""Stage2 training logic for VQ-LoRD3."""

import os

import torch
from tqdm import tqdm

from train_vq_lord3 import (
    _get_image_token_id,
    maybe_set_dataloader_epoch,
    sanitize_image_sizes,
    save_stage2_checkpoint,
)


def _get_stage2_vq_loss(model, device: str, dtype: torch.dtype) -> torch.Tensor:
    vq_loss = model._vq_loss_container.get("loss", None)
    if isinstance(vq_loss, torch.Tensor):
        return vq_loss
    return torch.zeros((), device=device, dtype=dtype)


def _get_stage2_vq_stats(model) -> tuple[float, int, int]:
    perplexity = model._vq_loss_container.get("perplexity", None)
    if isinstance(perplexity, torch.Tensor):
        perplexity_val = float(perplexity.detach().item())
    else:
        perplexity_val = 0.0
    dead_code_resets = int(model._vq_loss_container.get("dead_code_resets", 0))
    dead_code_count = int(model._vq_loss_container.get("dead_code_count", 0))
    return perplexity_val, dead_code_resets, dead_code_count


def train_stage2_vision(model, dataloader, args, tb_writer):
    """
    阶段 2: 视觉能力蒸馏

    目标：在固定 Stage1 codebook 的前提下，利用答案强监督 + 解释弱监督
    将视觉能力迁移到可被 Stage3 继续优化的生成策略。
    """
    print("\n" + "=" * 50)
    print("阶段 2: 视觉能力蒸馏")
    print("=" * 50)

    image_token_id = _get_image_token_id(model)
    model.train()

    main_params = []
    prepost_params = []
    vision_params = []

    for name, param in model.named_parameters():
        if not torch.is_floating_point(param):
            param.requires_grad = False
            continue

        name_l = name.lower()
        is_lora = "lora_" in name_l or "modules_to_save" in name_l
        is_projector = "projector" in name_l or "multi_modal_projector" in name_l
        is_prepost = "pre_quant" in name_l or "post_quant" in name_l
        is_vq_embedding = "vq.embedding.weight" in name_l
        is_vision = "vision" in name_l and not args.freeze_vision_tower

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

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"可训练参数量: {trainable_params:,}")

    param_groups = []
    if main_params:
        param_groups.append({"params": main_params, "lr": args.lr})
    if prepost_params:
        param_groups.append({"params": prepost_params, "lr": args.lr * float(args.stage2_prepost_lr_scale)})
    if vision_params:
        param_groups.append({"params": vision_params, "lr": args.lr * float(args.stage2_vision_lr_scale)})

    if not param_groups:
        raise RuntimeError("Stage2 没有可训练参数，请检查冻结策略")

    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    grad_accum = max(1, int(getattr(args, "stage2_grad_accum", 0) or getattr(args, "grad_accum", 1)))
    print(
        f"[Stage2] grad_accum={grad_accum}, "
        f"answer_w={args.stage2_answer_weight}, rationale_w={args.stage2_rationale_weight}, "
        f"prepost_lr_scale={args.stage2_prepost_lr_scale}, "
        f"vision_lr_scale={args.stage2_vision_lr_scale}, grad_clip={args.stage2_grad_clip}"
    )

    global_step = 0
    for epoch in range(args.epochs):
        maybe_set_dataloader_epoch(dataloader, epoch)
        epoch_total = 0.0
        epoch_answer = 0.0
        epoch_rationale = 0.0
        epoch_vq = 0.0
        epoch_vq_ratio = 0.0
        epoch_vq_perplexity = 0.0
        epoch_count = 0

        optimizer.zero_grad(set_to_none=True)
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"视觉蒸馏 Epoch {epoch+1}")):
            global_step += 1

            pixel_values = batch["pixel_values"].to(args.device)
            image_sizes = sanitize_image_sizes(batch.get("image_sizes"), batch_size=pixel_values.shape[0])

            answer_input_ids = batch["answer_input_ids"].to(args.device)
            answer_attention_mask = batch["answer_attention_mask"].to(args.device)
            answer_labels = batch["answer_labels"].to(args.device)

            full_input_ids = batch["full_input_ids"].to(args.device)
            full_attention_mask = batch["full_attention_mask"].to(args.device)
            full_labels = batch["full_labels"].to(args.device)
            has_rationale = batch["has_rationale"].to(args.device).bool()
            rationale_labels = full_labels.masked_fill(~has_rationale.unsqueeze(1), -100)
            has_any_rationale = bool(has_rationale.any().item())

            if global_step == 1:
                n_img = (full_input_ids == image_token_id).sum(dim=1)
                if (n_img == 0).any():
                    raise RuntimeError(
                        f"首个 batch 缺少 image token: counts={n_img.tolist()}, id={image_token_id}"
                    )

            # 前向1：答案监督
            outputs_answer = model(
                input_ids=answer_input_ids,
                attention_mask=answer_attention_mask,
                pixel_values=pixel_values,
                image_sizes=image_sizes,
                labels=answer_labels,
            )
            answer_loss = outputs_answer.loss
            answer_vq_loss = _get_stage2_vq_loss(model, args.device, answer_loss.dtype)

            # 前向2：解释/完整回答监督（无解释样本可退化为 0）
            if has_any_rationale:
                outputs_full = model(
                    input_ids=full_input_ids,
                    attention_mask=full_attention_mask,
                    pixel_values=pixel_values,
                    image_sizes=image_sizes,
                    labels=rationale_labels,
                )
                rationale_loss = outputs_full.loss
                # 同一张图的 VQ 路径应近似一致；有第二次图像前向时取最新值。
                vq_loss = _get_stage2_vq_loss(model, args.device, answer_loss.dtype)
            else:
                rationale_loss = torch.zeros((), device=args.device, dtype=answer_loss.dtype)
                vq_loss = answer_vq_loss

            total_loss = (
                float(args.stage2_answer_weight) * answer_loss
                + float(args.stage2_rationale_weight) * rationale_loss
                + args.beta * vq_loss
            )

            if not torch.isfinite(total_loss):
                print(
                    f"[Stage2][Warn] step={global_step} 非有限损失，跳过。"
                )
                optimizer.zero_grad(set_to_none=True)
                continue

            (total_loss / grad_accum).backward()

            if (batch_idx + 1) % grad_accum == 0 or batch_idx == len(dataloader) - 1:
                if args.stage2_grad_clip > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        max_norm=float(args.stage2_grad_clip),
                    )
                    if isinstance(grad_norm, torch.Tensor) and not torch.isfinite(grad_norm):
                        print(f"[Stage2][Warn] step={global_step} 梯度范数非有限，跳过 optimizer.step()")
                        optimizer.zero_grad(set_to_none=True)
                        continue
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            answer_loss_val = float(answer_loss.item())
            rationale_loss_val = float(rationale_loss.item())
            vq_loss_val = float(vq_loss.item())
            total_loss_val = float(total_loss.item())
            weighted_vq_ratio = float(args.beta) * vq_loss_val / max(answer_loss_val, 1e-8)
            perplexity_val, dead_code_resets, dead_code_count = _get_stage2_vq_stats(model)

            epoch_total += total_loss_val
            epoch_answer += answer_loss_val
            epoch_rationale += rationale_loss_val
            epoch_vq += vq_loss_val
            epoch_vq_ratio += weighted_vq_ratio
            epoch_vq_perplexity += perplexity_val
            epoch_count += 1

            if global_step % args.log_step == 0:
                print(
                    f"Step {global_step}, Total: {total_loss_val:.4f}, "
                    f"Answer: {answer_loss_val:.4f}, Rationale: {rationale_loss_val:.4f}, "
                    f"VQ: {vq_loss_val:.4f}, VQ/Answer: {weighted_vq_ratio:.4f}, "
                    f"PPL: {perplexity_val:.2f}, Dead: {dead_code_count}"
                )
                tb_writer.add_scalar("stage2/total_loss", total_loss_val, global_step)
                tb_writer.add_scalar("stage2/answer_loss", answer_loss_val, global_step)
                tb_writer.add_scalar("stage2/rationale_loss", rationale_loss_val, global_step)
                tb_writer.add_scalar("stage2/vq_loss", vq_loss_val, global_step)
                tb_writer.add_scalar("stage2/vq_answer_ratio", weighted_vq_ratio, global_step)
                tb_writer.add_scalar("stage2/vq_perplexity", perplexity_val, global_step)
                tb_writer.add_scalar("stage2/dead_code_resets", dead_code_resets, global_step)
                tb_writer.add_scalar("stage2/dead_code_count", dead_code_count, global_step)

        denom = max(1, epoch_count)
        print(
            f"Epoch {epoch+1} 平均损失: Total={epoch_total/denom:.4f}, "
            f"Answer={epoch_answer/denom:.4f}, Rationale={epoch_rationale/denom:.4f}, "
            f"VQ={epoch_vq/denom:.4f}, VQ/Answer={epoch_vq_ratio/denom:.4f}, "
            f"PPL={epoch_vq_perplexity/denom:.2f}"
        )
        tb_writer.add_scalar("stage2_epoch/total_loss", epoch_total / denom, epoch + 1)
        tb_writer.add_scalar("stage2_epoch/answer_loss", epoch_answer / denom, epoch + 1)
        tb_writer.add_scalar("stage2_epoch/rationale_loss", epoch_rationale / denom, epoch + 1)
        tb_writer.add_scalar("stage2_epoch/vq_loss", epoch_vq / denom, epoch + 1)
        tb_writer.add_scalar("stage2_epoch/vq_answer_ratio", epoch_vq_ratio / denom, epoch + 1)
        tb_writer.add_scalar("stage2_epoch/vq_perplexity", epoch_vq_perplexity / denom, epoch + 1)
        if int(getattr(args, "save_each_epoch", 0)) == 1:
            epoch_ckpt_path = os.path.join(args.save_path, f"stage2_vision_epoch{epoch+1}")
            save_stage2_checkpoint(model, args, epoch_ckpt_path)

    return model
