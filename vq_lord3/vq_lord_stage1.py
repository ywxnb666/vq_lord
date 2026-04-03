"""Stage1 training logic for VQ-LoRD3."""

import os

import torch
import torch.nn.functional as F
from tqdm import tqdm

from train_vq_lord3 import maybe_set_dataloader_epoch, save_vq_codebook


def train_stage1_vq(model, dataloader, args, tb_writer):
    """
    阶段 1: VQ Codebook 预训练
    
    目标：参考 VQGAN 的 encode->quantize->decode 闭环，
    在固定 vision tower 的前提下训练出可重建原始视觉特征的 VQ stack。
    """
    print("\n" + "=" * 50)
    print("阶段 1: VQ Codebook 预训练")
    print("=" * 50)

    model.eval()
    vq_encoder = model.vq_vision_encoder

    for param in model.parameters():
        param.requires_grad = False
    for param in vq_encoder.stage1_parameters():
        if torch.is_floating_point(param):
            param.requires_grad = True

    vq_encoder.pre_quant.train()
    vq_encoder.vq.train()
    vq_encoder.post_quant.train()
    vq_encoder.vision_tower.eval()
    
    # 统计可训练参数
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"可训练参数量: {trainable_params:,}")

    stage1_lr = args.stage1_lr if args.stage1_lr > 0 else args.lr * 5
    print(
        f"[Stage1] lr={stage1_lr}, recon_w={args.stage1_recon_weight}, "
        f"cos_w={args.stage1_cosine_weight}, vq_w={args.stage1_vq_weight}, "
        f"grad_clip={args.stage1_grad_clip}"
    )

    # 对齐 taming VQGAN：Stage1 使用 Adam(无 weight decay)，避免 codebook 被 AdamW 拉向 0。
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=stage1_lr,
        betas=(0.5, 0.9),
    )
    
    log_dir = os.path.join(args.save_path, "logs")
    os.makedirs(log_dir, exist_ok=True)
    metrics_path = os.path.join(log_dir, "vq_metrics.csv")

    if not os.path.exists(metrics_path):
        with open(metrics_path, "w", encoding="utf-8") as f:
            f.write(
                "stage,step,total_loss,recon_loss,cosine_loss,vq_loss,loss_ema,"
                "codebook_used,perplexity,dead_code_resets,dead_code_count\n"
            )

    global_step = 0
    loss_ema = None
    for epoch in range(args.epochs):
        maybe_set_dataloader_epoch(dataloader, epoch)
        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_cosine = 0.0
        epoch_vq = 0.0
        for batch in tqdm(dataloader, desc=f"VQ预训练 Epoch {epoch+1}"):
            global_step += 1
            
            pixel_values = batch["pixel_values"].to(args.device)

            if pixel_values.dim() == 5:
                batch_size, num_patches, channels, height, width = pixel_values.shape
                pixel_values = pixel_values.view(batch_size * num_patches, channels, height, width)
            
            optimizer.zero_grad(set_to_none=True)
            stage1_out = vq_encoder.stage1_forward(pixel_values)

            target_features = stage1_out["target_features"]
            reconstructed_features = stage1_out["reconstructed_features"]
            vq_loss = stage1_out["vq_loss"]
            vq_indices = stage1_out.get("indices")

            recon_loss = F.mse_loss(reconstructed_features, target_features)
            cosine_loss = 1.0 - F.cosine_similarity(
                reconstructed_features.float(),
                target_features.float(),
                dim=-1,
            ).mean()
            total_loss = (
                args.stage1_recon_weight * recon_loss
                + args.stage1_cosine_weight * cosine_loss
                + args.stage1_vq_weight * vq_loss
            )

            if not torch.isfinite(total_loss):
                print(
                    f"[Stage1][Warn] step={global_step} 出现非有限损失，"
                    f"跳过该 batch: recon={recon_loss.item()}, cos={cosine_loss.item()}, "
                    f"vq={vq_loss.item() if isinstance(vq_loss, torch.Tensor) else vq_loss}"
                )
                optimizer.zero_grad(set_to_none=True)
                continue

            total_loss.backward()
            grad_norm = None
            if args.stage1_grad_clip > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_([
                    p for p in model.parameters() if p.requires_grad
                ], max_norm=args.stage1_grad_clip)
                if isinstance(grad_norm, torch.Tensor) and not torch.isfinite(grad_norm):
                    print(f"[Stage1][Warn] step={global_step} 梯度范数非有限，跳过 optimizer.step()")
                    optimizer.zero_grad(set_to_none=True)
                    continue
            optimizer.step()

            total_loss_val = total_loss.item()
            recon_loss_val = recon_loss.item()
            cosine_loss_val = cosine_loss.item()
            vq_loss_val = vq_loss.item() if vq_loss is not None else 0.0

            epoch_loss += total_loss_val
            epoch_recon += recon_loss_val
            epoch_cosine += cosine_loss_val
            epoch_vq += vq_loss_val
            loss_ema = total_loss_val if loss_ema is None else (0.95 * loss_ema + 0.05 * total_loss_val)

            if vq_indices is not None:
                codebook_used = torch.unique(vq_indices).numel()
            else:
                codebook_used = 0

            perplexity = vq_encoder.vq_cache.get("perplexity")
            if isinstance(perplexity, torch.Tensor):
                perplexity_val = float(perplexity.detach().item())
            else:
                perplexity_val = 0.0
            dead_code_resets = int(vq_encoder.vq_cache.get("dead_code_resets", 0))
            dead_code_count = int(vq_encoder.vq_cache.get("dead_code_count", 0))

            if global_step % args.log_step == 0:
                print(
                    f"Step {global_step}, Total: {total_loss_val:.4f}, "
                    f"Recon: {recon_loss_val:.4f}, Cos: {cosine_loss_val:.4f}, VQ: {vq_loss_val:.4f}, "
                    f"used={codebook_used}, ppl={perplexity_val:.2f}, reset={dead_code_resets}, dead={dead_code_count}"
                )
                tb_writer.add_scalar("stage1/total_loss", total_loss_val, global_step)
                tb_writer.add_scalar("stage1/recon_loss", recon_loss_val, global_step)
                tb_writer.add_scalar("stage1/cosine_loss", cosine_loss_val, global_step)
                tb_writer.add_scalar("stage1/vq_loss", vq_loss_val, global_step)
                tb_writer.add_scalar("stage1/loss_ema", loss_ema, global_step)
                tb_writer.add_scalar("stage1/codebook_used", codebook_used, global_step)
                tb_writer.add_scalar("stage1/perplexity", perplexity_val, global_step)
                tb_writer.add_scalar("stage1/dead_code_resets", dead_code_resets, global_step)
                tb_writer.add_scalar("stage1/dead_code_count", dead_code_count, global_step)
                with open(metrics_path, "a", encoding="utf-8") as f:
                    f.write(
                        f"stage1,{global_step},{total_loss_val:.6f},{recon_loss_val:.6f},"
                        f"{cosine_loss_val:.6f},{vq_loss_val:.6f},{loss_ema:.6f},{codebook_used},"
                        f"{perplexity_val:.6f},{dead_code_resets},{dead_code_count}\n"
                    )
        
        num_batches = len(dataloader)
        avg_loss = epoch_loss / num_batches
        print(
            f"Epoch {epoch+1} 平均损失: Total={avg_loss:.4f}, "
            f"Recon={epoch_recon / num_batches:.4f}, Cos={epoch_cosine / num_batches:.4f}, "
            f"VQ={epoch_vq / num_batches:.4f}"
        )
        if int(getattr(args, "save_each_epoch", 0)) == 1:
            epoch_dir = os.path.join(args.save_path, f"stage1_vq_epoch{epoch+1}")
            epoch_codebook_path = os.path.join(epoch_dir, "vq_codebook.pt")
            save_vq_codebook(model, epoch_codebook_path)
            print(f"Stage1 Epoch 检查点已保存至 {epoch_dir}")
    
    return model


