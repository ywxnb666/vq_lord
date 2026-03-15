"""
======================================================================
VQ_MODULE ---

Vector Quantization 模块，用于将视觉特征离散化为可蒸馏的 tokens

核心功能:
1. VectorQuantizer: 将连续特征量化为离散 codebook 索引
2. VQVisionEncoder: 包装 Vision Encoder，添加 VQ 层

    Author: VQ-LoRD Project
    Created: January 2026
======================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from typing import Dict, Tuple, Optional


class VectorQuantizer2(nn.Module):
    """
    参考taming-transformers/taming/modules/vqvae/quantize.py的VectorQuantizer2实现
    主要修改部分：

    """
    def __init__(
        self, 
        num_embeddings: int = 8192, 
        embedding_dim: int = 1024, 
        commitment_cost: float = 0.25,
        dead_code_threshold: float = 0.0,
        usage_decay: float = 0.99,
        dead_code_reset_interval: int = 0,
        remap=None, 
        unknown_index="random",
        sane_index_shape=False, 
        legacy=True
    ):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.beta = commitment_cost
        self.legacy = legacy
        self.dead_code_threshold = max(0.0, float(dead_code_threshold))
        self.usage_decay = float(usage_decay)
        self.dead_code_reset_interval = int(dead_code_reset_interval)

        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.num_embeddings, 1.0 / self.num_embeddings)
        self.register_buffer("ema_cluster_size", torch.zeros(self.num_embeddings))
        self.register_buffer("usage_update_steps", torch.zeros(1, dtype=torch.long))
        self.register_buffer("last_dead_code_resets", torch.zeros(1, dtype=torch.long))

        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index # "random" or "extra" or integer
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed+1
            print(f"Remapping {self.num_embeddings} indices to {self.re_embed} indices. "
                  f"Using {self.unknown_index} for unknown indices.")
        else:
            self.re_embed = num_embeddings

        self.sane_index_shape = sane_index_shape

    @torch.no_grad()
    def _update_usage_and_maybe_refresh_codes(
        self,
        min_encoding_indices: torch.Tensor,
        z_flattened: torch.Tensor,
    ) -> torch.Tensor:
        counts = torch.bincount(
            min_encoding_indices,
            minlength=self.num_embeddings,
        ).to(self.ema_cluster_size.dtype)

        self.usage_update_steps += 1
        self.last_dead_code_resets.zero_()
        self.ema_cluster_size.mul_(self.usage_decay).add_(counts, alpha=1.0 - self.usage_decay)

        if (
            not self.training
            or self.dead_code_threshold <= 0.0
            or self.dead_code_reset_interval <= 0
            or int(self.usage_update_steps.item()) % self.dead_code_reset_interval != 0
            or z_flattened.numel() == 0
        ):
            return counts

        # 仅重置“本 batch 未命中且 EMA 使用率过低”的 code，避免扰动活跃 code。
        dead_mask = (self.ema_cluster_size < self.dead_code_threshold) & (counts <= 0)
        num_dead = int(dead_mask.sum().item())
        if num_dead <= 0:
            return counts

        sample_ids = torch.randint(
            low=0,
            high=z_flattened.shape[0],
            size=(num_dead,),
            device=z_flattened.device,
        )
        refreshed = z_flattened[sample_ids].to(dtype=self.embedding.weight.dtype)
        refreshed = refreshed + 1e-3 * torch.randn_like(refreshed)
        self.embedding.weight.data[dead_mask] = refreshed
        self.ema_cluster_size[dead_mask] = max(1.0, self.dead_code_threshold)
        self.last_dead_code_resets.fill_(num_dead)

        return counts

    def remap_to_used(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        match = (inds[:,:,None]==used[None,None,...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2)<1
        if self.unknown_index == "random":
            new[unknown]=torch.randint(0,self.re_embed,size=new[unknown].shape).to(device=new.device)
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        if self.re_embed > self.used.shape[0]: # extra token
            inds[inds>=self.used.shape[0]] = 0 # simply set to zero
        back=torch.gather(used[None,:][inds.shape[0]*[0],:], 1, inds)
        return back.reshape(ishape)

    def forward(self, z, temp=None, rescale_logits=False, return_logits=False):
        assert temp is None or temp==1.0, "Only for interface compatible with Gumbel"
        assert rescale_logits==False, "Only for interface compatible with Gumbel"
        if z.dim() == 4:
            input_kind = "image"
            z_work = rearrange(z, 'b c h w -> b h w c').contiguous()
            flat_shape = z_work.shape[:-1]
        elif z.dim() == 3:
            input_kind = "sequence"
            z_work = z.contiguous()
            flat_shape = z_work.shape[:-1]
        else:
            raise ValueError(f"VectorQuantizer2 期望 3D/4D 输入，实际得到 shape={tuple(z.shape)}")

        z_flattened = z_work.view(-1, self.embedding_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, rearrange(self.embedding.weight, 'n d -> d n'))

        min_encoding_indices = torch.argmin(d, dim=1)
        counts = self._update_usage_and_maybe_refresh_codes(min_encoding_indices, z_flattened)
        z_q = self.embedding(min_encoding_indices).view(*flat_shape, self.embedding_dim)
        avg_probs = counts / counts.sum().clamp(min=1.0)
        perplexity = torch.exp(-(avg_probs * torch.log(avg_probs + 1e-10)).sum())
        min_encodings = None

        # compute loss for embedding
        if not self.legacy:
            loss = self.beta * torch.mean((z_q.detach()-z_work)**2) + \
                   torch.mean((z_q - z_work.detach()) ** 2)
        else:
            loss = torch.mean((z_q.detach()-z_work)**2) + self.beta * \
                   torch.mean((z_q - z_work.detach()) ** 2)

        # preserve gradients
        z_q = z_work + (z_q - z_work).detach()

        # reshape back to match original input shape
        if input_kind == "image":
            z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()
        else:
            z_q = z_q.contiguous()

        if self.remap is not None:
            min_encoding_indices = min_encoding_indices.reshape(z_work.shape[0],-1) # add batch axis
            min_encoding_indices = self.remap_to_used(min_encoding_indices)
            min_encoding_indices = min_encoding_indices.reshape(-1,1) # flatten

        if self.sane_index_shape:
            if input_kind == "image":
                min_encoding_indices = min_encoding_indices.reshape(
                    z_q.shape[0], z_q.shape[2], z_q.shape[3]
                )
            else:
                min_encoding_indices = min_encoding_indices.reshape(z_q.shape[0], z_q.shape[1])

        if return_logits:
            logits = -d.view(*flat_shape, self.num_embeddings)
            return z_q, loss, (perplexity, min_encodings, min_encoding_indices), logits

        return z_q, loss, (perplexity, min_encodings, min_encoding_indices)

    def get_codebook_logits(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() == 4:
            z_work = rearrange(z, 'b c h w -> b h w c').contiguous()
            flat_shape = z_work.shape[:-1]
        elif z.dim() == 3:
            z_work = z.contiguous()
            flat_shape = z_work.shape[:-1]
        else:
            raise ValueError(f"VectorQuantizer2 期望 3D/4D 输入，实际得到 shape={tuple(z.shape)}")

        z_flattened = z_work.view(-1, self.embedding_dim)
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, rearrange(self.embedding.weight, 'n d -> d n'))
        return -d.view(*flat_shape, self.num_embeddings)

    def get_codebook_entry(self, indices, shape):
        # shape specifying (batch, height, width, channel)
        if self.remap is not None:
            indices = indices.reshape(shape[0],-1) # add batch axis
            indices = self.unmap_to_all(indices)
            indices = indices.reshape(-1) # flatten again

        # get quantized latent vectors
        z_q = self.embedding(indices)

        if shape is not None:
            z_q = z_q.view(shape)
            # reshape back to match original input shape
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q

class VQVisionEncoder(nn.Module):
    """
    带有 VQ 层的视觉编码器包装器
    
    包装 LLaVA 的 Vision Tower，添加 VQ 离散化层，
    使视觉特征具有可蒸馏的 token logits。
    
    Args:
        vision_tower: 原始的视觉编码器 (如 CLIP ViT)
        num_embeddings: VQ codebook 大小
        freeze_vision_tower: 是否冻结原始视觉编码器
    """
    
    def __init__(
        self,
        vision_tower: nn.Module,
        num_embeddings: int = 8192,
        commitment_cost: float = 0.25,
        legacy: bool = False,
        dead_code_threshold: float = 0.0,
        usage_decay: float = 0.99,
        dead_code_reset_interval: int = 0,
        freeze_vision_tower: bool = False,
    ):
        super().__init__()
        
        self.vision_tower = vision_tower
        self.config = getattr(vision_tower, "config", None)
        self.freeze_vision_tower = freeze_vision_tower
        
        # 获取视觉编码器的输出维度
        # LLaVA 的 vision_tower 通常输出 1024 或 1152 维
        self.hidden_size = getattr(
            vision_tower.config, 
            "hidden_size", 
            1024
        )

        # 参考 VQGAN 的 quant_conv/post_quant_conv，在量化前后加入可学习投影。
        self.pre_quant = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, self.hidden_size),
        )
        self.post_quant = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )
        self.compute_dtype = torch.float32
        
        # VQ 层
        self.vq = VectorQuantizer2(
            num_embeddings=num_embeddings,
            embedding_dim=self.hidden_size,
            commitment_cost=commitment_cost,
            legacy=legacy,
            dead_code_threshold=dead_code_threshold,
            usage_decay=usage_decay,
            dead_code_reset_interval=dead_code_reset_interval,
        )
        
        if freeze_vision_tower:
            for param in self.vision_tower.parameters():
                param.requires_grad = False

        self.vq_cache = {
            "loss": None,
            "logits": None,
            "indices": None,
            "features": None,
            "pre_quant_features": None,
            "reconstructed_features": None,
            "target_features": None,
            "perplexity": None,
            "dead_code_resets": 0,
            "dead_code_count": 0,
        }

    def get_vq_state(self) -> Dict[str, Dict[str, torch.Tensor]]:
        return {
            "vq": self.vq.state_dict(),
            "pre_quant": self.pre_quant.state_dict(),
            "post_quant": self.post_quant.state_dict(),
        }

    def load_vq_state(self, state: Dict[str, Dict[str, torch.Tensor]]) -> None:
        if not isinstance(state, dict):
            raise TypeError("VQ state 必须是 dict")

        if "vq" in state:
            self.vq.load_state_dict(state["vq"], strict=False)
        if "pre_quant" in state:
            self.pre_quant.load_state_dict(state["pre_quant"], strict=False)
        if "post_quant" in state:
            self.post_quant.load_state_dict(state["post_quant"], strict=False)

    def stage1_parameters(self):
        for module in (self.pre_quant, self.vq, self.post_quant):
            yield from module.parameters()

    def _align_vq_modules(self, reference: torch.Tensor) -> None:
        if not isinstance(reference, torch.Tensor):
            return
        target_device = reference.device
        for module in (self.pre_quant, self.vq, self.post_quant):
            module.to(device=target_device, dtype=self.compute_dtype)

    def _prepare_vq_input(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.dtype]:
        original_dtype = features.dtype
        compute_features = features.to(dtype=self.compute_dtype)
        return compute_features, original_dtype

    def encode_features(self, pixel_values: torch.Tensor, *args, **kwargs):
        if self.freeze_vision_tower:
            with torch.no_grad():
                tower_output = self.vision_tower(pixel_values, *args, **kwargs)
        else:
            tower_output = self.vision_tower(pixel_values, *args, **kwargs)
        return tower_output, self._extract_hidden_states(tower_output)

    def _extract_hidden_states(self, tower_output):
        if hasattr(tower_output, "last_hidden_state"):
            return tower_output.last_hidden_state
        if isinstance(tower_output, tuple):
            return tower_output[0]
        return tower_output

    def _replace_hidden_states(self, tower_output, quantized_features):
        if hasattr(tower_output, "last_hidden_state"):
            tower_output.last_hidden_state = quantized_features
            return tower_output
        if isinstance(tower_output, tuple):
            return (quantized_features,) + tower_output[1:]
        return quantized_features

    def quantize_features(
        self,
        vision_features: torch.Tensor,
        return_vq_logits: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if return_vq_logits:
            quantized, vq_loss, info, logits = self.vq(
                vision_features,
                return_logits=True,
            )
        else:
            quantized, vq_loss, info = self.vq(
                vision_features,
                return_logits=False,
            )
            logits = None

        indices = info[2]
        if isinstance(indices, torch.Tensor) and vision_features.dim() == 3 and indices.dim() == 1:
            indices = indices.view(vision_features.shape[0], vision_features.shape[1])

        self.vq_cache["loss"] = vq_loss
        self.vq_cache["logits"] = logits
        self.vq_cache["indices"] = indices
        self.vq_cache["features"] = quantized
        self.vq_cache["perplexity"] = info[0] if isinstance(info, tuple) else None

        dead_code_resets = int(getattr(self.vq, "last_dead_code_resets", torch.zeros(1)).item())
        self.vq_cache["dead_code_resets"] = dead_code_resets

        if self.vq.dead_code_threshold > 0:
            dead_code_count = int((self.vq.ema_cluster_size < self.vq.dead_code_threshold).sum().item())
        else:
            dead_code_count = 0
        self.vq_cache["dead_code_count"] = dead_code_count

        return quantized, indices, vq_loss, logits

    def stage1_forward(
        self,
        pixel_values: torch.Tensor,
        *args,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            _, target_features = self.encode_features(pixel_values, *args, **kwargs)

        target_features = target_features.detach()
        self._align_vq_modules(target_features)
        target_features_compute, _ = self._prepare_vq_input(target_features)
        pre_quant_features = self.pre_quant(target_features_compute)
        quantized, indices, vq_loss, logits = self.quantize_features(
            pre_quant_features,
            return_vq_logits=True,
        )
        reconstructed_features = self.post_quant(quantized)

        self.vq_cache["pre_quant_features"] = pre_quant_features
        self.vq_cache["reconstructed_features"] = reconstructed_features
        self.vq_cache["target_features"] = target_features_compute
        self.vq_cache["features"] = reconstructed_features

        return {
            "target_features": target_features_compute,
            "pre_quant_features": pre_quant_features,
            "quantized_features": quantized,
            "reconstructed_features": reconstructed_features,
            "indices": indices,
            "vq_loss": vq_loss,
            "logits": logits,
        }
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        return_vq_logits: bool = True,
        return_details: bool = False,
        *args,
        **kwargs,
    ):
        """
        前向传播
        
        Args:
            pixel_values: 输入图像 [batch, channels, height, width]
            return_vq_logits: 是否返回 VQ logits
            
        Returns:
            quantized_features: 量化后的视觉特征
            indices: VQ token 索引
            vq_loss: VQ 损失
            logits: VQ logits (可用于蒸馏)
        """
        tower_output, vision_features = self.encode_features(pixel_values, *args, **kwargs)
        self._align_vq_modules(vision_features)
        vision_features_compute, original_dtype = self._prepare_vq_input(vision_features)
        pre_quant_features = self.pre_quant(vision_features_compute)
        quantized, indices, vq_loss, logits = self.quantize_features(
            pre_quant_features,
            return_vq_logits=return_vq_logits,
        )
        reconstructed_features = self.post_quant(quantized)
        reconstructed_output = reconstructed_features.to(dtype=original_dtype)

        self.vq_cache["pre_quant_features"] = pre_quant_features
        self.vq_cache["reconstructed_features"] = reconstructed_output
        self.vq_cache["target_features"] = vision_features.detach()
        self.vq_cache["features"] = reconstructed_output

        if return_details:
            return reconstructed_output, indices, vq_loss, logits

        return self._replace_hidden_states(tower_output, reconstructed_output)
    
    def get_vision_logits(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        只获取 VQ logits，用于蒸馏
        """
        with torch.no_grad():
            _, vision_features = self.encode_features(pixel_values)
            self._align_vq_modules(vision_features)
            vision_features, _ = self._prepare_vq_input(vision_features)
            vision_features = self.pre_quant(vision_features)
        
        return self.vq.get_codebook_logits(vision_features)


def load_pretrained_vqgan_codebook(
    vq_module: VectorQuantizer2,
    checkpoint_path: str,
) -> VectorQuantizer2:
    """
    加载预训练的 VQGAN codebook
    
    Args:
        vq_module: VectorQuantizer2 实例
        checkpoint_path: VQGAN 检查点路径
        
    Returns:
        加载了预训练权重的 VQ 模块
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # VQGAN 的权重键可能不同，需要适配
    if "quantize.embedding.weight" in checkpoint:
        vq_module.embedding.weight.data.copy_(
            checkpoint["quantize.embedding.weight"]
        )
    elif "codebook" in checkpoint:
        vq_module.embedding.weight.data.copy_(checkpoint["codebook"])
    else:
        print("Warning: Could not find codebook weights in checkpoint")
    
    return vq_module


# 测试代码
if __name__ == "__main__":
    # 测试 VectorQuantizer
    vq = VectorQuantizer2(num_embeddings=8192, embedding_dim=1024)
    
    # 模拟视觉特征 [batch=2, seq_len=196, dim=1024]
    fake_features = torch.randn(2, 196, 1024)
    
    quantized, loss, info, logits = vq(fake_features, return_logits=True)
    indices = info[2]
    
    print(f"Input shape: {fake_features.shape}")
    print(f"Quantized shape: {quantized.shape}")
    print(f"Indices shape: {indices.shape}")
    print(f"VQ Loss: {loss.item():.4f}")
    print(f"Logits shape: {logits.shape}")
    print(f"Logits 可用于 softmax 得到概率分布，用于 LoRD 蒸馏")
