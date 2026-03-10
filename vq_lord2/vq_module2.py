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
from typing import Tuple, Optional


class VectorQuantizer2(nn.Module):
    """
    参考taming-transformers/taming/modules/vqvae/quantize.py的VectorQuantizer2实现
    主要修改部分：

    """
    # NOTE: due to a bug the beta term was applied to the wrong term. for
    # backwards compatibility we use the buggy version by default, but you can
    # specify legacy=False to fix it.
    def __init__(
        self, 
        num_embeddings: int = 8192, 
        embedding_dim: int = 1024, 
        commitment_cost: float = 0.25, 
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

        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.num_embeddings, 1.0 / self.num_embeddings)

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
        z_q = self.embedding(min_encoding_indices).view(*flat_shape, self.embedding_dim)
        perplexity = None
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
        
        # VQ 层
        self.vq = VectorQuantizer2(
            num_embeddings=num_embeddings,
            embedding_dim=self.hidden_size,
            commitment_cost=commitment_cost,
        )
        
        if freeze_vision_tower:
            for param in self.vision_tower.parameters():
                param.requires_grad = False

        self.vq_cache = {
            "loss": None,
            "logits": None,
            "indices": None,
            "features": None,
        }

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

        return quantized, indices, vq_loss, logits
    
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
        # 通过视觉编码器
        if self.freeze_vision_tower:
            with torch.no_grad():
                tower_output = self.vision_tower(pixel_values, *args, **kwargs)
        else:
            tower_output = self.vision_tower(pixel_values, *args, **kwargs)

        vision_features = self._extract_hidden_states(tower_output)
        quantized, indices, vq_loss, logits = self.quantize_features(
            vision_features,
            return_vq_logits=return_vq_logits,
        )

        if return_details:
            return quantized, indices, vq_loss, logits

        return self._replace_hidden_states(tower_output, quantized)
    
    def get_vision_logits(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        只获取 VQ logits，用于蒸馏
        """
        with torch.no_grad():
            tower_output = self.vision_tower(pixel_values)
            vision_features = self._extract_hidden_states(tower_output)
        
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
