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
from typing import Tuple, Optional


class VectorQuantizer(nn.Module):
    """
    向量量化模块 (Vector Quantization)
    
    将连续的视觉特征映射到离散的 codebook 索引，
    使其具有类似文本 token 的 logit，可用于 LoRD 蒸馏。
    
    参数:
        num_embeddings: codebook 大小 (离散 token 数量)
        embedding_dim: 每个 code 的维度
        commitment_cost: commitment loss 的权重
        use_ema: 是否使用指数移动平均更新 codebook
        ema_decay: EMA 衰减率
    """
    
    def __init__(
        self,
        num_embeddings: int = 8192,
        embedding_dim: int = 1024,
        commitment_cost: float = 0.25,
        use_ema: bool = True,
        ema_decay: float = 0.99,
    ):
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        
        # Codebook: 离散 token 的嵌入表
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)
        
        if use_ema:
            # EMA 更新所需的缓冲区
            self.register_buffer("ema_cluster_size", torch.zeros(num_embeddings))
            self.register_buffer("ema_embedding_sum", self.embedding.weight.clone())
        
    def forward(
        self, 
        z: torch.Tensor,
        return_logits: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播
        
        Args:
            z: 输入特征 [batch, seq_len, embedding_dim]
            return_logits: 是否返回到各个 code 的距离 logits
            
        Returns:
            quantized: 量化后的特征 [batch, seq_len, embedding_dim]
            indices: 量化索引 [batch, seq_len]
            loss: VQ 损失 (commitment + codebook loss)
            logits: 到各个 code 的距离 logits [batch, seq_len, num_embeddings]
        """
        batch_size, seq_len, dim = z.shape
        
        # 展平为 [batch * seq_len, embedding_dim]
        z_flat = z.reshape(-1, dim)
        
        # 计算到每个 codebook embedding 的距离
        # [batch * seq_len, num_embeddings]
        distances = (
            torch.sum(z_flat ** 2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight ** 2, dim=1)
            - 2 * torch.matmul(z_flat, self.embedding.weight.t())
        )
        
        # 找到最近的 code
        indices = torch.argmin(distances, dim=1)
        
        # 获取量化后的嵌入
        quantized = self.embedding(indices).view(batch_size, seq_len, dim)
        
        # 计算损失
        if self.training:
            if self.use_ema:
                # EMA 更新 codebook
                self._ema_update(z_flat, indices)
                # 只有 commitment loss
                loss = self.commitment_cost * F.mse_loss(z, quantized.detach())
            else:
                # Codebook loss + Commitment loss
                codebook_loss = F.mse_loss(quantized, z.detach())
                commitment_loss = self.commitment_cost * F.mse_loss(z, quantized.detach())
                loss = codebook_loss + commitment_loss
        else:
            loss = torch.tensor(0.0, device=z.device)
        
        # Straight-through estimator: 梯度直接通过
        quantized = z + (quantized - z).detach()
        
        # 计算 logits (负距离，可用于 softmax)
        if return_logits:
            logits = -distances.view(batch_size, seq_len, self.num_embeddings)
        else:
            logits = None
        
        indices = indices.view(batch_size, seq_len)
        
        return quantized, indices, loss, logits
    
    def _ema_update(self, z_flat: torch.Tensor, indices: torch.Tensor):
        """使用 EMA 更新 codebook"""
        # One-hot 编码
        encodings = F.one_hot(indices, self.num_embeddings).float()
        
        # 更新聚类大小
        cluster_size = encodings.sum(0)
        self.ema_cluster_size.data.mul_(self.ema_decay).add_(
            cluster_size, alpha=1 - self.ema_decay
        )
        
        # 更新嵌入和
        embedding_sum = encodings.t() @ z_flat
        self.ema_embedding_sum.data.mul_(self.ema_decay).add_(
            embedding_sum, alpha=1 - self.ema_decay
        )
        
        # Laplace 平滑
        n = self.ema_cluster_size.sum()
        cluster_size = (
            (self.ema_cluster_size + 1e-5)
            / (n + self.num_embeddings * 1e-5)
            * n
        )
        
        # 更新嵌入
        self.embedding.weight.data.copy_(
            self.ema_embedding_sum / cluster_size.unsqueeze(1)
        )
    
    def get_codebook_logits(self, z: torch.Tensor) -> torch.Tensor:
        """
        仅计算 logits，不进行量化
        用于蒸馏时获取软标签
        """
        batch_size, seq_len, dim = z.shape
        z_flat = z.reshape(-1, dim)
        
        distances = (
            torch.sum(z_flat ** 2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight ** 2, dim=1)
            - 2 * torch.matmul(z_flat, self.embedding.weight.t())
        )
        
        logits = -distances.view(batch_size, seq_len, self.num_embeddings)
        return logits


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
        self.freeze_vision_tower = freeze_vision_tower
        
        # 获取视觉编码器的输出维度
        # LLaVA 的 vision_tower 通常输出 1024 或 1152 维
        self.hidden_size = getattr(
            vision_tower.config, 
            "hidden_size", 
            1024
        )
        
        # VQ 层
        self.vq = VectorQuantizer(
            num_embeddings=num_embeddings,
            embedding_dim=self.hidden_size,
            commitment_cost=commitment_cost,
        )
        
        if freeze_vision_tower:
            for param in self.vision_tower.parameters():
                param.requires_grad = False
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        return_vq_logits: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
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
                vision_features = self.vision_tower(pixel_values)
        else:
            vision_features = self.vision_tower(pixel_values)
        
        # 处理不同的输出格式
        if hasattr(vision_features, "last_hidden_state"):
            vision_features = vision_features.last_hidden_state
        elif isinstance(vision_features, tuple):
            vision_features = vision_features[0]
        
        # VQ 离散化
        quantized, indices, vq_loss, logits = self.vq(
            vision_features, 
            return_logits=return_vq_logits
        )
        
        return quantized, indices, vq_loss, logits
    
    def get_vision_logits(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        只获取 VQ logits，用于蒸馏
        """
        with torch.no_grad():
            vision_features = self.vision_tower(pixel_values)
            if hasattr(vision_features, "last_hidden_state"):
                vision_features = vision_features.last_hidden_state
            elif isinstance(vision_features, tuple):
                vision_features = vision_features[0]
        
        return self.vq.get_codebook_logits(vision_features)


def load_pretrained_vqgan_codebook(
    vq_module: VectorQuantizer,
    checkpoint_path: str,
) -> VectorQuantizer:
    """
    加载预训练的 VQGAN codebook
    
    Args:
        vq_module: VectorQuantizer 实例
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
    vq = VectorQuantizer(num_embeddings=8192, embedding_dim=1024)
    
    # 模拟视觉特征 [batch=2, seq_len=196, dim=1024]
    fake_features = torch.randn(2, 196, 1024)
    
    quantized, indices, loss, logits = vq(fake_features)
    
    print(f"Input shape: {fake_features.shape}")
    print(f"Quantized shape: {quantized.shape}")
    print(f"Indices shape: {indices.shape}")
    print(f"VQ Loss: {loss.item():.4f}")
    print(f"Logits shape: {logits.shape}")
    print(f"Logits 可用于 softmax 得到概率分布，用于 LoRD 蒸馏")
