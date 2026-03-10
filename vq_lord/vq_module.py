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
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        # 使用 N(0,1) 初始化，而非 uniform(-1/N,1/N)（后者范围过小导致 codebook collapse）
        self.embedding.weight.data.normal_(0, 1.0)

        # 首 batch 数据驱动初始化标记
        self.register_buffer("_initialized", torch.tensor(0, dtype=torch.long))

        if use_ema:
            # EMA 更新所需的缓冲区
            self.register_buffer("ema_cluster_size", torch.zeros(num_embeddings))
            self.register_buffer("ema_embedding_sum", self.embedding.weight.clone())

        # Dead code 检测：记录每个 code 连续未被选中的步数
        self.register_buffer("_code_idle_steps", torch.zeros(num_embeddings, dtype=torch.long))
        self.dead_code_threshold = 100  # 连续 100 步未使用则重置

    def _codebook_weight_for_compute(self, ref: torch.Tensor) -> torch.Tensor:
        """按输入特征的 dtype/device 暂时投影 codebook，避免 matmul dtype mismatch。"""
        weight = self.embedding.weight
        if weight.device != ref.device or weight.dtype != ref.dtype:
            weight = weight.to(device=ref.device, dtype=ref.dtype)
        return weight
        
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

        # ===== 首 batch 数据驱动初始化 =====
        # 用实际视觉特征初始化 codebook，避免尺度不匹配导致 collapse
        if self.training and self._initialized.item() == 0:
            self._init_codebook_from_data(z_flat)
            self._initialized.fill_(1)

        codebook_weight = self._codebook_weight_for_compute(z_flat)
        
        # 计算到每个 codebook embedding 的距离
        # [batch * seq_len, num_embeddings]
        distances = (
            torch.sum(z_flat ** 2, dim=1, keepdim=True)
            + torch.sum(codebook_weight ** 2, dim=1)
            - 2 * torch.matmul(z_flat, codebook_weight.t())
        )
        
        # 找到最近的 code
        indices = torch.argmin(distances, dim=1)
        
        # 获取量化后的嵌入
        quantized = F.embedding(indices, codebook_weight).view(batch_size, seq_len, dim)
        
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

            # Dead code restart：将长期未使用的 code 重置为当前 batch 中的随机特征
            self._restart_dead_codes(z_flat, indices)
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
        z_update = z_flat.to(dtype=self.embedding.weight.dtype)

        # One-hot 编码
        encodings = F.one_hot(indices, self.num_embeddings).to(dtype=self.embedding.weight.dtype)
        
        # 更新聚类大小
        cluster_size = encodings.sum(0).to(dtype=self.ema_cluster_size.dtype)
        self.ema_cluster_size.data.mul_(self.ema_decay).add_(
            cluster_size, alpha=1 - self.ema_decay
        )
        
        # 更新嵌入和
        embedding_sum = (encodings.t() @ z_update).to(dtype=self.ema_embedding_sum.dtype)
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
    
    @torch.no_grad()
    def _init_codebook_from_data(self, z_flat: torch.Tensor):
        """用首 batch 的实际视觉特征初始化 codebook（随机采样 + 扰动）。"""
        n_data = z_flat.shape[0]
        n_codes = self.num_embeddings

        if n_data >= n_codes:
            # 数据够多：随机采样 n_codes 个不重复的特征向量
            perm = torch.randperm(n_data, device=z_flat.device)[:n_codes]
            self.embedding.weight.data.copy_(z_flat[perm])
        else:
            # 数据不够：循环采样 + 添加小扰动避免重复
            repeats = (n_codes + n_data - 1) // n_data
            pool = z_flat.repeat(repeats, 1)[:n_codes]
            noise = torch.randn_like(pool) * pool.std() * 0.05
            self.embedding.weight.data.copy_(pool + noise)

        # 同步 EMA 缓冲区
        if self.use_ema:
            self.ema_embedding_sum.data.copy_(self.embedding.weight.data)
            self.ema_cluster_size.fill_(1.0)  # 初始假设每个 code 被使用过 1 次

        used = min(n_data, n_codes)
        print(f"[VQ] 首 batch 数据驱动初始化 codebook: "
              f"{n_data} 个特征 → {n_codes} 个 code (unique={used})")

    @torch.no_grad()
    def _restart_dead_codes(self, z_flat: torch.Tensor, indices: torch.Tensor):
        """将长期未使用的 dead code 重置为当前 batch 中的随机特征向量。"""
        # 统计本次使用的 code
        used_mask = torch.zeros(self.num_embeddings, dtype=torch.bool, device=z_flat.device)
        used_mask.scatter_(0, indices, True)

        # 更新 idle 计数
        self._code_idle_steps[used_mask] = 0
        self._code_idle_steps[~used_mask] += 1

        # 找到 dead codes
        dead_mask = self._code_idle_steps >= self.dead_code_threshold
        n_dead = dead_mask.sum().item()
        if n_dead == 0:
            return

        # 从当前 batch 随机采样替换 dead codes
        dead_indices = dead_mask.nonzero(as_tuple=False).squeeze(-1)
        n_data = z_flat.shape[0]
        replace_idx = torch.randint(0, n_data, (n_dead,), device=z_flat.device)
        noise = torch.randn(n_dead, z_flat.shape[1], device=z_flat.device) * 0.01
        self.embedding.weight.data[dead_indices] = z_flat[replace_idx] + noise

        # 同步 EMA 缓冲区
        if self.use_ema:
            self.ema_embedding_sum.data[dead_indices] = self.embedding.weight.data[dead_indices]
            self.ema_cluster_size[dead_indices] = 1.0

        # 重置 idle 计数
        self._code_idle_steps[dead_indices] = 0

        print(f"[VQ] Dead code restart: 重置了 {n_dead} 个 dead codes "
              f"(threshold={self.dead_code_threshold})")

    def get_codebook_logits(self, z: torch.Tensor) -> torch.Tensor:
        """
        仅计算 logits，不进行量化
        用于蒸馏时获取软标签
        """
        batch_size, seq_len, dim = z.shape
        z_flat = z.reshape(-1, dim)
        codebook_weight = self._codebook_weight_for_compute(z_flat)
        
        distances = (
            torch.sum(z_flat ** 2, dim=1, keepdim=True)
            + torch.sum(codebook_weight ** 2, dim=1)
            - 2 * torch.matmul(z_flat, codebook_weight.t())
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
