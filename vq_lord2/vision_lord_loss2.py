"""
======================================================================
VISION_LORD_LOSS ---

视觉 LoRD 损失函数模块

核心功能:
1. VisionLoRDLoss: 综合损失函数，包含文本蒸馏 + 视觉蒸馏 + VQ 损失
2. compute_vision_distillation_loss: 计算视觉特征蒸馏损失

    Author: VQ-LoRD Project
    Created: January 2026
======================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class VisionLoRDLoss(nn.Module):
    """
    VQ-LoRD 综合损失函数
    
    L_total = L_text + alpha * L_vision + beta * L_vq
    
    其中:
    - L_text: 原始 LoRD 文本蒸馏损失 (对比学习 + KL散度)
    - L_vision: 视觉特征蒸馏损失 (VQ logits 的 KL散度)
    - L_vq: VQ 重建损失 (commitment loss)
    
    Args:
        alpha: 视觉损失权重
        beta: VQ 损失权重
        temperature: 蒸馏温度
        use_contrastive: 是否使用对比学习损失
    """
    
    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.25,
        temperature: float = 1.5,
        use_contrastive: bool = True,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature
        self.use_contrastive = use_contrastive
        
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")
    
    def forward(
        self,
        # 文本相关
        student_text_logits: torch.Tensor,
        teacher_text_logits: Optional[torch.Tensor] = None,
        student_old_logits: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        # 视觉相关
        student_vq_logits: Optional[torch.Tensor] = None,
        teacher_vq_logits: Optional[torch.Tensor] = None,
        vq_loss: Optional[torch.Tensor] = None,
        # 对比学习相关
        positive_logits: Optional[torch.Tensor] = None,
        negative_logits: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        计算综合损失
        
        Args:
            student_text_logits: 学生模型文本 logits [bs, seq, vocab]
            teacher_text_logits: 教师模型文本 logits (软标签)
            student_old_logits: 学生模型上一轮的 logits (用于对比)
            text_mask: 文本 attention mask
            student_vq_logits: 学生 VQ logits [bs, num_patches, codebook_size]
            teacher_vq_logits: 教师 VQ logits (或目标分布)
            vq_loss: VQ 模块返回的重建损失
            positive_logits: 正样本 logits (对比学习)
            negative_logits: 负样本 logits (对比学习)
            
        Returns:
            包含各项损失的字典
        """
        losses = {}
        total_loss = torch.tensor(0.0, device=student_text_logits.device)
        
        # 1. 文本蒸馏损失 (原 LoRD)
        if teacher_text_logits is not None:
            l_text_kl = self._compute_text_kl_loss(
                student_text_logits, 
                teacher_text_logits, 
                text_mask
            )
            losses["text_kl"] = l_text_kl
            total_loss = total_loss + l_text_kl
        
        # 2. 对比学习损失
        if self.use_contrastive and positive_logits is not None:
            l_contrastive = self._compute_contrastive_loss(
                student_text_logits,
                positive_logits,
                negative_logits,
                student_old_logits,
                text_mask,
            )
            losses["contrastive"] = l_contrastive
            total_loss = total_loss + l_contrastive
        
        # 3. 视觉蒸馏损失
        if student_vq_logits is not None and teacher_vq_logits is not None:
            l_vision = self._compute_vision_kl_loss(
                student_vq_logits, 
                teacher_vq_logits
            )
            losses["vision_kl"] = l_vision
            total_loss = total_loss + self.alpha * l_vision
        
        # 4. VQ 重建损失
        if vq_loss is not None:
            losses["vq"] = vq_loss
            total_loss = total_loss + self.beta * vq_loss
        
        losses["total"] = total_loss
        return losses
    
    def _compute_text_kl_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """计算文本 KL 散度损失"""
        # 温度缩放
        student_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
        
        # KL 散度
        kl = F.kl_div(student_probs, teacher_probs, reduction="none")
        
        # 应用 mask
        if mask is not None:
            mask = mask.unsqueeze(-1).expand_as(kl)
            kl = kl * mask
            loss = kl.sum() / mask.sum()
        else:
            loss = kl.mean()
        
        # 温度平方缩放 (蒸馏标准做法)
        return loss * (self.temperature ** 2)
    
    def _compute_vision_kl_loss(
        self,
        student_vq_logits: torch.Tensor,
        teacher_vq_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算视觉 VQ logits 的 KL 散度损失
        
        这是 VQ-LoRD 的核心创新：
        将图像特征离散化后，使用 KL 散度蒸馏视觉理解能力
        """
        # 温度缩放
        student_probs = F.log_softmax(student_vq_logits / self.temperature, dim=-1)
        teacher_probs = F.softmax(teacher_vq_logits / self.temperature, dim=-1)
        
        # 对所有 patch 位置计算 KL 散度
        kl = F.kl_div(student_probs, teacher_probs, reduction="batchmean")
        
        return kl * (self.temperature ** 2)
    
    def _compute_contrastive_loss(
        self,
        current_logits: torch.Tensor,
        positive_logits: torch.Tensor,
        negative_logits: Optional[torch.Tensor],
        old_logits: Optional[torch.Tensor],
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        计算对比学习损失 (类似原 LoRD)
        
        目标：让模型更偏好正样本（更好的回答）
        """
        # 计算 log-likelihood
        current_ll = self._masked_log_likelihood(current_logits, mask)
        positive_ll = self._masked_log_likelihood(positive_logits, mask)
        
        # 对比损失：正样本应该有更高的 likelihood
        loss = -torch.mean(positive_ll - current_ll)
        
        if negative_logits is not None:
            negative_ll = self._masked_log_likelihood(negative_logits, mask)
            # 负样本应该有更低的 likelihood
            loss = loss + torch.mean(torch.relu(negative_ll - current_ll + 0.1))
        
        return loss
    
    def _masked_log_likelihood(
        self, 
        logits: torch.Tensor, 
        mask: Optional[torch.Tensor],
        target_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        计算带 mask 的平均 log-likelihood
        
        Args:
            logits: 模型输出的 logits [bs, seq, vocab]
            mask: attention mask [bs, seq]
            target_tokens: 目标 token ids [bs, seq]，如果为 None 则使用 argmax
        """
        log_probs = F.log_softmax(logits, dim=-1)
        
        if target_tokens is not None:
            # 获取目标 token 位置的 log probability
            # gather 需要 [bs, seq, 1] 的索引
            target_tokens = target_tokens.unsqueeze(-1)
            ll = log_probs.gather(dim=-1, index=target_tokens).squeeze(-1)
        else:
            # 如果没有目标 token，使用 argmax (退化情况)
            ll = log_probs.max(dim=-1).values
        
        if mask is not None:
            ll = ll * mask
            return ll.sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)
        else:
            return ll.mean(dim=-1)


def compute_vision_distillation_loss(
    student_features: torch.Tensor,
    teacher_descriptions: list,
    text_encoder: nn.Module,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    通过文本描述间接蒸馏视觉能力
    
    思路：
    1. GPT-4V 生成详细的图像描述 (teacher_descriptions)
    2. 将描述编码为文本特征
    3. 让学生的视觉特征与描述特征对齐
    
    Args:
        student_features: 学生模型的视觉特征 [bs, num_patches, dim]
        teacher_descriptions: 教师生成的图像描述列表
        text_encoder: 文本编码器 (用于编码描述)
        temperature: 对比学习温度
        
    Returns:
        对齐损失
    """
    # 编码教师描述
    with torch.no_grad():
        teacher_text_features = text_encoder(teacher_descriptions)
    
    # 全局池化学生视觉特征
    student_global = student_features.mean(dim=1)  # [bs, dim]
    
    # 归一化
    student_global = F.normalize(student_global, dim=-1)
    teacher_text_features = F.normalize(teacher_text_features, dim=-1)
    
    # 对比损失 (InfoNCE)
    logits = torch.matmul(student_global, teacher_text_features.t()) / temperature
    labels = torch.arange(len(logits), device=logits.device)
    
    loss = F.cross_entropy(logits, labels)
    
    return loss


class VisualQADistillationLoss(nn.Module):
    """
    视觉问答蒸馏损失
    
    通过让学生模型回答视觉问题，间接学习教师的视觉理解能力
    
    流程:
    1. 对每张图像生成视觉细节问题
    2. GPT-4V 回答这些问题
    3. 训练学生模型模仿这些回答
    """
    
    def __init__(
        self,
        visual_question_types: list = None,
        temperature: float = 1.5,
    ):
        super().__init__()
        
        self.question_types = visual_question_types or [
            "describe_objects",      # 描述图中物体
            "count_objects",         # 计数
            "spatial_relations",     # 空间关系
            "colors_attributes",     # 颜色和属性
            "scene_understanding",   # 场景理解
        ]
        
        self.temperature = temperature
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")
    
    def get_visual_questions(self, question_type: str) -> str:
        """根据问题类型生成视觉问题"""
        templates = {
            "describe_objects": "请详细描述这张图片中的所有物体，包括它们的外观特征。",
            "count_objects": "请数一数图片中共有多少个主要物体？列出每种物体的数量。",
            "spatial_relations": "请描述图片中各个物体之间的空间位置关系。",
            "colors_attributes": "请描述图片中各个物体的颜色、大小、材质等属性。",
            "scene_understanding": "这张图片展示的是什么场景？请解释你的判断依据。",
        }
        return templates.get(question_type, "请描述这张图片。")
    
    def forward(
        self,
        student_answers_logits: torch.Tensor,
        teacher_answers_tokens: torch.Tensor,
        question_type: str,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        计算视觉问答蒸馏损失
        
        Args:
            student_answers_logits: 学生回答的 logits [bs, seq, vocab]
            teacher_answers_tokens: 教师回答的 token ids [bs, seq]
            question_type: 问题类型
            mask: attention mask
            
        Returns:
            蒸馏损失
        """
        # 交叉熵损失 (让学生模仿教师回答)
        vocab_size = student_answers_logits.size(-1)
        
        loss = F.cross_entropy(
            student_answers_logits.view(-1, vocab_size),
            teacher_answers_tokens.view(-1),
            ignore_index=-100,  # padding
        )
        
        return loss


# 测试代码
if __name__ == "__main__":
    # 测试 VisionLoRDLoss
    loss_fn = VisionLoRDLoss(alpha=1.0, beta=0.25)
    
    # 模拟输入
    bs, seq, vocab = 2, 64, 32000
    codebook_size = 8192
    num_patches = 196
    
    student_text = torch.randn(bs, seq, vocab)
    teacher_text = torch.randn(bs, seq, vocab)
    student_vq = torch.randn(bs, num_patches, codebook_size)
    teacher_vq = torch.randn(bs, num_patches, codebook_size)
    vq_loss = torch.tensor(0.1)
    
    losses = loss_fn(
        student_text_logits=student_text,
        teacher_text_logits=teacher_text,
        student_vq_logits=student_vq,
        teacher_vq_logits=teacher_vq,
        vq_loss=vq_loss,
    )
    
    print("损失项:")
    for name, value in losses.items():
        print(f"  {name}: {value.item():.4f}")
