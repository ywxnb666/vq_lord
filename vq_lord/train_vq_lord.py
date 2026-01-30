"""
======================================================================
TRAIN_VQ_LORD ---

VQ-LoRD 主训练脚本

结合 VQ 离散化和 LoRD 蒸馏，窃取多模态模型的图像识别能力

训练流程:
1. 加载预训练 LLaVA 模型
2. 添加 VQ 层到 Vision Encoder
3. 加载 GPT-4V 收集的视觉数据
4. 三阶段训练: VQ预训练 → 视觉蒸馏 → LoRD联合训练

    Author: VQ-LoRD Project
    Created: January 2026
======================================================================
"""

import os
import torch
import argparse
from typing import Optional, List
from pprint import pprint
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from torch.utils.tensorboard import SummaryWriter
from datasets import load_dataset

from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from vq_module import VectorQuantizer, VQVisionEncoder
from vision_lord_loss import VisionLoRDLoss, VisualQADistillationLoss
from data_collector import GPT4VDataCollector, VQLORDDataset

import torch.nn.functional as F


def build_scienceqa_samples(
    split: str,
    train_num: int,
    seed: int,
) -> List[dict]:
    dataset = load_dataset("derek-thomas/ScienceQA", split=split)
    dataset_with_images = [item for item in dataset if item.get("image") is not None]

    import random

    all_indices = list(range(len(dataset_with_images)))
    random.seed(seed)
    random.shuffle(all_indices)
    if train_num > 0 and len(all_indices) > train_num:
        all_indices = all_indices[:train_num]

    samples = []
    for idx in all_indices:
        item = dataset_with_images[idx]
        question = item.get("question", "")
        choices = item.get("choices", [])
        choices_text = ""
        for idx, choice in enumerate(choices):
            choices_text += f"({chr(65 + idx)}) {choice}\n"

        answer_idx = item.get("answer", 0)
        answer = choices[answer_idx] if choices and answer_idx < len(choices) else ""
        lecture = item.get("lecture", "")
        solution = item.get("solution", "")

        instruction = f"<image>\nQuestion: {question}\nOptions:\n{choices_text}Answer:"
        if lecture:
            response = f"Explanation: {lecture}\nSolution: {solution}\nAnswer: {answer}"
        else:
            response = f"Solution: {solution}\nAnswer: {answer}"

        samples.append({
            "image": item.get("image"),
            "instruction": instruction,
            "response": response,
        })

    return samples


class ScienceQADataset(torch.utils.data.Dataset):
    """ScienceQA 多模态数据集包装"""

    def __init__(
        self,
        processor,
        split: str = "train",
        train_num: int = 500,
        max_length: int = 512,
        samples: Optional[List[dict]] = None,
        seed: int = 20240306,
    ):
        self.processor = processor
        self.max_length = max_length

        if samples is None:
            samples = build_scienceqa_samples(
                split=split,
                train_num=train_num,
                seed=seed,
            )

        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        image = item["image"]
        image_sizes_default = torch.tensor([image.height, image.width], dtype=torch.long)

        full_text = f"{item['instruction']}\n{item['response']}"
        inputs = self.processor(
            text=full_text,
            images=image,
            return_tensors="pt",
            padding="longest",
            truncation=False,
        )
        image_sizes = inputs.get("image_sizes") if isinstance(inputs, dict) else None
        if image_sizes is None:
            image_sizes = image_sizes_default
        else:
            image_sizes = image_sizes.squeeze(0)

        pad_id = self.processor.tokenizer.pad_token_id
        labels = inputs["input_ids"].squeeze(0)
        labels = labels.masked_fill(labels == pad_id, -100)

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "image_sizes": image_sizes,
            "labels": labels,
            "data_type": "scienceqa",
        }


def setup_args():
    """设置训练参数"""
    parser = argparse.ArgumentParser(description="VQ-LoRD 训练脚本")
    
    # 模型参数
    parser.add_argument("--model_path", type=str, 
                       default="/root/workspace/models/llama3-llava-next-8b-hf",
                       help="LLaVA 模型路径")
    parser.add_argument("--victim_model", type=str,
                       default="gpt-4-vision-preview",
                       help="教师模型 (API)")
    
    # VQ 参数
    parser.add_argument("--vq_codebook_size", type=int, default=8192,
                       help="VQ codebook 大小")
    parser.add_argument("--vq_commitment_cost", type=float, default=0.25,
                       help="VQ commitment loss 权重")
    parser.add_argument("--freeze_vision_tower", type=int, default=0,
                       help="是否冻结原始 vision tower")
    
    # 损失权重
    parser.add_argument("--alpha", type=float, default=1.0,
                       help="视觉蒸馏损失权重")
    parser.add_argument("--beta", type=float, default=0.25,
                       help="VQ 损失权重")
    parser.add_argument("--temperature", type=float, default=1.5,
                       help="蒸馏温度")
    
    # 训练参数
    parser.add_argument("--stage", type=int, default=3,
                       help="训练阶段 (1=VQ预训练, 2=视觉蒸馏, 3=联合训练)")
    parser.add_argument("--epochs", type=int, default=3,
                       help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="批次大小")
    parser.add_argument("--lr", type=float, default=3e-5,
                       help="学习率")
    parser.add_argument("--max_length", type=int, default=512,
                       help="最大序列长度")
    
    # LoRD 超参数
    parser.add_argument("--tau1", type=float, default=0.1,
                       help="LoRD 置信度阈值")
    parser.add_argument("--tau_delta", type=float, default=0.01,
                       help="LoRD delta 阈值")
    parser.add_argument("--max_new_tokens", type=int, default=64,
                       help="LoRD 生成最大新 token 数")
    
    # LoRA 参数
    parser.add_argument("--use_lora", type=int, default=1,
                       help="是否使用 LoRA")
    parser.add_argument("--lora_rank", type=int, default=32,
                       help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=64,
                       help="LoRA alpha")
    
    # 量化参数
    parser.add_argument("--use_4bit", type=int, default=1,
                       help="是否使用 4-bit 量化")
    
    # 数据参数
    parser.add_argument("--data_dir", type=str, default="./vq_lord_data",
                       help="数据目录")
    parser.add_argument("--train_num", type=int, default=500,
                       help="训练样本数")
    parser.add_argument("--dataset_name", type=str, default="vq_lord",
                       choices=["vq_lord", "scienceqa"],
                       help="训练数据来源 (vq_lord 或 scienceqa)")
    parser.add_argument("--scienceqa_split", type=str, default="train",
                       help="ScienceQA 数据集 split")
    parser.add_argument("--scienceqa_seed", type=int, default=20240306,
                       help="ScienceQA 划分随机种子")
    parser.add_argument("--scienceqa_eval_split", type=str, default="validation",
                       help="ScienceQA 验证/测试 split")
    parser.add_argument("--reuse_vq_codebook", type=int, default=1,
                       help="若存在已保存的VQ codebook则复用并跳过Stage1")
    parser.add_argument("--vq_codebook_path", type=str, default="",
                       help="VQ codebook 保存/加载路径 (为空则使用save_path/stage1_vq)")
    parser.add_argument("--reuse_stage2", type=int, default=1,
                       help="若存在已保存的Stage2模型则复用并跳过Stage2")
    parser.add_argument("--stage2_ckpt_path", type=str, default="",
                       help="Stage2 checkpoint 保存/加载路径 (为空则使用save_path/stage2_vision)")
    
    # 保存参数
    parser.add_argument("--save_path", type=str, default="./vq_lord_ckpts",
                       help="模型保存路径")
    parser.add_argument("--log_step", type=int, default=10,
                       help="日志间隔")
    parser.add_argument("--save_step", type=int, default=100,
                       help="保存间隔")
    
    parser.add_argument("--device", type=str, default="cuda",
                       help="训练设备")
    
    return parser.parse_args()


def load_model_and_processor(args):
    """加载模型和处理器"""
    print(f"加载模型: {args.model_path}")
    
    # 量化配置
    if args.use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
        model = LlavaNextForConditionalGeneration.from_pretrained(
            args.model_path,
            device_map="auto",
            quantization_config=quantization_config,
            trust_remote_code=True,
        )
    else:
        model = LlavaNextForConditionalGeneration.from_pretrained(
            args.model_path,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
    
    processor = LlavaNextProcessor.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )
    
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    # 训练场景下关闭 KV cache，避免与梯度检查点冲突
    if hasattr(model, "config"):
        model.config.use_cache = False

    # 显式设置 gradient checkpointing 的 use_reentrant，消除警告
    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            # 兼容不支持该参数的版本
            model.gradient_checkpointing_enable()
    
    return model, processor


def add_vq_to_model(model, args):
    """
    为模型添加 VQ 层
    
    关键：使用 forward hook 将 VQ 量化后的特征替换原始视觉特征
    """
    print("添加 VQ 离散化层...")
    
    # 获取 vision tower
    vision_tower = model.vision_tower
    
    # 创建 VQ Vision Encoder
    vq_vision_encoder = VQVisionEncoder(
        vision_tower=vision_tower,
        num_embeddings=args.vq_codebook_size,
        freeze_vision_tower=bool(args.freeze_vision_tower),
    )

    try:
        vision_device = next(vision_tower.parameters()).device
        vq_vision_encoder.to(vision_device)
    except StopIteration:
        pass
    
    # 保存到模型
    model.vq_vision_encoder = vq_vision_encoder
    
    # 保存 VQ 损失的容器 (用于训练时获取)
    model._vq_loss_container = {"loss": None, "logits": None}
    
    # 注册 forward hook，在 vision_tower 输出后应用 VQ
    def vq_hook(module, input, output):
        """在 vision tower 输出后应用 VQ 量化"""
        # 处理不同的输出格式
        if hasattr(output, "last_hidden_state"):
            vision_features = output.last_hidden_state
        elif isinstance(output, tuple):
            vision_features = output[0]
        else:
            vision_features = output
        
        # 应用 VQ 量化
        if model.vq_vision_encoder.vq.embedding.weight.device != vision_features.device:
            model.vq_vision_encoder.vq.to(vision_features.device)
        quantized, indices, vq_loss, logits = model.vq_vision_encoder.vq(
            vision_features, return_logits=True
        )
        
        # 保存 VQ 损失供训练使用
        model._vq_loss_container["loss"] = vq_loss
        model._vq_loss_container["logits"] = logits
        model._vq_loss_container["indices"] = indices
        
        # 返回量化后的特征（替换原始特征）
        if hasattr(output, "last_hidden_state"):
            # 如果是 BaseModelOutputWithPooling 类型
            output.last_hidden_state = quantized
            return output
        elif isinstance(output, tuple):
            return (quantized,) + output[1:]
        else:
            return quantized
    
    # 注册 hook (注意：这会修改原模型行为)
    model._vq_hook_handle = vision_tower.register_forward_hook(vq_hook)
    
    print(f"VQ 层已添加，codebook 大小: {args.vq_codebook_size}")
    return model


def apply_lora(model, args):
    """应用 LoRA"""
    print("应用 LoRA...")
    
    if args.use_4bit:
        model = prepare_model_for_kbit_training(model)
    
    # LoRA 配置 - 包含视觉部分
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        target_modules=[
            # 语言模型层
            "q_proj", "v_proj", "k_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
            # 多模态投影层 (如果需要训练)
            # "multi_modal_projector.linear_1",
            # "multi_modal_projector.linear_2",
        ],
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model


def train_stage1_vq(model, dataloader, args, tb_writer):
    """
    阶段 1: VQ Codebook 预训练
    
    目标：训练 VQ 层学习好的图像离散表示
    注意：这个阶段只训练 VQ codebook，不训练其他组件
    """
    print("\n" + "=" * 50)
    print("阶段 1: VQ Codebook 预训练")
    print("=" * 50)

    original_use_ema = model.vq_vision_encoder.vq.use_ema
    model.vq_vision_encoder.vq.use_ema = False
    
    # 设置模型为训练模式
    model.train()
    
    # 只训练 VQ 层
    for name, param in model.named_parameters():
        if "vq" in name.lower() and torch.is_floating_point(param):
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    # 统计可训练参数
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"可训练参数量: {trainable_params:,}")
    
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr * 5,  # VQ 预训练用更大学习率
    )
    
    log_dir = os.path.join(args.save_path, "logs")
    os.makedirs(log_dir, exist_ok=True)
    metrics_path = os.path.join(log_dir, "vq_metrics.csv")

    if not os.path.exists(metrics_path):
        with open(metrics_path, "w", encoding="utf-8") as f:
            f.write("stage,step,loss,loss_ema,codebook_used\n")

    global_step = 0
    loss_ema = None
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for batch in tqdm(dataloader, desc=f"VQ预训练 Epoch {epoch+1}"):
            global_step += 1
            
            pixel_values = batch["pixel_values"].to(args.device)

            if pixel_values.dim() == 5:
                batch_size, num_patches, channels, height, width = pixel_values.shape
                pixel_values = pixel_values.view(batch_size * num_patches, channels, height, width)
            
            # 前向传播（会触发 VQ hook）
            _ = model.vision_tower(pixel_values)
            
            # 获取 VQ 损失与索引
            vq_loss = model._vq_loss_container["loss"]
            vq_indices = model._vq_loss_container.get("indices")
            
            if vq_loss is None or not vq_loss.requires_grad:
                # 如果 hook 中计算的 loss 不需要梯度，需要重新计算
                vision_features = model.vision_tower(pixel_values)
                if hasattr(vision_features, "last_hidden_state"):
                    vision_features = vision_features.last_hidden_state
                _, indices, vq_loss, _ = model.vq_vision_encoder.vq(vision_features)
                vq_indices = indices
            
            if vq_loss.requires_grad:
                optimizer.zero_grad()
                vq_loss.backward()
                optimizer.step()
            
            vq_loss_val = vq_loss.item()
            epoch_loss += vq_loss_val
            loss_ema = vq_loss_val if loss_ema is None else (0.95 * loss_ema + 0.05 * vq_loss_val)

            if vq_indices is not None:
                codebook_used = torch.unique(vq_indices).numel()
            else:
                codebook_used = 0
            
            if global_step % args.log_step == 0:
                print(f"Step {global_step}, VQ Loss: {vq_loss_val:.4f}")
                tb_writer.add_scalar("stage1/vq_loss", vq_loss_val, global_step)
                tb_writer.add_scalar("stage1/vq_loss_ema", loss_ema, global_step)
                tb_writer.add_scalar("stage1/codebook_used", codebook_used, global_step)
                with open(metrics_path, "a", encoding="utf-8") as f:
                    f.write(f"stage1,{global_step},{vq_loss_val:.6f},{loss_ema:.6f},{codebook_used}\n")
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} 平均 VQ Loss: {avg_loss:.4f}")
    
    return model


def train_stage2_vision(model, dataloader, args, tb_writer):
    """
    阶段 2: 视觉能力蒸馏
    
    目标：通过视觉问答蒸馏，让学生模型学习 GPT-4V 的视觉理解能力
    训练：VQ 层 + Vision Encoder（如果未冻结）+ Projector
    """
    print("\n" + "=" * 50)
    print("阶段 2: 视觉能力蒸馏")
    print("=" * 50)
    
    model.train()
    
    # 训练 VQ + Vision (如果未冻结) + Projector
    for name, param in model.named_parameters():
        if not torch.is_floating_point(param):
            param.requires_grad = False
            continue
        if any(key in name.lower() for key in ["vq", "projector", "multi_modal_projector"]):
            param.requires_grad = True
        elif "vision" in name.lower() and not args.freeze_vision_tower:
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"可训练参数量: {trainable_params:,}")
    
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
    )
    
    global_step = 0
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for batch in tqdm(dataloader, desc=f"视觉蒸馏 Epoch {epoch+1}"):
            global_step += 1
            
            input_ids = batch["input_ids"].to(args.device)
            attention_mask = batch["attention_mask"].to(args.device)
            pixel_values = batch["pixel_values"].to(args.device)
            image_sizes = batch.get("image_sizes")
            if image_sizes is not None:
                image_sizes = image_sizes.to(args.device)
            labels = batch["labels"].to(args.device)
            
            # 前向传播（VQ hook 会自动应用）
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_sizes=image_sizes,
                labels=labels,
            )
            
            # 获取 VQ 损失
            vq_loss = model._vq_loss_container.get("loss", torch.tensor(0.0, device=args.device))
            if vq_loss is None:
                vq_loss = torch.tensor(0.0, device=args.device)
            
            # 计算总损失
            text_loss = outputs.loss
            total_loss = text_loss + args.beta * vq_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += total_loss.item()
            
            if global_step % args.log_step == 0:
                vq_val = vq_loss.item() if hasattr(vq_loss, 'item') else vq_loss
                print(f"Step {global_step}, Total: {total_loss.item():.4f}, "
                      f"Text: {text_loss.item():.4f}, VQ: {vq_val:.4f}")
                tb_writer.add_scalar("stage2/total_loss", total_loss.item(), global_step)
                tb_writer.add_scalar("stage2/text_loss", text_loss.item(), global_step)
                tb_writer.add_scalar("stage2/vq_loss", vq_val, global_step)
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} 平均损失: {avg_loss:.4f}")
    
    return model


def log_clip(tnsr, epsilon=0.2):
    """对数裁剪函数，防止数值不稳定"""
    one_tensor = torch.ones_like(tnsr)
    one_tensor = one_tensor.to(tnsr.device)
    tnsr = torch.min(tnsr, torch.log(one_tensor * (1 + epsilon)))
    tnsr = torch.max(tnsr, torch.log(one_tensor * (1 - epsilon)))
    return tnsr


def compute_sequence_log_prob(model, input_ids, attention_mask, pixel_values, image_sizes):
    """
    计算序列的 log probability
    
    Args:
        model: 学生模型
        input_ids: token 序列 [bs, seq_len]
        attention_mask: attention mask
        pixel_values: 图像特征
        image_sizes: 图像尺寸
        
    Returns:
        log_probs: 每个样本的平均 log probability [bs]
    """
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_sizes=image_sizes,
        )
    
    logits = outputs.logits[:, :-1, :]  # [bs, seq-1, vocab]
    log_probs = F.log_softmax(logits, dim=-1)
    
    # 获取目标 token 的 log prob
    bs, seq_len = input_ids.shape
    target_ids = input_ids[:, 1:seq_len]  # [bs, seq-1]
    
    # gather: [bs, seq-1]
    token_log_probs = log_probs.gather(
        dim=-1, 
        index=target_ids.unsqueeze(-1)
    ).squeeze(-1)
    
    # 使用 attention_mask 计算平均 (忽略 padding)
    mask = attention_mask[:, 1:seq_len].float()
    seq_log_prob = (token_log_probs * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)
    
    return seq_log_prob, token_log_probs, mask


def train_stage3_lord(model, dataloader, args, tb_writer):
    """
    阶段 3: LoRD 联合训练 (真正的 LoRD 方法)
    
    实现 LoRD 核心算法：
    1. 双样本生成：学生模型生成两个独立序列 S1, S2
    2. 局部性排序：根据 log prob 确定正负样本 y+, y-
    3. 冷启动策略：置信度低时使用教师标签 y_vic
    4. LoRD 损失：对比损失 + 正则化损失
    
    损失公式:
    L_obj = -log(P(y+|x) / P(y-|x)) = -(log P(y+) - log P(y-))
    L_reg = -clip(log(P(y_vic|x) / P(y-|x)))
    L_total = L_obj + L_reg + beta * L_vq
    """
    print("\n" + "=" * 50)
    print("阶段 3: LoRD 联合训练 (真正的 LoRD 方法)")
    print("=" * 50)
    
    # LoRD 超参数
    tau1 = getattr(args, 'tau1', 0.1)  # 置信度阈值
    tau_delta = getattr(args, 'tau_delta', 0.01)  # delta 阈值
    max_new_tokens = getattr(args, 'max_new_tokens', 64)  # 生成长度
    
    print(f"LoRD 参数: tau1={tau1}, tau_delta={tau_delta}, max_new_tokens={max_new_tokens}")
    
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
    )
    
    # 用于记录上一轮的 log prob (用于计算 delta)
    prev_log_probs = {}
    
    global_step = 0
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        epoch_obj_loss = 0.0
        epoch_reg_loss = 0.0
        epoch_vq_loss = 0.0
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"LoRD Epoch {epoch+1}")):
            global_step += 1
            
            # 获取输入数据
            input_ids = batch["input_ids"].to(args.device)  # prompt + y_vic
            attention_mask = batch["attention_mask"].to(args.device)
            pixel_values = batch["pixel_values"].to(args.device)
            image_sizes = batch.get("image_sizes")
            if image_sizes is not None:
                image_sizes = image_sizes.to(args.device)
            labels = batch["labels"].to(args.device)  # y_vic
            
            bs = input_ids.shape[0]
            
            # ========== Step 1: 双样本生成 ==========
            # 找到 prompt 部分 (labels == -100 的部分是 prompt)
            # 注意：部分数据集未标注 prompt，会导致 prompt_len=0
            prompt_len = (labels == -100).sum(dim=-1).min().item()
            prompt_ids = input_ids[:, :max(prompt_len, 1)]
            prompt_mask = attention_mask[:, :max(prompt_len, 1)]

            # 保障 prompt 中包含 image token，否则生成会报错
            image_token_id = getattr(model.config, "image_token_index", None)
            if image_token_id is None:
                image_token_id = getattr(model.config, "image_token_id", None)

            has_image_token = True
            if image_token_id is not None:
                has_image_token = (prompt_ids == image_token_id).any().item()

            if prompt_len <= 0 or not has_image_token:
                prompt_ids = input_ids
                prompt_mask = attention_mask
            
            model.eval()
            with torch.no_grad():
                # 生成阶段无需梯度，临时关闭 checkpoint 以消除警告
                gc_enabled = getattr(model, "is_gradient_checkpointing", False)
                if gc_enabled and hasattr(model, "gradient_checkpointing_disable"):
                    model.gradient_checkpointing_disable()
                # 生成样本 S1
                gen_output_1 = model.generate(
                    input_ids=prompt_ids,
                    attention_mask=prompt_mask,
                    pixel_values=pixel_values,
                    image_sizes=image_sizes,
                    do_sample=True,
                    max_new_tokens=max_new_tokens,
                    temperature=0.7,
                    pad_token_id=model.config.pad_token_id or 0,
                )
                
                # 生成样本 S2
                gen_output_2 = model.generate(
                    input_ids=prompt_ids,
                    attention_mask=prompt_mask,
                    pixel_values=pixel_values,
                    image_sizes=image_sizes,
                    do_sample=True,
                    max_new_tokens=max_new_tokens,
                    temperature=0.7,
                    pad_token_id=model.config.pad_token_id or 0,
                )

                if gc_enabled and hasattr(model, "gradient_checkpointing_enable"):
                    try:
                        model.gradient_checkpointing_enable(
                            gradient_checkpointing_kwargs={"use_reentrant": False}
                        )
                    except TypeError:
                        model.gradient_checkpointing_enable()
            
            model.train()
            
            # 对齐序列长度
            max_len = max(gen_output_1.shape[1], gen_output_2.shape[1], input_ids.shape[1])
            
            def pad_to_length(tensor, length, pad_value=0):
                if tensor.shape[1] < length:
                    padding = torch.full(
                        (tensor.shape[0], length - tensor.shape[1]),
                        pad_value,
                        dtype=tensor.dtype,
                        device=tensor.device
                    )
                    return torch.cat([tensor, padding], dim=1)
                return tensor[:, :length]
            
            s1_ids = pad_to_length(gen_output_1, max_len)
            s2_ids = pad_to_length(gen_output_2, max_len)
            y_vic_ids = pad_to_length(input_ids, max_len)
            
            s1_mask = (s1_ids != 0).long()
            s2_mask = (s2_ids != 0).long()
            y_vic_mask = pad_to_length(attention_mask, max_len)
            
            # ========== Step 2: 计算 log probability ==========
            # 计算 S1 的 log prob
            log_prob_s1, token_lp_s1, mask_s1 = compute_sequence_log_prob(
                model, s1_ids, s1_mask, pixel_values, image_sizes
            )
            
            # 计算 S2 的 log prob
            log_prob_s2, token_lp_s2, mask_s2 = compute_sequence_log_prob(
                model, s2_ids, s2_mask, pixel_values, image_sizes
            )
            
            # 计算 y_vic 的 log prob
            log_prob_vic, token_lp_vic, mask_vic = compute_sequence_log_prob(
                model, y_vic_ids, y_vic_mask, pixel_values, image_sizes
            )
            
            # ========== Step 3: 局部性排序 + 冷启动 ==========
            # 确定 y+ 和 y-
            y_plus_ids = s1_ids.clone()
            y_minus_ids = s2_ids.clone()
            y_plus_mask = s1_mask.clone()
            y_minus_mask = s2_mask.clone()
            log_prob_plus = log_prob_s1.clone()
            log_prob_minus = log_prob_s2.clone()
            
            for i in range(bs):
                p1 = torch.exp(log_prob_s1[i]).item()
                p2 = torch.exp(log_prob_s2[i]).item()
                
                # 获取上一轮的 log prob (用于计算 delta)
                key = f"{batch_idx}_{i}"
                prev_p1 = prev_log_probs.get(f"{key}_1", p1)
                prev_p2 = prev_log_probs.get(f"{key}_2", p2)
                
                delta1 = p1 - prev_p1
                delta2 = p2 - prev_p2
                
                # 更新记录
                prev_log_probs[f"{key}_1"] = p1
                prev_log_probs[f"{key}_2"] = p2
                
                # 排序：确保 y+ 有更高的 log prob
                if p2 > p1:
                    y_plus_ids[i] = s2_ids[i]
                    y_minus_ids[i] = s1_ids[i]
                    y_plus_mask[i] = s2_mask[i]
                    y_minus_mask[i] = s1_mask[i]
                    log_prob_plus[i] = log_prob_s2[i]
                    log_prob_minus[i] = log_prob_s1[i]
                    delta1, delta2 = delta2, delta1
                
                # 冷启动策略：置信度低时使用 y_vic
                if max(p1, p2) < tau1 and delta1 < tau_delta:
                    y_plus_ids[i] = y_vic_ids[i]
                    y_plus_mask[i] = y_vic_mask[i]
                    log_prob_plus[i] = log_prob_vic[i]
            
            # ========== Step 4: 计算 LoRD 损失 ==========
            # 重新计算 y+, y- 的 log prob (需要梯度)
            outputs_plus = model(
                input_ids=y_plus_ids,
                attention_mask=y_plus_mask,
                pixel_values=pixel_values,
                image_sizes=image_sizes,
            )
            logits_plus = outputs_plus.logits[:, :-1, :]
            log_probs_plus = F.log_softmax(logits_plus, dim=-1)
            token_lp_plus = log_probs_plus.gather(
                dim=-1,
                index=y_plus_ids[:, 1:].unsqueeze(-1)
            ).squeeze(-1)
            mask_plus = y_plus_mask[:, 1:].float()
            seq_lp_plus = (token_lp_plus * mask_plus).sum(dim=-1) / mask_plus.sum(dim=-1).clamp(min=1)
            
            outputs_minus = model(
                input_ids=y_minus_ids,
                attention_mask=y_minus_mask,
                pixel_values=pixel_values,
                image_sizes=image_sizes,
            )
            logits_minus = outputs_minus.logits[:, :-1, :]
            log_probs_minus = F.log_softmax(logits_minus, dim=-1)
            token_lp_minus = log_probs_minus.gather(
                dim=-1,
                index=y_minus_ids[:, 1:].unsqueeze(-1)
            ).squeeze(-1)
            mask_minus = y_minus_mask[:, 1:].float()
            seq_lp_minus = (token_lp_minus * mask_minus).sum(dim=-1) / mask_minus.sum(dim=-1).clamp(min=1)
            
            outputs_vic = model(
                input_ids=y_vic_ids,
                attention_mask=y_vic_mask,
                pixel_values=pixel_values,
                image_sizes=image_sizes,
            )
            logits_vic = outputs_vic.logits[:, :-1, :]
            log_probs_vic = F.log_softmax(logits_vic, dim=-1)
            token_lp_vic_grad = log_probs_vic.gather(
                dim=-1,
                index=y_vic_ids[:, 1:].unsqueeze(-1)
            ).squeeze(-1)
            mask_vic_grad = y_vic_mask[:, 1:].float()
            seq_lp_vic = (token_lp_vic_grad * mask_vic_grad).sum(dim=-1) / mask_vic_grad.sum(dim=-1).clamp(min=1)
            
            # L_obj = -(log P(y+) - log P(y-))
            # 目标：最大化 y+ 与 y- 的差距
            L_obj = -torch.mean(seq_lp_plus - seq_lp_minus)
            
            # L_reg = -clip(log P(y_vic) - log P(y-))
            # 使用 y_vic 作为锚点
            L_reg = -torch.mean(log_clip(seq_lp_vic - seq_lp_minus))
            
            # VQ 损失
            vq_loss = model._vq_loss_container.get("loss", torch.tensor(0.0, device=args.device))
            if vq_loss is None:
                vq_loss = torch.tensor(0.0, device=args.device)
            
            # 总损失
            total_loss = L_obj + L_reg + args.beta * vq_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # 记录损失
            epoch_loss += total_loss.item()
            epoch_obj_loss += L_obj.item()
            epoch_reg_loss += L_reg.item()
            if hasattr(vq_loss, 'item'):
                epoch_vq_loss += vq_loss.item()
            
            if global_step % args.log_step == 0:
                print(f"Step {global_step}, Total: {total_loss.item():.4f}, "
                      f"L_obj: {L_obj.item():.4f}, L_reg: {L_reg.item():.4f}, "
                      f"VQ: {vq_loss.item() if hasattr(vq_loss, 'item') else 0:.4f}")
                tb_writer.add_scalar("stage3/total_loss", total_loss.item(), global_step)
                tb_writer.add_scalar("stage3/L_obj", L_obj.item(), global_step)
                tb_writer.add_scalar("stage3/L_reg", L_reg.item(), global_step)
                tb_writer.add_scalar("stage3/vq_loss", vq_loss.item() if hasattr(vq_loss, 'item') else 0, global_step)
            
            if global_step % args.save_step == 0:
                save_checkpoint(model, args, f"step_{global_step}")
        
        # Epoch 统计
        num_batches = len(dataloader)
        print(f"Epoch {epoch+1} 平均损失: Total={epoch_loss/num_batches:.4f}, "
              f"L_obj={epoch_obj_loss/num_batches:.4f}, L_reg={epoch_reg_loss/num_batches:.4f}, "
              f"VQ={epoch_vq_loss/num_batches:.4f}")
    
    return model


def save_checkpoint(model, args, suffix=""):
    """保存检查点"""
    save_path = os.path.join(args.save_path, suffix)
    os.makedirs(save_path, exist_ok=True)
    
    model.save_pretrained(save_path)
    print(f"模型已保存至 {save_path}")


def save_vq_codebook(model, codebook_path: str):
    os.makedirs(os.path.dirname(codebook_path), exist_ok=True)
    torch.save(model.vq_vision_encoder.vq.embedding.weight.detach().cpu(), codebook_path)


def load_vq_codebook(model, codebook_path: str):
    codebook = torch.load(codebook_path, map_location="cpu")
    model.vq_vision_encoder.vq.embedding.weight.data.copy_(codebook)


def save_stage2_checkpoint(model, args, ckpt_path: str):
    """保存 Stage2 检查点（包括 VQ codebook 和 LoRA 权重）"""
    os.makedirs(ckpt_path, exist_ok=True)
    
    # 保存 VQ codebook
    vq_path = os.path.join(ckpt_path, "vq_codebook.pt")
    torch.save(model.vq_vision_encoder.vq.embedding.weight.detach().cpu(), vq_path)
    
    # 保存 LoRA 适配器
    model.save_pretrained(ckpt_path)
    
    # 保存配置信息
    import json
    config_info = {
        "vq_codebook_size": args.vq_codebook_size,
        "freeze_vision_tower": args.freeze_vision_tower,
        "alpha": args.alpha,
        "beta": args.beta,
        "stage": 2,
    }
    with open(os.path.join(ckpt_path, "stage2_config.json"), "w") as f:
        json.dump(config_info, f, indent=2)
    
    print(f"Stage2 检查点已保存至 {ckpt_path}")


def load_stage2_checkpoint(model, ckpt_path: str):
    """加载 Stage2 检查点"""
    # 加载 VQ codebook
    vq_path = os.path.join(ckpt_path, "vq_codebook.pt")
    if os.path.exists(vq_path):
        codebook = torch.load(vq_path, map_location="cpu")
        model.vq_vision_encoder.vq.embedding.weight.data.copy_(codebook)
        print(f"已加载 VQ codebook: {vq_path}")
    
    # 加载 LoRA 适配器
    from peft import PeftModel
    adapter_config = os.path.join(ckpt_path, "adapter_config.json")
    if os.path.exists(adapter_config):
        # 如果模型已经是 PeftModel，则加载权重
        if hasattr(model, 'load_adapter'):
            model.load_adapter(ckpt_path, adapter_name="stage2")
            model.set_adapter("stage2")
        else:
            # 直接加载状态字典
            adapter_weights = os.path.join(ckpt_path, "adapter_model.safetensors")
            if os.path.exists(adapter_weights):
                from safetensors.torch import load_file
                state_dict = load_file(adapter_weights)
                model.load_state_dict(state_dict, strict=False)
            else:
                adapter_weights_bin = os.path.join(ckpt_path, "adapter_model.bin")
                if os.path.exists(adapter_weights_bin):
                    state_dict = torch.load(adapter_weights_bin, map_location="cpu")
                    model.load_state_dict(state_dict, strict=False)
        print(f"已加载 LoRA 权重: {ckpt_path}")
    
    print(f"Stage2 检查点加载完成")
    return model


def main():
    args = setup_args()
    
    print("=" * 60)
    print("VQ-LoRD 训练")
    print("=" * 60)
    pprint(vars(args))
    print("=" * 60)
    
    # 创建保存目录
    os.makedirs(args.save_path, exist_ok=True)
    if not args.vq_codebook_path:
        args.vq_codebook_path = os.path.join(args.save_path, "stage1_vq", "vq_codebook.pt")
    tb_writer = SummaryWriter(log_dir=os.path.join(args.save_path, "logs"))
    
    # 加载模型
    model, processor = load_model_and_processor(args)
    
    # 添加 VQ 层
    model = add_vq_to_model(model, args)
    
    # 应用 LoRA
    if args.use_lora:
        model = apply_lora(model, args)
    
    # 加载数据
    print("\n加载训练数据...")
    if args.dataset_name == "scienceqa":
        train_samples = build_scienceqa_samples(
            split=args.scienceqa_split,
            train_num=args.train_num,
            seed=args.scienceqa_seed,
        )

        train_dataset = ScienceQADataset(
            processor=processor,
            max_length=args.max_length,
            samples=train_samples,
            seed=args.scienceqa_seed,
        )
        test_dataset = ScienceQADataset(
            processor=processor,
            split=args.scienceqa_eval_split,
            train_num=0,
            max_length=args.max_length,
            seed=args.scienceqa_seed,
        )
    else:
        collector = GPT4VDataCollector(save_dir=args.data_dir)

        # 尝试加载已收集的数据
        visual_qa_data = collector.load_collected_data("visual_qa_data.json")
        image_descriptions = collector.load_collected_data("image_descriptions.json")

        if not visual_qa_data and not image_descriptions:
            print("警告：未找到预收集的数据，请先运行数据收集脚本")
            print("示例：python data_collector.py --collect_data")
            return

        # 创建数据集
        from data_collector import VisualQAItem, ImageDescriptionItem
        visual_qa_items = [VisualQAItem(**item) for item in visual_qa_data]
        desc_items = [ImageDescriptionItem(**item) for item in image_descriptions]

        train_dataset = VQLORDDataset(
            visual_qa_data=visual_qa_items,
            image_descriptions=desc_items,
            processor=processor,
            max_length=args.max_length,
        )
        test_dataset = None
    
    def vq_lord_collate(batch):
        batch = [b for b in batch if b is not None]
        if len(batch) == 0:
            return None

        image_sizes_list = [b.get("image_sizes") for b in batch]
        for idx, size in enumerate(image_sizes_list):
            if size is None:
                pixel_values = batch[idx].get("pixel_values")
                if isinstance(pixel_values, torch.Tensor) and pixel_values.dim() >= 3:
                    height = int(pixel_values.shape[-2])
                    width = int(pixel_values.shape[-1])
                    image_sizes_list[idx] = torch.tensor([height, width], dtype=torch.long)
        base = []
        for b in batch:
            item = dict(b)
            item.pop("image_sizes", None)
            base.append(item)

        collated = default_collate(base)

        if all(s is None for s in image_sizes_list):
            collated["image_sizes"] = None
        else:
            tensor_list = [s for s in image_sizes_list if s is not None]
            collated["image_sizes"] = torch.stack(tensor_list, dim=0)

        return collated

    dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=vq_lord_collate,
    )
    
    print(f"训练数据量: {len(train_dataset)}")
    if test_dataset is not None:
        print(f"测试数据量: {len(test_dataset)}")
    
    # 根据阶段训练
    if args.stage >= 1:
        if args.reuse_vq_codebook and os.path.exists(args.vq_codebook_path):
            print(f"检测到VQ codebook，跳过Stage1并加载: {args.vq_codebook_path}")
            load_vq_codebook(model, args.vq_codebook_path)
        else:
            model = train_stage1_vq(model, dataloader, args, tb_writer)
            save_checkpoint(model, args, "stage1_vq")
            save_vq_codebook(model, args.vq_codebook_path)
    
    if args.stage >= 2:
        # Stage2 复用逻辑
        if not args.stage2_ckpt_path:
            args.stage2_ckpt_path = os.path.join(args.save_path, "stage2_vision")
        
        if args.reuse_stage2 and os.path.exists(args.stage2_ckpt_path) and os.path.exists(os.path.join(args.stage2_ckpt_path, "vq_codebook.pt")):
            print(f"检测到Stage2检查点，跳过Stage2并加载: {args.stage2_ckpt_path}")
            model = load_stage2_checkpoint(model, args.stage2_ckpt_path)
        else:
            model = train_stage2_vision(model, dataloader, args, tb_writer)
            save_stage2_checkpoint(model, args, args.stage2_ckpt_path)
            save_checkpoint(model, args, "stage2_vision")
    
    if args.stage >= 3:
        model = train_stage3_lord(model, dataloader, args, tb_writer)
        save_checkpoint(model, args, "stage3_lord_final")
    
    print("\n" + "=" * 60)
    print("VQ-LoRD 训练完成!")
    print("=" * 60)
    
    tb_writer.close()


if __name__ == "__main__":
    main()
