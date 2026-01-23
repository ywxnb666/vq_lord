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
from pprint import pprint
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from vq_module import VectorQuantizer, VQVisionEncoder
from vision_lord_loss import VisionLoRDLoss, VisualQADistillationLoss
from data_collector import GPT4VDataCollector, VQLORDDataset

import torch.nn.functional as F


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
        quantized, indices, vq_loss, logits = model.vq_vision_encoder.vq(
            vision_features, return_logits=True
        )
        
        # 保存 VQ 损失供训练使用
        model._vq_loss_container["loss"] = vq_loss
        model._vq_loss_container["logits"] = logits
        
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
    
    # 设置模型为训练模式
    model.train()
    
    # 只训练 VQ 层
    for name, param in model.named_parameters():
        if "vq" in name.lower():
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    # 统计可训练参数
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"可训练参数量: {trainable_params:,}")
    
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr * 10,  # VQ 预训练用更大学习率
    )
    
    global_step = 0
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for batch in tqdm(dataloader, desc=f"VQ预训练 Epoch {epoch+1}"):
            global_step += 1
            
            pixel_values = batch["pixel_values"].to(args.device)
            
            # 前向传播（会触发 VQ hook）
            with torch.no_grad():
                _ = model.vision_tower(pixel_values)
            
            # 获取 VQ 损失
            vq_loss = model._vq_loss_container["loss"]
            
            if vq_loss is None or not vq_loss.requires_grad:
                # 如果 hook 中计算的 loss 不需要梯度，需要重新计算
                vision_features = model.vision_tower(pixel_values)
                if hasattr(vision_features, "last_hidden_state"):
                    vision_features = vision_features.last_hidden_state
                _, _, vq_loss, _ = model.vq_vision_encoder.vq(vision_features)
            
            optimizer.zero_grad()
            vq_loss.backward()
            optimizer.step()
            
            epoch_loss += vq_loss.item()
            
            if global_step % args.log_step == 0:
                print(f"Step {global_step}, VQ Loss: {vq_loss.item():.4f}")
                tb_writer.add_scalar("stage1/vq_loss", vq_loss.item(), global_step)
        
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
            labels = batch["labels"].to(args.device)
            
            # 前向传播（VQ hook 会自动应用）
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
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


def train_stage3_lord(model, dataloader, args, tb_writer):
    """
    阶段 3: LoRD 联合训练
    
    目标：端到端微调，结合文本蒸馏和视觉蒸馏
    类似原始 LoRD，但增加了视觉组件
    """
    print("\n" + "=" * 50)
    print("阶段 3: LoRD 联合训练")
    print("=" * 50)
    
    # 使用 LoRA，大部分参数冻结
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
    )
    
    loss_fn = VisionLoRDLoss(
        alpha=args.alpha,
        beta=args.beta,
        temperature=args.temperature,
        use_contrastive=True,
    )
    
    global_step = 0
    for epoch in range(args.epochs):
        for batch in tqdm(dataloader, desc=f"LoRD联合训练 Epoch {epoch+1}"):
            global_step += 1
            
            input_ids = batch["input_ids"].to(args.device)
            attention_mask = batch["attention_mask"].to(args.device)
            pixel_values = batch["pixel_values"].to(args.device)
            labels = batch["labels"].to(args.device)
            
            # 前向传播
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                labels=labels,
            )
            
            # VQ 损失
            if hasattr(model, "vq_vision_encoder"):
                _, _, vq_loss, vq_logits = model.vq_vision_encoder(
                    pixel_values, return_vq_logits=True
                )
            else:
                vq_loss = torch.tensor(0.0, device=args.device)
            
            # 综合损失
            total_loss = outputs.loss + args.beta * vq_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            if global_step % args.log_step == 0:
                print(f"Step {global_step}, Loss: {total_loss.item():.4f}")
                tb_writer.add_scalar("stage3/total_loss", total_loss.item(), global_step)
            
            if global_step % args.save_step == 0:
                save_checkpoint(model, args, f"step_{global_step}")
    
    return model


def save_checkpoint(model, args, suffix=""):
    """保存检查点"""
    save_path = os.path.join(args.save_path, suffix)
    os.makedirs(save_path, exist_ok=True)
    
    model.save_pretrained(save_path)
    print(f"模型已保存至 {save_path}")


def main():
    args = setup_args()
    
    print("=" * 60)
    print("VQ-LoRD 训练")
    print("=" * 60)
    pprint(vars(args))
    print("=" * 60)
    
    # 创建保存目录
    os.makedirs(args.save_path, exist_ok=True)
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
    
    dataset = VQLORDDataset(
        visual_qa_data=visual_qa_items,
        image_descriptions=desc_items,
        processor=processor,
        max_length=args.max_length,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    
    print(f"训练数据量: {len(dataset)}")
    
    # 根据阶段训练
    if args.stage >= 1:
        model = train_stage1_vq(model, dataloader, args, tb_writer)
        save_checkpoint(model, args, "stage1_vq")
    
    if args.stage >= 2:
        model = train_stage2_vision(model, dataloader, args, tb_writer)
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
