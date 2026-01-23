"""
======================================================================
MERGE_LORA_MUL ---

Merge LoRA weights with LLaVA multimodal models.
Adapted for LLaVA-Llama-3 architecture.

    Author: Adapted for multimodal learning
    Copyright © 2024, ZiLiang, all rights reserved.
    Created: December 2024
======================================================================
"""


# ------------------------ Code --------------------------------------

import os
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["TORCH_USE_CUDA_DSA"]="1"

import torch
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
from peft import PeftModel
import argparse


def main():
    """
    主函数：合并 LoRA 权重到 LLaVA 基础模型
    
    使用示例：
    python merge_lora_mul.py \
        --base_model /root/workspace/models/llama3-llava-next-8b-hf \
        --lora_path ./sciqa_ckpts/SCIQAscienceqa1281LoRD-VI___period64 \
        --save_path ./sciqa_ckpts/MERGED/llava-llama3-8b-sciqa-lord6
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_model",
        default="/root/workspace/models/llama3-llava-next-8b-hf",
        type=str,
        help="Base LLaVA model path"
    )
    parser.add_argument(
        "--lora_path",
        default="./sciqa_ckpts/SCIQAscienceqa1281LoRD-VI___period64",
        type=str,
        help="LoRA checkpoint path"
    )
    parser.add_argument(
        "--save_path",
        default="./sciqa_ckpts/MERGED/llava-llama3-8b-sciqa",
        type=str,
        help="Path to save merged model"
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("Merging LoRA weights with LLaVA model")
    print(f"Base model: {args.base_model}")
    print(f"LoRA path: {args.lora_path}")
    print(f"Save path: {args.save_path}")
    print("="*70)
    
    merge_lora_multimodal(
        args.base_model,
        args.lora_path,
        args.save_path
    )


def merge_lora_multimodal(
        pretrained_path,
        lora_path,
        save_path
):
    """
    合并 LoRA 权重到多模态模型
    
    关键修改：
    1. 使用 LlavaNextForConditionalGeneration 而不是 AutoModelForCausalLM
    2. 使用 LlavaNextProcessor 处理多模态输入
    3. 确保 vision_tower 和 multi_modal_projector 参数被正确保存
    
    Args:
        pretrained_path: 基础 LLaVA 模型路径
        lora_path: LoRA checkpoint 路径
        save_path: 保存合并后模型的路径
    """
    
    print("\nStep 1: Loading base LLaVA-Next model...")
    print(f"Loading from: {pretrained_path}")
    
    # 关键修改1：使用 LlavaNextForConditionalGeneration
    base_model = LlavaNextForConditionalGeneration.from_pretrained(
        pretrained_path,
        device_map="auto",
        dtype=torch.float16,
        trust_remote_code=True,
    )
    
    print(f"Base model loaded. Type: {type(base_model)}")
    print(f"Number of parameters: {base_model.num_parameters():,}")
    
    print("\nStep 2: Loading LoRA weights...")
    print(f"Loading from: {lora_path}")
    
    # 加载 LoRA 权重
    model = PeftModel.from_pretrained(
        base_model,
        lora_path,
        device_map="auto",
    )
    
    print("LoRA weights loaded successfully")
    print("Model structure:")
    print(f"  - Vision Tower: {hasattr(model, 'vision_tower')}")
    print(f"  - Multi-modal Projector: {hasattr(model, 'multi_modal_projector')}")
    print(f"  - Language Model: {hasattr(model, 'language_model')}")
    
    print("\nStep 3: Loading processor...")
    # 关键修改2：加载 Processor（包含 tokenizer 和 image_processor）
    processor = LlavaNextProcessor.from_pretrained(
        pretrained_path  # 从基础模型加载，因为 lora_path 可能没有 processor
    )
    
    print("Processor loaded successfully")
    
    print("\nStep 4: Merging LoRA weights...")
    # 合并 LoRA 权重
    merged_model = model.merge_and_unload()
    
    print("LoRA weights merged successfully")
    
    print("\nStep 5: Saving merged model...")
    os.makedirs(save_path, exist_ok=True)
    
    # 保存合并后的模型
    # 关键修改3：确保所有组件都被保存
    merged_model.save_pretrained(
        save_path,
        safe_serialization=True,  # 使用 safetensors 格式
    )
    
    # 保存 processor
    processor.save_pretrained(save_path)
    
    print(f"Merged model saved to: {save_path}")
    
    # 验证保存的文件
    print("\nStep 6: Verifying saved files...")
    saved_files = os.listdir(save_path)
    print(f"Saved files ({len(saved_files)}):")
    for f in sorted(saved_files):
        file_path = os.path.join(save_path, f)
        if os.path.isfile(file_path):
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"  - {f} ({size_mb:.2f} MB)")
    
    print("\n" + "="*70)
    print("✓ MERGE COMPLETED SUCCESSFULLY")
    print("="*70)
    
    # 可选：测试加载合并后的模型
    print("\nStep 7: Testing merged model loading...")
    try:
        test_model = LlavaForConditionalGeneration.from_pretrained(
            save_path,
            device_map="cpu",  # 使用 CPU 测试以节省 GPU 内存
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        test_processor = AutoProcessor.from_pretrained(
            save_path,
            trust_remote_code=True
        )
        print("✓ Merged model loads successfully!")
        print(f"  Model type: {type(test_model)}")
        print(f"  Processor type: {type(test_processor)}")
        del test_model
        del test_processor
    except Exception as e:
        print(f"✗ Error loading merged model: {e}")
        print("  Please check the saved model manually")
    
    return save_path


def batch_merge(config_file="merge_config.json"):
    """
    批量合并多个 LoRA checkpoint
    
    config_file 格式:
    {
        "base_model": "/path/to/base/model",
        "checkpoints": [
            {
                "lora_path": "./ckpt1",
                "save_path": "./merged1"
            },
            {
                "lora_path": "./ckpt2",
                "save_path": "./merged2"
            }
        ]
    }
    """
    import json
    
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    base_model = config['base_model']
    checkpoints = config['checkpoints']
    
    print(f"Batch merging {len(checkpoints)} checkpoints...")
    
    for i, ckpt in enumerate(checkpoints):
        print(f"\n{'='*70}")
        print(f"Processing checkpoint {i+1}/{len(checkpoints)}")
        print(f"{'='*70}")
        
        merge_lora_multimodal(
            base_model,
            ckpt['lora_path'],
            ckpt['save_path']
        )
    
    print(f"\n{'='*70}")
    print(f"✓ ALL {len(checkpoints)} CHECKPOINTS MERGED SUCCESSFULLY")
    print(f"{'='*70}")


# running entry
if __name__ == "__main__":
    main()
    print("\nMERGE_LORA_MUL DONE.")
