"""
======================================================================
LORD_TRAIN_MUL ---

Multimodal Model Extraction using LoRD method.
Adapted for LLaVA-Llama-3 architecture on ScienceQA dataset.

    Author: Adapted for multimodal learning
    Copyright © 2024, ZiLiang, all rights reserved.
    Created: December 2024
======================================================================
"""

# ------------------------ Code --------------------------------------
import torch
import argparse
from pprint import pprint as ppp

# 关键修改1：使用 LlavaNextForConditionalGeneration
# LlavaNext 是 LLaVA v1.6 系列的正确类
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
from transformers import BitsAndBytesConfig
from transformers import AutoTokenizer

from sciqa_process import load_scienceqa_data

import torch.nn.functional as F


def setup_train_args():
    """
    设置训练参数（保留原有参数，添加多模态相关参数）
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda", type=str, required=False)
    parser.add_argument("--epoch", default=2, type=int, required=False)
    parser.add_argument("--period_num", default=3, type=int, required=False)
    parser.add_argument("--sub_stage_num", default=10, type=int, required=False)
    parser.add_argument("--sub_set_num", default=16, type=int, required=False)
    parser.add_argument("--train_num", default=100, type=int, required=False)
    parser.add_argument("--acc_step", default=4, type=int, required=False)
    parser.add_argument("--log_step", default=1, type=int, required=False)
    parser.add_argument("--save_step", default=64, type=int, required=False)

    parser.add_argument("--lambda1", default=0.5, type=float, required=False)
    parser.add_argument("--tau1", default=0.99, type=float, required=False)
    parser.add_argument("--tau_delta", default=0.999, type=float, required=False)
    parser.add_argument("--tau2", default=0.998, type=float, required=False)
    parser.add_argument("--LR", default=3e-4, type=float, required=False)
    parser.add_argument("--beta", default=0.7, type=float, required=False)
    parser.add_argument("--temperature", default=0.8, type=float, required=False)
    parser.add_argument("--T", default=1.0, type=float, required=False)

    parser.add_argument("--use_lora", default=1, type=int, required=False)
    parser.add_argument("--use_4bit", default=1, type=int, required=False, help="Enable 4-bit quantization (1=True, 0=False)")
    parser.add_argument("--rank", default=64, type=int, required=False)
    parser.add_argument("--lora_alpha", default=128, type=int, required=False)

    parser.add_argument("--batch_size", default=1, type=int, required=False)
    parser.add_argument("--infer_batch_size", default=1, type=int, required=False)
    
    parser.add_argument(
        "--task",
        default="LoRD-VI",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--use_old_logits",
        default="1",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--use_vic_logits",
        default="1",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--use_kld",
        default="1",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--use_entropy",
        default="1",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--is_black_box",
        default=1,
        type=int,
        required=False,
    )
    parser.add_argument(
        "--dataset_task",
        default="scienceqa",
        type=str,
        required=False,
    )
    parser.add_argument("--max_length", default=512, type=int, required=False)
    parser.add_argument("--max_new_tokens", default=256, type=int, required=False)  # 多模态通常需要更长回复
    parser.add_argument("--with_early_shut", default=0, type=int, required=False)

    parser.add_argument(
        "--victim_path",
        default="gpt-4-vision-preview",
        type=str,
        required=False,
    )
    
    # 关键修改2：默认学生模型路径指向兼容的 LLaVA 模型
    # 使用本地的 llama3-llava-next-8b-hf 模型
    parser.add_argument(
        "--from_path",
        default="/root/workspace/models/llama3-llava-next-8b-hf",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--save_path",
        default="./sciqa_ckpts/",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--temp_save_path",
        default="./sciqa_ckpts/",
        type=str,
        required=False,
    )

    return parser.parse_args()


def main():
    args = setup_train_args()

    print("----------------------------------------------------------")
    ppp(args)
    print("----------------------------------------------------------")

    # 关键修改3：使用 LlavaNextForConditionalGeneration
    # 参考 run_llava.py 的加载方式
    print(f"Loading LLaVA-Next model from {args.from_path}...")
    
    # 根据 use_4bit 参数决定是否使用4bit量化
    if args.use_4bit == 1:
        print(">>> Enabling 4-bit quantization...")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
        lm = LlavaNextForConditionalGeneration.from_pretrained(
            args.from_path,
            device_map="auto",
            quantization_config=quantization_config,
            trust_remote_code=True,
        )
    else:
        print(">>> Loading model without quantization (float16)...")
        lm = LlavaNextForConditionalGeneration.from_pretrained(
            args.from_path,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
    
    # 关键修改4：使用 LlavaNextProcessor 处理多模态输入
    print(f"Loading processor from {args.from_path}...")
    processor = LlavaNextProcessor.from_pretrained(
        args.from_path,
        trust_remote_code=True,
    )
    
    # tokenizer 从 processor 中获取
    lm_tokenizer = processor.tokenizer
    
    if lm_tokenizer.pad_token is None:
        lm_tokenizer.pad_token = lm_tokenizer.eos_token
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    # 关键修复：同步模型的 generation_config
    # Llama-3 如果 pad_token_id 未设置或设置错误，容易在生成时产生 NaN
    if lm.generation_config.pad_token_id is None:
        lm.generation_config.pad_token_id = lm_tokenizer.pad_token_id
        print(f"Set model generation_config.pad_token_id to {lm_tokenizer.pad_token_id}")

    # 关键修复：检查并调整词表大小
    # LLaVA-Next/Llama-3 可能有额外的 token (如 <image>)
    # 如果 tokenizer 的词表比模型 embedding 层大，会导致 index out of bounds 错误
    if len(lm_tokenizer) > lm.get_input_embeddings().weight.shape[0]:
        print(f"WARNING: Tokenizer vocab size ({len(lm_tokenizer)}) > Model embedding size ({lm.get_input_embeddings().weight.shape[0]}). Resizing...")
        lm.resize_token_embeddings(len(lm_tokenizer))
        print(f"Resized model embedding to {lm.get_input_embeddings().weight.shape[0]}")

    print("---------------------------------------------------------")
    print(f"Running ScienceQA multimodal extraction task")
    print(f"Student model: {args.from_path}")
    print(f"Teacher model: {args.victim_path}")
    
    # 关键修改5：加载 ScienceQA 数据
    if args.dataset_task == "scienceqa":
        print(f"Loading ScienceQA dataset...")
        raw_train_datals = load_scienceqa_data(
            processor,
            task_name=args.dataset_task,
            train_num=args.train_num,
            max_length=args.max_length,
        )
        
        if raw_train_datals is None:
            print("ERROR: Failed to load ScienceQA data")
            return -1
    else:
        print(f"ERROR: Unsupported task {args.dataset_task}")
        return -1

    print("Data LOADING done.")
    print(f">>>> Num of params: {lm.num_parameters()}")
    
    # 关键修改6：对于 LLaVA 模型，强制使用 LoRA
    # 决定是否冻结 vision tower
    if args.use_lora == 1 or float(lm.num_parameters()) > 6e9:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        
        print(">>>> Applying LoRA to LLaVA model")
        print(">>>> Note: Vision tower will be frozen")
        
        # 如果启用了4bit量化，需要准备模型
        if args.use_4bit == 1:
            print(">>>> Preparing model for k-bit training...")
            lm = prepare_model_for_kbit_training(lm)
        
        # 关键修改7：配置 LoRA 目标模块
        # LLaVA 包含 language_model 和 vision_tower
        # 通常只对 language_model 的线性层应用 LoRA
        lora_config = LoraConfig(
            r=args.rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.0,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", 
                          "gate_proj", "up_proj", "down_proj"],  # LLaMA架构的注意力和MLP层
            # 不包含 vision_tower 的模块，从而冻结视觉编码器
        )
        
        model = get_peft_model(lm, lora_config)
        lm = model
        print(f">>>> Type of the model: {type(lm)}")
        lm.print_trainable_parameters()

    # 关键修改8：调用多模态训练模块
    if "LoRD" in args.task or "lord" in args.task.lower():
        print("TRAIN WITH MULTIMODAL LORD!!!")
        from train_pod_mul import train
        
        # 获取image_processor
        image_processor = processor.image_processor
        
        train(
            lm,
            lm_tokenizer,  # tokenizer
            image_processor,  # image_processor
            args,
            raw_train_datals,
            max_new_tokens=args.max_new_tokens,
        )
    else:
        print(f"ERROR: Task {args.task} not supported for multimodal training")
        return -1

    print("EVERYTHING in the MULTIMODAL TRAINING now DONE.")
    return 0


# running entry
if __name__ == "__main__":
    main()
    print("EVERYTHING DONE.")
