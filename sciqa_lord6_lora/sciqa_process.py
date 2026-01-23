"""
======================================================================
SCIQA_PROCESS ---

ScienceQA dataset's process for multimodal model extraction.

    Author: Adapted for multimodal learning
    Copyright © 2024, ZiLiang, all rights reserved.
    Created: December 2024
======================================================================
"""

# ------------------------ Code --------------------------------------
import os

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["TORCH_USE_CUDA_DSA"] = "1"

from datasets import load_dataset
import json
import random
from tqdm import tqdm
import numpy as np
from collections import OrderedDict
from pprint import pprint

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from transformers import LlavaNextProcessor
from PIL import Image
import torch


def safe_decode(text):
    """修复模型输出中的编码问题"""
    if isinstance(text, bytes):
        try:
            return text.decode("utf-8")
        except UnicodeDecodeError:
            try:
                return text.decode("latin-1")
            except UnicodeDecodeError:
                return text.decode("utf-8", errors="ignore")
    
    if isinstance(text, str):
        try:
            if '\\x' in text:
                import re
                pattern = re.compile(r'\\x([0-9a-fA-F]{2})')
                def replace_hex(match):
                    return chr(int(match.group(1), 16))
                text = pattern.sub(replace_hex, text)
            
            try:
                return text.encode('latin-1').decode('utf-8')
            except (UnicodeEncodeError, UnicodeDecodeError):
                try:
                    return text.encode('utf-8').decode('utf-8')
                except:
                    return text
        except:
            return text
    
    return text


def load_scienceqa_data(
    processor,
    task_name="scienceqa",
    train_num=100,
    model_name="gpt-4-vision-preview",  # 教师模型使用支持视觉的OpenAI API
    topk=5,
    max_length=256,
    openai_tmp_save_pth="./STEALED_PKLS/sciqa_data_saveto_",
):
    """
    加载 ScienceQA 数据集并构造多模态输入
    
    关键修改：
    1. 使用 LlavaNextProcessor 处理图像和文本
    2. Prompt 包含 <image> 占位符
    3. Label 组合 Lecture + Solution + Answer
    """
    
    # 加载 ScienceQA 数据集
    try:
        dataset = load_dataset("derek-thomas/ScienceQA", split="train")
        # 关键修改：优先选择有图像的样本
        dataset_with_images = [item for item in dataset if item.get("image") is not None]
        print(f"Found {len(dataset_with_images)} samples with images out of {len(dataset)} total")
        
        if len(dataset_with_images) < train_num:
            print(f"Warning: Only {len(dataset_with_images)} samples with images, but {train_num} requested")
            print("Using all available samples with images")
            selected_data = dataset_with_images
        else:
            # 随机打乱并选择
            import random
            random.seed(20240306)
            random.shuffle(dataset_with_images)
            selected_data = dataset_with_images[:train_num]
    except Exception as e:
        print(f"Failed to load ScienceQA from HuggingFace: {e}")
        print("Please ensure the dataset is available or provide local path")
        return None
    
    inp_ls = []
    image_ls = []
    label_ls = []
    
    for item in tqdm(selected_data, desc="Processing ScienceQA data"):
        # 提取问题和选项
        question = item.get("question", "")
        choices = item.get("choices", [])
        
        # 构造选项文本
        choices_text = ""
        for idx, choice in enumerate(choices):
            choices_text += f"({chr(65+idx)}) {choice}\n"
        
        # 构造输入 prompt (包含图像占位符)
        # 注意：LLaVA 格式通常需要 <image> 标记
        image = item.get("image")
        if image is not None:
            text = f"<image>\nQuestion: {question}\nOptions:\n{choices_text}Answer:"
            image_ls.append(image)
        else:
            # 这个分支理论上不会执行（因为我们已经过滤了）
            text = f"Question: {question}\nOptions:\n{choices_text}Answer:"
            image_ls.append(None)
        
        inp_ls.append(text)
        
        # 构造标签：Lecture + Solution + Answer
        lecture = item.get("lecture", "")
        solution = item.get("solution", "")
        answer = choices[item.get("answer", 0)] if choices else ""
        
        # 组合成教师的 Ground Truth
        if lecture:
            label_text = f"Explanation: {lecture}\nSolution: {solution}\nAnswer: {answer}"
        else:
            label_text = f"Solution: {solution}\nAnswer: {answer}"
        
        label_ls.append(label_text)
    
    # 处理图像和文本，生成 token IDs 和 pixel_values
    p_idxls = []
    pixel_values_ls = []
    image_sizes_ls = []  # 关键修改：添加 image_sizes
    idx2ls = []  # 关键修改：存储 tokenized 的标签
    
    for text, image, label in zip(inp_ls, image_ls, label_ls):
        if image is not None:
            # 多模态输入：使用 processor 处理图像和文本
            inputs = processor(text=text, images=image, return_tensors="pt")
            p_idxls.append(inputs['input_ids'][0])
            # 关键修改：保留 processor 返回的原始格式
            pv = inputs.get('pixel_values')
            if pv is not None:
                # 保持原样,不去掉任何维度
                # LLaVA-Next 的 pixel_values 可能是 [1, num_patches, C, H, W]
                # 我们需要存储去掉batch维度后的版本: [num_patches, C, H, W] 或 [C, H, W]
                pixel_values_ls.append(pv.squeeze(0))  # 只去掉第一个维度(batch)
            else:
                pixel_values_ls.append(None)
            
            # 关键修改：添加 image_sizes (LlavaNext 必需)
            # image.size 是 (width, height), 需要转换为 (height, width)
            # 直接创建 [2] 形状的 tensor，不要额外的维度
            image_sizes_ls.append(torch.tensor(list(image.size[::-1])))
        else:
            # 纯文本输入
            inputs = processor(text=text, return_tensors="pt")
            p_idxls.append(inputs['input_ids'][0])
            pixel_values_ls.append(None)
            image_sizes_ls.append(None)
        
        # 关键修改：将标签也 tokenize
        # 只对 label 进行 tokenize（不包含图像），然后拼接到 prompt tokens 后面
        label_inputs = processor.tokenizer(label, return_tensors="pt", add_special_tokens=False)
        label_tokens = label_inputs['input_ids'][0].tolist()
        
        # 拼接 prompt tokens + label tokens
        prompt_tokens = inputs['input_ids'][0].tolist()
        full_tokens = prompt_tokens + label_tokens
        idx2ls.append(full_tokens)
    
    # 保存路径
    task_name1 = task_name.replace("/", "_")
    openai_tmp_save_pth += f"SCIQAtask_{task_name1}-trainNUM_{train_num}.pkl"
    
    # 调用教师模型获取输出（需要支持多模态的API）
    # 这里需要使用支持视觉的 OpenAI API (如 gpt-4-vision)
    # 注意：这里的 commonly_used_openai_post_process 需要修改以支持多模态
    # 或者使用专门的多模态API调用函数
    # 暂时返回基本数据结构
    
    print(f"Loaded {len(inp_ls)} samples from ScienceQA")
    if len(inp_ls) > 0:
        print(f"Sample prompt text: {inp_ls[0][:100]}...")
        print(f"Sample label text: {label_ls[0][:100]}...")
        print(f"Sample prompt tokens length: {len(p_idxls[0])}")
        print(f"Sample full tokens length: {len(idx2ls[0])}")
        print(f"Sample idx2ls[0] type: {type(idx2ls[0])}")
        print(f"Sample idx2ls[0][:10]: {idx2ls[0][:10]}")
        if pixel_values_ls[0] is not None:
            print(f"Sample pixel_values shape: {pixel_values_ls[0].shape}")
        else:
            print(f"Sample pixel_values: None (no image)")
    
    # 返回格式：(prompts, tokenized_full_sequences, teacher_logits, candidate_tokens, pixel_values, image_sizes)
    # 关键修改：返回 tokenized 的完整序列和 image_sizes
    return (p_idxls, idx2ls, None, None, pixel_values_ls, image_sizes_ls)


def infer_scienceqa(
    model,
    processor,
    task_name="scienceqa",
    res_pth="./sciqa_infer_res.json",
    test_set_take_num=100,
    max_new_tokens=128,
    base_model_name=None,
):
    """
    在 ScienceQA 测试集上进行推理
    
    关键修改：
    1. 支持图像输入 (pixel_values)
    2. 使用 processor 处理输入
    3. 在模型生成时传入图像特征
    """
    save_pth = res_pth
    
    # 加载测试集
    try:
        testset = load_dataset("derek-thomas/ScienceQA", split="validation")
        testset = testset.shuffle(seed=20240307).select(range(min(test_set_take_num, len(testset))))
    except Exception as e:
        print(f"Failed to load ScienceQA test set: {e}")
        return []
    
    inp_ls = []
    
    for item in testset:
        question = item.get("question", "")
        choices = item.get("choices", [])
        
        choices_text = ""
        for idx, choice in enumerate(choices):
            choices_text += f"({chr(65+idx)}) {choice}\n"
        
        answer_idx = item.get("answer", 0)
        label = chr(65 + answer_idx)  # A, B, C, D...
        
        if item.get("image") is not None:
            text = f"<image>\nQuestion: {question}\nOptions:\n{choices_text}Answer:"
            inp_ls.append((text, label, item["image"]))
        else:
            text = f"Question: {question}\nOptions:\n{choices_text}Answer:"
            inp_ls.append((text, label, None))
    
    res_ls = []
    
    # 推理循环
    for text, label, image in tqdm(inp_ls, desc="Inferencing on ScienceQA"):
        try:
            if image is not None:
                # 多模态推理
                inputs = processor(text=text, images=image, return_tensors="pt")
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                    )
                
                # 解码输出
                output_text = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
                
                # 提取生成部分（去掉输入prompt）
                if text in output_text:
                    output_text = output_text.split(text)[-1]
                
                output_text = safe_decode(output_text)
                print(f"Generated: {output_text}")
                res_ls.append((output_text, label))
            else:
                # 纯文本推理
                inputs = processor(text=text, return_tensors="pt")
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                    )
                
                output_text = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
                
                if text in output_text:
                    output_text = output_text.split(text)[-1]
                
                output_text = safe_decode(output_text)
                print(f"Generated: {output_text}")
                res_ls.append((output_text, label))
                
        except Exception as e:
            print(f"Error during inference: {e}")
            res_ls.append(("", label))
    
    # 保存结果
    with open(save_pth, 'w', encoding='utf8') as f:
        json.dump(res_ls, f, ensure_ascii=False, indent=4)
    print(f"Saved inference results to {save_pth}")
    
    return res_ls


def eval_sciqa_acc(res_ls):
    """
    评估 ScienceQA 的准确率
    
    从生成的文本中提取答案选项 (A, B, C, D...)
    """
    predictions = []
    labels = []
    
    for generated_text, true_label in res_ls:
        generated_text = generated_text.strip().upper()
        
        # 尝试从生成文本中提取答案
        # 寻找 "Answer: X" 或单独的 "A", "B", "C", "D"
        predicted_label = None
        
        if "ANSWER:" in generated_text:
            parts = generated_text.split("ANSWER:")
            if len(parts) > 1:
                answer_part = parts[1].strip()
                if answer_part and answer_part[0] in ['A', 'B', 'C', 'D', 'E']:
                    predicted_label = answer_part[0]
        
        if predicted_label is None:
            # 尝试找第一个出现的选项字母
            for char in generated_text:
                if char in ['A', 'B', 'C', 'D', 'E']:
                    predicted_label = char
                    break
        
        if predicted_label is None:
            predicted_label = "A"  # 默认值
        
        predictions.append(predicted_label)
        labels.append(true_label)
    
    # 计算准确率
    accuracy = accuracy_score(labels, predictions)
    
    print(f"Accuracy: {accuracy:.4f}")
    
    return accuracy


# running entry
if __name__ == "__main__":
    print("ScienceQA processing module")
    print("This module should be imported by training scripts")
    print("EVERYTHING DONE.")
