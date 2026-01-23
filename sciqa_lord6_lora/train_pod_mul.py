"""
======================================================================
TRAIN_POD_MUL ---

Multimodal stealing mechanism for LLaVA models.
Based on train_pod2.py with multimodal extensions.

    Author: Adapted for multimodal learning
    Copyright © 2024, ZiLiang, all rights reserved.
    Created: December 2024
======================================================================
"""


# ------------------------ Code --------------------------------------
import math
import torch
# import json
from torch.utils.tensorboard import SummaryWriter
# from torch.distributions import Categorical
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import math
import time
# import argparse
# from transformers import AutoModelForCausalLM
# from transformers import AutoModelForSequenceClassification
# from transformers import AutoModelForTokenClassification
# from transformers import AutoTokenizer, AutoConfig, AutoModel

# from training_data_collecting_openai import load_raw_train_datals
# from training_data_collecting_openai import load_steal_datals
# from glue_process import load_glue_datals

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sequence_utils import my_padding, my_padding_logits
from sequence_utils import my_padding_token_dist
from sequence_utils import my_padding_logit
from sequence_utils import left_pad
from sequence_utils import random_shut 

import torch.nn.functional as F

# from rlhf_train import clip, log_clip
import random

from rlhf_train import clip, log_clip


from peft import (
    LoraConfig,
    # PeftConfig,
    # PeftModel,
    get_peft_model,
    # prepare_model_for_kbit_training,
)
from peft import PeftModel
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer, AutoConfig, AutoModel


import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
printt=print
print = logger.info


def random_take(num, ls, seed,):
    random.seed(seed)
    random.shuffle(ls)
    if num >= len(ls):
        return ls
    return ls[:num]


def train(lm, lm_tokenizer, image_processor, args,
          raw_train_datals, max_new_tokens=16):
    """
    多模态训练主函数
    关键修改：添加 image_processor 参数
    """
    sub_stage_num = args.sub_stage_num

    steps = sub_stage_num*args.sub_set_num *\
        args.period_num
    print(f"OVERALL STEPS: {steps}.")

    p_i_11_ls = None
    p_i_12_ls = None
    p_m_11_ls = None
    p_m_12_ls = None
    p_logits_11_ls = None
    p_logits_12_ls = None

    p_i2ls = None
    pmask2s = None
    p_logits2ls = None
    p_vic_logits2ls = None

    pp11ls=None
    pp12ls=None
    
    # 多模态特有：存储图像数据
    p_pixel_values_ls = None
    p_image_sizes_ls = None

    preset_subset_num=args.sub_set_num
    subset_pointer=0

    for ssn in tqdm(range(sub_stage_num)):

        print(f"stage_num: {ssn+1}.")
        if (ssn+1)%args.save_step==0:
            print(f" ------>NOW save the ckpt in stage {ssn+1}.")
            args.temp_save_path=args.save_path+"___period"+str(ssn+1)
            lm_tokenizer.save_pretrained(args.temp_save_path)
            lm.save_pretrained(args.temp_save_path)

        print(f"subset_num: {args.sub_set_num}.")

        lm, p_i_11_ls, p_i_12_ls, p_m_11_ls,\
            p_m_12_ls, p_logits_11_ls, p_logits_12_ls,\
            p_i2ls, pmask2s, p_logits2ls, p_vic_logits2ls, \
            pp11ls, pp12ls, p_pixel_values_ls, p_image_sizes_ls = train_pod(
                lm, lm_tokenizer, image_processor,
                        args, raw_train_datals,
                        max_new_tokens,
                        p_i_11_ls, p_i_12_ls, p_m_11_ls,
                        p_m_12_ls, p_logits_11_ls, p_logits_12_ls,
                        p_i2ls, pmask2s, p_logits2ls, p_vic_logits2ls,
                        pp11ls, pp12ls, p_pixel_values_ls, p_image_sizes_ls,
                        subset_pointer,
                        )
        subset_pointer+=1


from accelerate import load_checkpoint_and_dispatch

def train_pod(lm,
              lm_tokenizer,
              image_processor,  # 多模态关键修改：添加图像处理器
              args, raw_train_datals,
              max_new_tokens,
              p_i_11_ls, p_i_12_ls, p_m_11_ls,
              p_m_12_ls, p_logits_11_ls, p_logits_12_ls,
              p_i2ls, pmask2s, p_logits2ls, p_vic_logits2ls,
              pp11ls, pp12ls, 
              p_pixel_values_ls,  # 多模态关键修改：添加像素值列表
              p_image_sizes_ls,  # 多模态关键修改：添加图像尺寸列表
              subset_pointer,
              ):

    tau1 = args.tau1
    tau2 = args.tau2
    tau_delta = args.tau_delta
    print(f" Tau1: {tau1}\t Tau2: {tau2}.")
    print(f"MAX NEW TOKENS: {max_new_tokens}.")
    
    ITER_num = args.period_num
    tb_writer = SummaryWriter(log_dir=args.save_path+"___log_writer")
    
    # 多模态数据解包：包含图像和 image_sizes
    op_ls, oidx2ls, ologits2ls, oidx2_dist, opixel_values_ls, oimage_sizes_ls = raw_train_datals

    subset_num = args.sub_set_num

    seed = time.time()
    if subset_pointer>= math.floor(len(op_ls)/subset_num)-1:
        subset_pointer=subset_pointer%(math.floor(len(op_ls)/subset_num))
    
    p_ls = op_ls[subset_pointer*subset_num:\
                (subset_pointer+1)*subset_num]
    idx2ls = oidx2ls[subset_pointer*subset_num:\
                    (subset_pointer+1)*subset_num]
    
    # 多模态关键修改：提取对应的图像数据和 image_sizes
    pixel_values_ls = opixel_values_ls[subset_pointer*subset_num:\
                                       (subset_pointer+1)*subset_num]
    image_sizes_ls = oimage_sizes_ls[subset_pointer*subset_num:\
                                     (subset_pointer+1)*subset_num]

    if ologits2ls is not None:
        vic_logits2ls = ologits2ls[subset_pointer*subset_num:\
                                   (subset_pointer+1)*subset_num]
        idx2_dist = oidx2_dist[subset_pointer*subset_num:\
                                   (subset_pointer+1)*subset_num]
    else:
        vic_logits2ls = [None for _ in range(subset_num)]
        idx2_dist = [None for _ in range(subset_num)]

    need_pre_store = 0
    if p_i_11_ls is None:
        p_i_11_ls = []
        p_i_12_ls = []
        p_logits_11_ls = []
        p_logits_12_ls = []
        p_m_11_ls = []
        p_m_12_ls = []

        need_pre_store = 1
        period_break = 0

        p_i2ls = []
        pmask2s = []
        p_logits2ls = []
        p_vic_logits2ls = []
        p_pixel_values_ls = []  # 多模态修改
        p_image_sizes_ls = []  # 多模态修改
    else:
        period_break = 1

    for iter_idx in range(ITER_num):
        tensorboard_name = f"Period {iter_idx}"

        idxs11_ls = []
        idxs12_ls = []
        old_logits11_ls = []
        old_logits12_ls = []
        old_logits2_ls = []
        # 多模态关键修改：保存每个生成序列对应的 pixel_values 和 image_sizes
        gen_pixel_values_ls = []
        gen_image_sizes_ls = []

        # 2. generate
        with torch.no_grad():

            # 首先计算 old_logits2（教师样本的logits）
            for i in range(len(idx2ls)):
                # 关键修改：确保 idx2ls[i] 是列表或tensor，不是字符串
                if isinstance(idx2ls[i], str):
                    print(f"Warning: idx2ls[{i}] is a string, skipping...")
                    continue
                
                idxs2 = torch.tensor(idx2ls[i], dtype=torch.long).unsqueeze(0).to("cuda")
                
                # 调试/安全检查：确保 input_ids 在词表范围内
                # 关键修复：LlavaNextConfig 的 vocab_size 在 text_config 中
                if hasattr(lm.config, 'vocab_size'):
                    vocab_size = lm.config.vocab_size
                elif hasattr(lm.config, 'text_config') and hasattr(lm.config.text_config, 'vocab_size'):
                    vocab_size = lm.config.text_config.vocab_size
                else:
                    vocab_size = lm.get_input_embeddings().weight.shape[0]
                
                if lm.get_input_embeddings().weight.shape[0] > vocab_size:
                    vocab_size = lm.get_input_embeddings().weight.shape[0]
                
                if idxs2.max() >= vocab_size:
                    print(f"ERROR: Input IDs exceed vocab size! Max ID: {idxs2.max()}, Vocab Size: {vocab_size}")
                    # 尝试截断或替换 (仅用于防止崩溃，实际应修复 tokenizer/model)
                    idxs2 = torch.clamp(idxs2, max=vocab_size-1)
                
                # 关键修改：处理可能为 None 的 pixel_values
                pixel_vals = pixel_values_ls[i]
                image_sizes = image_sizes_ls[i]
                
                if pixel_vals is not None and image_sizes is not None:
                    # pixel_vals 应该已经是 [num_patches, C, H, W] 或 [C, H, W] 格式
                    # 添加 batch 维度: [1, ...]
                    pixel_vals = pixel_vals.unsqueeze(0).to("cuda")
                    # image_sizes 也需要添加 batch 维度
                    if image_sizes.dim() == 2:  # 已经是 [1, 2] 格式
                        image_sizes = image_sizes.to("cuda")
                    else:  # [2] 格式，需要添加batch维度
                        image_sizes = image_sizes.unsqueeze(0).to("cuda")
                    
                    # 调试：检查 pixel_values 和 image_sizes 形状
                    # print(f"Debug: pixel_values shape: {pixel_vals.shape}, image_sizes: {image_sizes}")
                else:
                    # 如果没有图像,跳过这个样本(因为LLaVA需要图像输入)
                    print(f"Warning: No pixel_values or image_sizes for sample {i}, skipping...")
                    continue
                
                # 多模态关键修改：传入 pixel_values 和 image_sizes
                # 修复：显式传入 attention_mask，防止 NoneType 错误
                attention_mask = torch.ones_like(idxs2)
                old_logits2 = lm(
                    input_ids=idxs2, 
                    pixel_values=pixel_vals,
                    image_sizes=image_sizes,  # 关键：添加 image_sizes
                    attention_mask=attention_mask
                ).logits
                old_logits2 = old_logits2[:,:-1]
                old_logits2 = F.log_softmax(old_logits2, dim=-1)
                bs, sql2 = idxs2.shape
                old_logits2 = old_logits2[
                    torch.arange(1).unsqueeze(1),
                    torch.arange(sql2-1).unsqueeze(0),
                    idxs2[:, 1:sql2]
                ]
                old_logits2_ls.append(old_logits2.squeeze(0).to("cpu"))

            if args.with_early_shut==1:
                print("EXECUTE EARLY SHUT...")
                p_ls=random_shut(p_ls)

            chunked_size=args.infer_batch_size
            num_chunks=math.floor(len(p_ls)/chunked_size)
            if num_chunks*chunked_size - len(p_ls)==0.0:
                num_range=num_chunks
            else:
                num_range=num_chunks+1

            # 生成两个不同的样本
            for i_chunked in range(num_range):
                print(f"Chunks: {i_chunked}/{num_chunks}")
                if i_chunked == num_chunks:
                    if i_chunked*chunked_size!=len(p_ls):
                        prompt=p_ls[i_chunked*chunked_size:]
                        chunk_pixel_vals_list = pixel_values_ls[i_chunked*chunked_size:]
                        chunk_image_sizes_list = image_sizes_ls[i_chunked*chunked_size:]
                        prompt=left_pad(prompt,lm_tokenizer.bos_token_id)
                        prompt=prompt.to(args.device)
                else:
                    prompt=p_ls[i_chunked*chunked_size:\
                                (i_chunked+1)*chunked_size]
                    chunk_pixel_vals_list = pixel_values_ls[i_chunked*chunked_size:\
                                                        (i_chunked+1)*chunked_size]
                    chunk_image_sizes_list = image_sizes_ls[i_chunked*chunked_size:\
                                                        (i_chunked+1)*chunked_size]
                    prompt=left_pad(prompt,lm_tokenizer.bos_token_id)
                    prompt=prompt.to(args.device)

                # 多模态关键修改：堆叠图像张量和 image_sizes
                # pixel_values 应该是 [num_patches, C, H, W] 或 [C, H, W] 格式
                valid_pixel_vals = []
                valid_image_sizes = []
                valid_indices = []
                for idx, (pv, img_size) in enumerate(zip(chunk_pixel_vals_list, chunk_image_sizes_list)):
                    if pv is not None and img_size is not None:
                        if not isinstance(pv, torch.Tensor):
                            pv = torch.tensor(pv)
                        # 如果是 3D: [C, H, W],保持不变
                        # 如果是 4D: [num_patches, C, H, W],也保持不变
                        # 只有在有batch维度时才squeeze
                        while pv.dim() > 4:
                            pv = pv.squeeze(0)
                        valid_pixel_vals.append(pv)
                        valid_image_sizes.append(img_size)
                        valid_indices.append(idx)
                    else:
                        print(f"Warning: None pixel_value or image_size in chunk, skipping this sample")
                        continue
                
                if len(valid_pixel_vals) == 0:
                    print("Warning: No valid pixel_values in chunk, skipping")
                    continue
                
                # 过滤 prompt，只保留有图像的样本
                if len(valid_indices) < len(prompt):
                    prompt = prompt[valid_indices]

                # 堆叠: 如果是 [C, H, W] -> [B, C, H, W]
                #      如果是 [num_patches, C, H, W] -> [B, num_patches, C, H, W]
                chunk_pixel_vals = torch.stack(valid_pixel_vals).to(args.device).to(dtype=lm.dtype)
                chunk_image_sizes = torch.stack(valid_image_sizes).to(args.device)
                
                # 检查 pixel_values 是否有 NaN/Inf
                if torch.isnan(chunk_pixel_vals).any() or torch.isinf(chunk_pixel_vals).any():
                    print("ERROR: pixel_values contains NaN or Inf!")
                    # 尝试修复: 将 NaN 替换为 0
                    chunk_pixel_vals = torch.nan_to_num(chunk_pixel_vals)
                
                # 检查 prompt 是否有越界 token
                # 关键修复：LlavaNextConfig 的 vocab_size 在 text_config 中
                if hasattr(lm.config, 'vocab_size'):
                    vocab_size = lm.config.vocab_size
                elif hasattr(lm.config, 'text_config') and hasattr(lm.config.text_config, 'vocab_size'):
                    vocab_size = lm.config.text_config.vocab_size
                else:
                    vocab_size = lm.get_input_embeddings().weight.shape[0]
                
                if lm.get_input_embeddings().weight.shape[0] > vocab_size:
                    vocab_size = lm.get_input_embeddings().weight.shape[0]
                
                if prompt.max() >= vocab_size:
                    print(f"ERROR: Prompt IDs exceed vocab size! Max ID: {prompt.max()}, Vocab Size: {vocab_size}")
                    prompt = torch.clamp(prompt, max=vocab_size-1)

                printt("++++++++++++++++++++++++++++++++++++++++++++++++")
                printt(f"prompt.shape: {prompt.shape}")
                printt(f"pixel_values.shape: {chunk_pixel_vals.shape}")
                printt(f"image_sizes.shape: {chunk_image_sizes.shape}")
                printt(f"temperature: {args.T}")

                pad_token_id = lm_tokenizer.pad_token_id
                attention_mask = (prompt != pad_token_id).long().to(args.device)

                # 多模态关键修改：生成时传入 pixel_values 和 image_sizes
                gen_idx = lm.generate(
                    input_ids=prompt,
                    pixel_values=chunk_pixel_vals,  # 关键修改
                    image_sizes=chunk_image_sizes,  # 关键修改：添加 image_sizes
                    attention_mask=attention_mask,
                    do_sample=True,
                    max_new_tokens=max_new_tokens,
                    num_return_sequences=2,
                    temperature=args.T,
                    top_p=0.98,
                    use_cache=True,
                    pad_token_id=pad_token_id,
                )

                print("*********************************************")
                print(f"Shape of generated indexes: {gen_idx.shape}")

                idxs11=gen_idx[0::2,:]
                idxs12=gen_idx[1::2,:]

                bs, sqqql = idxs11.shape
                print("========================================")
                print(f"idxs11 {lm_tokenizer.decode(idxs11[0])}")
                print(f"idxs12 {lm_tokenizer.decode(idxs12[0])}")

                # 多模态关键修改：计算logits时传入pixel_values和image_sizes
                # 需要为每个样本复制pixel_values（因为每个prompt生成了2个序列）
                expanded_pixel_vals = chunk_pixel_vals.repeat_interleave(2, dim=0)
                expanded_image_sizes = chunk_image_sizes.repeat_interleave(2, dim=0)
                
                # 修复：显式传入 attention_mask 和 image_sizes
                attention_mask11 = torch.ones_like(idxs11[:, :-1])
                old_logits11 = lm(input_ids=idxs11[:, :-1], 
                                 pixel_values=expanded_pixel_vals[0::2],
                                 image_sizes=expanded_image_sizes[0::2],  # 添加 image_sizes
                                 attention_mask=attention_mask11).logits
                old_logits11 = F.log_softmax(old_logits11, dim=-1)
                old_logits11 = old_logits11[
                    torch.arange(bs).unsqueeze(1),
                    torch.arange(sqqql-1).unsqueeze(0),
                    idxs11[:, 1:sqqql]
                ]

                bs, sqqql2 = idxs12.shape
                # 修复：显式传入 attention_mask 和 image_sizes
                attention_mask12 = torch.ones_like(idxs12[:, :-1])
                old_logits12 = lm(input_ids=idxs12[:, :-1],
                                 pixel_values=expanded_pixel_vals[1::2],
                                 image_sizes=expanded_image_sizes[1::2],  # 添加 image_sizes
                                 attention_mask=attention_mask12).logits
                old_logits12 = F.log_softmax(old_logits12, dim=-1)
                old_logits12 = old_logits12[
                    torch.arange(bs).unsqueeze(1),
                    torch.arange(sqqql2-1).unsqueeze(0),
                    idxs12[:, 1:sqqql2]
                ]

                idxs11_ls.extend([x for x in idxs11.to("cpu")])
                idxs12_ls.extend([x for x in idxs12.to("cpu")])
                old_logits11_ls.extend([x for x in old_logits11.to("cpu")])
                old_logits12_ls.extend([x for x in old_logits12.to("cpu")])
                
                # 多模态关键修改：保存每个样本对应的 pixel_values 和 image_sizes
                # idxs11 和 idxs12 来自同一批样本，所以用相同的 pixel_values
                for pv, isz in zip(chunk_pixel_vals.to("cpu"), chunk_image_sizes.to("cpu")):
                    gen_pixel_values_ls.append(pv)
                    gen_image_sizes_ls.append(isz)

        # do truncations and paddings.
        cmax_token_num_2 = min(args.max_length,
                               max([len(x) for x in idx2ls]))
        cmax_token_num_11 = min(args.max_length,
                                max([len(x) for x in idxs11_ls]))
        cmax_token_num_12 = min(args.max_length,
                                max([len(x) for x in idxs12_ls]))

        max_token_num = max(cmax_token_num_2, cmax_token_num_11)
        max_token_num = max(max_token_num, cmax_token_num_12)

        print(f"max_token_num: {max_token_num}")
        pad_idx = lm_tokenizer.pad_token_id

        idx2ls, mask2 = my_padding(idx2ls, p_ls,
                                   max_token_num, pad_idx)
        idxs11_ls, mask11 = my_padding(idxs11_ls,
                                       p_ls, max_token_num, pad_idx)
        idxs12_ls, mask12 = my_padding(idxs12_ls,
                                       p_ls, max_token_num, pad_idx)
        if args.with_early_shut==1:
            mask11=torch.ones_like(mask11)
            mask12=torch.ones_like(mask12)

        old_logits11_ls = my_padding_logit(old_logits11_ls,
                                           max_token_num-1, pad_idx)
        old_logits12_ls = my_padding_logit(old_logits12_ls,
                                           max_token_num-1, pad_idx)
        old_logits2_ls = my_padding_logit(old_logits2_ls,
                                          max_token_num-1, pad_idx)

        if vic_logits2ls[0] is not None:
            newvic_logits2ls = []
            for per_data in vic_logits2ls:
                sl = len(per_data)
                v = len(per_data[0])
                tmp_ts = torch.ones((sl, v), dtype=torch.float)
                for jjjj, per_token_logit in enumerate(per_data):
                    tmp_ts[jjjj] = torch.tensor(per_token_logit,)
                newvic_logits2ls.append(tmp_ts)

            vic_logits2ls = my_padding_logits(newvic_logits2ls,
                                            max_token_num-1, pad_idx)
            idxs2_dist = my_padding_token_dist(idx2_dist,
                                            max_token_num-1, pad_idx)

        # 构造正负样本对
        if need_pre_store == 1:
            need_pre_store = 0
            p_i_11_ls = idx2ls
            p_i_12_ls = idxs12_ls

            p_logits_11_ls = old_logits2_ls
            p_logits_12_ls = old_logits12_ls

            p_m_11_ls = mask2
            p_m_12_ls = mask12

            p_i2ls = idx2ls
            pmask2s = mask2
            p_logits2ls = old_logits2_ls
            p_vic_logits2ls = vic_logits2ls
            
            # 多模态关键修改：使用生成时保存的 pixel_values 和 image_sizes
            # 这样保证与 idxs11_ls/idxs12_ls 的对应关系
            p_pixel_values_ls = gen_pixel_values_ls
            p_image_sizes_ls = gen_image_sizes_ls

            pp11ls=[]
            pp12ls=[]
            for i, prompt in enumerate(p_ls):
                p11=float(torch.exp(
                    torch.sum(
                        p_logits_11_ls[i]*p_m_11_ls[i, :-1]
                        )/torch.sum(p_m_11_ls[i, :-1])))
                p12=float(torch.exp(
                    torch.sum(
                        p_logits_12_ls[i]*p_m_12_ls[i, :-1]
                        )/torch.sum(p_m_12_ls[i, :-1])))
                
                pp11ls.append(p11)
                pp12ls.append(p12)

        else:
            # 后续轮次：使用前一轮的数据构造对比对
            for i, prompt in enumerate(p_ls):
                p11=float(torch.exp(
                    torch.sum(
                        old_logits2_ls[i]*mask2[i, :-1]
                        )/torch.sum(mask2[i, :-1])))
                p12=float(torch.exp(
                    torch.sum(
                        old_logits12_ls[i]*mask12[i, :-1]
                        )/torch.sum(mask12[i, :-1])))

                if p11>p12:
                    p_i_11_ls[i]=idx2ls[i]
                    p_i_12_ls[i]=idxs12_ls[i]
                    p_logits_11_ls[i]=old_logits2_ls[i]
                    p_logits_12_ls[i]=old_logits12_ls[i]
                    p_m_11_ls[i]=mask2[i]
                    p_m_12_ls[i]=mask12[i]
                else:
                    p_i_11_ls[i]=idxs12_ls[i]
                    p_i_12_ls[i]=idx2ls[i]
                    p_logits_11_ls[i]=old_logits12_ls[i]
                    p_logits_12_ls[i]=old_logits2_ls[i]
                    p_m_11_ls[i]=mask12[i]
                    p_m_12_ls[i]=mask2[i]

                p_i2ls[i]=idx2ls[i]
                pmask2s[i]=mask2[i]
                p_logits2ls[i]=old_logits2_ls[i]
                if vic_logits2ls[0] is not None:
                    p_vic_logits2ls[i]=vic_logits2ls[i]

                pp11ls[i]=max(p11,p12)
                pp12ls[i]=min(p11,p12)

        # 创建数据集
        if vic_logits2ls[0] is not None:
            trainset = TensorDataset(
                p_i_11_ls, p_i_12_ls, p_i2ls,
                p_m_11_ls, p_m_12_ls, pmask2s,
                p_logits_11_ls, p_logits_12_ls,
                p_logits2ls, p_vic_logits2ls,
            )
        else:
            trainset = TensorDataset(
                p_i_11_ls, p_i_12_ls, p_i2ls,
                p_m_11_ls, p_m_12_ls, pmask2s,
                p_logits_11_ls, p_logits_12_ls,
                p_logits2ls,
            )

        if period_break == 1:
            print("\n\n NOW BREAK SINCE ENOUGH TRAINING\n\n")
            return lm, p_i_11_ls, p_i_12_ls, p_m_11_ls,\
                p_m_12_ls, p_logits_11_ls, p_logits_12_ls,\
                p_i2ls, pmask2s, p_logits2ls, p_vic_logits2ls,\
                pp11ls, pp12ls, p_pixel_values_ls, p_image_sizes_ls

        loader = DataLoader(trainset,
                            batch_size=args.batch_size,
                            shuffle=True,
                            )

        print(">>>> Period {}".format(iter_idx))
        
        # 多模态关键修改：传入image_processor、pixel_values和image_sizes
        lm = one_period(args, lm,
                        lm_tokenizer,
                        image_processor,  # 新增参数
                        p_pixel_values_ls,  # 新增参数
                        p_image_sizes_ls,  # 新增参数
                        loader,
                        args.epoch, args.device,
                        tb_writer,
                        tensorboard_name,
                        args.save_path,
                        args.LR,
                        args.acc_step, args.log_step,
                        args.save_step,
                        args.beta,
                        is_black_box=0,
                        method=args.task,
                        )

    return lm, p_i_11_ls, p_i_12_ls, p_m_11_ls,\
        p_m_12_ls, p_logits_11_ls, p_logits_12_ls,\
        p_i2ls, pmask2s, p_logits2ls, p_vic_logits2ls, \
        pp11ls, pp12ls, p_pixel_values_ls, p_image_sizes_ls


def one_period(args, lm,
               lm_tokenizer,
               image_processor,  # 多模态新增
               pixel_values_ls,  # 多模态新增
               image_sizes_ls,  # 多模态新增
               loader, epoch, device,
               tb_writer,
               tensorboard_name,
               save_path,
               LR=3e-5,
               acc_step=1,
               log_step=100,
               save_step=1000,
               beta=0.7,
               epsln=1e-6,
               is_black_box=0,
               method="LORD-II",
               ):
    """
    一个训练周期
    多模态关键修改：在前向传播时传入pixel_values和image_sizes
    """

    overall_loss = 0.
    overall_step = 0
    pad_token_id = lm_tokenizer.pad_token_id
    kl_loss = torch.nn.KLDivLoss(reduction="none")
    sigmoid = torch.nn.Sigmoid()

    opt1 = torch.optim.AdamW(lm.parameters(), lr=LR)

    for e in tqdm(range(epoch), desc="epoch"):
        for batch_idx, item in enumerate(tqdm(loader, desc="loader")):
            overall_step += 1

            if len(item) == 10:
                idxs11, idxs12, idxs2, mask11, mask12,\
                    mask2, old_logits11, old_logits12,\
                    old_logits2, vic_logits2 = item
            else:
                idxs11, idxs12, idxs2, mask11, mask12,\
                    mask2, old_logits11, old_logits12,\
                    old_logits2 = item
                vic_logits2 = None

            bs, sqlen1 = idxs11.shape
            sqlen = sqlen1

            idxs11 = idxs11.to(device)
            idxs12 = idxs12.to(device)
            idxs2 = idxs2.to(device)
            mask11 = mask11.to(device)
            mask12 = mask12.to(device)
            mask2 = mask2.to(device)

            old_logits11 = old_logits11.to(device)
            old_logits12 = old_logits12.to(device)
            old_logits2 = old_logits2.to(device)

            if vic_logits2 is not None and args.is_black_box==0:
                vic_logits2 = vic_logits2.to(device)

            # 多模态关键修改：获取当前batch对应的pixel_values
            # 多模态关键修改：收集batch的pixel_values和image_sizes
            batch_pixel_values = []
            batch_image_sizes = []
            batch_start_idx = batch_idx * args.batch_size
            for i in range(bs):
                sample_idx = batch_start_idx + i
                if sample_idx < len(pixel_values_ls):
                    batch_pixel_values.append(pixel_values_ls[sample_idx])
                    batch_image_sizes.append(image_sizes_ls[sample_idx])
                    
            if len(batch_pixel_values) > 0:
                batch_pixel_values = torch.stack(batch_pixel_values).to(device)
                batch_image_sizes = torch.stack(batch_image_sizes).to(device)
            else:
                print(f"Warning: No pixel values for batch {batch_idx}")
                continue

            # 多模态关键修改：前向传播时的处理
            # 注意：由于 idxs11/idxs12 来自 generate() 的输出，
            # 其中的 image tokens 已经被扩展处理过了
            # 在训练阶段，我们不再传入 pixel_values，让模型直接处理已扩展的序列
            # 这样可以避免 "Image features and image tokens do not match" 错误
            logits11 = lm(input_ids=idxs11, attention_mask=mask11).logits[:, :-1, :]
            logits11 = F.log_softmax(logits11, dim=-1)
            logits11 = logits11[torch.arange(bs).unsqueeze(1),
                                torch.arange(sqlen-1).unsqueeze(0),
                                idxs11[:, 1:sqlen]]

            logits12 = lm(input_ids=idxs12, attention_mask=mask12).logits[:, :-1, :]
            logits12 = F.log_softmax(logits12, dim=-1)
            logits12 = logits12[torch.arange(bs).unsqueeze(1),
                                torch.arange(sqlen-1).unsqueeze(0),
                                idxs12[:, 1:sqlen]]

            # idxs2 也是已处理过的序列，不需要 pixel_values
            logits2 = lm(input_ids=idxs2, attention_mask=mask2).logits[:, :-1, :]
            logits2 = torch.log_softmax(logits2, dim=-1)
            logits2_cons = logits2[torch.arange(bs).unsqueeze(1),
                                   torch.arange(sqlen-1).unsqueeze(0),
                                   idxs2[:, 1:sqlen]]

            # LoRD损失计算（保持原有逻辑）
            term1 = torch.mean(log_clip(-old_logits12+logits12))
            term2 = torch.mean(log_clip(old_logits11-logits11))

            if args.is_black_box == 0 and vic_logits2 is not None:
                term3 = \
                    (vic_logits2[:, :, 0]+old_logits2-2*logits2_cons)
            else:
                term3 = old_logits2 - logits2_cons

            term3 = torch.sum(term3 * mask2[:, :-1])

            # 根据方法选择损失函数
            if method in ["LoRD-VI", "LoRD-VII", "LoRD-VIII", "LoRD-IX"]:
                if args.is_black_box == 0 and vic_logits2 is not None:
                    term3 = (vic_logits2[:, :, 0]-logits2_cons)
                else:
                    term3 = - logits2_cons
                term3 = torch.mean(log_clip(term3))

                los2=-1*torch.mean(logits2_cons)
                loss11=-1*torch.mean(logits11)
                loss12=1*torch.mean(logits12)

                loss = los2+loss11+2*loss12

                if method=="LoRD-VII":
                    loss=sigmoid(loss/(los2))
                elif method=="LoRD-VIII":
                    loss=2*sigmoid(args.lambda1*(los2+loss12)/loss11+(1-args.lambda1)*(loss11+loss12)/los2)
                elif method=="LoRD-IX":
                    loss=args.lambda1*log_clip(logits2_cons-logits12)+(1-args.lambda1)*log_clip(logits11-logits12)
                    loss=torch.mean(loss)
                else:
                    loss=sigmoid(loss)

                print(f"term2: {-1*torch.sum(logits2_cons*mask2[:,:-1])/torch.sum(mask2[:,:-1])}")
                print(f"term11: {-1*torch.sum(logits11*mask11[:,:-1])/torch.sum(mask11[:,:-1])}")
                print(f"term12: {-2*torch.sum(logits12*mask12[:,:-1])/torch.sum(mask12[:,:-1])}")
                print(f"LOSS: {loss}\n\n")
            else:
                loss_1 = term1 + term2
                loss_2 = term3
                loss = loss_1 + loss_2

            overall_loss += loss
            overall_loss = loss
            
            if overall_step % log_step == 0:
                print(" LOSS: {}".format(overall_loss))
                print(" Neg Loss: {}".format(term1))
                print(" Pos Loss: {}".format(term2))
                print(" Standard Loss: {}".format(term3))
                tb_writer.add_scalar("loss", overall_loss, overall_step)
                tb_writer.add_scalar("term1", term1, overall_step)
                tb_writer.add_scalar("term2", term2, overall_step)
                tb_writer.add_scalar("term3", term3, overall_step)

            if overall_step % save_step == 0:
                print(" -->Regular Saving.")
                print(f"in epoch {e}, step {overall_step}.")
                lm_tokenizer.save_pretrained(save_path+"___"+str(overall_step))
                lm.save_pretrained(save_path+"___"+str(overall_step))

            if overall_step % acc_step == 0:
                opt1.zero_grad()
                overall_loss.backward()
                opt1.step()
                overall_loss = 0.

    print(" -->Finally Saving.")
    print("ONE PERIOD TRAINING DONE!")
    return lm


if __name__ == "__main__":
    print("TRAIN_POD_MUL READY.")
