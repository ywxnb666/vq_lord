# 多模态模型窃取 (Multimodal Model Extraction)

## 📋 概述

本项目扩展了原有的 LoRD（Learning from Outputs for Black-box Distillation）方法，支持**多模态模型窃取**。具体来说，使用 **LLaVA-Llama-3-8B** 作为学生模型，通过 LoRD 方法在 **ScienceQA** 数据集上窃取教师模型（如 GPT-4V）的多模态推理能力。

### 🎯 核心特性

- ✅ 支持图像+文本的多模态输入
- ✅ 基于 LLaVA-Llama-3 架构的学生模型
- ✅ LoRD-VI 方法的多模态适配
- ✅ LoRA 高效微调（冻结 Vision Tower）
- ✅ ScienceQA 数据集自动下载与处理
- ✅ 模型权重合并工具

---

## 📂 文件结构

```
align/
├── sciqa_process.py              # ScienceQA 数据处理
├── lord_train_mul.py             # 多模态训练主脚本
├── train_pod_mul.py              # 多模态 LoRD 训练逻辑
├── merge_lora_mul.py             # LoRA 权重合并工具
├── scripts/
│   └── 6.0.sciqa_lord6_lora.sh  # 训练启动脚本
└── README_MULTIMODAL.md          # 本文档
```

---

## 🚀 快速开始

### 1. 环境准备

确保已安装以下依赖：

```bash
conda activate align  # 激活环境
pip install transformers>=4.40.0 datasets pillow peft accelerate
```

### 2. 模型准备

确保学生模型路径正确：

```bash
# 默认路径：/root/workspace/models/llama3-llava-next-8b-hf
# 如需修改，编辑 lord_train_mul.py 中的 from_path 参数
```

### 3. 运行训练

```bash
cd /root/workspace/align
bash scripts/6.0.sciqa_lord6_lora.sh
```

训练完成后，模型将保存在 `./sciqa_ckpts/` 目录下。

---

## 📝 核心脚本说明

### 1. `sciqa_process.py` - 数据处理

**功能**：
- 自动下载 ScienceQA 数据集
- 构造多模态 Prompt（包含 `<image>` 占位符）
- 将 Lecture + Solution + Answer 组合成教师标签

**关键函数**：

```python
def load_scienceqa_data(processor, task_name="scienceqa", train_num=100, ...):
    """
    返回格式: (prompt_ids, labels, teacher_logits, candidate_tokens, pixel_values)
    """
```

**Prompt 格式示例**：

```
<image>
Question: What is the capital of France?
Options:
(A) London
(B) Paris
(C) Berlin
(D) Madrid
Answer:
```

---

### 2. `lord_train_mul.py` - 训练主脚本

**关键修改**：

| 原版代码 | 多模态版本 |
|---------|-----------|
| `AutoModelForCausalLM` | `LlavaNextForConditionalGeneration` |
| `AutoTokenizer` | `AutoProcessor` (包含 tokenizer + image_processor) |
| 纯文本数据 | 文本 + 图像 (pixel_values) |

**LoRA 配置**：

```python
lora_config = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", 
                    "gate_proj", "up_proj", "down_proj"],
    # 注意：不包含 vision_tower，从而冻结视觉编码器
)
```

---

### 3. `train_pod_mul.py` - LoRD 训练逻辑

**核心修改点**：

#### a. 数据加载
```python
# 原版：(prompts, labels, teacher_logits, candidates)
# 多模态：(prompts, labels, teacher_logits, candidates, pixel_values)
op_ls, oidx2ls, ologits2ls, oidx2_dist, opixel_values_ls = raw_train_datals
```

#### b. 模型生成
```python
# 多模态生成时传入 pixel_values
gen_idx = lm.generate(
    input_ids=prompt,
    pixel_values=chunk_pixel_vals,  # 关键！
    attention_mask=attention_mask,
    max_new_tokens=max_new_tokens,
    ...
)
```

#### c. 前向传播
```python
# 计算 logits 时也需要传入 pixel_values
logits = lm(input_ids=idxs, pixel_values=batch_pixel_values).logits
```

#### d. collate_fn
```python
# 在 DataLoader 的 collate 函数中堆叠图像张量
if len(batch_pixel_values) > 0:
    batch_pixel_values = torch.stack(batch_pixel_values).to(device)
```

---

### 4. `merge_lora_mul.py` - 权重合并

**用途**：将训练好的 LoRA 权重合并回基础模型

**使用方法**：

```bash
python merge_lora_mul.py \
    --base_model /root/workspace/models/llama3-llava-next-8b-hf \
    --lora_path ./sciqa_ckpts/SCIQAscienceqa1281LoRD-VI___period64 \
    --save_path ./sciqa_ckpts/MERGED/llava-sciqa
```

**输出**：
- 完整的可直接推理的 LLaVA 模型
- processor（包含 tokenizer 和 image_processor）

---

## ⚙️ 参数说明

### 训练参数（`scripts/6.0.sciqa_lord6_lora.sh`）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `from_path` | `/root/workspace/models/llama3-llava-next-8b-hf` | 学生模型路径 |
| `victim_model` | `gpt-4-vision-preview` | 教师模型（API） |
| `train_num` | `128` | 训练样本数量 |
| `epoch` | `2` | 训练轮数 |
| `period` | `3` | LoRD 训练周期数 |
| `sub_stage_num` | `64` | 每个周期的阶段数 |
| `max_new_tokens` | `128` | 生成的最大 token 数 |
| `batch_size` | `1` | 训练批次大小（多模态通常为1） |
| `LR` | `3e-5` | 学习率 |
| `temperature` | `1.5` | 生成温度 |
| `rank` | `64` | LoRA 秩 |
| `lora_alpha` | `128` | LoRA alpha |

---

## 🔍 与纯文本版本的对比

| 特性 | 纯文本版本 | 多模态版本 |
|------|-----------|-----------|
| 学生模型 | Llama-3-8B | LLaVA-Llama-3-8B |
| 输入模态 | 文本 | 文本 + 图像 |
| 模型类 | `AutoModelForCausalLM` | `LlavaNextForConditionalGeneration` |
| 处理器 | `AutoTokenizer` | `AutoProcessor` |
| 数据格式 | `(ids, masks, logits)` | `(ids, masks, logits, pixel_values)` |
| 生成参数 | `input_ids` | `input_ids` + `pixel_values` |
| LoRA 目标 | 所有线性层 | 仅 language_model 层 |

---

## 📊 实验流程

### 完整流程图

```
┌─────────────────────┐
│  ScienceQA 数据集   │
│   (图像 + 问题)     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  数据预处理         │
│  - 格式化 Prompt    │
│  - 处理图像         │
│  - Tokenization     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  LoRD 训练循环      │
│  1. 生成双样本      │
│  2. 排序            │
│  3. 计算损失        │
│  4. 反向传播        │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  保存 LoRA 权重     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  合并权重           │
│  (merge_lora_mul)   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  完整模型           │
│  可直接推理         │
└─────────────────────┘
```

---

## 🐛 常见问题

### 1. 内存不足

**问题**：训练时 CUDA OOM

**解决方案**：
- 减小 `batch_size`（已默认为1）
- 减小 `max_length` 或 `max_new_tokens`
- 使用梯度检查点（在 `lord_train_mul.py` 中添加）

### 2. 图像加载失败

**问题**：部分 ScienceQA 样本没有图像

**解决方案**：
- 在 `sciqa_process.py` 中已添加跳过逻辑
- 确保只处理有图像的样本

### 3. 模型生成速度慢

**问题**：多模态生成比纯文本慢很多

**原因**：
- 视觉编码器增加计算量
- 图像特征提取需要时间

**优化方案**：
- 使用更小的视觉编码器
- 预计算图像特征并缓存

### 4. LoRA 权重不兼容

**问题**：合并权重时报错

**解决方案**：
- 确保 `base_model` 路径正确
- 检查 LoRA 的 `target_modules` 是否与模型架构匹配

---

## 📈 性能优化建议

### 1. 训练效率

- **混合精度训练**：已使用 `torch.bfloat16`
- **梯度累积**：通过 `acc_step` 参数控制
- **分布式训练**：可修改脚本支持多 GPU

### 2. 数据处理

- **预处理缓存**：预先处理图像并保存
- **动态 Padding**：根据 batch 长度动态调整
- **多进程加载**：在 DataLoader 中设置 `num_workers`

### 3. 模型优化

- **量化**：使用 `bitsandbytes` 进行 4-bit/8-bit 量化
- **Flash Attention**：启用 Flash Attention 2
- **选择性冻结**：根据任务决定是否冻结 Vision Tower

---

## 📖 相关论文

- **LoRD**: *Learning from Outputs for Black-box Distillation* (2024)
- **LLaVA**: *Visual Instruction Tuning* (NeurIPS 2023)
- **ScienceQA**: *A Multimodal Reasoning Challenge* (NeurIPS 2022)

---

## 👨‍💻 代码维护

### 关键修改标记

代码中所有多模态相关的修改都用注释标记：

```python
# 多模态关键修改：说明修改内容
# 关键修改1、2、3...：按顺序标记
```

### 文件对应关系

| 纯文本版本 | 多模态版本 |
|-----------|-----------|
| `qa_process.py` | `sciqa_process.py` |
| `lord_train.py` | `lord_train_mul.py` |
| `train_pod2.py` | `train_pod_mul.py` |
| `merge_lora.py` | `merge_lora_mul.py` |
| `6.2.qa_lord6_lora.sh` | `6.0.sciqa_lord6_lora.sh` |

---

## 🎓 使用建议

### 1. 初次使用

建议先用小数据集测试：

```bash
# 修改 scripts/6.0.sciqa_lord6_lora.sh
export TRAIN_NUMS=(16)  # 从 16 个样本开始
export sub_stage_num=8   # 减少训练步数
```

### 2. 生产环境

逐步增加数据量和训练步数：

```bash
export TRAIN_NUMS=(64 128 256)
export sub_stage_num=64
export period=3
```

### 3. 评估

训练完成后，使用 `sciqa_process.py` 中的 `infer_scienceqa` 函数评估模型性能。

---

## 📞 联系与支持

如有问题，请检查：
1. GPU 内存是否充足（建议 >= 24GB）
2. 模型路径是否正确
3. 数据集是否成功下载

---

**最后更新**: 2024年12月

**版本**: v1.0

**许可**: 遵循原项目许可协议
