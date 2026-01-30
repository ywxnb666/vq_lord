# VQ-LoRD 模块文档

> **用途**: 本文档详细说明 VQ-LoRD 项目各脚本的功能、设计思路和实现细节，供后续开发和调试参考。
> 
> **最后更新**: 2026-01-21

---

## 目录

1. [项目概述](#项目概述)
2. [vq_module.py — VQ 离散化模块](#vq_modulepy--vq-离散化模块)
3. [vision_lord_loss.py — 损失函数模块](#vision_lord_losspy--损失函数模块)
4. [data_collector.py — 数据收集模块](#data_collectorpy--数据收集模块)
5. [train_vq_lord.py — 主训练脚本](#train_vq_lordpy--主训练脚本)
6. [启动脚本和配置文件](#启动脚本和配置文件)
7. [已知问题与注意事项](#已知问题与注意事项)
8. [sciqa_process.py — ScienceQA 验证脚本](#sciqa_processpy--scienceqa-验证脚本)

---

## 项目概述

**VQ-LoRD** (Vector Quantization - Locality Reinforced Distillation) 是一种扩展 LoRD 方法以窃取多模态模型图像识别能力的技术方案。

### 核心思路

原始 LoRD 方法通过文本 token 的 logit 进行知识蒸馏，但图像不是离散的 token 序列。VQ-LoRD 的创新在于：

1. **VQ 离散化**：使用 VQ-VAE 技术将连续的视觉特征量化为离散的 codebook 索引
2. **获取 logits**：量化过程产生到各 codebook 向量的距离，可转换为类似文本的 logit 分布
3. **统一蒸馏**：文本和视觉都有 logit，可使用相同的 KL 散度蒸馏框架

### 三阶段训练流程

| 阶段 | 目标 | 训练模块 |
|------|------|----------|
| Stage 1 | 学习好的图像离散表示 | 仅 VQ codebook |
| Stage 2 | 蒸馏视觉理解能力 | VQ + Vision Encoder + Projector |
| Stage 3 | 端到端联合优化 | 全部（LoRA 微调 LLM） |

---

## vq_module.py — VQ 离散化模块

### 功能概述

该模块实现了向量量化（Vector Quantization）核心功能，将视觉编码器输出的连续特征映射到有限的离散 codebook 中。

### 核心类

#### `VectorQuantizer`

**作用**：将输入特征量化到最近的 codebook 向量

**关键参数**：
- `num_embeddings`：codebook 大小，默认 8192。越大表示能力越高但训练越难
- `embedding_dim`：每个 code 的维度，需与 Vision Encoder 输出维度匹配
- `commitment_cost`：commitment loss 权重，平衡编码器与 codebook 的学习
- `use_ema`：是否使用指数移动平均更新 codebook（推荐开启，训练更稳定）

**前向传播流程**：
1. 计算输入特征到每个 codebook 向量的欧氏距离
2. 选择距离最近的 codebook 索引（argmin）
3. 获取对应的量化嵌入
4. 使用 Straight-Through Estimator 传递梯度
5. 返回量化特征、索引、VQ 损失和 logits

**输出的 logits**：负距离值，经过 softmax 后表示"属于各 code 的概率"，可用于 KL 散度蒸馏。

#### `VQVisionEncoder`

**作用**：包装原始 Vision Tower，自动添加 VQ 层

**关键设计**：
- 自动检测 Vision Encoder 输出维度
- 支持冻结原始 Vision Tower（仅训练 VQ 层）
- 处理多种输出格式（BaseModelOutput、tuple、tensor）

### 注意事项

1. **Codebook 坍塌**：如果大量 code 从未被使用，说明 codebook 利用率低。可通过增大初始化范围或使用 EMA 缓解
2. **维度匹配**：`embedding_dim` 必须与 Vision Encoder 的隐藏维度完全一致

---

## vision_lord_loss.py — 损失函数模块

### 功能概述

实现 VQ-LoRD 的综合损失函数，统一文本蒸馏、视觉蒸馏和 VQ 重建损失。

### 核心类

#### `VisionLoRDLoss`

**损失公式**：
```
L_total = L_text + α × L_vision + β × L_vq
```

**各项损失说明**：

| 损失项 | 计算方式 | 作用 |
|--------|----------|------|
| `L_text` | 学生与教师文本 logits 的 KL 散度 | 蒸馏文本生成能力 |
| `L_vision` | 学生与教师 VQ logits 的 KL 散度 | 蒸馏视觉理解能力 |
| `L_vq` | VQ 模块的 commitment loss | 保证量化质量 |
| `L_contrastive` | 正负样本对比损失 | 原 LoRD 的迭代改进机制 |

**关键参数**：
- `alpha`：视觉损失权重，默认 1.0
- `beta`：VQ 损失权重，默认 0.25
- `temperature`：蒸馏温度，较高温度使分布更平滑

#### `VisualQADistillationLoss`

**作用**：通过视觉问答任务间接蒸馏视觉能力

**设计思路**：
1. 对图像生成视觉细节问题（物体描述、空间关系、计数等）
2. GPT-4V 回答这些问题
3. 训练学生模型模仿教师回答
4. 间接学习教师的视觉理解能力

### 注意事项

1. **温度选择**：温度过低会导致硬标签，过高会丢失关键信息。1.5 是经验值
2. **权重平衡**：`alpha` 和 `beta` 需要根据任务调整，视觉任务可增大 `alpha`

---

## data_collector.py — 数据收集模块

### 功能概述

自动化收集 GPT-4V 的视觉理解数据，构建 VQ-LoRD 训练集。

### 核心类

#### `GPT4VDataCollector`

**主要功能**：
1. 调用 GPT-4V API 进行视觉问答
2. 支持多种问题类型：物体描述、计数、空间关系、颜色属性、场景理解、OCR、动作检测、情绪表达
3. 自动保存收集结果为 JSON 文件
4. 支持断点续传（检查已存在文件）

**API 调用细节**：
- 图片编码为 base64 格式发送
- 使用 `detail: high` 获取高质量视觉分析
- 失败时指数退避重试（最多 3 次）

#### `VQLORDDataset`

**作用**：整合视觉问答数据和图像描述，构建 PyTorch 数据集

**数据处理**：
- 使用 LLaVA Processor 处理多模态输入
- 自动 padding 到固定长度
- 返回 input_ids、attention_mask、pixel_values、labels

### 注意事项

1. **API 成本**：GPT-4V API 调用费用较高，建议先用少量样本测试
2. **问题选择**：不同任务选择不同问题类型，ScienceQA 重点使用 `describe_objects` 和 `scene_understanding`
3. **数据质量**：GPT-4V 回答可能有幻觉，可人工抽检

---

## train_vq_lord.py — 主训练脚本

### 功能概述

VQ-LoRD 的完整训练流程实现，包含模型加载、VQ 层添加、三阶段训练和检查点保存。

### 关键函数

#### `add_vq_to_model(model, args)`

**核心设计**：使用 PyTorch 的 `register_forward_hook` 机制

**工作原理**：
1. 在 Vision Tower 上注册 forward hook
2. 每次 Vision Tower 前向传播后，hook 自动应用 VQ 量化
3. 量化后的特征替换原始特征，流入后续模块
4. VQ 损失保存到 `model._vq_loss_container` 供训练使用

**优势**：无需修改原模型代码，VQ 层透明集成。

#### `train_stage1_vq`

**训练目标**：学习好的 codebook

**策略**：
- 仅设置 VQ 层 `requires_grad=True`
- 使用较大学习率（10×）加速收敛
- 损失：纯 VQ commitment loss

#### `train_stage2_vision`

**训练目标**：蒸馏视觉理解能力

**策略**：
- 训练 VQ + Vision Encoder（可选）+ Projector
- 使用 GPT-4V 收集的视觉 QA 数据
- 损失：Text CE loss + VQ loss

#### `train_stage3_lord`

**训练目标**：端到端联合优化

**策略**：
- 使用 LoRA 微调 LLM
- 结合对比学习（原 LoRD 机制）
- 损失：综合损失函数

### 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--vq_codebook_size` | 8192 | VQ 词表大小 |
| `--alpha` | 1.0 | 视觉损失权重 |
| `--beta` | 0.25 | VQ 损失权重 |
| `--stage` | 3 | 训练阶段（1/2/3） |
| `--freeze_vision_tower` | 0 | 是否冻结 Vision Encoder |

---

## 启动脚本和配置文件

### 7.0.vq_lord_train.sh

**功能**：配置环境变量并启动训练

**关键配置**：
- CUDA 环境设置
- 自动选择空闲 GPU
- 训练参数导出
- 目录创建

### vq_lord_config.yaml

**功能**：集中管理所有可配置参数

**包含配置**：
- 模型路径
- VQ 参数
- 损失权重
- 训练超参
- LoRA 配置
- 量化配置
- 数据配置

---

## 已知问题与注意事项

### 潜在问题

1. **Hook 副作用**：forward hook 会修改 Vision Tower 的输出，如果保存/加载模型需要重新注册 hook

2. **内存占用**：VQ logits 大小为 `[batch, num_patches, codebook_size]`，8192 codebook 会显著增加显存

3. **梯度问题**：VQ 使用 Straight-Through Estimator，可能导致梯度不精确

### 调试建议

1. **验证 VQ 工作**：检查 `model._vq_loss_container["loss"]` 是否非零

2. **监控 codebook 利用率**：统计被使用的 codebook 索引数量

3. **对比有无 VQ**：移除 hook 对比性能差异

### 后续优化方向

1. 使用预训练 VQGAN codebook 初始化
2. 增加 codebook 正则化避免坍塌
3. 多尺度 VQ：不同层使用不同粒度的量化

---

**文档结束**

---

## sciqa_process.py — ScienceQA 验证脚本

### 功能概述

该脚本用于对训练后的多模态学生模型进行 ScienceQA 验证，支持图像 + 文本输入，输出选择题准确率并保存详细结果。

### 核心功能

1. **多模态推理**：使用 `LlavaNextProcessor` 构建 `<image> + Question + Options` 的提示词，并传入图像进行生成。
2. **答案解析**：优先解析 `Answer: X` 形式的字母答案，若无则尝试匹配选项文本。
3. **评测输出**：输出准确率，保存 `metrics` 和逐样本结果到 JSON 文件。

### 主要参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model_path` | 基础模型路径 | 必填 |
| `--adapter_path` | LoRA 适配器路径 | 空 |
| `--split` | ScienceQA split | `validation` |
| `--max_samples` | 最大评测样本数 | `200` |
| `--max_new_tokens` | 生成长度 | `64` |
| `--use_4bit` | 是否 4-bit 加载 | `1` |
| `--save_path` | 结果保存路径 | `./sciqa_eval.json` |

### 结果文件示例

结果文件为 JSON，包含：

- `metrics`: 准确率、总样本数、正确数
- `results`: 每条样本的 `question / choices / pred_idx / answer_idx / output`

### 划分一致性说明

训练使用官方 `train` split，验证使用官方 `validation`/`test` split，保持数据划分一致性。

