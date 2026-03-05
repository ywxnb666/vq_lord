## VQ-LoRD 当前问题修复 ToDo（2026-03-05）

> 目标：让训练不仅“能跑”，而且真正发生对 `gpt-4-vision-preview` 的蒸馏。

### 一、核心问题清单

1. **`victim_model` 参数未被训练流程实际使用**
	- 现状：脚本传了 `--victim_model`，但训练中没有基于该模型生成教师标签。
	- 影响：训练退化为 ScienceQA 监督学习/自蒸馏，不是 GPT-4V 蒸馏。

2. **Stage2 未使用视觉蒸馏损失框架**
	- 现状：`VisionLoRDLoss` 被导入但未使用，Stage2 仅 `text_loss + beta * vq_loss`。
	- 影响：`alpha` / `temperature` 等蒸馏参数基本失效。

3. **Stage3 可能未训练 LoRA 参数**
	- 现状：Stage2 冻结了大量参数，Stage3 未重新设置 `requires_grad`。
	- 影响：Stage3 可能只在少量模块上更新，LoRD 联合训练效果受限。

4. **`labels` 未正确屏蔽 prompt 区域**
	- 现状：仅屏蔽 padding，未屏蔽 instruction/prompt token。
	- 影响：Stage3 通过 `(labels==-100)` 推导 prompt 长度失真，LoRD 正负样本构造偏差。

5. **VQ 参数未完整生效**
	- 现状：`--vq_commitment_cost` 定义了但未真正传入 `VectorQuantizer`。
	- 影响：脚本中的超参与实际训练行为不一致。

6. **复用开关默认会跳过前置阶段**
	- 现状：`reuse_vq_codebook=1`、`reuse_stage2=1` 默认开启。
	- 影响：容易误以为全流程重训，实际可能仅跑 Stage3。

7. **训练脚本注释与配置不一致**
	- 现状：`use_4bit=1` 但注释写“关闭 4-bit”。
	- 影响：实验记录与复现实验时易混淆。

---

### 二、修复任务（执行顺序）

- [x] 在 ScienceQA 训练流中接入 GPT-4V 教师回答采集与缓存（真正使用 `victim_model`）
- [x] 让 `labels` 正确仅监督 response 部分（屏蔽 prompt）
- [x] Stage3 进入前重新设置可训练参数，确保 LoRA 参与更新
- [x] 把 `vq_commitment_cost` 传入 VQ 模块
- [x] 校正脚本中 `use_4bit` 注释与行为不一致问题
- [x] 训练后增加“是否真的使用 GPT-4V 教师数据”的日志标识

---

### 三、后续增强（可选）

- [ ] Stage2 引入真正的视觉蒸馏项（如 VQ logits KL，或教师视觉描述蒸馏）
- [ ] Stage1 先做 ImageNet 预训练，再做 ScienceQA 迁移
- [ ] 增加 ablation：`base / +LoRA / +VQ / +LoRD / +GPT4V teacher`

