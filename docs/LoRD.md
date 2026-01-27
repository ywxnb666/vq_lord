# LoRD (Locality Reinforced Distillation) 方法核心原理解析

## 1. 简介
LoRD 是一种针对大语言模型（LLM）的**模型窃取（Model Extraction）**算法。它的核心目标是在**黑盒（Black-box）**环境下，让一个较小的“学生模型（Local Model）”高效地学习“教师模型（Victim Model）”的能力。

与传统的监督微调（SFT/MLE）或知识蒸馏（KD）不同，LoRD 并不直接最小化学生输出与教师输出的 token 差异，而是模拟了 **RLHF（基于人类反馈的强化学习）** 的过程，但无需显式的奖励模型（Reward Model）或人工标注。

## 2. 核心假设与动机
* **对齐不一致问题**：现有的攻击方法（如 MLE）将教师模型的输出视为静态标签，忽略了现代 LLM 是经过 RL 对齐的。直接模仿会导致查询效率低，且容易遭受水印（Watermark）防御。
* **解决方案**：LoRD 引入了一种类似 Policy Gradient 的训练范式，利用教师模型的回复作为信号，指导学生模型在局部生成空间中建立“偏好（Preference）”。

## 3. LoRD 算法流程 (Workflow)

LoRD 的训练过程是一个迭代的强化学习循环。对于每一条查询输入 $x$ 和教师模型的标准回答 $y_{vic}$，LoRD 执行以下步骤：

### 第一步：双样本生成 (Dual Sampling)
学生模型 $P_{\theta}$ 基于输入 $x$ 自行生成两个不同的回复样本，记为 $y_{t-1}^+$ 和 $y_{t-1}^-$。
* 这两个样本代表了学生模型当前的探索方向。

### 第二步：局部性排序 (Locality Ranking)
LoRD 需要判断这两个样本中，哪一个更接近教师模型的分布方向。它计算样本在更新前后概率的变化量 $\Delta$：

$$
\Delta = \log P_{\theta_t}(y) - \log P_{\theta_{t-1}}(y)
$$

* 如果 $\Delta^+ < \Delta^-$，则交换两个样本，确保 $y^+$ 代表了更符合优化方向的“正样本”，而 $y^-$ 为“负样本”。

### 第三步：冷启动策略 (Cold Start / Anchor Strategy)
为了防止学生模型在训练初期生成的样本质量过差（导致 $y^+$ 和 $y^-$ 都是垃圾），算法引入了阈值检查：
* 如果学生模型对 $y^+$ 的置信度低于阈值 $\tau_1$，或者概率增量 $\Delta^+$ 低于 $\tau_2$，**强制将正样本 $y^+$ 替换为教师模型的真实回复 $y_{vic}$**。
* **关键点**：这意味着训练目标在“模仿教师”和“自我探索”之间动态切换。

### 第四步：计算 LoRD 损失函数 (Optimization)
LoRD 的 Loss 由两部分组成：

$$
\mathcal{L}_{LoRD} = \mathcal{L}_{obj} + \mathcal{L}_{reg}
$$

**1. 目标函数 (Objective Function):**
最大化正样本与负样本的概率比率（类似于对比学习）：

$$
\mathcal{L}_{obj} = - \log \frac{P_{\theta}(y^+|x)}{P_{\theta}(y^-|x)} = - (\log P_{\theta}(y^+|x) - \log P_{\theta}(y^-|x))
$$

*这鼓励模型拉大好样本与坏样本的差距。*

**2. 正则化项 (Regularization):**
约束模型不要偏离教师模型的分布太远（防止 RL 训练崩溃），使用了类似 PPO 的 clip 机制：

$$
\mathcal{L}_{reg} = - \text{clip} \left( \log \frac{P_{\theta}(y_{vic}|x)}{P_{\theta}(y^-|x)} \right)
$$

*这利用教师的 Ground Truth ($y_{vic}$) 作为锚点来纠正负样本。*

### 第五步：参数更新
基于上述 Loss 反向传播更新学生模型参数 $\theta$。

## 4. 为什么 LoRD 适合多模态窃取？
在多模态场景（如 ScienceQA）中，输入包含图像和文本。
1.  **去幻觉 (Hallucination Mitigation)**：多模态模型容易产生视觉幻觉。LoRD 通过 $y^+$ 和 $y^-$ 的对比（$- \log P(y^+) + 2 \log P(y^-)$），能有效抑制那些置信度低、包含幻觉的生成路径（即 $y^-$）。
2.  **黑盒友好**：LoRD 只需要教师模型输出文本（$y_{vic}$），**不需要访问教师模型的 Logits 或梯度**，这完全符合攻击商业多模态 API（如 GPT-4V）的设定。
3.  **长文本推理**：ScienceQA 需要生成长推理链（CoT）。LoRD 的强化学习机制比简单的 MLE 更适合优化这种长序列生成的逻辑一致性。

<!-- ## 5. 总结：给 Agent 的执行指令
在编写代码时，请遵循以下逻辑映射：
* **Teacher Ground Truth**: $y_{vic}$ (来自数据集的正确答案/解析)。
* **Student Generation**: 使用 `model.generate` 生成两个独立的序列 $S_1, S_2$。
* **Loss Calculation**:
    * 计算 $S_1, S_2$ 以及 $y_{vic}$ 在当前学生模型下的 Log Probability。
    * 根据 Log Prob 的数值或变化量确定哪个是 $y^+$ (Better)，哪个是 $y^-$ (Worse)。
    * 代入 $\mathcal{L}_{LoRD}$ 公式计算梯度。
* **Inputs**: 必须同时处理 Image Tensor (pixel_values) 和 Text Prompt。 -->