# Stage3 LoRD 重构计划

---

## 一、现有 Stage3 的问题诊断

当前 `train_stage3_lord()` 存在以下核心问题：

1. **缺失跨 period 状态**  
   现实现是 batch 内“生成 -> 排序 -> 更新”的扁平循环，没有为同一样本跨 period 保存候选及其旧概率，因此无法实现 LoRD-II 所需的“旧策略 vs 新策略”比较。

2. **排序与冷启动口径不统一**  
   当前实现用绝对 log prob 比较 `lp2 > lp1` 决定正负样本，但计划草稿中又写成“Delta 排序”。这两个定义不能并存，必须统一。

3. **缺少 token-level old logits / old log-probs**  
   LoRD-II 的 `term1/term2/term3` 是逐 token 的相对变化约束。仅保存序列级标量平均 log prob，不足以支撑标准 LoRD-II 损失。

4. **当前损失不是标准 LoRD-II**  
   现实现本质上还是绝对值式目标：拉高 `y+`、压低 `y-`、并用 `y_vic` 做正则，但没有显式使用上一轮 detached 的 token-level 旧值，丢失了 LoRD 的“跨 period 改进”语义。

5. **Stage3 数据重复预处理，效率差**  
   目前 Stage3 若反复从 `ScienceQADataset.__getitem__()` 取样，会重复执行 processor 的文本和图像预处理。多 period 场景下，这个 CPU 路径会成为明显瓶颈。

6. **Stage2 -> Stage3 工件契约不清晰**  
   Stage3 依赖 Stage2 的 LoRA、VQ 栈以及 projector，但当前计划没有把 projector 的显式保存/加载写成强约束，存在依赖隐式 `save_pretrained()` 行为的风险。

7. **Stage3 中继续加入 VQ loss 的收益不清晰**  
   若 VQ codebook、pre_quant、post_quant 全冻结，则 `beta * vq_loss` 不再服务于一个明确的可训练目标，容易引入无关梯度信号。

8. **当前 `train_stage3_lord()` 入口形式与多 period 架构不匹配**  
   现有主流程将标准 `DataLoader` 直接传给 Stage3；而新的 LoRD-II 结构需要同时支持样本级 Phase A 和 batch 级 Phase B，因此 Stage3 训练入口需要从“dataloader 直传”重构为“dataset/cache 驱动”。

---

## 二、设计决策（已确认）

| 决策项 | 选择 |
|--------|------|
| 排序策略 | **绝对概率排序 + delta 双条件 cold-start** |
| 损失变体 | **LoRD-II**（token-level old log-probs） |
| VQ 策略 | **冻结 VQ 全部**（codebook + pre_quant + post_quant） |
| 默认可训练模块 | **只训 LoRA** |
| 教师数据 | **纯离线 y_vic**，不在线调用 API |
| Stage3 目标 | 在不破坏 Stage2 视觉能力的前提下，提升生成质量与答案准确率 |

排序口径的最终定义：

- `y+ / y-` 的排序依据：**当前模型下的绝对概率比较**
- `delta` 的作用：**仅用于双条件 cold-start**
- 计划中不再出现“Delta 排序”表述

---

## 三、LoRD-II 的理论定义与工程实现选择

LoRD-II 的标准形式需要使用上一轮 detached 的 token-level 旧值，对当前 token-level log prob 的变化进行约束。工程实现上，我们不做序列级降级，而是采用：

- **token-level old log-probs 缓存到 CPU**
- 每个候选仅缓存“生成段”的 token-level log-probs 和对应 mask
- `float32` 保存，按 ScienceQA + `max_new_tokens <= 64` 的规模，内存可接受

这意味着我们的工程实现是对 LoRD-II 的忠实落地，而不是近似版。

---

## 四、新 Stage3 整体架构

### 4.1 多 Period 结构

```python
train_stage3_lord(model, train_dataset, args, tb_writer, bucket_meta=None):
    sample_cache = build_stage3_sample_cache(dataset)

    for sub_stage in range(sub_stage_num):
        active_indices = sample_subset(sample_cache, sub_set_num)
        period_states = bootstrap_period_states(...)  # 每个 sub_stage 开始时重新初始化

        for period in range(period_num):
            # Step 1. 用当前 theta_t 对旧候选重新评分
            # Step 2. 绝对概率排序，必要时触发 cold-start
            # Step 3. 计算并保存 old token log-probs（供本 period 训练使用）
            # Step 4. 用当前 theta_t 生成下一 period 的新候选
            # Step 5. Phase B 训练，将 theta_t 更新到 theta_(t+1)
```

### 4.2 一个 period 的精确时序

为避免歧义，一个 period 内固定按以下顺序执行：

1. 使用当前参数 `theta_t` 对上一 period 保留下来的候选 `y11 / y12` 重新评分。
2. 依据当前绝对概率比较，得到 `y+ / y-`。
3. 基于 `max(p11, p12) < tau1` 且 `delta11 < tau_delta` 判断是否 cold-start；若触发，则 `y+ = y_vic`。
4. 用当前 `theta_t` 对 `y+ / y- / y_vic` 做 no-grad forward，得到本 period 训练所需的 **old token log-probs**。
5. 用当前 `theta_t` 重新生成下一 period 要用的新候选，并把它们保存为下一轮的 `PeriodState`。
6. 用第 4 步保存的 old token log-probs 执行本 period 的 LoRD-II 训练，把模型从 `theta_t` 更新到 `theta_(t+1)`。

关键结论：

- 本 period 的 `old_*` 和“下一 period 的候选”都由 **训练前的 `theta_t`** 产生。
- 下一 period 开头再用 `theta_(t+1)` 回看这些候选，才能形成有意义的概率变化。
- 新 `sub_stage` 开始时：**模型参数连续、`PeriodState` 重置并重新 bootstrap**。

### 4.3 与当前 `main()` 的对接方式

当前工程的 `main()` 仍然是：

```python
stage3_dataloader = build_train_dataloader(stage_id=3)
model = train_stage3_lord(model, stage3_dataloader, args, tb_writer)
```

这与新架构不兼容。新 Stage3 的对接方式应改为：

```python
train_dataset = ...
bucket_meta = ...
model = train_stage3_lord(model, train_dataset, args, tb_writer, bucket_meta=bucket_meta)
```

原因：

- Phase A 需要按样本访问 cache，而不是按普通 batch 训练
- Phase B 需要根据当前 period 动态构造训练对，再生成临时 DataLoader
- 不应继续把“原始 stage3_dataloader”当作 Stage3 主入口

---

## 五、数据契约

### 5.1 Stage3SampleCache

`Stage3SampleCache` 表示整个 Stage3 期间不变的样本缓存，一次构建，多次复用。

```python
@dataclass
class Stage3SampleCacheItem:
    sample_idx: int
    prompt_ids: Tensor         # (prompt_len,)
    prompt_mask: Tensor        # (prompt_len,)
    y_vic_ids: Tensor          # (seq_len,)
    y_vic_mask: Tensor         # (seq_len,)
    pixel_values: Tensor       # 预处理后的视觉输入
    image_sizes: Tensor
    prompt_len: int
```

职责：

- 避免每个 period 反复走 `processor`
- 为 Phase A / Phase B 提供稳定的 prompt、教师目标和视觉输入
- 支持按 `sample_idx` 取样，而不是反复访问原始 dataset

额外约束：

- `prompt_len` 必须直接取自 `prompt_input_ids.shape[0]`
- 不使用 `labels == -100` 的反推方式
- `y_vic_ids` 对应当前 dataset 中的 `full_input_ids`，训练时通过 `prompt_len` mask 掉 prompt 段

### 5.2 PeriodState

`PeriodState` 表示“某个样本在当前 period 结束后，供下一 period 使用的候选状态”。

```python
@dataclass
class PeriodState:
    sample_idx: int

    y11_ids: Tensor                 # (seq_len,)
    y11_mask: Tensor                # (seq_len,)
    y12_ids: Tensor                 # (seq_len,)
    y12_mask: Tensor                # (seq_len,)

    # 用于下一 period 计算 delta 的序列级统计
    avg_lp_11: float
    avg_lp_12: float
    prob_11: float
    prob_12: float
```

### 5.3 PeriodTrainingPairs

`PeriodTrainingPairs` 表示“当前 period 的 Phase B 训练对”，生命周期仅限当前 period。

```python
@dataclass
class PeriodTrainingItem:
    sample_idx: int

    y_plus_ids: Tensor              # (seq_len,)
    y_plus_mask: Tensor             # (seq_len,)
    y_minus_ids: Tensor             # (seq_len,)
    y_minus_mask: Tensor            # (seq_len,)
    y_vic_ids: Tensor               # (seq_len,)
    y_vic_mask: Tensor              # (seq_len,)

    old_token_lp_plus: Tensor       # (gen_len,)
    old_token_mask_plus: Tensor     # (gen_len,)
    old_token_lp_minus: Tensor      # (gen_len,)
    old_token_mask_minus: Tensor    # (gen_len,)
    old_token_lp_vic: Tensor        # (gen_len,)
    old_token_mask_vic: Tensor      # (gen_len,)
```

生命周期说明：

- `Stage3SampleCache`：Stage3 入口构建，训练全程不变
- `PeriodState`：每个 period 结束后更新一次
- `PeriodTrainingPairs`：每个 period 的 Phase A 结束后生成，Phase B 结束后立即丢弃
- `old_token_lp_*`：属于 `PeriodTrainingPairs`，由当前 `theta_t` 的 no-grad 评分产生，不跨 period 持久保存

---

## 六、Phase A：排序、Cold-Start 与候选更新

### 6.1 排序规则

最终排序规则固定为：

```python
if p12 > p11:
    swap(y11, y12)
```

含义：

- `y11` 始终表示当前绝对概率更高的候选
- `y12` 始终表示当前绝对概率更低的候选
- `swap_ratio` 的定义是“按绝对概率比较后发生交换的比例”

### 6.2 Delta 的角色

`delta` 不参与排序，只参与 cold-start：

```python
delta11 = p11 - prev_prob_11
delta12 = p12 - prev_prob_12

if max(p11, p12) < tau1 and delta11 < tau_delta:
    y_plus = y_vic
else:
    y_plus = y11
y_minus = y12
```

解释：

- `max(p11, p12) < tau1`：当前两个候选都不够好
- `delta11 < tau_delta`：高概率候选相对上一轮也没有明显改善
- 两者同时满足，才说明模型在当前样本上确实生成无力，应使用教师回答兜底

### 6.3 Period 0（Bootstrap）

首轮没有上一 period 候选，因此采用 bootstrap：

1. 用 `theta_0` 对每个样本生成两个候选 `y11 / y12`
2. 计算它们的当前绝对概率
3. 排序得到生成候选中的高低顺序
4. **写入 `PeriodState` 时固定定义：`y11 = y_vic`，`y12 = 低概率生成候选`**
5. 训练时令 `y+ = y_vic`，`y- = y12`
6. 同时计算 `y_vic`、`y12` 的 old token log-probs，并写入 `PeriodTrainingPairs`

这样可以平滑进入后续 period，而不需要人为定义“虚假的上一个 delta”。

### 6.4 后续 Period

后续 period 对每个样本执行：

1. 取上一 period 保留的 `y11 / y12`
2. 用当前模型重新评分
3. 用绝对概率排序
4. 用 `delta + tau1` 判断是否 cold-start
5. 计算本 period 训练所需的 old token log-probs，并写入 `PeriodTrainingPairs`
6. 用当前模型重新生成下一 period 候选，写入新的 `PeriodState`

关键约束：

- 步骤 1-5 得到的 `y+ / y-` **只服务于本 period 的 Phase B 训练**
- 步骤 6 生成的新候选 **完全替换** 下一 period 的 `y11 / y12`
- 除 bootstrap 的 `period 0 -> period 1` 过渡外，`y_vic` 不应长期残留在 `PeriodState.y11`

---

## 七、Phase B：LoRD-II 损失与训练

### 7.1 核心工具函数

```python
def compute_token_log_probs(
    model,
    input_ids,
    attention_mask,
    pixel_values,
    image_sizes,
    prompt_lens,
):
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        image_sizes=image_sizes,
    )

    logits = outputs.logits[:, :-1, :]
    log_probs = F.log_softmax(logits, dim=-1)
    token_lp = log_probs.gather(
        dim=-1,
        index=input_ids[:, 1:].unsqueeze(-1),
    ).squeeze(-1)

    gen_mask = attention_mask[:, 1:].float()
    for b in range(input_ids.shape[0]):
        gen_start = max(int(prompt_lens[b].item()) - 1, 0)
        gen_mask[b, :gen_start] = 0.0

    return token_lp, gen_mask
```

该函数默认返回：

- 逐 token log-probs
- 生成段 mask

序列级平均 log prob 由上层显式计算：

```python
avg_lp = (token_lp * gen_mask).sum(dim=-1) / gen_mask.sum(dim=-1).clamp(min=1)
```

### 7.2 LoRD-II 损失

```python
def masked_mean(values, mask):
    return (values * mask).sum() / mask.sum().clamp(min=1)


def lord_ii_loss(
    cur_lp_plus,
    cur_mask_plus,
    old_lp_plus,
    old_mask_plus,
    cur_lp_minus,
    cur_mask_minus,
    old_lp_minus,
    old_mask_minus,
    cur_lp_vic,
    cur_mask_vic,
    old_lp_vic,
    old_mask_vic,
):
    term1 = masked_mean(log_clip(-old_lp_minus + cur_lp_minus), old_mask_minus)
    term2 = masked_mean(log_clip(old_lp_plus - cur_lp_plus), old_mask_plus)
    term3 = masked_mean(old_lp_vic - cur_lp_vic, old_mask_vic)

    L_obj = term1 + term2
    L_reg = term3
    return L_obj + L_reg, L_obj, L_reg
```

说明：

- `term1`：惩罚 `y-` 概率相对旧值上升
- `term2`：惩罚 `y+` 概率相对旧值下降
- `term3`：约束模型不要偏离教师回答 `y_vic` 太远
- `term3` 使用 **按有效 token 数归一化的 mean**，避免 rationale 长度偏置

### 7.3 Phase B 训练循环

```python
def one_period_train(model, optimizer, train_loader, args):
    model.train()

    for batch in train_loader:
        # batch 中已包含：
        # y_plus / y_minus / y_vic
        # old_token_lp_* / old_token_mask_*
        # pixel_values / image_sizes / prompt_lens

        loss, L_obj, L_reg = lord_ii_loss(...)
        (loss / grad_accum).backward()

        if should_step:
            clip_grad_norm_(trainable_params, max_norm=args.stage3_grad_clip)
            optimizer.step()
            optimizer.zero_grad()
```

`train_loader` 的 collate 需要保证：

- `cur_*` 与 `old_*` 在 loss 计算前已经按**同一 padding 方案**对齐
- 样本级保存的 `old_token_lp_*` 不以 batch padded 形式持久化
- 若 `cur_lp_*` 与 `old_lp_*` 长度不一致，应在 collate 阶段统一 pad，而不是在 loss 中隐式截断

---

## 八、可训练模块与解冻策略

### 8.1 默认训练策略

默认只训练 LoRA：

```python
for name, param in model.named_parameters():
    name_l = name.lower()
    is_lora = "lora_" in name_l or "modules_to_save" in name_l
    param.requires_grad = bool(is_lora)
```

默认冻结：

- projector
- vision tower
- pre_quant / post_quant
- VQ codebook

理由：

- Stage3 的对比信号波动较大
- 先保护 Stage2 建立的视觉桥接
- 先验证“只训 LoRA”能否带来可观测提升，再决定是否逐步解冻

### 8.2 Plateau 时的解冻顺序

若 Stage3 多个 period 后 `val_answer_acc` 停滞：

1. 第一优先：`stage3_train_projector=1`，并令 `projector_lr = stage3_lr * 0.1`
2. 第二优先：解冻 `pre_quant / post_quant`，并令其 `lr = stage3_lr * 0.05`
3. 不建议：解冻 vision tower

---

## 九、VQ 在 Stage3 中的角色

- VQ codebook、pre_quant、post_quant **全部冻结**
- 视觉特征仍然经过 VQ 路径，保持 Stage1/2 学到的离散视觉表征
- **Stage3 不再把 `vq_loss` 加入总损失**

原因：

- 冻结后 `vq_loss` 不再服务于明确的 Stage3 训练目标
- 当前唯一主优化目标应是 LoRD-II 的相对变化约束
- 避免将无关梯度混入 LoRA 更新

---

## 十、Stage2 -> Stage3 工件契约

Stage3 启动时，必须显式获取并恢复以下工件：

1. `LoRA` 权重
2. `VQ` 全栈：codebook + pre_quant + post_quant
3. `projector` 权重
4. `stage2_config.json`
5. Stage3 所需数据集

### 10.1 显式保存要求

计划中将 projector 保存/加载提升为 **P0 级工程约束**，不依赖 `save_pretrained()` 的隐式行为。

同时，这也是 **Stage3 开工前的前置 blocking 任务**：若 Stage2 仍沿用当前保存逻辑，则其训练得到的 projector 参数可能无法正确交接给 Stage3。

Stage2 结束时：

```python
torch.save(projector.state_dict(), stage2_ckpt_path / "projector.pt")
```

Stage3 加载时：

```python
projector.load_state_dict(torch.load(projector_path, map_location="cpu"))
```

同理，VQ 栈继续使用显式保存/显式恢复。

### 10.2 启动前健全性检查

- LoRA 参数存在且 `requires_grad=True`
- `vq.*` 参数全部 `requires_grad=False`
- projector 默认 `requires_grad=False`
- 能在少量样本上完成 smoke generation
- projector 工件存在且可成功恢复

### 10.3 与当前 Stage2 代码的关系

当前工程中：

- Stage2 训练时 projector 会被设置为可训练
- 但现有保存逻辑只显式保存了 VQ 栈和 PEFT adapter
- 因此 projector 显式保存/加载应先补到 Stage2 代码里，再进入 Stage3 实现

结论：这是 **Stage2 的前置修复任务**，不是可延后的 Stage3 小优化。

---

## 十一、效率策略

### 11.1 基线原则

正确性优先，效率优化后置：

- `P0`：支持 `sub_set_num` 子集运行
- `P0`：采用逐样本生成
- `P0`：预先构建 `Stage3SampleCache`
- `P3`：再做 batched generation

关于 `Stage3SampleCache` 的实现约束：

- smoke 配置下可直接采用内存缓存
- full 配置下需评估是否切换到磁盘缓存或分块加载
- 计划默认只承诺 smoke 场景稳定，不假设 full 配置可直接把全部 `pixel_values` 常驻内存

### 11.2 生成是主要时间瓶颈之一

计划不再假设“生成不是主要瓶颈”。多 period 下，`model.generate()` 本身就是主要耗时来源之一，尤其在全量 ScienceQA 上更明显。

### 11.3 Batched Generation（后续优化）

后续可按 prompt 长度近似分组做 batched generation，但这不是 P0 阶段的必需项。P0 阶段优先保证：

- 时序正确
- 状态正确
- 损失正确
- 能稳定跑通 smoke 配置

---

## 十二、评估与日志

### 12.1 训练中评估

建议按 `period` 粒度插入验证：

- `answer_acc`
- `format_rate`
- `cold_start_ratio`

预期趋势：

- `answer_acc` 不低于 Stage2
- `cold_start_ratio` 随训练下降
- 若 `answer_acc` 下降，优先降低 `stage3_lr_scale`

### 12.2 日志项

```text
stage3/L_obj
stage3/L_reg
stage3/total_loss
stage3/cold_start_ratio
stage3/swap_ratio
stage3/avg_lp_plus
stage3/avg_lp_minus
stage3/phase_a_seconds
stage3/val_answer_acc
stage3/period
stage3/sub_stage
```

日志说明：

- `swap_ratio`：按**绝对概率排序**后发生交换的比例
- 不再使用 `delta_ranking` 相关命名

---

## 十三、运行配置

### 13.1 Smoke 配置

用于验证代码正确性、loss 趋势和状态流转：

- `sub_stage_num = 1`
- `period_num = 2`
- `sub_set_num = 500`

### 13.2 Full 配置

不在计划阶段写死。待 smoke 结果稳定后，再在以下范围内搜索：

- `sub_stage_num = 2 ~ 3`
- `period_num = 3 ~ 5`
- `sub_set_num = 2000 ~ 5000`

是否进入更大规模配置，应由以下信号共同决定：

- smoke 阶段 loss 正常下降
- `cold_start_ratio` 有下降趋势
- `val_answer_acc` 不退化

---

## 十四、实现优先级

1. **P0 — 核心 LoRD-II 循环**
   包括：多 period 结构、绝对概率排序、delta 双条件 cold-start、token-level old log-probs、LoRD-II loss、Stage3SampleCache、PeriodTrainingPairs

2. **P0 — Stage2 前置修复**
   包括：projector 显式 save/load、VQ 栈显式恢复契约补齐、Stage2 产物健全性检查

3. **P0 — Stage3 训练入口重构**
   将 Stage3 从“接收普通 dataloader”改为“接收 dataset/cache，并在内部构建 period 训练对”

4. **P0 — 参数接口补齐**
   新增 `stage3_grad_clip` 等 Stage3 专用参数；若实现早期未补齐，代码层需使用 `getattr(args, "stage3_grad_clip", 1.0)` 兜底

5. **P1 — 断点续训（已实现）**
   已实现保存/恢复 `PeriodState`、optimizer state、RNG state（python/numpy/torch/cuda）以及训练进度（`next_sub_stage_idx`、`next_period_idx`、`global_step`）。
   默认断点目录：`save_path/stage3_resume_latest`；可通过 `--stage3_resume_path` 指定恢复目录，通过 `--stage3_resume_save_path` 指定保存目录。

6. **P2 — 评估集成**
   周期性答案准确率和格式率评估

7. **P3 — 生成效率优化**
   batched generation、长度分组

---

## 十五、验收标准

1. Stage3 训练全程无 NaN / Inf
2. `cold_start_ratio` 随 period 下降
3. `val_answer_acc` 在 Stage3 结束时不低于 Stage2
4. `val_answer_acc` 相比 Stage2 有可观测提升
5. 生成回答格式率 > 80%
6. 断点续训结果与连续训练一致
7. projector / VQ / LoRA 工件可显式保存并恢复

---

以上就是收口后的 Stage3 LoRD 重构计划。当前版本的重点不是一次性把效率拉满，而是先把：

- 算法口径
- 状态表示
- period 时序
- Stage2 -> Stage3 工件契约

这四件事彻底定清楚，再在此基础上做后续优化。
