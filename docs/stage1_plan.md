# Stage1 重构与执行计划

## Maintenance Rule
- 以后每次修改 `stage1` 相关实现时，必须同步更新本文件，确保 `docs/stage1_plan.md` 与 `vq_lord2/train_vq_lord2.py` 保持一致。

## Summary
本计划定义 `stage1` 的唯一核心职责：
- 在冻结大模型主干的前提下，训练稳定、可重建视觉特征的 `VQ stack`（`pre_quant + codebook + post_quant`）。
- 为 `stage2` 提供可复用、可加载、统计健康的视觉离散词表与量化路径。

重构后的 `stage1` 设计边界：
- `stage1` 不负责答案能力提升，不优化语言对齐，不引入 LoRD 目标。
- `stage1` 只优化视觉特征重建与量化稳定性。
- `stage1` 产物以轻量保存为主，避免整模型落盘。

默认设计结论：
- 冻结除 `stage1_parameters()` 外的全部参数。
- 保持 `vision_tower` 前向可用但不更新权重。
- 使用 `Adam`（非 `AdamW`）训练 Stage1 参数，避免权重衰减把 codebook 拉向 0。
- 训练监控以 `recon/cosine/vq/perplexity/dead_code` 为主，不以文本 loss 为目标。

## Implementation Changes
### 1. 重新定义 Stage1 目标
`train_stage1_vq` 的目标函数固定为：
`L_stage1 = w_recon * L_recon + w_cos * L_cos + w_vq * L_vq`

默认权重：
- `w_recon = args.stage1_recon_weight`（默认 `1.0`）
- `w_cos = args.stage1_cosine_weight`（默认 `0.25`）
- `w_vq = args.stage1_vq_weight`（默认 `1.0`）

语义定义：
- `L_recon`：`reconstructed_features` 与 `target_features` 的 MSE。
- `L_cos`：`1 - cosine_similarity` 的均值，约束方向一致性。
- `L_vq`：来自量化器的承诺损失/码本相关损失。

### 2. 固定 Stage1 参数冻结策略
Stage1 参数策略固定如下：
- 先对 `model.parameters()` 全量 `requires_grad=False`。
- 仅对 `model.vq_vision_encoder.stage1_parameters()` 中浮点参数置 `requires_grad=True`。
- `vq_encoder.pre_quant/train`、`vq_encoder.vq/train`、`vq_encoder.post_quant/train`。
- `vq_encoder.vision_tower.eval()`，避免污染原始视觉塔。

理由：
- Stage1 目标是建立稳定量化桥，不应提前扰动语言侧或 LoRA 参数。
- 视觉主干冻结可减少漂移，把学习容量集中在量化适配层。

### 3. 固定 Stage1 优化器与学习率策略
优化器策略固定如下：
- 使用 `torch.optim.Adam`。
- `betas=(0.5, 0.9)`。
- 不使用 weight decay。

学习率规则固定如下：
- `stage1_lr = args.stage1_lr if args.stage1_lr > 0 else args.lr * 5`。
- 该放大策略用于让 Stage1 快速收敛至可用 codebook 区域。

### 4. 固定数值稳定与梯度更新规则
每个 batch 的训练流程固定如下：
- 取 `pixel_values`，若维度为 5（`[bs, patches, c, h, w]`）则 reshape 成 4 维。
- 调用 `stage1_forward(pixel_values)` 获取 `target/reconstructed/vq_loss/indices`。
- 计算总损失并做有限值检查；若非有限值，直接跳过该 batch。
- `total_loss.backward()` 后按需执行梯度裁剪。
- 当 `args.stage1_grad_clip > 0` 时，执行 `clip_grad_norm_`；若梯度范数非有限，跳过 `optimizer.step()`。

关键约束：
- 非有限损失和非有限梯度都必须显式跳过，不能静默继续 step。
- Stage1 不使用梯度累积。

### 5. 固定 Stage1 监控指标与日志
Stage1 在线监控必须包含：
- `stage1/total_loss`
- `stage1/recon_loss`
- `stage1/cosine_loss`
- `stage1/vq_loss`
- `stage1/loss_ema`
- `stage1/codebook_used`
- `stage1/perplexity`
- `stage1/dead_code_resets`
- `stage1/dead_code_count`

监控来源定义：
- `codebook_used`：当前 batch `indices` 的 unique 数。
- `perplexity`、`dead_code_resets`、`dead_code_count`：从 `vq_encoder.vq_cache` 读取。
- `loss_ema`：`0.95 * ema + 0.05 * current_loss`。

CSV 记录固定写入：
- 文件：`save_path/logs/vq_metrics.csv`
- 表头：`stage,step,total_loss,recon_loss,cosine_loss,vq_loss,loss_ema,codebook_used,perplexity,dead_code_resets,dead_code_count`

### 6. 固定 Stage1 保存与复用策略
保存策略固定如下：
- Stage1 正常完成后，调用 `save_stage1_checkpoint`。
- `save_stage1_checkpoint` 仅保存 VQ stack，不保存整模型。
- 默认路径：`save_path/stage1_vq/vq_codebook.pt`（可由 `--vq_codebook_path` 覆盖）。
- 同目录下额外保存 `vq_encoder_state.pt`（usage/dead-code 等状态）。

按 epoch 额外保存：
- 当 `save_each_epoch=1`，每个 epoch 结束写入 `save_path/stage1_vq_epoch{N}/vq_codebook.pt`。

复用策略固定如下：
- 当 `reuse_vq_codebook=1` 且 `vq_codebook_path` 存在时，跳过 Stage1 训练并直接加载 codebook 与 VQ state。

### 7. DataLoader 与分桶采样策略
Stage1 的 DataLoader 构建规则固定如下：
- 若 `dataset_name=scienceqa` 且提供了合法 `scienceqa_preprocessed_path`，优先使用分桶采样。
- 分桶采样基于 `ScienceQABucketBatchSampler`，并记录 batch 统计。
- 否则使用常规 `DataLoader(batch_size=args.batch_size, shuffle=True)`。

健康检查固定包含：
- `num_batches`
- `num_samples`
- `batch_size_hist`
- `mean_batch_size`
- `full_batch_ratio`
- `top bucket`（若启用分桶）

## Public Interface Changes
`setup_args()` 中 Stage1 相关参数及默认值：
- `--stage1_lr`：默认 `0.0`（表示自动回退到 `lr*5`）
- `--stage1_recon_weight`：默认 `1.0`
- `--stage1_cosine_weight`：默认 `0.25`
- `--stage1_vq_weight`：默认 `1.0`
- `--stage1_grad_clip`：默认 `5.0`
- `--reuse_vq_codebook`：默认 `1`
- `--vq_codebook_path`：默认空（运行时自动补成 `save_path/stage1_vq/vq_codebook.pt`）

相关 VQ 结构参数：
- `--vq_codebook_size`
- `--vq_commitment_cost`
- `--vq_dead_code_threshold`
- `--vq_usage_decay`
- `--vq_dead_code_reset_interval`
- `--vq_legacy_loss`
- `--freeze_vision_tower`

## Test Plan
### 1. 静态与参数级检查
- 冻结检查：除 `stage1_parameters()` 外，其他参数均为 `requires_grad=False`。
- 训练模式检查：`pre_quant/vq/post_quant` 为 train，`vision_tower` 为 eval。
- 优化器检查：`Adam` 且不包含无关参数。

### 2. Smoke 训练检查
使用 8 到 16 条样本跑 1 个 epoch：
- 训练过程无 `NaN/Inf`。
- `total_loss/recon_loss` 有下降趋势。
- `codebook_used` 非零并逐步增加或稳定。
- `perplexity` 非零且无崩塌到极低值。

### 3. 量化健康检查
至少观察以下信号：
- `dead_code_count` 不应长期单调上升。
- `dead_code_resets` 在开启 reset 时有触发但不失控。
- `vq_loss` 不能持续主导到压制 `recon/cosine`（需结合权重判断）。

### 4. 保存与恢复检查
- `save_stage1_checkpoint` 产物齐全：`vq_codebook.pt` 与 `vq_encoder_state.pt`。
- 通过 `load_vq_codebook` 成功恢复，且后续 Stage2 可直接启动。
- `reuse_vq_codebook=1` 时应命中跳过逻辑并打印加载信息。

## Acceptance Criteria
以下条件全部满足才算完成本计划：
- Stage1 已稳定执行“冻结主干 + 训练 VQ stack”的单一目标。
- 训练日志能完整反映 `recon/cosine/vq/perplexity/dead_code` 指标。
- 保存产物最小化且可被 Stage2 直接复用。
- 同配置下，开启复用时 Stage1 可被可靠跳过，行为可重复。

## Assumptions
- `VQVisionEncoder.stage1_forward` 已正确提供重建与量化相关输出。
- Stage1 使用的数据集分支（`scienceqa` 或 `vq_lord`）均可提供合法 `pixel_values`。
- Stage1 的目标不是最终任务准确率，而是构建稳定视觉离散表示。

## Proven Practice Notes
- 当前稳定路径：VQ stack 计算保持 `float32`，并坚持“仅保存 VQ stack，不保存整模型”的 Stage1 落盘策略。
- ScienceQA smoke（`train_num=64`）中，`stage1_lr=3e-5` 且 `stage1_vq_weight=1.0` 的表现优于把 `stage1_vq_weight` 降到 `0.5`。
- 经验上第 2 个 epoch 往往出现 `vq_loss` 明显抬升（`freeze_vision_tower=0/1` 都会出现），当前推荐优先采用 `stage1_epochs=1`。
- `freeze_vision_tower` 在当前 smoke 范围影响较小，建议默认保持 `0`，除非后续大规模实验给出相反证据。

## Stage1 Tuning Rubric
### Priority P0: Training Health Gate
- 全程无 `NaN/Inf`。
- 非有限损失/梯度被正确跳过并有日志提示。
- 每个 epoch 能正常结束并输出平均损失。

### Priority P1: Reconstruction Quality
- `recon_loss` 稳定下降。
- `cosine_loss` 稳定下降或保持低位。
- `total_loss` 不出现持续震荡失控。

### Priority P2: Codebook Utilization
- `codebook_used` 在训练初期明显提升，后期趋于稳定。
- `perplexity` 不应长期贴近 1。
- `dead_code_count` 可控，reset 机制有效但不应频繁异常触发。

### Priority P3: Handoff Readiness
- 保存出的 Stage1 VQ stack 能被 Stage2 直接加载。
- Stage2 启动后 VQ 路径无形状错误或 state 缺失。

### Tuning Decision Tree
1. `recon_loss` 降不下去：
- 优先提高 `stage1_lr`（或确认是否仍在 `lr*5` 回退策略）。
- 适当提高 `stage1_recon_weight`。
- 检查 `stage1_grad_clip` 是否过小导致更新受限。

2. `cosine_loss` 高且波动大：
- 提高 `stage1_cosine_weight`（如 `0.25 -> 0.5`）。
- 降低 `stage1_lr` 以提升方向收敛稳定性。

3. `codebook_used` 长期偏低或 `perplexity` 过低：
- 调整 `vq_dead_code_threshold` 与 `vq_dead_code_reset_interval`。
- 检查 `vq_usage_decay` 是否过高导致更新过慢。
- 必要时增大 `vq_codebook_size` 前先确认样本规模是否匹配。

4. `dead_code_count` 持续恶化：
- 降低学习率或减小 `stage1_vq_weight`。
- 放宽 reset 间隔，避免频繁重置引入噪声。
- 先确保 `recon/cosine` 稳定，再追求更高 code 使用率。
