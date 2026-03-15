# Stage2 重构执行计划

## Maintenance Rule
- 以后每次修改 `stage2` 相关实现时，必须同步更新本文件，确保 `docs/stage2_plan.md` 与 `vq_lord2/train_vq_lord2.py` 保持一致。

## Summary
本计划将 `stage2` 从“仅训练视觉侧的长文本 SFT”重构为“基于稳定 `vq_codebook` 的视觉能力蒸馏 + 为 `stage3` 提供可采样 warm start policy”的阶段。

重构后的 `stage2` 必须同时满足两件事：
- 让学生模型学会利用 `stage1` 产出的离散视觉表示做出正确视觉判断，而不是主要依赖语言先验。
- 让 `stage3` 一开始就能从学生策略中采样出质量可排序的候选回答，降低 `cold_start` 依赖。

默认设计结论：
- `stage2` 要训练 `LoRA`，不能只训练 `vq / projector / vision`。
- `stage2` 要拆分监督目标：强监督答案，弱监督解释。
- `stage2` 默认冻结 `vq.embedding`，把 `stage1` 学到的 codebook 当作稳定视觉词表使用。
- `stage2` 的主评估不再只看总 loss，要同时看答案能力和对 `stage3` 的 warm start 效果。

## Implementation Changes
### 1. 重新定义 Stage2 目标
将 `train_stage2_vision` 的目标改为：
`L_stage2 = w_answer * L_answer + w_rationale * L_rationale + w_vq * L_vq`

默认权重：
- `w_answer = 1.0`
- `w_rationale = 0.3`
- `w_vq = args.beta`

语义定义：
- `L_answer`：只监督最终答案段，目标是把视觉判断能力蒸馏出来。
- `L_rationale`：监督教师解释文本，但权重低于答案，防止训练目标被文风和长文本噪声主导。
- `L_vq`：继续保持量化路径与上游视觉表征一致，但不再把它当作 `stage2` 的主目标。

### 2. 修改 Stage2 的可训练参数集合
`stage2` 的参数策略改为固定规则，不留实现者自行决定：
- 训练 `LoRA` 参数：名字包含 `lora_` 或 `modules_to_save`。
- 训练多模态桥接层：名字包含 `projector` 或 `multi_modal_projector`。
- 训练 `pre_quant` 与 `post_quant`。
- 如果 `freeze_vision_tower=0`，训练整个 vision tower，但使用更小学习率。
- 默认冻结 `vq.embedding.weight`。
- 不训练语言模型原始全参。

优化器分组固定如下：
- `LoRA + projector`：学习率 `args.lr`
- `pre_quant + post_quant`：学习率 `args.lr * 0.5`
- `vision_tower`：学习率 `args.lr * 0.2`
- `vq.embedding`：不加入 optimizer

理由：
- `stage1` 已经把 codebook 训稳，`stage2` 的职责是“利用 codebook 学视觉能力”，不是继续大幅改写 codebook。
- `stage3` 真正会更新 `LoRA`，所以 `stage2` 必须先把 `LoRA` 训成可用策略。
- `pre_quant/post_quant` 虽然不直接修改 codebook，但会间接改变量化输入分布；给它们更小学习率更符合“稳定视觉词表 + 允许有限适配”的目标。

### 3. 修改 Stage2 数据目标构造
在 ScienceQA 样本中显式保存并使用以下字段：
- `answer_idx`
- `answer_letter`
- `answer_text`
- `teacher_response`

其中：
- `answer_letter` 固定使用 `A/B/C/...`
- `answer_text` 为选项原文
- `teacher_response` 继续保留教师解释文本

Stage2 为每个样本构造两个监督目标：
- `answer_target`
  固定模板：`Answer: <letter>`
- `rationale_target`
  固定模板：`Explanation: ...\nAnswer: <letter>` 或中文对应格式

规则固定如下：
- 若教师回答中存在解释段，则使用解释段加上规范化后的 `Answer: <letter>`。
- 若教师回答缺失解释段，则 `rationale_target` 只保留 `Answer: <letter>`，此时 `L_rationale` 允许退化为 0。
- `answer_target` 一律使用数据集 canonical answer，不从教师自由文本中解析。
- 若 explanation 过长需要截断，必须优先保留尾部 `Answer: <letter>` 锚点；宁可丢弃 explanation，也不能生成一个没有最终答案锚点的 `rationale_target`。

### 4. 修改 Dataset 输出结构
`ScienceQADataset` 输出从“单一 full sequence”改为“一条 prompt 对应两条监督序列”：
- `prompt_input_ids`
- `prompt_attention_mask`
- `pixel_values`
- `image_sizes`
- `full_input_ids`
- `full_attention_mask`
- `full_labels`
- `answer_input_ids`
- `answer_attention_mask`
- `answer_labels`

构造规则固定如下：
- `prompt_*` 仅包含用户提问与图像。
- `full_*` 对应 `rationale_target`。
- `answer_*` 对应 `answer_target`。
- `full_labels` 仅监督 assistant 回答段。
- `answer_labels` 监督完整 `Answer: <letter>` 后缀，而不是只监督最终答案字母单 token。
- 所有 `label` 中 prompt 和 pad 位置都置为 `-100`。
- 数据集只显式输出 `has_rationale` 标记；是否跳过 `L_rationale` 由训练循环根据 `full_labels` 和 `has_rationale` 动态决定，不额外持久化 `rationale_labels`。

`collate_fn` 必须支持对 `full_*` 和 `answer_*` 两套序列分别 padding。

### 5. 修改 Stage2 训练循环
每个 batch 中默认执行两条监督路径：
- 前向 1：`answer_input_ids + answer_labels`，得到 `L_answer`
- 前向 2：`full_input_ids + full_labels`，得到 `L_rationale`

解释路径的执行规则固定如下：
- 若当前 batch 中存在有效 explanation 监督，则执行第二次前向。
- 若当前 batch 所有样本都缺失 explanation，则跳过第二次前向，并令 `L_rationale = 0`。

两次前向都传入同一份 `pixel_values` 和 `image_sizes`，都走相同 VQ 视觉路径。

损失合成规则固定如下：
- `L_answer` 直接用 `outputs.loss`
- `L_rationale` 直接用 `outputs.loss`
- `L_vq` 从 `_vq_loss_container["loss"]` 读取
- 若执行了第二次前向，则 `L_vq` 使用第二次前向后的最新值
- 若 explanation 路径被跳过，则 `L_vq` 回退为答案路径前向后的值
- `L_total = 1.0 * L_answer + 0.3 * L_rationale + args.beta * L_vq`

反向传播固定为：
- 将 `L_total / grad_accum` 做一次 backward
- 保持现有 gradient accumulation 机制
- 加入梯度裁剪，默认阈值 `1.0`
- 梯度裁剪只在 `grad_accum` 完成、且 `optimizer.step()` 之前执行一次，不能在中间累积步提前裁剪

日志固定增加：
- `stage2/answer_loss`
- `stage2/rationale_loss`
- `stage2/vq_loss`
- `stage2/vq_answer_ratio`
- `stage2/vq_perplexity`
- `stage2/dead_code_resets`
- `stage2/dead_code_count`
- `stage2/total_loss`
- `stage2/answer_only_accuracy_proxy`
  说明：使用答案字母位点的 top-1 命中率作为在线代理指标；计算时应忽略末尾 `eos`，避免把 `eot/eos` 误当作答案 token

VQ 监控规则固定如下：
- `stage2/vq_answer_ratio = beta * vq_loss / answer_loss`
- 若 `vq_answer_ratio` 长时间过高，说明 VQ 正则开始主导 `pre_quant` 梯度，应优先考虑调低 `beta`
- 需同时观察 `vq_perplexity` 与 `dead_code_count`，防止 `pre_quant` 把输入分布拉向少数 code

### 6. 调整 Stage2 与 Stage3 的衔接
`stage3` 的输入假设不变，但要以新的 `stage2` 产物为 warm start。

交接目标固定为：
- `stage2` 结束时，模型已经能在同一 prompt 上生成或打分出较可靠的答案字母。
- `stage3` 进入 LoRD 时，`cold_start_count / total_samples` 必须比当前主干实现更低，不能更高。
- `stage3` 仍冻结 `vq.embedding`，不改现有冻结逻辑。
- `stage3` 默认使用更小学习率继续训练 LoRA，避免对比学习初期破坏 Stage2 学到的答案能力。
- `stage3` 默认冻结 projector，仅在显式需要时再放开。

`save_stage2_checkpoint` 需额外记录以下配置到 `stage2_config.json`：
- `stage2_answer_weight`
- `stage2_rationale_weight`
- `stage2_train_lora=1`
- `stage2_train_codebook=0`
- `stage2_answer_format="letter"`

`load_stage2_checkpoint` 规则固定如下：
- Stage2 LoRA 权重应直接加载回现有默认 adapter。
- 不应在恢复 Stage2 checkpoint 时额外创建新的 adapter name，以免干扰 Stage3 的参数遍历、优化器状态和后续保存语义。

## Public Interface Changes
`setup_args()` 中新增参数，默认值固定如下：
- `--stage2_answer_weight`，默认 `1.0`
- `--stage2_rationale_weight`，默认 `0.3`
- `--stage2_prepost_lr_scale`，默认 `0.5`
- `--stage2_vision_lr_scale`，默认 `0.2`
- `--stage2_grad_clip`，默认 `1.0`
- `--stage3_lr_scale`，默认 `0.2`
- `--stage3_train_projector`，默认 `0`

现有参数语义调整：
- `--beta` 在 `stage2` 中仅表示 `L_vq` 的权重，不再隐含“唯一辅助项”。
- `--reuse_stage2` 的复用前提不变，但加载后的行为默认假设该 checkpoint 已包含 `LoRA + projector + VQ stack` 的联合 warm start。

## Test Plan
### 1. 静态与单元级检查
- Dataset 样本检查：
  - `answer_labels` 应覆盖完整 `Answer: <letter>` 后缀，而不只是最终字母
  - `full_labels` 覆盖 explanation 和 answer 段
  - `answer_target` 始终是 canonical `Answer: <letter>`
  - 当 `has_rationale=0` 时，训练循环中的 `L_rationale` 应退化为 `0`
- 参数检查：
  - `LoRA` 参数必须为 `requires_grad=True`
  - `vq.embedding.weight` 必须为 `requires_grad=False`
  - `vision_tower` 在 `freeze_vision_tower=0` 时为 trainable 且位于单独 optimizer group
- Collate 检查：
  - `full_*` 和 `answer_*` 都能正确 padding
  - `image_sizes` 与 batch 对齐不丢样本

### 2. Smoke 训练检查
使用 8 到 16 条 ScienceQA 样本跑 1 个 epoch：
- 训练过程无 `image features and image tokens do not match`
- `answer_loss` 能下降
- `total_loss` 有限且无 NaN
- checkpoint 可保存、可加载、可继续推理

### 3. Stage2 效果检查
在固定 200 条验证样本上，使用 `answer_mode=logits` 评测：
- 新版 `stage2` 的准确率不得低于当前主干 `stage2`
- 新版 `stage2` 的答案字母 top-1 代理准确率应高于旧版
- 若 explanation 生成质量下降但答案准确率明显提高，视为可接受
- 建议追加 10 到 20 条样本的 `model.generate()` 诊断，确认 Stage2 warm start 已能生成包含正确答案字母的文本；该诊断用于评估与 Stage3 `tau1` 的衔接，不作为硬阻塞条件。

### 4. Stage3 衔接检查
在同一 smoke 配置上跑 `stage3` 前若干 step：
- `cold_start_count / total_samples` 相比旧版不应恶化，且应呈下降趋势
- `stage3` 初期 `L_obj` 和 `L_reg` 为有限值
- 生成样本不应大面积退化为空回答或只复制 prompt
- 若 Stage3 初期答案能力明显回退，优先检查 `stage3_lr_scale` 是否过大，以及是否误开启了 `stage3_train_projector`

## Acceptance Criteria
以下条件全部满足才算完成本计划：
- `stage2` 代码已从单一 `text_loss + beta*vq_loss` 重构为双监督目标。
- `LoRA` 已进入 `stage2` 训练图。
- `vq.embedding` 默认不在 `stage2` 中更新。
- Stage2 checkpoint 能作为 `stage3` 的直接 warm start 使用。
- 同配置下，新版 `stage2` 至少不弱于旧版 `stage2` 的 ScienceQA 准确率。
- 同配置下，新版 `stage3` 初期 `cold_start` 比例低于旧版。

## Assumptions
- `stage1` 产出的 `vq_codebook`、`pre_quant`、`post_quant` 已经足够稳定，本计划不再重新设计 `stage1`。
- `stage2` 只针对 `ScienceQA` 主路径设计，不处理 `vq_lord` 老数据集分支的额外增强。
- 教师回答的主要价值是提供解释性监督；最终答案监督以 ScienceQA canonical label 为准。
- 当前实现优先保证“答案能力 + stage3 可衔接性”，不追求 explanation 文风与教师完全一致。

## Stage2 Tuning Rubric
### Priority P0: Training Health Gate
- `stage2/total_loss`、`stage2/answer_loss`、`stage2/rationale_loss` 全程有限，无 `NaN/Inf`。
- 训练中因非有限梯度导致的 step 跳过应接近 0。
- Stage2 checkpoint 必须可保存、可加载、可推理。

### Priority P1: Core Answer Ability (Primary Objective)
- `stage2/answer_only_accuracy_proxy` 持续上升并收敛；目标优先看该指标。
- `stage2/answer_loss` 稳定下降；后期一般应显著低于初始值（常见可到 `<1.0`）。
- checkpoint 选择顺序：先看 `answer_only_accuracy_proxy`，再看 `answer_loss`，最后看 `total_loss`。

### Priority P2: VQ Stability Constraints
- `stage2/vq_answer_ratio = beta * vq_loss / answer_loss`。
- 经验健康区间：`0.05-0.15`；持续 `>0.30` 视为 VQ 正则过强风险。
- `stage2/vq_perplexity` 不应相对 Stage1 末值明显下跌（建议跌幅不超过 `20%`）。
- `stage2/dead_code_count` 不应持续上升。

### Priority P3: Auxiliary and Handoff Checks
- `stage2/rationale_loss` 仅作辅助参考，不作为主优化目标。
- Stage2 结束后做 50 到 100 条样本 `model.generate()` 诊断：
- 输出包含 `Answer: [A-D]` 的比例建议 `>80%`。
- 字母准确率建议 `>50%`（显著高于随机 `25%`）。
- Stage3 前 100 step 的 cold-start 比例不应恶化，建议 `<20%`，理想 `<5%`。

### Tuning Decision Tree
1. `answer_only_accuracy_proxy` 不涨或涨得很慢：
- 若 `vq_answer_ratio > 0.30`：先降 `beta`（如 `0.25 -> 0.10/0.05`）。
- 若 `vq_answer_ratio` 正常但 `answer_loss` 震荡：降 `lr` 或增 `grad_accum`。
- 若 `answer_loss` 稳降但 acc 仍低：增 `lora_rank`（如 `64 -> 128`）或增加 Stage2 epoch。

2. VQ 稳定性恶化（`vq_perplexity` 下跌明显或 `dead_code_count` 异常）：
- 若 `dead_code_count` 上升：优先排查冻结逻辑 bug。
- 若冻结正常：降 `stage2_prepost_lr_scale`（`0.5 -> 0.2 -> 0.1`），必要时临时冻结 pre/post quant。
- 同时可降 `stage2_vision_lr_scale`（`0.2 -> 0.1`）。

3. `rationale_loss` 下降快于 `answer_loss` 且答案能力无提升：
- 降 `stage2_rationale_weight`（`0.3 -> 0.1`）。
- `stage2_answer_weight` 默认保持 `1.0`。

4. Stage2 指标正常但 Stage3 cold-start 仍高：
- 优先降低 `stage3_lr_scale`（如 `0.2 -> 0.1`）。
- 默认保持 `stage3_train_projector=0`（冻结 projector）。
- 再考虑增加 Stage2 epoch。

### Recommended Tuning Order
1. `beta`
2. `stage2_prepost_lr_scale`
3. `lr` and `grad_accum`
4. `lora_rank`
5. `stage2_rationale_weight`
6. `stage2_vision_lr_scale`

## First Tuning Plan (Round-1)
### Goal
- 在不引入教师 token 概率蒸馏的前提下，建立可信 Stage2 评估闭环，并完成第一轮可复现调参。

### Step 1: Fix Evaluation Reliability (Blocking)
1. 修复 `vq_lord2/sciqa_process.py` 的 VQ 推理加载一致性：
- 除 `vq_codebook.pt` 外，必须同时加载 `vq_encoder_state.pt`（含 `pre_quant/post_quant`）。
- 若缺失 `vq_encoder_state.pt`，必须打印告警，提示评估不可信。
2. 修复 `stage2/answer_only_accuracy_proxy` 位点定义：
- 从“最后监督 token”改为“答案字母位点（忽略 eos）”。
3. 增加 epoch 级日志：
- 新增 `stage2_epoch/*` 指标，用于观察跨 epoch 趋势，避免单 batch 噪声误判。

### Step 2: Round-1 Training (No New Objective)
1. 训练配置：
- `epochs=3`
- `beta=0.05`
- `stage2_answer_weight=1.0`
- `stage2_rationale_weight=0.2`
- `stage2_prepost_lr_scale=0.2`
- `stage2_vision_lr_scale=0.2`
2. 本轮不引入新损失项：
- 不加入 `choice_ce_loss`
- 不加入教师 top-token 概率相关蒸馏
- 不改 Stage1 codebook size
3. 每轮结束评估：
- 保存 Stage2 checkpoint
- 使用修复后的 `vq_lord2/sciqa_process.py` 在 `validation` 上跑 `answer_mode=hybrid`，记录 `val_acc` 与答案格式率。

### Step 3: Go/No-Go Decision
1. 若 `val_acc` 随 epoch 上升且最终 `>50%`：
- 视为 Stage2 warm-start 基本达标，进入 Stage3 小步验证（前 100-200 step 冷启动率）。
2. 若 `val_acc` 停滞：
- 第二轮再引入 `warmup+cosine` 调度，保持其余配置不变复跑。
3. 若 `val_acc` 明显偏低（如 `<45%`）且格式率正常：
- 优先排查 Stage1 表征利用率（`vq_perplexity/dead_code_count`），再决定是否调整 Stage1，而非直接增大 codebook。
