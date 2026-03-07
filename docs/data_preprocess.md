# ScienceQA 分桶训练数据预处理方案

## 目标
在不改变任务语义的前提下，提升训练吞吐和显存利用率。

当前痛点是样本尺寸差异导致 `pixel_values` patch 数不同，混合成同一 batch 时 padding 浪费较大，导致有效 `batch_size` 常退化为 1。

本方案通过"按视觉 patch 规模分桶 + 分桶批采样"，使 `batch_size=4` 在 Stage1/2 更可行。

## 核心原则
1. 优先按 patch 分桶，而不是仅按原图 `(H, W)` 分桶。
2. 分桶后仍保留随机性: 桶内打乱 + 桶间打乱 + 每个 epoch 重排。
3. 尾包保留: 不满 4 的批次也训练，不丢样本。
4. 逐阶段放量: Stage1/2 先提 batch，Stage3 保守推进。

## 为什么按 patch 分桶
`LLaVA-Next` 的视觉计算开销和 `pixel_values` 的 patch 数更直接相关。

只按原图尺寸分桶有两个问题:
1. 不同原图可能经过 processor 后得到相同 patch 数，分得过细会降低桶利用率。
2. 相同原图尺寸在不同预处理策略下也可能出现 patch 差异。

建议分桶键顺序:
1. 首选: `patch_count = pixel_values.shape[0]`
2. 次选: `(H, W)`
3. 兜底: `none`（不分桶，走原始随机）

## 预处理与采样流程
每个 epoch 的建议流程如下。

1. 为每个样本计算分桶键
`bucket_key = patch_count` 或 `bucket_key = (H, W)`。

2. 构建桶映射
`bucket_key -> [sample_idx, ...]`。

3. 桶内打乱
每个桶内 `shuffle(indices)`。

4. 生成批次
按 `batch_size=4` 对每个桶切片。尾包 `len < 4` 保留。

5. 桶间打乱
把所有桶产出的批次列表再做一次 `shuffle(batches)`。

6. 交给 `DataLoader(batch_sampler=...)`
由现有 `collate_fn` 继续处理文本 padding 和少量视觉对齐。

## 推荐的数据结构
建议在数据集初始化后缓存桶键，避免每个 epoch 重算 processor。

1. `dataset.bucket_keys: List[Hashable]`
2. `dataset.patch_counts: List[int]`（可选，便于日志）
3. `dataset.image_sizes_hw: List[Tuple[int, int]]`（可选）

缓存时机:
1. 训练前一次性构建。
2. 若 processor 配置变化，需要强制重建。

## Sampler 设计要点
可实现一个 `BucketBatchSampler`，只负责吐出索引批次。

建议参数:
1. `batch_size`
2. `drop_last=False`
3. `shuffle=True`
4. `seed`
5. `bucket_by in {patches, size, none}`

行为定义:
1. `shuffle=False` 时，桶内和桶间都不打乱。
2. `shuffle=True` 时，桶内先乱序，再乱序批次顺序。
3. 每个 epoch 用 `seed + epoch` 保证可复现的随机。

## 与现有代码的衔接点
目标文件: `vq_lord/train_vq_lord.py`

建议改动点:
1. 在 `ScienceQADataset` 增加桶键构建逻辑。
2. 新增 `BucketBatchSampler` 类。
3. `DataLoader` 从 `batch_size + shuffle` 切换为 `batch_sampler`。
4. 保留现有 `vq_lord_collate`，无需推翻。

建议新增参数:
1. `--bucket_by` 默认 `patches`
2. `--bucket_batch_size` 默认 `4`
3. `--bucket_drop_last` 默认 `0`
4. `--bucket_seed` 默认 `scienceqa_seed`
5. `--disable_bucket_for_stage3` 默认 `1`

## Stage 落地顺序
分阶段上线，避免一次性改动引入定位困难。

### Stage1
目标: 验证吞吐提升和 codebook 使用稳定性。

建议:
1. `batch_size=4`（或用 `bucket_batch_size=4`）
2. 启用 `bucket_by=patches`
3. 保持原损失与学习率，先不改优化器策略

观察指标:
1. step time
2. `stage1/codebook_used`
3. loss 曲线是否异常抖动

### Stage2
目标: 在视觉蒸馏阶段扩大有效 batch。

建议:
1. 沿用 Stage1 同一分桶策略
2. 若显存仍富余，可尝试 `bucket_batch_size=6`（谨慎）
3. 保持 `drop_last=False`，避免样本损失

观察指标:
1. `stage2/text_loss`
2. `stage2/vq_loss`
3. 吞吐和显存峰值

### Stage3
目标: 避免 LoRD 多次 forward/generate 导致 OOM。

建议:
1. 首轮保持 `batch_size=1` 或 `2`
2. 可以仅保留"轻分桶"，不强推到 4
3. 通过 `grad_accum` 维持等效总 batch

原因:
1. Stage3 每 step 包含两次采样生成和多次前向/反向，显存压力远高于 Stage1/2。
2. 强行拉到 4 的收益未必覆盖稳定性风险。

## 训练与复现实验建议
建议做三组对照，保证结论可解释。

1. Baseline: `bucket_by=none`, `batch_size=1`
2. Bucket-S: `bucket_by=size`, `batch_size=4`
3. Bucket-P: `bucket_by=patches`, `batch_size=4`

固定条件:
1. 相同数据子集
2. 相同随机种子
3. 相同学习率和 epoch

记录内容:
1. 平均 step time
2. 最大显存
3. 每阶段关键 loss
4. 最终评测指标与 fallback 比例

## 风险与规避
1. 风险: 桶过细导致很多小尾包。
规避: 对低频桶可合并到邻近 patch 桶，或设置最小桶样本阈值。

2. 风险: 顺序偏置。
规避: 桶内与桶间双重 shuffle，每 epoch 重排。

3. 风险: Stage3 显存突增。
规避: Stage3 默认降 batch，必要时开启更高 `grad_accum`。

4. 风险: 文本长度仍差异大。
规避: 保持现有文本 right-padding；后续可追加长度分桶（第二维）。

## 最小可执行版本
第一版优先做最小变更:

1. `ScienceQADataset` 在 `__getitem__` 或初始化阶段暴露 `patch_count`。
2. 增加 `BucketBatchSampler`。
3. Stage1/2 启用分桶，Stage3 先关闭。
4. 用已有 `collate_fn` 直接跑通。

该版本跑通后，再考虑二维分桶（`patch_count + text_len`）进一步减少 padding。

## 结论
该方案工程上可行，且与当前代码结构兼容度高。

最推荐路径是:
1. 先在 Stage1/2 启用 `patch` 分桶并把 batch 提到 4。
2. Stage3 保守（1 或 2）+ `grad_accum`。
3. 用三组对照实验确认吞吐提升是否转化为稳定收益。
