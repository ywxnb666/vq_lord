# 🚀 多模态模型窃取 - 快速开始指南

## 一、文件清单 ✅

所有必需文件已创建完毕：

```bash
# 核心代码文件 (5个)
sciqa_process.py              # 数据处理模块
lord_train_mul.py             # 主训练脚本  
train_pod_mul.py              # 训练逻辑实现
merge_lora_mul.py             # 权重合并工具
scripts/6.0.sciqa_lord6_lora.sh  # 启动脚本

# 文档文件 (2个)
README_MULTIMODAL.md          # 详细文档
MULTIMODAL_FILES_CHECKLIST.md # 完整性检查清单
```

## 二、验证文件完整性

```bash
cd /root/workspace/align

# 检查文件是否存在
ls -lh sciqa_process.py lord_train_mul.py train_pod_mul.py merge_lora_mul.py scripts/6.0.sciqa_lord6_lora.sh

# 预期输出：
# -rw-r--r-- 1 root root 11K sciqa_process.py
# -rw-r--r-- 1 root root 8.1K lord_train_mul.py
# -rw-r--r-- 1 root root 26K train_pod_mul.py
# -rw-r--r-- 1 root root 7.0K merge_lora_mul.py
# -rwxr-xr-x 1 root root 5.2K scripts/6.0.sciqa_lord6_lora.sh
```

## 三、环境准备

### 3.1 检查Python环境

```bash
# 检查Python版本 (需要 >= 3.8)
python --version

# 检查必要的包
python << 'PYEOF'
try:
    import transformers
    print(f"✓ transformers {transformers.__version__}")
    import peft
    print(f"✓ peft installed")
    import datasets
    print(f"✓ datasets installed")
    from PIL import Image
    print(f"✓ PIL installed")
    import torch
    print(f"✓ torch {torch.__version__}")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✓ CUDA devices: {torch.cuda.device_count()}")
except ImportError as e:
    print(f"✗ Missing package: {e}")
PYEOF
```

### 3.2 检查模型文件

```bash
# 检查LLaVA模型是否存在
MODEL_PATH="/root/workspace/models/llama3-llava-next-8b-hf"

if [ -d "$MODEL_PATH" ]; then
    echo "✓ LLaVA model found at $MODEL_PATH"
    ls -lh $MODEL_PATH/ | head -10
else
    echo "✗ LLaVA model NOT found at $MODEL_PATH"
    echo "Please download the model first:"
    echo "  git lfs install"
    echo "  git clone https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf $MODEL_PATH"
fi
```

## 四、快速测试（小样本）

### 4.1 创建测试脚本

```bash
cd /root/workspace/align

# 创建测试配置
cat > test_multimodal_quick.sh << 'TESTEOF'
#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export PYTHONIOENCODING=utf-8
export TORCH_USE_CUDA_DSA="1"
export HF_ENDPOINT="https://hf-mirror.com"

python=${HOME}/anaconda3/envs/align/bin/python3
root_dir="/root/workspace/align/"

# 测试参数（最小配置）
export from_path="/root/workspace/models/llama3-llava-next-8b-hf"
export save_path="${root_dir}sciqa_ckpts/TEST_quick"
export train_num=8        # 只用8个样本快速测试
export epoch=1
export period=1
export sub_stage_num=2
export max_new_tokens=32
export max_length=256

echo "========================================="
echo "快速测试配置:"
echo "  - 训练样本数: $train_num"
echo "  - 训练轮数: $epoch"
echo "  - 训练周期: $period"
echo "========================================="

$python ${root_dir}lord_train_mul.py \
    --dataset_task=scienceqa \
    --use_lora=1 \
    --rank=32 \
    --lora_alpha=64 \
    --from_path=$from_path \
    --is_black_box=1 \
    --sub_set_num=2 \
    --sub_stage_num=$sub_stage_num \
    --infer_batch_size=1 \
    --tau1=0.80 \
    --tau2=0.85 \
    --task=LoRD-VI \
    --device=cuda \
    --epoch=$epoch \
    --period_num=$period \
    --acc_step=1 \
    --log_step=1 \
    --save_step=100 \
    --train_num=$train_num \
    --max_new_tokens=$max_new_tokens \
    --LR=3e-5 \
    --beta=1.0 \
    --temperature=1.5 \
    --batch_size=1 \
    --use_old_logits=1 \
    --use_vic_logits=1 \
    --use_kld=0 \
    --max_length=$max_length \
    --save_path=$save_path

echo "测试完成！检查输出:"
ls -lh $save_path/
TESTEOF

chmod +x test_multimodal_quick.sh
```

### 4.2 运行快速测试

```bash
# 运行测试（预计5-10分钟）
bash test_multimodal_quick.sh

# 如果成功，应该看到:
# - 数据加载信息
# - 模型加载信息  
# - 训练进度条
# - checkpoint 保存信息
```

### 4.3 检查测试结果

```bash
# 查看生成的文件
ls -lh sciqa_ckpts/TEST_quick/

# 预期输出：
# - adapter_config.json
# - adapter_model.safetensors
# - (其他配置文件)

# 查看日志
tail -n 50 sciqa_ckpts/TEST_quick___log_writer/events.out.*
```

## 五、正式训练

如果快速测试通过，可以开始正式训练：

### 5.1 编辑训练参数

```bash
vim scripts/6.0.sciqa_lord6_lora.sh

# 关键参数：
# - TRAIN_NUMS=(128)      # 训练样本数
# - epoch=2               # 训练轮数
# - period=3              # 训练周期
# - sub_stage_num=64      # 子阶段数
```

### 5.2 启动训练

```bash
cd /root/workspace/align

# 方法1: 直接运行
bash scripts/6.0.sciqa_lord6_lora.sh

# 方法2: 后台运行并记录日志
nohup bash scripts/6.0.sciqa_lord6_lora.sh > training_multimodal.log 2>&1 &

# 查看进程
ps aux | grep lord_train_mul

# 实时查看日志
tail -f training_multimodal.log
```

### 5.3 监控训练进度

```bash
# 查看GPU使用情况
watch -n 1 nvidia-smi

# 查看checkpoint
watch -n 60 'ls -lht sciqa_ckpts/ | head -10'

# 查看TensorBoard (如果安装了)
tensorboard --logdir=sciqa_ckpts/SCIQAscienceqa*___log_writer --port=6006
```

## 六、训练后处理

### 6.1 合并LoRA权重

```bash
cd /root/workspace/align

# 找到最佳checkpoint (通常是最后一个period)
BEST_CKPT=$(ls -td sciqa_ckpts/SCIQAscienceqa*___period* | head -1)
echo "Best checkpoint: $BEST_CKPT"

# 合并权重
python merge_lora_mul.py \
    --base_model /root/workspace/models/llama3-llava-next-8b-hf \
    --lora_path $BEST_CKPT \
    --save_path ./sciqa_ckpts/MERGED/llava-sciqa-final

echo "合并完成！模型保存在: ./sciqa_ckpts/MERGED/llava-sciqa-final"
```

### 6.2 测试合并后的模型

```python
# test_merged_model.py
import torch
from transformers import LlavaNextForConditionalGeneration, AutoProcessor
from PIL import Image
import requests

# 加载模型
model_path = "./sciqa_ckpts/MERGED/llava-sciqa-final"
model = LlavaNextForConditionalGeneration.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.bfloat16
)
processor = AutoProcessor.from_pretrained(model_path)

# 测试推理
def test_inference(image_url, question):
    image = Image.open(requests.get(image_url, stream=True).raw)
    prompt = f"<image>\nQuestion: {question}\nAnswer:"
    
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=100)
    
    output_text = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
    print(f"Q: {question}")
    print(f"A: {output_text}")
    print("-" * 50)

# 运行测试
print("Testing merged model...")
test_inference(
    "https://example.com/science_image.jpg",
    "What is shown in this image?"
)
```

```bash
# 运行测试
python test_merged_model.py
```

## 七、常见问题排查

### 问题1: CUDA Out of Memory

```bash
# 解决方案: 减小批次大小和序列长度
vim scripts/6.0.sciqa_lord6_lora.sh

# 修改:
export batch_size=1
export infer_batch_size=1  
export max_length=256       # 从512降到256
export max_new_tokens=64    # 从128降到64
```

### 问题2: 数据集下载失败

```bash
# 使用HuggingFace镜像
export HF_ENDPOINT="https://hf-mirror.com"

# 或手动下载数据集
python << 'PYEOF'
from datasets import load_dataset
dataset = load_dataset("derek-thomas/ScienceQA", cache_dir="./data/scienceqa")
PYEOF
```

### 问题3: 模型加载失败

```bash
# 检查模型文件完整性
python << 'PYEOF'
from transformers import LlavaNextForConditionalGeneration
try:
    model = LlavaNextForConditionalGeneration.from_pretrained(
        "/root/workspace/models/llama3-llava-next-8b-hf",
        torch_dtype="auto",
        device_map="cpu"  # 先用CPU测试
    )
    print("✓ Model loaded successfully")
except Exception as e:
    print(f"✗ Error: {e}")
PYEOF
```

### 问题4: 语法错误

```bash
# 检查所有新文件的语法
cd /root/workspace/align
for file in sciqa_process.py lord_train_mul.py train_pod_mul.py merge_lora_mul.py; do
    echo "Checking $file..."
    python -m py_compile $file && echo "✓ $file OK" || echo "✗ $file has syntax errors"
done
```

## 八、性能优化建议

### 8.1 提高训练速度

```bash
# 1. 使用混合精度训练（已默认启用 bfloat16）
# 2. 启用gradient checkpointing
# 在 lord_train_mul.py 中添加:
# model.gradient_checkpointing_enable()

# 3. 增加子集大小（减少模型加载次数）
export sub_set_num=8  # 从4增加到8

# 4. 使用更快的数据加载
export infer_batch_size=2  # 如果显存允许
```

### 8.2 提高模型质量

```bash
# 1. 增加训练样本
export TRAIN_NUMS=(256 512)

# 2. 增加训练周期
export period=5

# 3. 调整学习率
export LR="5e-5"  # 可尝试不同值

# 4. 使用更大的LoRA rank
export rank=128
export lora_alpha=256
```

## 九、清理与重启

### 清理测试文件

```bash
# 删除测试checkpoint
rm -rf sciqa_ckpts/TEST_quick

# 清理测试脚本
rm test_multimodal_quick.sh
```

### 完全重新开始

```bash
# 警告: 这会删除所有训练结果！
rm -rf sciqa_ckpts/
rm -f training_multimodal.log
rm -f nohup.out
```

## 十、下一步

训练完成后，你可以：

1. **评估模型**: 在ScienceQA测试集上测试
2. **迁移到其他数据集**: 修改`sciqa_process.py`支持新数据
3. **优化模型**: 尝试不同的超参数组合
4. **部署模型**: 将合并后的模型部署到生产环境

详细信息请参考 `README_MULTIMODAL.md`

---

**最后更新**: 2024-12-18
**状态**: ✅ 所有文件已验证完整
**准备就绪**: 可以开始训练！
