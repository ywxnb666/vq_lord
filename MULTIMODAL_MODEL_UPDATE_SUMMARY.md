# 多模态脚本模型路径更新总结

## 更新日期
2024-12-24

## 更新目标
将所有多模态相关脚本中的学生模型路径统一更新为：
```
/root/workspace/models/llama3-llava-next-8b-hf
```

并参考 `run_llava.py` 的成功加载方式，修正模型加载代码。

---

## 更新的文件清单

### 1. `/root/workspace/align/lord_train_mul.py`
**主要修改**:
- ✅ 将 `LlavaForConditionalGeneration` 改为 `LlavaNextForConditionalGeneration`
- ✅ 将 `AutoProcessor` 改为 `LlavaNextProcessor`
- ✅ 默认 `--from_path` 参数改为 `/root/workspace/models/llama3-llava-next-8b-hf`
- ✅ 模型加载使用 `dtype=torch.float16` (参考 `run_llava.py`)
- ✅ Processor 加载去除 `trust_remote_code=True` (LlavaNext 不需要)

**修改前**:
```python
from transformers import LlavaForConditionalGeneration, AutoProcessor
parser.add_argument("--from_path", default="llava-hf/llava-1.5-7b-hf", ...)
lm = LlavaForConditionalGeneration.from_pretrained(
    args.from_path,
    torch_dtype=torch.bfloat16,
)
processor = AutoProcessor.from_pretrained(args.from_path, trust_remote_code=True)
```

**修改后**:
```python
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
parser.add_argument("--from_path", default="/root/workspace/models/llama3-llava-next-8b-hf", ...)
lm = LlavaNextForConditionalGeneration.from_pretrained(
    args.from_path,
    dtype=torch.float16,
)
processor = LlavaNextProcessor.from_pretrained(args.from_path)
```

---

### 2. `/root/workspace/align/sciqa_process.py`
**主要修改**:
- ✅ 将 `AutoProcessor` 改为 `LlavaNextProcessor`
- ✅ 更新函数文档字符串

**修改前**:
```python
from transformers import AutoProcessor
```

**修改后**:
```python
from transformers import LlavaNextProcessor
```

---

### 3. `/root/workspace/align/merge_lora_mul.py`
**主要修改**:
- ✅ 更新文档字符串，说明使用 `LlavaNextForConditionalGeneration`
- ✅ 默认 `--base_model` 参数保持为 `/root/workspace/models/llama3-llava-next-8b-hf`

**修改前**:
```python
"""
关键修改：
1. 使用 LlavaForConditionalGeneration 而不是 AutoModelForCausalLM
2. 使用 AutoProcessor 处理多模态输入
"""
```

**修改后**:
```python
"""
关键修改：
1. 使用 LlavaNextForConditionalGeneration 而不是 AutoModelForCausalLM
2. 使用 LlavaNextProcessor 处理多模态输入
"""
```

---

### 4. `/root/workspace/align/quick_start_multimodal.sh`
**主要修改**:
- ✅ 更新 `MODEL_PATH` 为 `/root/workspace/models/llama3-llava-next-8b-hf`

**修改前**:
```bash
MODEL_PATH="/root/workspace/models/llama3-llava-next-8b"
```

**修改后**:
```bash
MODEL_PATH="/root/workspace/models/llama3-llava-next-8b-hf"
```

---

### 5. `/root/workspace/align/scripts/6.0.sciqa_lord6_lora.sh`
**状态**: ✅ 已经正确使用 `/root/workspace/models/llama3-llava-next-8b-hf`

**无需修改** - 该文件已经使用正确的路径：
```bash
export from_path="/root/workspace/models/llama3-llava-next-8b-hf"
```

---

## 参考的正确加载方式

基于 `/root/workspace/align/run_llava.py` 的成功实践：

```python
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch

# 1. 加载 Processor
processor = LlavaNextProcessor.from_pretrained(
    "/root/workspace/models/llama3-llava-next-8b-hf"
)

# 2. 加载模型
model = LlavaNextForConditionalGeneration.from_pretrained(
    "/root/workspace/models/llama3-llava-next-8b-hf",
    dtype=torch.float16,  # 使用 float16 节省显存
    device_map="auto",
)
```

**关键点**:
1. ✅ 使用 `LlavaNextForConditionalGeneration` (不是 `LlavaForConditionalGeneration`)
2. ✅ 使用 `LlavaNextProcessor` (不是 `AutoProcessor`)
3. ✅ 使用 `dtype=torch.float16` (不是 `torch_dtype=torch.bfloat16`)
4. ✅ 不需要 `trust_remote_code=True`

---

## 模型路径对比

| 原路径 | 新路径 | 状态 |
|--------|--------|------|
| `llava-hf/llava-1.5-7b-hf` | `/root/workspace/models/llama3-llava-next-8b-hf` | ✅ 已更新 |
| `/root/workspace/models/llama3-llava-next-8b` | `/root/workspace/models/llama3-llava-next-8b-hf` | ✅ 已更新 |

**区别说明**:
- `-hf` 后缀表示这是 HuggingFace 官方转换的兼容版本
- LlavaNext 是 LLaVA v1.6 的正式名称
- Llama-3 基底模型，支持 128K 词汇表

---

## 验证步骤

### 1. 验证模型路径存在
```bash
ls -lh /root/workspace/models/llama3-llava-next-8b-hf
```

应该看到以下文件:
- `config.json`
- `model.safetensors` 或 `pytorch_model.bin`
- `preprocessor_config.json`
- `tokenizer_config.json`
- ...

### 2. 测试快速加载
```bash
cd /root/workspace/align
python run_llava.py
```

应该能成功加载并生成输出。

### 3. 测试训练脚本
```bash
cd /root/workspace/align
bash quick_start_multimodal.sh
```

应该能成功启动训练。

---

## 未修改的文件

以下文件**已经使用正确路径**，无需修改：
- ✅ `/root/workspace/align/run_llava.py`
- ✅ `/root/workspace/align/scripts/6.0.sciqa_lord6_lora.sh`
- ✅ `/root/workspace/align/test_sciqa_data.py`
- ✅ `/root/workspace/align/test_llava_loading.py`
- ✅ `/root/workspace/align/debug_forward_pass.py`
- ✅ `/root/workspace/align/test_pixel_format.py`
- ✅ `/root/workspace/align/test_single_forward.py`

---

## train_pod_mul.py 的特殊说明

`/root/workspace/align/train_pod_mul.py` **不需要直接修改**，因为：

1. ✅ 它接收 `image_processor` 作为参数，由 `lord_train_mul.py` 传入
2. ✅ 它使用的是 `processor.image_processor`，而不是直接加载模型
3. ✅ 只要 `lord_train_mul.py` 使用正确的 `LlavaNextProcessor`，`train_pod_mul.py` 就会自动正确工作

**调用链**:
```
lord_train_mul.py (加载 LlavaNextProcessor)
    ↓
提取 processor.image_processor
    ↓
传递给 train_pod_mul.py 的 train() 函数
```

---

## 兼容性检查

### LlavaNext vs Llava
- ✅ `LlavaNextForConditionalGeneration` 是 LLaVA v1.6 的官方类
- ✅ `LlavaForConditionalGeneration` 是 LLaVA v1.5 的类
- ✅ 两者架构相似，但 LlavaNext 支持更高分辨率和动态分块

### Processor 类型
- ✅ `LlavaNextProcessor` 是专门为 LlavaNext 模型设计的
- ✅ `AutoProcessor` 可以自动检测，但显式指定更可靠
- ✅ 两者都包含 `tokenizer` 和 `image_processor` 属性

### dtype 选择
- ✅ `torch.float16` - 通用，节省显存，推荐用于推理和训练
- ⚠️ `torch.bfloat16` - 更稳定，但需要 Ampere 架构 GPU (A100/3090/4090)
- 📝 根据 `run_llava.py` 的成功经验，使用 `float16`

---

## 总结

✅ **所有必要的文件都已更新**

✅ **模型路径已统一为**: `/root/workspace/models/llama3-llava-next-8b-hf`

✅ **加载方式已参考**: `run_llava.py` 的成功模式

✅ **类型已修正**: `LlavaNextForConditionalGeneration` + `LlavaNextProcessor`

✅ **dtype 已优化**: `torch.float16` (节省显存)

**下一步建议**:
1. 运行 `run_llava.py` 验证模型加载
2. 运行 `quick_start_multimodal.sh` 验证训练流程
3. 如果成功，运行完整训练 `scripts/6.0.sciqa_lord6_lora.sh`

---

**更新人**: GitHub Copilot  
**更新日期**: 2024-12-24  
**版本**: v1.1
