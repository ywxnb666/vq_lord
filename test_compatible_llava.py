#!/root/anaconda3/envs/align/bin/python3
"""
测试使用HuggingFace兼容的LLaVA模型
使用 llava-hf/llava-1.5-7b-hf 替代不兼容的模型
"""

import torch
from transformers import LlavaForConditionalGeneration, AutoProcessor
from PIL import Image
import numpy as np

print("=" * 70)
print("测试 HuggingFace 兼容的 LLaVA 模型")
print("模型: llava-hf/llava-1.5-7b-hf")
print("=" * 70)

model_path = "llava-hf/llava-1.5-7b-hf"

print("\n[1/5] 加载 Processor...")
processor = AutoProcessor.from_pretrained(model_path)
print("✅ Processor 加载成功")

print("\n[2/5] 加载模型...")
model = LlavaForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    low_cpu_mem_usage=True,
)
print("✅ 模型加载成功")

# 检查模型结构
print("\n[3/5] 检查模型结构...")
print(f"模型类型: {type(model).__name__}")
if hasattr(model, 'language_model'):
    if hasattr(model.language_model, 'model') and hasattr(model.language_model.model, 'embed_tokens'):
        embed_shape = model.language_model.model.embed_tokens.weight.shape
        print(f"✅ Embedding 权重形状: {embed_shape}")
        print(f"   词汇表大小: {embed_shape[0]}")
        print(f"   隐藏维度: {embed_shape[1]}")

print("\n[4/5] 测试前向传播...")
# 创建测试输入
dummy_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
text = "USER: <image>\nWhat is shown in this image?\nASSISTANT:"

inputs = processor(text=text, images=dummy_image, return_tensors="pt")
print(f"Input IDs shape: {inputs['input_ids'].shape}")
print(f"Pixel values shape: {inputs['pixel_values'].shape}")

inputs = {k: v.to("cuda") for k, v in inputs.items()}

try:
    with torch.no_grad():
        outputs = model(**inputs)
    print(f"✅ 前向传播成功!")
    print(f"   Logits shape: {outputs.logits.shape}")
    print(f"   Logits dtype: {outputs.logits.dtype}")
except Exception as e:
    print(f"❌ 前向传播失败: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n[5/5] 测试生成...")
try:
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,
        )
    generated_text = processor.batch_decode(output_ids, skip_special_tokens=False)[0]
    print(f"✅ 生成成功!")
    print(f"生成的文本: {generated_text}")
except Exception as e:
    print(f"❌ 生成失败: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "=" * 70)
print("✅ 所有测试通过! 这个模型可以用于训练!")
print("=" * 70)
