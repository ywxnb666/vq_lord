#!/root/anaconda3/envs/align/bin/python3
"""Test different approaches to load LLaVA model"""

import torch
from transformers import LlavaForConditionalGeneration, AutoProcessor
from PIL import Image
import numpy as np

model_path = "/root/workspace/models/llama3-llava-next-8b-hf"

print("=" * 60)
print("Approach 1: Standard loading with LlavaForConditionalGeneration")
print("=" * 60)

try:
    processor = AutoProcessor.from_pretrained(model_path)
    print("✅ Processor loaded")
    
    # 加载模型,但先不移到GPU
    model = LlavaForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",  # 先加载到CPU检查权重
        low_cpu_mem_usage=True,
    )
    print("✅ Model loaded to CPU")
    
    # 检查关键权重是否存在
    print("\n=== Checking key weights ===")
    
    # 检查language model的embed层
    if hasattr(model, 'language_model'):
        print("✅ model.language_model exists")
        if hasattr(model.language_model, 'model'):
            if hasattr(model.language_model.model, 'embed_tokens'):
                embed_weight = model.language_model.model.embed_tokens.weight
                print(f"✅ model.language_model.model.embed_tokens.weight: {embed_weight.shape}")
            else:
                print("❌ model.language_model.model.embed_tokens NOT found")
        else:
            print("❌ model.language_model.model NOT found")
    else:
        print("❌ model.language_model NOT found")
        
    # 检查是否有扁平命名的权重
    if hasattr(model, 'model'):
        if hasattr(model.model, 'embed_tokens'):
            print(f"✅ Found flat naming: model.model.embed_tokens")
    
    # 打印模型结构的前几层
    print("\n=== Model structure (first level) ===")
    for name, module in model.named_children():
        print(f"  {name}: {type(module).__name__}")
    
    # 检查所有参数名
    print("\n=== Sample parameter names (first 20) ===")
    param_names = list(model.state_dict().keys())[:20]
    for name in param_names:
        print(f"  {name}")
    
    # 移到GPU测试前向传播
    print("\n=== Testing forward pass ===")
    model = model.to("cuda")
    
    # 创建测试输入
    dummy_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    text = "<image>\nQuestion: Test?\nAnswer:"
    
    inputs = processor(text=text, images=dummy_image, return_tensors="pt")
    print(f"Input IDs shape: {inputs['input_ids'].shape}")
    print(f"Pixel values shape: {inputs['pixel_values'].shape}")
    print(f"Input IDs: {inputs['input_ids'][0].tolist()}")
    
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    print(f"✅ Forward pass successful!")
    print(f"Logits shape: {outputs.logits.shape}")
    print(f"Logits range: [{outputs.logits.min():.2f}, {outputs.logits.max():.2f}]")
    
    # 测试生成
    print("\n=== Testing generation ===")
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=10)
    generated_text = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
    print(f"Generated text: {generated_text[:200]}")
    print("✅ Generation successful!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    
print("\n" + "=" * 60)
