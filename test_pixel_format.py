#!/root/anaconda3/envs/align/bin/python3
"""Minimal test for pixel_values dimensions"""

import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

print("Loading processor...")
model_path = "/root/workspace/models/llama3-llava-next-8b-hf"
processor = AutoProcessor.from_pretrained(model_path)

print("\nTesting processor output format...")
from PIL import Image
import numpy as np

# Create a dummy image
image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
text = "<image>\nQuestion: What is this?\nAnswer:"

inputs = processor(text=text, images=image, return_tensors="pt")

print(f"input_ids shape: {inputs['input_ids'].shape}")
print(f"pixel_values shape: {inputs['pixel_values'].shape}")
print(f"pixel_values dtype: {inputs['pixel_values'].dtype}")

# Test what happens when we squeeze(0)
pv_squeezed = inputs['pixel_values'].squeeze(0)
print(f"\nAfter squeeze(0): {pv_squeezed.shape}")

# Test loading model with minimal memory
print("\nLoading model (this may take a while)...")
model = LlavaForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    low_cpu_mem_usage=True
)

print("\nTesting forward pass with original format...")
try:
    with torch.no_grad():
        outputs = model(**inputs.to("cuda"))
    print(f"✅ Original format works! Logits shape: {outputs.logits.shape}")
except Exception as e:
    print(f"❌ Original format failed: {e}")

print("\nTesting forward pass with squeezed+unsqueezed format...")
try:
    with torch.no_grad():
        outputs = model(
            input_ids=inputs['input_ids'].to("cuda"),
            pixel_values=pv_squeezed.unsqueeze(0).to("cuda")
        )
    print(f"✅ Squeezed format works! Logits shape: {outputs.logits.shape}")
except Exception as e:
    print(f"❌ Squeezed format failed: {e}")

print("\n=== Test Complete ===")
