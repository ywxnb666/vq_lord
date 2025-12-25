#!/root/anaconda3/envs/align/bin/python3
"""Debug forward pass with minimal example"""

import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image
import numpy as np

print("Loading processor and model...")
model_path = "/root/workspace/models/llama3-llava-next-8b-hf"
processor = AutoProcessor.from_pretrained(model_path)
model = LlavaForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="cuda"
)

print(f"Model vocab size: {model.config.vocab_size}")
print(f"Text config vocab size: {model.config.text_config.vocab_size}")

# Create test image
image = Image.new('RGB', (336, 336), color='red')

# Test 1: Process with processor (正确方式)
print("\n=== Test 1: Using processor ===")
prompt = "<image>\nQuestion: What is the capital?\nOptions:\n(A) A\n(B) B\nAnswer:"
inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda")
print(f"Input keys: {inputs.keys()}")
print(f"input_ids shape: {inputs['input_ids'].shape}")
print(f"pixel_values shape: {inputs['pixel_values'].shape}")
print(f"pixel_values dtype: {inputs['pixel_values'].dtype}")
print(f"pixel_values min/max: {inputs['pixel_values'].min():.3f}/{inputs['pixel_values'].max():.3f}")

try:
    outputs = model(**inputs)
    print(f"✅ Forward pass succeeded! Logits shape: {outputs.logits.shape}")
except Exception as e:
    print(f"❌ Forward pass failed: {e}")

# Test 2: Manual pixel_values (我们当前的方式)
print("\n=== Test 2: Manual pixel_values ===")
text_inputs = processor(text=prompt, return_tensors="pt").to("cuda")
image_inputs = processor(images=image, return_tensors="pt")
pixel_values = image_inputs['pixel_values'].to("cuda")

print(f"text input_ids shape: {text_inputs['input_ids'].shape}")
print(f"manual pixel_values shape: {pixel_values.shape}")
print(f"manual pixel_values dtype: {pixel_values.dtype}")

try:
    outputs = model(
        input_ids=text_inputs['input_ids'],
        pixel_values=pixel_values
    )
    print(f"✅ Manual forward pass succeeded! Logits shape: {outputs.logits.shape}")
except Exception as e:
    print(f"❌ Manual forward pass failed: {e}")

# Test 3: Check what train_pod_mul.py is doing
print("\n=== Test 3: Simulate train_pod_mul.py ===")
# Load actual data from sciqa_process
from sciqa_process import load_scienceqa_data

p_idxls, idx2ls, _, _, pixel_values_ls = load_scienceqa_data(
    model_path=model_path,
    train_num=4
)

print(f"Loaded {len(idx2ls)} samples")
print(f"First sample idx2ls type: {type(idx2ls[0])}")
print(f"First sample idx2ls length: {len(idx2ls[0])}")
print(f"First sample pixel_values type: {type(pixel_values_ls[0])}")
if pixel_values_ls[0] is not None:
    print(f"First sample pixel_values shape: {pixel_values_ls[0].shape}")
    print(f"First sample pixel_values dtype: {pixel_values_ls[0].dtype}")

# Try forward pass with real data
idxs2 = torch.tensor(idx2ls[0], dtype=torch.long).unsqueeze(0).to("cuda")
pixel_vals = pixel_values_ls[0].unsqueeze(0).to("cuda")

print(f"\nPrepared input_ids shape: {idxs2.shape}")
print(f"Prepared pixel_values shape: {pixel_vals.shape}")
print(f"Prepared pixel_values dtype: {pixel_vals.dtype}")

try:
    outputs = model(input_ids=idxs2, pixel_values=pixel_vals)
    print(f"✅ Real data forward pass succeeded! Logits shape: {outputs.logits.shape}")
except Exception as e:
    print(f"❌ Real data forward pass failed: {e}")
    import traceback
    traceback.print_exc()
