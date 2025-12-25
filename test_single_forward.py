#!/root/anaconda3/envs/align/bin/python3
"""Test single sample forward pass"""

import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from sciqa_process import load_scienceqa_data

print("Loading model...")
model_path = "/root/workspace/models/llama3-llava-next-8b-hf"
processor = AutoProcessor.from_pretrained(model_path)
model = LlavaForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    low_cpu_mem_usage=True
)
model.eval()

print("\nLoading data...")
p_idxls, idx2ls, _, _, pixel_values_ls = load_scienceqa_data(
    processor=processor,
    train_num=4
)

print(f"Loaded {len(idx2ls)} samples\n")

# Test with first sample
idx = 0
print(f"=== Testing sample {idx} ===")
print(f"idx2ls[{idx}] type: {type(idx2ls[idx])}")
print(f"idx2ls[{idx}] length: {len(idx2ls[idx])}")
print(f"idx2ls[{idx}][:20]: {idx2ls[idx][:20]}")
print(f"pixel_values_ls[{idx}] shape: {pixel_values_ls[idx].shape if pixel_values_ls[idx] is not None else None}")

# Prepare inputs
idxs2 = torch.tensor(idx2ls[idx], dtype=torch.long).unsqueeze(0).to("cuda")
pixel_vals = pixel_values_ls[idx]
if pixel_vals is not None:
    pixel_vals = pixel_vals.unsqueeze(0).to("cuda")
else:
    pixel_vals = torch.zeros((1, 3, 336, 336), device="cuda", dtype=torch.bfloat16)

print(f"\nPrepared input_ids shape: {idxs2.shape}")
print(f"Prepared pixel_values shape: {pixel_vals.shape}")
print(f"input_ids dtype: {idxs2.dtype}")
print(f"pixel_values dtype: {pixel_vals.dtype}")
print(f"Max token ID: {idxs2.max().item()}")
print(f"Min token ID: {idxs2.min().item()}")
print(f"Model vocab size: {model.config.text_config.vocab_size}")

# Test forward pass
print("\n=== Testing forward pass ===")
try:
    with torch.no_grad():
        outputs = model(input_ids=idxs2, pixel_values=pixel_vals)
    print(f"✅ Forward pass succeeded!")
    print(f"Logits shape: {outputs.logits.shape}")
except Exception as e:
    print(f"❌ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    
    # Try with processor (correct way)
    print("\n=== Retrying with processor ===")
    # Need to get original image and text
    from datasets import load_dataset
    dataset = load_dataset("derek-thomas/ScienceQA", split="train")
    filtered = [item for item in dataset if item['image'] is not None]
    
    sample = filtered[idx]
    choices_str = "\n".join([f"({chr(65+i)}) {choice}" for i, choice in enumerate(sample['choices'])])
    prompt = f"<image>\nQuestion: {sample['question']}\nOptions:\n{choices_str}\nAnswer:"
    label = f"Lecture: {sample['lecture']}\nSolution: {sample['solution']}\nAnswer: {chr(65+sample['answer'])}"
    
    inputs = processor(text=prompt, images=sample['image'], return_tensors="pt").to("cuda")
    full_inputs = processor(text=prompt + " " + label, images=sample['image'], return_tensors="pt").to("cuda")
    
    print(f"Processor input_ids shape: {inputs['input_ids'].shape}")
    print(f"Processor pixel_values shape: {inputs['pixel_values'].shape}")
    print(f"Processor full_text input_ids shape: {full_inputs['input_ids'].shape}")
    
    try:
        with torch.no_grad():
            outputs = model(**full_inputs)
        print(f"✅ Processor forward pass succeeded!")
        print(f"Logits shape: {outputs.logits.shape}")
    except Exception as e2:
        print(f"❌ Processor forward pass also failed: {e2}")
        traceback.print_exc()
