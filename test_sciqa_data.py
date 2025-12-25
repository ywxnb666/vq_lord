"""
测试 ScienceQA 数据加载
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from transformers import AutoProcessor
from sciqa_process import load_scienceqa_data

# 加载 processor
processor_path = "/root/workspace/models/llama3-llava-next-8b-hf"
print(f"Loading processor from {processor_path}...")
processor = AutoProcessor.from_pretrained(processor_path, trust_remote_code=True)

# 测试数据加载
print("\nTesting data loading...")
raw_data = load_scienceqa_data(
    processor,
    task_name="scienceqa",
    train_num=5,
    max_length=256,
)

if raw_data is not None:
    p_idxls, idx2ls, _, _, pixel_values_ls = raw_data
    
    print(f"\n=== Data Structure ===")
    print(f"Number of samples: {len(p_idxls)}")
    print(f"Number of labels: {len(idx2ls)}")
    print(f"Number of pixel_values: {len(pixel_values_ls)}")
    
    print(f"\n=== First Sample ===")
    print(f"Prompt tokens type: {type(p_idxls[0])}")
    print(f"Prompt tokens length: {len(p_idxls[0])}")
    print(f"Prompt tokens[:20]: {p_idxls[0][:20]}")
    
    print(f"\nLabel tokens type: {type(idx2ls[0])}")
    print(f"Label tokens length: {len(idx2ls[0])}")
    print(f"Label tokens[:20]: {idx2ls[0][:20]}")
    
    print(f"\nPixel values type: {type(pixel_values_ls[0])}")
    if pixel_values_ls[0] is not None:
        print(f"Pixel values shape: {pixel_values_ls[0].shape}")
        print(f"Pixel values dtype: {pixel_values_ls[0].dtype}")
    
    print("\n✅ Data loading test passed!")
else:
    print("\n❌ Data loading failed!")
