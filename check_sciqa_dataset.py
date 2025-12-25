"""
检查 ScienceQA 数据集结构
"""
from datasets import load_dataset

print("Loading ScienceQA dataset...")
dataset = load_dataset("derek-thomas/ScienceQA", split="train")

print(f"Total samples: {len(dataset)}")
print(f"\nDataset features: {dataset.features}")

# 检查前几个样本
for i in range(min(10, len(dataset))):
    item = dataset[i]
    has_image = item.get("image") is not None
    print(f"\nSample {i}:")
    print(f"  Has image: {has_image}")
    print(f"  Question: {item.get('question', '')[:50]}...")
    print(f"  Choices: {item.get('choices', [])}")
    print(f"  Answer: {item.get('answer')}")
    
# 统计有图像的样本数量
image_count = sum(1 for item in dataset if item.get("image") is not None)
print(f"\nTotal samples with images: {image_count}/{len(dataset)} ({100*image_count/len(dataset):.1f}%)")
