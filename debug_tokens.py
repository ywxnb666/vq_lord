"""
调试 token IDs 范围问题
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from transformers import AutoProcessor

# 加载 processor
processor_path = "/root/workspace/models/llama3-llava-next-8b"
print(f"Loading processor from {processor_path}...")
processor = AutoProcessor.from_pretrained(processor_path, trust_remote_code=True)

# 获取词汇表大小
vocab_size = len(processor.tokenizer)
print(f"Vocabulary size: {vocab_size}")
print(f"BOS token ID: {processor.tokenizer.bos_token_id}")
print(f"EOS token ID: {processor.tokenizer.eos_token_id}")
print(f"PAD token ID: {processor.tokenizer.pad_token_id}")
print(f"Image token ID: {processor.tokenizer.encode('<image>')}")

# 测试tokenization
text1 = "<image>\nQuestion: What is the capital?\nOptions:\n(A) A\n(B) B\nAnswer:"
text2 = " Solution: Test\nAnswer: A"

full_text = text1 + text2

inputs = processor(text=text1, return_tensors="pt")
print(f"\nText1 tokens: {inputs['input_ids'][0]}")
print(f"Max token ID in text1: {inputs['input_ids'][0].max().item()}")
print(f"Min token ID in text1: {inputs['input_ids'][0].min().item()}")

full_inputs = processor(text=full_text, return_tensors="pt")
print(f"\nFull text tokens: {full_inputs['input_ids'][0]}")
print(f"Max token ID in full: {full_inputs['input_ids'][0].max().item()}")
print(f"Min token ID in full: {full_inputs['input_ids'][0].min().item()}")

# 检查是否有超出范围的 token
if full_inputs['input_ids'][0].max().item() >= vocab_size:
    print(f"\n⚠️ WARNING: Token ID {full_inputs['input_ids'][0].max().item()} >= vocab_size {vocab_size}")
else:
    print(f"\n✅ All token IDs are within valid range")
