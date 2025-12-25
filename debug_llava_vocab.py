
from transformers import LlavaForConditionalGeneration, AutoProcessor
import torch

model_id = "llava-hf/llava-1.5-7b-hf"
print(f"Loading model: {model_id}")
model = LlavaForConditionalGeneration.from_pretrained(
    model_id, 
    device_map="cpu", # Load on CPU to avoid OOM for check
    torch_dtype=torch.float16
)
processor = AutoProcessor.from_pretrained(model_id)

print(f"Model vocab size: {model.config.vocab_size}")
print(f"Tokenizer vocab size: {processor.tokenizer.vocab_size}")
print(f"Tokenizer len: {len(processor.tokenizer)}")

text = "<image>\nQuestion: What is this?\nAnswer: A test."
inputs = processor(text=text, images=torch.randn(1, 3, 336, 336), return_tensors="pt")

print(f"Input IDs shape: {inputs.input_ids.shape}")
print(f"Max input ID: {inputs.input_ids.max()}")
print(f"Min input ID: {inputs.input_ids.min()}")

if inputs.input_ids.max() >= model.config.vocab_size:
    print("WARNING: Input IDs exceed model vocab size!")
else:
    print("Input IDs are within model vocab size.")

print("Pixel values shape:", inputs.pixel_values.shape)
