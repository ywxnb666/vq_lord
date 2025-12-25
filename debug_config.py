
from transformers import AutoConfig, LlavaForConditionalGeneration

model_path = "/root/workspace/models/llama3-llava-next-8b"
try:
    config = AutoConfig.from_pretrained(model_path)
    print(f"Config class: {config.__class__.__name__}")
    print(f"Vocab size in config: {config.vocab_size}")
    
    if hasattr(config, 'text_config'):
        print(f"Text config found: {config.text_config}")
        if config.text_config:
            print(f"Text config vocab size: {config.text_config.vocab_size}")
    else:
        print("No text_config found")
        
except Exception as e:
    print(f"Error loading config: {e}")
