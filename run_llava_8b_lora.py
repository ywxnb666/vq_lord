import argparse

from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from peft import PeftModel
import torch
from PIL import Image
import requests

DEFAULT_BASE_MODEL_PATH = "/root/autodl-tmp/models/llama3-llava-next-8b-hf"
DEFAULT_ADAPTER_PATH = "/root/workspace/align_vq/vq_lord_ckpts/stage3_lord_final"
DEFAULT_IMAGE_URL = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"


def parse_args():
    parser = argparse.ArgumentParser(description="Run LLaVA-Next with LoRA adapter")
    parser.add_argument("--base_model_path", type=str, default=DEFAULT_BASE_MODEL_PATH)
    parser.add_argument("--adapter_path", type=str, default=DEFAULT_ADAPTER_PATH)
    parser.add_argument("--image_url", type=str, default=DEFAULT_IMAGE_URL)
    parser.add_argument("--max_new_tokens", type=int, default=100)
    return parser.parse_args()


def main():
    args = parse_args()

    # 1) 加载基础模型 + processor
    processor = LlavaNextProcessor.from_pretrained(args.base_model_path, use_fast=False)
    base_model = LlavaNextForConditionalGeneration.from_pretrained(
        args.base_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    # 2) 挂载LoRA（微调后模型）
    model = PeftModel.from_pretrained(base_model, args.adapter_path)
    model.eval()

    # 3) 准备图像 + 对话模板
    image = Image.open(requests.get(args.image_url, stream=True).raw).convert("RGB")

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is shown in this image?"},
                {"type": "image"},
            ],
        },
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)

    # 4) 生成
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )

    print(processor.decode(output[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
