"""Student model backend helpers for VQ-LoRD.

This module keeps Stage2/Stage3 training logic model-agnostic.  The
loss code still consumes logits and labels exactly as before; only model
loading, multimodal preprocessing, and forward kwargs are dispatched here.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import torch
from transformers import BitsAndBytesConfig, LlavaNextForConditionalGeneration, LlavaNextProcessor

from vq_module2 import VQVisionEncoder


LLAVA_NEXT = "llava_next"
QWEN2_VL = "qwen2_vl"


def normalize_student_model_type(value: Optional[str]) -> str:
    raw = str(value or LLAVA_NEXT).strip().lower().replace("-", "_")
    aliases = {
        "llava": LLAVA_NEXT,
        "llava_next": LLAVA_NEXT,
        "llavanext": LLAVA_NEXT,
        "qwen2_vl": QWEN2_VL,
        "qwen2vl": QWEN2_VL,
    }
    if raw not in aliases:
        raise ValueError(
            f"不支持的 student_model_type={value!r}，"
            f"当前支持: {LLAVA_NEXT}, {QWEN2_VL}"
        )
    return aliases[raw]


def is_qwen_backend(student_model_type: Optional[str]) -> bool:
    return normalize_student_model_type(student_model_type) == QWEN2_VL


def _resolve_torch_dtype(dtype_name: str) -> torch.dtype:
    dtype_name = str(dtype_name).lower()
    mapping = {
        "auto": torch.bfloat16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if dtype_name not in mapping:
        raise ValueError(f"不支持的 model_dtype={dtype_name}")
    return mapping[dtype_name]


def _assert_readable_model_file(model_path: str, rel_path: str) -> None:
    path = os.path.join(model_path, rel_path)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Qwen 学生模型文件不可读: {path}。"
            "当前目录可能是断开的 symlink 快照，请先修复模型文件。"
        )


def _load_qwen_dependencies():
    try:
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    except Exception as exc:
        raise RuntimeError(
            "当前环境无法加载 Qwen2-VL 学生模型。"
            "需要 transformers 版本支持 Qwen2VLForConditionalGeneration。"
        ) from exc

    return AutoProcessor, Qwen2VLForConditionalGeneration


def load_student_model_and_processor(args):
    backend = normalize_student_model_type(getattr(args, "student_model_type", LLAVA_NEXT))
    if backend == LLAVA_NEXT:
        return _load_llava_next_model_and_processor(args)
    return _load_qwen2_vl_model_and_processor(args)


def _load_llava_next_model_and_processor(args):
    if getattr(args, "use_4bit", 0):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
        device_map = {"": args.local_rank} if getattr(args, "distributed", False) else "auto"
        model = LlavaNextForConditionalGeneration.from_pretrained(
            args.model_path,
            device_map=device_map,
            quantization_config=quantization_config,
            trust_remote_code=True,
        )
    else:
        model_dtype = _resolve_torch_dtype(args.model_dtype)
        model = LlavaNextForConditionalGeneration.from_pretrained(
            args.model_path,
            device_map=None,
            low_cpu_mem_usage=True,
            torch_dtype=model_dtype,
            trust_remote_code=True,
        )
        model.to(args.device)

    processor = LlavaNextProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    tok_img_id = processor.tokenizer.convert_tokens_to_ids("<image>")
    cfg_img_id = getattr(model.config, "image_token_index", None)
    if cfg_img_id != tok_img_id:
        print(f"[Info] 对齐 image_token_index: config={cfg_img_id} -> tokenizer={tok_img_id}")
        model.config.image_token_index = int(tok_img_id)

    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.student_model_type = LLAVA_NEXT
    return model, processor


def _load_qwen2_vl_model_and_processor(args):
    _assert_readable_model_file(args.model_path, "config.json")
    _assert_readable_model_file(args.model_path, "preprocessor_config.json")
    AutoProcessor, QwenModel = _load_qwen_dependencies()

    if getattr(args, "use_4bit", 0):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
        device_map = {"": args.local_rank} if getattr(args, "distributed", False) else "auto"
        model = QwenModel.from_pretrained(
            args.model_path,
            device_map=device_map,
            quantization_config=quantization_config,
            trust_remote_code=True,
        )
    else:
        model_dtype = _resolve_torch_dtype(args.model_dtype)
        model = QwenModel.from_pretrained(
            args.model_path,
            device_map=None,
            low_cpu_mem_usage=True,
            torch_dtype=model_dtype,
            trust_remote_code=True,
        )
        model.to(args.device)

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.student_model_type = QWEN2_VL
    return model, processor


def build_chat_messages(instruction_text: str, target_text: Optional[str] = None, image: Any = None):
    image_item = {"type": "image"}
    if image is not None:
        image_item["image"] = image
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": instruction_text},
                image_item,
            ],
        }
    ]
    if target_text is not None:
        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": target_text}],
            }
        )
    return messages


def encode_multimodal(
    processor,
    backend: str,
    instruction_text: str,
    image: Any,
    target_text: Optional[str] = None,
    add_generation_prompt: bool = False,
):
    backend = normalize_student_model_type(backend)
    if backend == LLAVA_NEXT:
        messages = build_chat_messages(instruction_text, target_text=target_text, image=None)
        text = processor.apply_chat_template(
            messages,
            add_generation_prompt=bool(add_generation_prompt),
        )
        return processor(
            text=text,
            images=image,
            return_tensors="pt",
            padding="longest",
            truncation=False,
        )

    if image is None:
        raise RuntimeError("Qwen2-VL 后端要求 image 不为空。")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": instruction_text},
            ],
        }
    ]
    if target_text is not None:
        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": target_text}],
            }
        )
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=bool(add_generation_prompt),
    )
    outputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
        padding=True,
    )
    if "pixel_values" not in outputs or "image_grid_thw" not in outputs:
        raise RuntimeError("Qwen2-VL processor 输出缺少 pixel_values 或 image_grid_thw。")
    return outputs


def get_image_token_id(model, processor=None, backend: Optional[str] = None) -> int:
    backend = normalize_student_model_type(
        backend or getattr(getattr(model, "config", None), "student_model_type", LLAVA_NEXT)
    )
    cfg = getattr(model, "config", None)
    if backend == LLAVA_NEXT:
        val = getattr(cfg, "image_token_index", None)
        if val is None:
            raise RuntimeError("model.config 中未找到 image_token_index，请检查 LLaVA 模型加载")
        return int(val)

    val = getattr(cfg, "image_token_id", None)
    if val is not None:
        return int(val)
    if processor is not None:
        tok_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        if tok_id is not None and int(tok_id) >= 0:
            return int(tok_id)
    raise RuntimeError("无法确定 Qwen2-VL image token id，请检查 config 和 tokenizer 配置")


def _get_vq_host_model(model):
    get_base_model = getattr(model, "get_base_model", None)
    if callable(get_base_model):
        base_model = get_base_model()
        if base_model is not None:
            return base_model
    return model


def _attach_vq_to_vision_module(model, attr_name: str, args):
    host_model = _get_vq_host_model(model)
    original_module = getattr(host_model, attr_name, None)
    if original_module is None:
        raise RuntimeError(f"学生模型缺少视觉模块 {attr_name}，无法挂载 VQ。")
    vq_vision_encoder = VQVisionEncoder(
        vision_tower=original_module,
        num_embeddings=args.vq_codebook_size,
        commitment_cost=args.vq_commitment_cost,
        legacy=bool(args.vq_legacy_loss),
        dead_code_threshold=args.vq_dead_code_threshold,
        usage_decay=args.vq_usage_decay,
        dead_code_reset_interval=args.vq_dead_code_reset_interval,
        freeze_vision_tower=bool(args.freeze_vision_tower),
    )
    vision_device = next(original_module.parameters()).device
    vq_vision_encoder.to(vision_device)
    host_model.vq_vision_encoder = vq_vision_encoder
    setattr(host_model, attr_name, vq_vision_encoder)
    host_model._vq_loss_container = vq_vision_encoder.vq_cache
    host_model._vq_hook_handle = None
    if host_model is not model:
        model.vq_vision_encoder = vq_vision_encoder
        model._vq_loss_container = vq_vision_encoder.vq_cache
        model._vq_hook_handle = None
    return model


def add_vq_to_student_model(model, args):
    backend = normalize_student_model_type(getattr(args, "student_model_type", LLAVA_NEXT))
    if backend == LLAVA_NEXT:
        return _attach_vq_to_vision_module(model, "vision_tower", args)
    host_model = _get_vq_host_model(model)
    if not hasattr(host_model, "visual"):
        raise RuntimeError("Qwen2-VL 模型缺少 visual 模块，无法挂载 VQ。")
    return _attach_vq_to_vision_module(model, "visual", args)


def lora_target_modules(student_model_type: Optional[str]):
    backend = normalize_student_model_type(student_model_type)
    if backend in (LLAVA_NEXT, QWEN2_VL):
        return ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    raise AssertionError(f"unreachable backend={backend}")


def is_projector_param(name: str, student_model_type: Optional[str]) -> bool:
    name_l = str(name).lower()
    patterns = ["projector", "multi_modal_projector"]
    if is_qwen_backend(student_model_type):
        patterns.extend(["merger", "visual.merger"])
    return any(pat in name_l for pat in patterns)


def is_vision_param(name: str, student_model_type: Optional[str]) -> bool:
    name_l = str(name).lower()
    if is_qwen_backend(student_model_type):
        return "visual" in name_l or "vision" in name_l
    return "vision" in name_l


def _move_optional_tensor(kwargs: Dict[str, Any], key: str, tensor: Optional[torch.Tensor], device: str):
    if isinstance(tensor, torch.Tensor):
        kwargs[key] = tensor.to(device)


def student_forward(
    model,
    args,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    pixel_values: torch.Tensor,
    image_sizes: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
    image_grid_thw: Optional[torch.Tensor] = None,
    use_cache: Optional[bool] = None,
):
    backend = normalize_student_model_type(getattr(args, "student_model_type", LLAVA_NEXT))
    kwargs: Dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values,
    }
    if labels is not None:
        kwargs["labels"] = labels
    if use_cache is not None:
        kwargs["use_cache"] = use_cache
    if backend == LLAVA_NEXT:
        kwargs["image_sizes"] = image_sizes
    else:
        if not isinstance(image_grid_thw, torch.Tensor):
            raise RuntimeError("Qwen2-VL 前向缺少 image_grid_thw。")
        image_grid_thw = image_grid_thw.to(input_ids.device)
        kwargs["image_grid_thw"] = image_grid_thw
    return model(**kwargs)


def student_generate(
    model,
    args,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    pixel_values: torch.Tensor,
    image_sizes: Optional[torch.Tensor] = None,
    image_grid_thw: Optional[torch.Tensor] = None,
    **kwargs,
):
    backend = normalize_student_model_type(getattr(args, "student_model_type", LLAVA_NEXT))
    gen_kwargs: Dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values,
        **kwargs,
    }
    if backend == LLAVA_NEXT:
        gen_kwargs["image_sizes"] = image_sizes
    else:
        if not isinstance(image_grid_thw, torch.Tensor):
            raise RuntimeError("Qwen2-VL generate 缺少 image_grid_thw。")
        image_grid_thw = image_grid_thw.to(input_ids.device)
        gen_kwargs["image_grid_thw"] = image_grid_thw
    return model.generate(**gen_kwargs)
