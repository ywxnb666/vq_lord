import json
import os
import types

import torch
from PIL import Image

from vq_lord3.sciqa_process2 import load_model_and_processor


MODEL_PATH = "/root/autodl-tmp/models/llama3-llava-next-8b-hf"
ADAPTER_PATH = "/root/workspace/vq_lord/vq_lord_ckpts/stage3_sub1_period7"
VQ_CODEBOOK_PATH = "/root/workspace/vq_lord/vq_lord_ckpts/stage3_sub1_period7/vq_codebook.pt"


def tensor_stats(a: torch.Tensor, b: torch.Tensor) -> dict:
    a = a.detach().float().cpu()
    b = b.detach().float().cpu()
    if a.shape != b.shape:
        return {
            "shape_a": list(a.shape),
            "shape_b": list(b.shape),
            "same_shape": False,
        }
    diff = (a - b).abs()
    return {
        "shape": list(a.shape),
        "same_shape": True,
        "max_abs_diff": float(diff.max().item()),
        "mean_abs_diff": float(diff.mean().item()),
        "allclose_1e-5_1e-4": bool(torch.allclose(a, b, atol=1e-5, rtol=1e-4)),
        "allclose_1e-4_1e-3": bool(torch.allclose(a, b, atol=1e-4, rtol=1e-3)),
    }


def maybe_strip_cls(x: torch.Tensor) -> torch.Tensor:
    if x.dim() >= 2 and x.shape[1] > 1:
        return x[:, 1:]
    return x


def main():
    captured = {}

    model, processor, load_info = load_model_and_processor(
        MODEL_PATH,
        ADAPTER_PATH,
        use_4bit=0,
        use_vq=1,
        vq_codebook_size=1024,
        freeze_vision_tower=0,
        vq_codebook_path=VQ_CODEBOOK_PATH,
    )
    base_model = model.get_base_model() if hasattr(model, "get_base_model") else model
    vision_tower = base_model.vision_tower
    projector = base_model.multi_modal_projector
    vision_feature_layer = int(getattr(base_model.config, "vision_feature_layer", -2))

    orig_forward = vision_tower.forward

    def wrapped_forward(self, pixel_values, return_vq_logits=True, return_details=False, *args, **kwargs):
        tower_output, vision_features = self.encode_features(pixel_values, *args, **kwargs)
        self._align_vq_modules(vision_features)
        vision_features_compute, original_dtype = self._prepare_vq_input(vision_features)
        pre_quant_features = self.pre_quant(vision_features_compute)
        quantized, indices, vq_loss, logits = self.quantize_features(
            pre_quant_features,
            return_vq_logits=return_vq_logits,
        )
        reconstructed_features = self.post_quant(quantized)
        reconstructed_output = reconstructed_features.to(dtype=original_dtype)

        captured["pre_quant_features"] = pre_quant_features.detach().float().cpu()
        captured["quantized_features"] = quantized.detach().float().cpu()
        captured["reconstructed_output"] = reconstructed_output.detach().float().cpu()
        captured["target_features"] = vision_features.detach().float().cpu()
        captured["vq_indices"] = indices.detach().cpu() if isinstance(indices, torch.Tensor) else None

        replaced = self._replace_hidden_states(tower_output, reconstructed_output)
        if hasattr(replaced, "last_hidden_state"):
            captured["last_hidden_state_after_replace"] = replaced.last_hidden_state.detach().float().cpu()
        if hasattr(replaced, "hidden_states") and replaced.hidden_states is not None:
            try:
                captured["selected_hidden_state_after_replace"] = (
                    replaced.hidden_states[vision_feature_layer].detach().float().cpu()
                )
            except Exception as exc:
                captured["selected_hidden_state_error"] = str(exc)

        if return_details:
            return reconstructed_output, indices, vq_loss, logits
        return replaced

    vision_tower.forward = types.MethodType(wrapped_forward, vision_tower)

    def projector_pre_hook(module, inputs):
        captured["projector_input"] = inputs[0].detach().float().cpu()

    hook_handle = projector.register_forward_pre_hook(projector_pre_hook)

    try:
        image = Image.new("RGB", (336, 336), color=(127, 127, 127))
        prompt = "<image>\nQuestion: What is shown in the image?\nOptions:\n(A) cat\n(B) dog\nAnswer:"
        inputs = processor(text=prompt, images=image, return_tensors="pt")
        device = next(base_model.parameters()).device
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        pixel_values = inputs["pixel_values"].to(device)
        image_sizes = inputs.get("image_sizes")
        if image_sizes is not None:
            image_sizes = image_sizes.to(device)

        with torch.no_grad():
            _ = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_sizes=image_sizes,
                use_cache=False,
            )
    finally:
        hook_handle.remove()

    result = {
        "load_info": load_info,
        "vision_feature_layer": vision_feature_layer,
        "captured_keys": sorted(captured.keys()),
        "comparisons": {
            "projector_vs_reconstructed": tensor_stats(
                captured["projector_input"], captured["reconstructed_output"]
            ),
            "projector_vs_reconstructed_strip_cls": tensor_stats(
                captured["projector_input"], maybe_strip_cls(captured["reconstructed_output"])
            ),
            "projector_vs_selected_hidden_state": tensor_stats(
                captured["projector_input"], captured["selected_hidden_state_after_replace"]
            ),
            "projector_vs_selected_hidden_state_strip_cls": tensor_stats(
                captured["projector_input"], maybe_strip_cls(captured["selected_hidden_state_after_replace"])
            ),
            "last_hidden_state_vs_reconstructed": tensor_stats(
                captured["last_hidden_state_after_replace"], captured["reconstructed_output"]
            ),
            "selected_hidden_state_vs_reconstructed": tensor_stats(
                captured["selected_hidden_state_after_replace"], captured["reconstructed_output"]
            ),
            "projector_vs_target_features": tensor_stats(
                captured["projector_input"], captured["target_features"]
            ),
        },
    }

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
