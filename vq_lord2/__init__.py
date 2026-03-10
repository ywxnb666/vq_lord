# VQ-LoRD: Vision Quantization for LoRD
# 用于窃取多模态模型图像识别能力的模块

from .vq_module2 import VectorQuantizer2, VQVisionEncoder
from .vision_lord_loss2 import VisionLoRDLoss, compute_vision_distillation_loss
from .data_collector import GPT4VDataCollector

VectorQuantizer = VectorQuantizer2

__all__ = [
    "VectorQuantizer",
    "VectorQuantizer2",
    "VQVisionEncoder", 
    "VisionLoRDLoss",
    "compute_vision_distillation_loss",
    "GPT4VDataCollector",
]
