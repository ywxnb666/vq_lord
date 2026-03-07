"""
======================================================================
DATA_COLLECTOR ---

GPT-4V 数据收集模块

核心功能:
1. 收集视觉问答数据 (从 GPT-4V 获取回答)
2. 收集详细图像描述
3. 构建 VQ-LoRD 训练数据

    Author: VQ-LoRD Project
    Created: January 2026
======================================================================
"""

import os
import json
import base64
import torch
from typing import List, Dict, Optional, Tuple
from PIL import Image
from io import BytesIO
from tqdm import tqdm
from dataclasses import dataclass, asdict
import time


@dataclass
class VisualQAItem:
    """视觉问答数据项"""
    image_path: str
    question: str
    question_type: str
    teacher_answer: str
    image_id: Optional[str] = None


@dataclass
class ImageDescriptionItem:
    """图像描述数据项"""
    image_path: str
    description: str
    description_type: str  # "brief", "detailed", "structured"
    image_id: Optional[str] = None


class GPT4VDataCollector:
    """
    GPT-4V 数据收集器
    
    用于从 GPT-4V 收集视觉理解数据，构建 VQ-LoRD 训练集
    
    Args:
        api_key: OpenAI API 密钥
        model: 模型名称 (默认 gpt-4-vision-preview)
        save_dir: 数据保存目录
        max_retries: API 调用最大重试次数
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "gpt-4-vision-preview",
        save_dir: str = "./vq_lord_data",
        max_retries: int = 3,
        http_timeout: float = 120.0,
    ):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE")
        self.model = model
        self.save_dir = save_dir
        self.max_retries = max_retries
        self.http_timeout = http_timeout
        self._client = None
        self._http_client = None
        
        os.makedirs(save_dir, exist_ok=True)
        
        # 视觉问题模板
        self.visual_questions = {
            "describe_objects": "请详细描述这张图片中的所有物体，包括它们的外观特征、位置和状态。",
            "count_objects": "请数一数图片中共有多少个主要物体？请列出每种物体及其数量。",
            "spatial_relations": "请描述图片中各个物体之间的空间位置关系（上下、左右、前后等）。",
            "colors_attributes": "请描述图片中各个物体的颜色、大小、形状、材质等属性。",
            "scene_understanding": "这张图片展示的是什么场景或情境？请解释你的判断依据。",
            "text_ocr": "如果图片中有文字，请识别并列出所有文字内容。",
            "action_detection": "图片中的人物或物体正在进行什么动作或活动？",
            "emotion_expression": "如果图片中有人物，请描述他们的表情和可能的情绪状态。",
        }

    def _get_openai_client(self):
        """
        构建 OpenAI 客户端。
        支持与如下形式等价的调用：
            _http_client = httpx.Client(base_url="...")
            client = OpenAI(api_key="...", base_url="...", http_client=_http_client)
        """
        if self._client is not None:
            return self._client

        if not self.api_key:
            print("未检测到 API Key，请设置 OPENAI_API_KEY")
            return None

        try:
            from openai import OpenAI as oa
        except ImportError:
            print("请安装 openai 库: pip install openai")
            return None

        kwargs = {"api_key": self.api_key}

        if self.base_url:
            kwargs["base_url"] = self.base_url
            try:
                import httpx
                self._http_client = httpx.Client(
                    base_url=self.base_url,
                    timeout=self.http_timeout,
                )
                kwargs["http_client"] = self._http_client
            except ImportError:
                print("警告: 未安装 httpx，将不传入自定义 http_client。可执行: pip install httpx")

        self._client = oa(**kwargs)
        return self._client

    def __del__(self):
        if self._http_client is not None:
            try:
                self._http_client.close()
            except Exception:
                pass
    
    def encode_image_base64(self, image_path: str) -> str:
        """将图片编码为 base64"""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def encode_pil_image_base64(self, image: Image.Image, image_format: str = "PNG") -> str:
        """将 PIL.Image 编码为 base64"""
        if image.mode != "RGB":
            image = image.convert("RGB")
        buffer = BytesIO()
        image.save(buffer, format=image_format)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    def query_gpt4v(
        self,
        image_path: str,
        prompt: str,
        max_tokens: int = 500,
    ) -> Optional[str]:
        """
        查询 GPT-4V
        
        Args:
            image_path: 图片路径
            prompt: 提示词
            max_tokens: 最大生成 token 数
            
        Returns:
            GPT-4V 的回答，失败返回 None
        """
        try:
            client = self._get_openai_client()
            if client is None:
                return None
            
            # 编码图片
            image_data = self.encode_image_base64(image_path)
            
            # 确定图片格式
            ext = image_path.lower().split(".")[-1]
            mime_type = {
                "jpg": "image/jpeg",
                "jpeg": "image/jpeg",
                "png": "image/png",
                "gif": "image/gif",
                "webp": "image/webp",
            }.get(ext, "image/jpeg")
            
            for attempt in range(self.max_retries):
                try:
                    response = client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:{mime_type};base64,{image_data}",
                                            "detail": "high",
                                        },
                                    },
                                    {
                                        "type": "text",
                                        "text": prompt,
                                    },
                                ],
                            }
                        ],
                        max_tokens=max_tokens,
                    )
                    return response.choices[0].message.content
                    
                except Exception as e:
                    print(f"API 调用失败 (尝试 {attempt + 1}/{self.max_retries}): {e}")
                    if attempt < self.max_retries - 1:
                        time.sleep(2 ** attempt)  # 指数退避
            
            return None

        except Exception as e:
            print(f"GPT-4V 查询失败: {e}")
            return None

    def query_gpt4v_image(
        self,
        image: Image.Image,
        prompt: str,
        max_tokens: int = 500,
        image_format: str = "PNG",
    ) -> Optional[str]:
        """
        使用 PIL.Image 直接查询 GPT-4V

        Args:
            image: PIL 图像
            prompt: 提示词
            max_tokens: 最大生成 token 数
            image_format: 编码格式（默认 PNG）

        Returns:
            GPT-4V 回答文本，失败返回 None
        """
        try:
            client = self._get_openai_client()
            if client is None:
                return None
            image_data = self.encode_pil_image_base64(image, image_format=image_format)
            mime_type = "image/png" if image_format.upper() == "PNG" else "image/jpeg"

            for attempt in range(self.max_retries):
                try:
                    response = client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:{mime_type};base64,{image_data}",
                                            "detail": "high",
                                        },
                                    },
                                    {
                                        "type": "text",
                                        "text": prompt,
                                    },
                                ],
                            }
                        ],
                        max_tokens=max_tokens,
                    )
                    return response.choices[0].message.content
                except Exception as e:
                    print(f"API 调用失败 (尝试 {attempt + 1}/{self.max_retries}): {e}")
                    if attempt < self.max_retries - 1:
                        time.sleep(2 ** attempt)

            return None
        except Exception as e:
            print(f"GPT-4V 查询失败: {e}")
            return None
    
    def collect_visual_qa_data(
        self,
        image_paths: List[str],
        question_types: Optional[List[str]] = None,
        save_filename: str = "visual_qa_data.json",
    ) -> List[VisualQAItem]:
        """
        收集视觉问答数据
        
        Args:
            image_paths: 图片路径列表
            question_types: 问题类型列表 (None 表示全部)
            save_filename: 保存文件名
            
        Returns:
            收集到的 VisualQAItem 列表
        """
        question_types = question_types or list(self.visual_questions.keys())
        collected_data = []
        
        for img_path in tqdm(image_paths, desc="收集视觉问答数据"):
            for q_type in question_types:
                question = self.visual_questions.get(q_type)
                if question is None:
                    continue
                
                answer = self.query_gpt4v(img_path, question)
                
                if answer:
                    item = VisualQAItem(
                        image_path=img_path,
                        question=question,
                        question_type=q_type,
                        teacher_answer=answer,
                        image_id=os.path.basename(img_path),
                    )
                    collected_data.append(item)
        
        # 保存数据
        save_path = os.path.join(self.save_dir, save_filename)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump([asdict(item) for item in collected_data], f, ensure_ascii=False, indent=2)
        
        print(f"已收集 {len(collected_data)} 条视觉问答数据，保存至 {save_path}")
        return collected_data
    
    def collect_detailed_descriptions(
        self,
        image_paths: List[str],
        description_type: str = "detailed",
        save_filename: str = "image_descriptions.json",
    ) -> List[ImageDescriptionItem]:
        """
        收集详细图像描述
        
        Args:
            image_paths: 图片路径列表
            description_type: 描述类型 ("brief", "detailed", "structured")
            save_filename: 保存文件名
            
        Returns:
            收集到的 ImageDescriptionItem 列表
        """
        prompts = {
            "brief": "请用一句话简要描述这张图片。",
            "detailed": "请非常详细地描述这张图片的每一个细节，包括所有物体、颜色、位置、动作、场景、光线、情绪等各个方面。",
            "structured": """请按以下格式结构化描述这张图片：
1. 场景类型：
2. 主要物体（列出所有）：
3. 每个物体的属性（颜色、大小、位置）：
4. 物体间的空间关系：
5. 动作或活动：
6. 整体氛围：""",
        }
        
        prompt = prompts.get(description_type, prompts["detailed"])
        collected_data = []
        
        for img_path in tqdm(image_paths, desc="收集图像描述"):
            description = self.query_gpt4v(img_path, prompt, max_tokens=800)
            
            if description:
                item = ImageDescriptionItem(
                    image_path=img_path,
                    description=description,
                    description_type=description_type,
                    image_id=os.path.basename(img_path),
                )
                collected_data.append(item)
        
        # 保存数据
        save_path = os.path.join(self.save_dir, save_filename)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump([asdict(item) for item in collected_data], f, ensure_ascii=False, indent=2)
        
        print(f"已收集 {len(collected_data)} 条图像描述，保存至 {save_path}")
        return collected_data
    
    def load_collected_data(
        self, 
        filename: str
    ) -> List[Dict]:
        """加载已收集的数据"""
        path = os.path.join(self.save_dir, filename)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return []


class VQLORDDataset(torch.utils.data.Dataset):
    """
    VQ-LoRD 训练数据集
    
    整合视觉问答数据和图像描述，用于训练
    """
    
    def __init__(
        self,
        visual_qa_data: List[VisualQAItem],
        image_descriptions: List[ImageDescriptionItem],
        processor,  # LLaVA processor
        max_length: int = 512,
    ):
        self.visual_qa_data = visual_qa_data
        self.image_descriptions = image_descriptions
        self.processor = processor
        self.max_length = max_length
        
        # 合并所有数据
        self.all_data = []
        
        for item in visual_qa_data:
            self.all_data.append({
                "type": "qa",
                "image_path": item.image_path,
                "instruction": item.question,
                "response": item.teacher_answer,
            })
        
        for item in image_descriptions:
            self.all_data.append({
                "type": "description",
                "image_path": item.image_path,
                "instruction": "请详细描述这张图片。",
                "response": item.description,
            })
    
    def __len__(self):
        return len(self.all_data)
    
    def __getitem__(self, idx):
        item = self.all_data[idx]
        
        # 加载图片
        image = Image.open(item["image_path"]).convert("RGB")
        image_sizes_default = torch.tensor([image.height, image.width], dtype=torch.long)
        
        # 构建对话格式
        # 使用 processor 处理 (拼接指令与回答，保证 labels 对齐)
        instruction = item["instruction"]
        if "<image>" not in instruction:
            instruction = f"<image>\n{instruction}"
        full_text = f"{instruction}\n{item['response']}"
        inputs = self.processor(
            text=full_text,
            images=image,
            return_tensors="pt",
            padding="longest",
            truncation=False,
        )
        prompt_inputs = self.processor(
            text=instruction,
            images=image,
            return_tensors="pt",
            padding="longest",
            truncation=False,
        )
        # 直接使用原图尺寸，避免 processor 返回形状差异触发下游 patch 计算异常
        image_sizes = image_sizes_default

        labels = inputs["input_ids"].clone()
        pad_id = self.processor.tokenizer.pad_token_id
        prompt_len = prompt_inputs["input_ids"].shape[1]
        prompt_len = min(prompt_len, labels.shape[1])
        labels[:, :prompt_len] = -100
        labels = labels.masked_fill(labels == pad_id, -100)
        
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "labels": labels.squeeze(0),
            "image_sizes": image_sizes,
            "data_type": item["type"],
        }


def collect_scienceqa_visual_data(
    scienceqa_data: List[Dict],
    collector: GPT4VDataCollector,
    max_samples: int = 500,
) -> Tuple[List[VisualQAItem], List[ImageDescriptionItem]]:
    """
    专门针对 ScienceQA 数据集收集视觉数据
    
    Args:
        scienceqa_data: ScienceQA 数据列表
        collector: GPT4V 数据收集器
        max_samples: 最大样本数
        
    Returns:
        (视觉问答数据, 图像描述数据)
    """
    # 筛选有图像的样本
    image_samples = [s for s in scienceqa_data if s.get("image") is not None]
    image_samples = image_samples[:max_samples]
    
    image_paths = [s["image"] for s in image_samples]
    
    # 收集视觉问答数据 (只用部分问题类型以节省 API 成本)
    qa_data = collector.collect_visual_qa_data(
        image_paths,
        question_types=["describe_objects", "spatial_relations", "scene_understanding"],
        save_filename="scienceqa_visual_qa.json",
    )
    
    # 收集详细描述
    desc_data = collector.collect_detailed_descriptions(
        image_paths,
        description_type="detailed",
        save_filename="scienceqa_descriptions.json",
    )
    
    return qa_data, desc_data


# 测试代码
if __name__ == "__main__":
    print("GPT4VDataCollector 模块测试")
    print("=" * 50)
    
    # 创建收集器 (需要设置 OPENAI_API_KEY 环境变量)
    collector = GPT4VDataCollector(save_dir="./test_vq_lord_data")
    
    print("可用的视觉问题类型:")
    for q_type, question in collector.visual_questions.items():
        print(f"  {q_type}: {question[:50]}...")
    
    print("\n数据收集器初始化成功!")
    print(f"数据保存目录: {collector.save_dir}")
