"""
======================================================================
DATA_COLLECTOR ---

教师模型数据收集模块

核心功能:
1. 收集视觉问答数据 (从教师模型获取回答)
2. 收集详细图像描述
3. 构建 VQ-LoRD 训练数据

    Author: VQ-LoRD Project
    Created: January 2026
======================================================================
"""

import os
import json
import base64
import argparse
import hashlib
import tempfile
import random
import re
import subprocess
import sys
import torch
from typing import List, Dict, Optional, Tuple, Any
from PIL import Image
from io import BytesIO
from tqdm import tqdm
from dataclasses import dataclass, asdict
import time
from datasets import load_dataset


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
    教师模型数据收集器
    
    用于从教师模型收集视觉理解数据，构建 VQ-LoRD 训练集
    
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
        查询教师模型
        
        Args:
            image_path: 图片路径
            prompt: 提示词
            max_tokens: 最大生成 token 数
            
        Returns:
            教师模型回答，失败返回 None
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
                        time.sleep(2)
            
            return None

        except Exception as e:
            print(f"教师模型查询失败: {e}")
            return None

    def query_gpt4v_image(
        self,
        image: Image.Image,
        prompt: str,
        max_tokens: int = 500,
        image_format: str = "PNG",
    ) -> Optional[str]:
        """
        使用 PIL.Image 直接查询教师模型

        Args:
            image: PIL 图像
            prompt: 提示词
            max_tokens: 最大生成 token 数
            image_format: 编码格式（默认 PNG）

        Returns:
            教师模型回答文本，失败返回 None
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
                        time.sleep(2)

            return None
        except Exception as e:
            print(f"教师模型查询失败: {e}")
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


TEACHER_SCHEMA_VERSION = "v2"
TEACHER_REQUIRED_FIELDS = (
    "observed_facts_visual",
    "context_textual",
    "reasoning",
    "answer",
)


def _safe_name(text: str) -> str:
    keep = []
    for ch in str(text):
        if ch.isalnum() or ch in ("-", "_", "."):
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep)


def _strip_image_tokens(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.replace("<image>", "")
    text = text.replace("< image >", "")
    text = text.replace("<Image>", "")
    return text.strip()


def _is_null_like_text(text: Any) -> bool:
    if text is None:
        return True
    if not isinstance(text, str):
        return False
    return text.strip().lower() in {"none", "null", "n/a", "na"}


def _truncate_text_by_budget_estimate(text: str, max_tokens: int) -> str:
    text = _strip_image_tokens(text)
    if not text:
        return ""
    if max_tokens <= 0:
        return text
    words = text.split()
    if len(words) > max_tokens:
        return " ".join(words[:max_tokens]).strip()
    if len(words) <= 1 and len(text) > max_tokens:
        return text[:max_tokens].strip()
    return text


def _extract_json_payload(text: str) -> Optional[dict]:
    if not isinstance(text, str):
        return None
    cleaned = text.strip()
    if not cleaned:
        return None
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        frag = cleaned[start:end + 1]
        try:
            data = json.loads(frag)
            if isinstance(data, dict):
                return data
        except Exception:
            return None
    return None


def _normalize_match_text(text: str) -> str:
    text = str(text or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _build_canonical_context(sample: Optional[dict]) -> str:
    if not isinstance(sample, dict):
        return ""
    question = _strip_image_tokens(sample.get("question", ""))
    hint = _strip_image_tokens(sample.get("hint", ""))
    choices = sample.get("choices", []) or []
    parts = []
    if question:
        parts.append(f"Question: {question}")
    if hint:
        parts.append(f"Hint: {hint}")
    if choices:
        option_lines = []
        for idx, choice in enumerate(choices):
            option_lines.append(f"({chr(65 + idx)}) {str(choice).strip()}")
        parts.append("Options:\n" + "\n".join(option_lines))
    return "\n".join(parts).strip()


def _normalize_choice_answer(answer: str, sample: Optional[dict], max_tokens: int) -> str:
    answer = _truncate_text_by_budget_estimate(answer, max_tokens)
    if not isinstance(sample, dict):
        return answer

    choices = sample.get("choices", []) or []
    if not choices:
        return answer

    max_letter = chr(65 + min(len(choices), 26) - 1)
    letter_match = re.search(
        rf"(?:option\s*)?\(?\s*([A-{max_letter}])\s*\)?",
        answer,
        flags=re.IGNORECASE,
    )
    if letter_match:
        letter = letter_match.group(1).upper()
        idx = ord(letter) - ord("A")
        if 0 <= idx < len(choices):
            return f"({letter}) {str(choices[idx]).strip()}".strip()

    norm_answer = _normalize_match_text(answer)
    for idx, choice in enumerate(choices):
        choice_text = str(choice).strip()
        norm_choice = _normalize_match_text(choice_text)
        if not norm_choice:
            continue
        if norm_answer == norm_choice or norm_answer in norm_choice or norm_choice in norm_answer:
            letter = chr(65 + idx)
            return f"({letter}) {choice_text}".strip()

    return answer


def _has_observed_leakage(text: str) -> bool:
    text_l = str(text or "").lower()
    banned_substrings = [
        "therefore",
        "thus",
        "hence",
        "best fit",
        "best matches",
        "aligns best",
        "option ",
        "the answer",
        "this means",
        "so the",
        "corresponds to",
    ]
    return any(s in text_l for s in banned_substrings)


def _semantic_issue_flags(annotation: Optional[dict], sample: Optional[dict]) -> List[str]:
    if not isinstance(annotation, dict):
        return ["invalid_json_or_missing_fields"]

    issues: List[str] = []
    observed = str(annotation.get("observed_facts_visual", "")).strip()
    answer = str(annotation.get("answer", "")).strip()
    if _has_observed_leakage(observed):
        issues.append("observed_leakage")

    if isinstance(sample, dict) and (sample.get("choices") or []):
        choices = sample.get("choices") or []
        max_letter = chr(65 + min(len(choices), 26) - 1)
        if not re.match(rf"^\(?[A-{max_letter}]\)?(?:\s+.*)?$", answer):
            issues.append("answer_not_choice_like")
    return issues


def _normalize_struct_payload(payload: Optional[dict], budget: dict, sample: Optional[dict] = None) -> Optional[dict]:
    if not isinstance(payload, dict):
        return None

    normalized = {
        "format_version": TEACHER_SCHEMA_VERSION,
        "observed_facts_visual": payload.get("observed_facts_visual", payload.get("observed_facts", "")),
        "context_textual": payload.get("context_textual", payload.get("context", "")),
        "reasoning": payload.get("reasoning", ""),
        "answer": payload.get("answer", ""),
    }
    canonical_context = _build_canonical_context(sample)
    if canonical_context:
        normalized["context_textual"] = canonical_context
    normalized["observed_facts_visual"] = _truncate_text_by_budget_estimate(
        normalized["observed_facts_visual"], int(budget.get("teacher_observed_max_tokens", 0))
    )
    normalized["context_textual"] = _truncate_text_by_budget_estimate(
        normalized["context_textual"], int(budget.get("teacher_context_max_tokens", 0))
    )
    normalized["reasoning"] = _truncate_text_by_budget_estimate(
        normalized["reasoning"], int(budget.get("teacher_reasoning_max_tokens", 0))
    )
    normalized["answer"] = _normalize_choice_answer(
        normalized["answer"],
        sample=sample,
        max_tokens=int(budget.get("teacher_answer_max_tokens", 0)),
    )

    for field in TEACHER_REQUIRED_FIELDS:
        value = normalized.get(field)
        if not isinstance(value, str) or len(value.strip()) == 0 or _is_null_like_text(value):
            return None
    return normalized


def _build_structured_teacher_prompt(instruction: str, lang: str, extra_strict: bool = False) -> str:
    instruction = _strip_image_tokens(instruction)
    if str(lang).lower() == "zh":
        base_prompt = (
            "你是严谨的视觉科学题解答助手。\n"
            "请仅输出严格 JSON，必须且只包含以下键：\n"
            "observed_facts_visual, context_textual, reasoning, answer。\n"
            "规则：\n"
            "1) observed_facts_visual 只写图像可见证据（可含 OCR），不要写推理、不要写选项匹配、不要写 '对应某地区/某国家/某答案' 之类判断。\n"
            "2) context_textual 必须完整重述题干、提示与选项文本条件。\n"
            "3) reasoning 写基于前两者的推理，可以引用选项，但不要把推理写进 observed_facts_visual。\n"
            "4) answer 输出最终选项，优先格式 '(A) option text'。\n"
            "不要输出 markdown，不要额外字段。\n\n"
        )
        extra_prompt = (
            "额外强调：如果 observed_facts_visual 中出现 therefore/option/对应/最佳匹配 这类推理或选项判断，视为错误。\n\n"
            if extra_strict else ""
        )
        return base_prompt + extra_prompt + instruction
    base_prompt = (
        "You are a rigorous visual science QA assistant.\n"
        "Return strict JSON only with exactly keys:\n"
        "observed_facts_visual, context_textual, reasoning, answer.\n"
        "Rules:\n"
        "1) observed_facts_visual: only image-observable evidence (OCR allowed); no inference, no option matching, no 'corresponds to', no final identification beyond what is directly visible.\n"
        "2) context_textual: restate the full textual conditions from question, hint, and options.\n"
        "3) reasoning: explicit reasoning from (1)+(2); option comparison belongs here, not in observed_facts_visual.\n"
        "4) answer: final selected option, preferably in the form '(A) option text'.\n"
        "No markdown, no extra fields.\n\n"
    )
    extra_prompt = (
        "Extra reminder: if observed_facts_visual contains words like therefore, best fit, aligns best, option, corresponds to, the output is invalid.\n\n"
        if extra_strict else ""
    )
    return base_prompt + extra_prompt + instruction


def _legacy_sample_key(sample: dict) -> str:
    instruction = sample.get("instruction", "")
    response = sample.get("response", "")
    image = sample.get("image")
    size = ""
    if image is not None and hasattr(image, "size"):
        size = f"{image.size[0]}x{image.size[1]}"
    raw = f"{instruction}\n{response}\n{size}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def _sample_key(sample: dict, split: Optional[str] = None) -> str:
    source_index = sample.get("source_index")
    if source_index is not None:
        split_name = str(split or sample.get("split") or "unknown")
        return f"scienceqa::{split_name}::{int(source_index)}"
    return _legacy_sample_key(sample)


def _dump_struct_cache(
    cache_path: str,
    cache_data: Dict[str, dict],
    args: argparse.Namespace,
    budget: dict,
) -> None:
    payload = {
        "format_version": TEACHER_SCHEMA_VERSION,
        "victim_model": args.victim_model,
        "scienceqa_split": args.scienceqa_split,
        "train_num": int(args.train_num),
        "scienceqa_seed": int(args.scienceqa_seed),
        "budget": budget,
        "samples": cache_data,
    }
    cache_dir = os.path.dirname(cache_path) or "."
    os.makedirs(cache_dir, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=cache_dir, delete=False) as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        temp_path = f.name
    os.replace(temp_path, cache_path)


def _normalize_loaded_cache_keys(
    cache_data: Dict[str, dict],
    split: str,
) -> Tuple[Dict[str, dict], Dict[str, int]]:
    normalized: Dict[str, dict] = {}
    normalized_priority: Dict[str, int] = {}
    stats = {
        "stable_kept": 0,
        "legacy_migrated": 0,
        "duplicate_collisions": 0,
        "stable_preferred": 0,
        "dropped_invalid": 0,
    }
    stable_prefix = f"scienceqa::{split}::"

    for old_key, old_value in cache_data.items():
        if not isinstance(old_value, dict):
            stats["dropped_invalid"] += 1
            continue

        new_key = None
        source_priority = 0
        if isinstance(old_key, str) and old_key.startswith(stable_prefix):
            new_key = old_key
            source_priority = 2
            stats["stable_kept"] += 1
        else:
            meta = old_value.get("meta", {}) if isinstance(old_value.get("meta", {}), dict) else {}
            source_index = meta.get("source_index")
            if source_index is None:
                stats["dropped_invalid"] += 1
                continue
            new_key = f"scienceqa::{split}::{int(source_index)}"
            source_priority = 1
            stats["legacy_migrated"] += 1

        if new_key in normalized:
            stats["duplicate_collisions"] += 1
            if source_priority > normalized_priority.get(new_key, 0):
                normalized[new_key] = old_value
                normalized_priority[new_key] = source_priority
                stats["stable_preferred"] += 1
            continue
        normalized[new_key] = old_value
        normalized_priority[new_key] = source_priority

    return normalized, stats


def _load_cache_as_stable_map(cache_path: str, split: str) -> Dict[str, dict]:
    if not cache_path or not os.path.exists(cache_path):
        return {}
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        samples = payload.get("samples", {}) if isinstance(payload, dict) else {}
        stable_map, _ = _normalize_loaded_cache_keys(samples or {}, split)
        return stable_map
    except Exception as exc:
        print(f"warning: failed to load cache {cache_path}: {exc}")
        return {}


def _shard_indices(total: int, num_shards: int, shard_id: int) -> List[int]:
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError("shard_id out of range")
    return [idx for idx in range(total) if idx % num_shards == shard_id]


def _build_scienceqa_samples(scienceqa_path: str, split: str, train_num: int, seed: int) -> List[dict]:
    dataset = load_dataset(scienceqa_path, split=split)
    dataset_with_images = []
    for raw_train_index, item in enumerate(dataset):
        if item.get("image") is not None:
            dataset_with_images.append((raw_train_index, item))

    all_indices = list(range(len(dataset_with_images)))
    random.seed(seed)
    random.shuffle(all_indices)
    if int(train_num) > 0 and len(all_indices) > int(train_num):
        all_indices = all_indices[: int(train_num)]

    samples = []
    for sampled_pos, dataset_idx in enumerate(all_indices):
        raw_train_index, item = dataset_with_images[dataset_idx]
        question = item.get("question", "")
        choices = item.get("choices", [])
        choices_text = ""
        for choice_idx, choice in enumerate(choices):
            choices_text += f"({chr(65 + choice_idx)}) {choice}\n"

        answer_idx = int(item.get("answer", 0))
        answer = choices[answer_idx] if choices and answer_idx < len(choices) else ""
        answer_letter = chr(65 + answer_idx) if 0 <= answer_idx < 26 else "A"
        hint = item.get("hint", "")
        lecture = item.get("lecture", "")
        solution = item.get("solution", "")

        hint_block = f"Hint: {hint}\n" if hint else ""
        instruction = f"<image>\nQuestion: {question}\n{hint_block}Options:\n{choices_text}Answer:"
        if lecture:
            response = f"Explanation: {lecture}\nSolution: {solution}\nAnswer: {answer}"
        else:
            response = f"Solution: {solution}\nAnswer: {answer}"

        samples.append(
            {
                "sample_id": sampled_pos,
                "source_index": dataset_idx,
                "filtered_source_index": dataset_idx,
                "raw_train_index": raw_train_index,
                "split": split,
                "image": item.get("image"),
                "question": question,
                "hint": item.get("hint", ""),
                "choices": choices,
                "answer_idx": answer_idx,
                "answer_letter": answer_letter,
                "answer_text": answer,
                "instruction": instruction,
                "response": response,
            }
        )
    return samples


def _build_budget(args: argparse.Namespace) -> dict:
    return {
        "teacher_observed_max_tokens": int(args.teacher_observed_max_tokens),
        "teacher_context_max_tokens": int(args.teacher_context_max_tokens),
        "teacher_reasoning_max_tokens": int(args.teacher_reasoning_max_tokens),
        "teacher_answer_max_tokens": int(args.teacher_answer_max_tokens),
        "teacher_max_new_tokens_total": int(args.teacher_max_new_tokens_total),
    }


def _resolve_cache_path(args: argparse.Namespace) -> str:
    if args.teacher_cache_path:
        return args.teacher_cache_path
    victim_tag = _safe_name(args.victim_model)
    return os.path.join(
        args.data_dir,
        f"scienceqa_teacher_{victim_tag}_{args.scienceqa_split}_"
        f"n{args.train_num}_seed{args.scienceqa_seed}_new.json",
    )


def _resolve_shard_output_path(final_cache_path: str, shard_id: int) -> str:
    return f"{final_cache_path}.shard{int(shard_id):02d}.json"


def _resolve_shard_index_path(final_cache_path: str, shard_id: int) -> str:
    return f"{final_cache_path}.missing_idx.shard{int(shard_id):02d}.json"


def _dump_index_list(index_path: str, indices: List[int]) -> None:
    index_dir = os.path.dirname(index_path) or "."
    os.makedirs(index_dir, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=index_dir, delete=False) as f:
        json.dump({"indices": indices}, f, ensure_ascii=False)
        temp_path = f.name
    os.replace(temp_path, index_path)


def _load_index_list(index_path: str) -> List[int]:
    if not index_path:
        return []
    with open(index_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    indices = payload.get("indices", []) if isinstance(payload, dict) else []
    if not isinstance(indices, list):
        return []
    return [int(v) for v in indices]


def _build_cache_item(
    ann: dict,
    raw_response: Optional[str],
    sample: dict,
    args: argparse.Namespace,
    budget: dict,
    existing_meta: Optional[dict] = None,
) -> dict:
    meta = dict(existing_meta or {})
    meta.update(
        {
            "teacher_model": args.victim_model,
            "teacher_lang": args.teacher_lang,
            "budget": budget,
            "source_index": int(sample.get("source_index", -1)),
            "filtered_source_index": int(sample.get("filtered_source_index", sample.get("source_index", -1))),
            "raw_train_index": int(sample.get("raw_train_index", -1)),
            "sample_id": int(sample.get("sample_id", -1)),
        }
    )
    return {
        "format_version": TEACHER_SCHEMA_VERSION,
        "observed_facts_visual": ann["observed_facts_visual"],
        "context_textual": ann["context_textual"],
        "reasoning": ann["reasoning"],
        "answer": ann["answer"],
        "raw_teacher_response": _strip_image_tokens(raw_response or ""),
        "meta": meta,
    }


def _validate_existing_entry(existing: Optional[dict], budget: dict, sample: dict) -> Optional[dict]:
    ann = _normalize_struct_payload(existing, budget, sample=sample)
    if ann is None:
        return None
    if _semantic_issue_flags(ann, sample):
        return None
    return ann


def _collect_teacher_annotation(
    collector: GPT4VDataCollector,
    sample: dict,
    args: argparse.Namespace,
    budget: dict,
) -> Tuple[Optional[dict], Optional[str], List[str]]:
    image = sample.get("image")
    raw_response = None
    prompt = _build_structured_teacher_prompt(
        sample.get("instruction", ""),
        args.teacher_lang,
        extra_strict=False,
    )
    if image is not None:
        raw_response = collector.query_gpt4v_image(
            image=image,
            prompt=prompt,
            max_tokens=int(args.teacher_max_new_tokens_total),
            image_format="PNG",
        )
    parsed = _extract_json_payload(raw_response or "")
    ann = _normalize_struct_payload(parsed, budget, sample=sample)
    issues = _semantic_issue_flags(ann, sample)
    return ann, raw_response, issues


def _run_scienceqa_struct_worker(args: argparse.Namespace) -> None:
    budget = _build_budget(args)
    shard_output_path = args.shard_output_path
    if not shard_output_path:
        raise ValueError("worker task requires --shard_output_path")

    final_cache_path = _resolve_cache_path(args)
    source_cache_path = args.source_cache_path or final_cache_path
    source_cache = _load_cache_as_stable_map(source_cache_path, args.scienceqa_split)
    samples = _build_scienceqa_samples(
        scienceqa_path=args.scienceqa_path,
        split=args.scienceqa_split,
        train_num=int(args.train_num),
        seed=int(args.scienceqa_seed),
    )
    if int(args.max_samples) > 0 and len(samples) > int(args.max_samples):
        samples = samples[: int(args.max_samples)]

    if args.index_file_path:
        shard_indices = _load_index_list(args.index_file_path)
    else:
        shard_indices = _shard_indices(len(samples), int(args.num_workers), int(args.shard_id))
    shard_cache: Dict[str, dict] = {}
    ready_from_cache = 0
    recollected = 0
    new_collected = 0
    failed = 0

    collector = None
    needs_api = int(args.collect_teacher_data) == 1 and len(shard_indices) > 0
    if needs_api:
        collector = GPT4VDataCollector(
            api_key=(args.teacher_api_key if args.teacher_api_key else None),
            base_url=(args.teacher_api_base if args.teacher_api_base else None),
            model=args.victim_model,
            save_dir=args.data_dir,
            max_retries=int(args.max_retries),
        )
        if not collector.api_key:
            raise RuntimeError("collect_teacher_data=1 but missing API key")

    for rank, sample_idx in enumerate(
        tqdm(shard_indices, desc=f"worker shard {args.shard_id}/{args.num_workers}"),
        start=1,
    ):
        sample = samples[sample_idx]
        key = _sample_key(sample, split=args.scienceqa_split)
        existing = source_cache.get(key)
        ann = _validate_existing_entry(existing, budget, sample)
        if ann is not None:
            existing_meta = existing.get("meta", {}) if isinstance(existing, dict) else {}
            existing_raw = existing.get("raw_teacher_response", "") if isinstance(existing, dict) else ""
            shard_cache[key] = _build_cache_item(
                ann,
                existing_raw,
                sample,
                args,
                budget,
                existing_meta=existing_meta,
            )
            ready_from_cache += 1
        else:
            if collector is None:
                failed += 1
            else:
                ann, raw_response, issues = _collect_teacher_annotation(collector, sample, args, budget)
                if ann is None or issues:
                    failed += 1
                else:
                    shard_cache[key] = _build_cache_item(ann, raw_response, sample, args, budget)
                    if existing is not None:
                        recollected += 1
                    else:
                        new_collected += 1

        if int(args.save_every) > 0 and (rank % int(args.save_every) == 0 or rank == len(shard_indices)):
            _dump_struct_cache(shard_output_path, shard_cache, args, budget)
        if float(args.sleep_sec) > 0:
            time.sleep(float(args.sleep_sec))

    _dump_struct_cache(shard_output_path, shard_cache, args, budget)
    print(
        f"worker_done shard={args.shard_id} total={len(shard_indices)} "
        f"ready_from_cache={ready_from_cache} recollected={recollected} "
        f"new_collected={new_collected} failed={failed} output={shard_output_path}"
    )


def _merge_shard_outputs(
    args: argparse.Namespace,
    final_cache_path: str,
    budget: dict,
    shard_artifacts: List[Tuple[str, str]],
) -> Dict[str, int]:
    merged: Dict[str, dict] = _load_cache_as_stable_map(args.source_cache_path or final_cache_path, args.scienceqa_split)
    duplicate_keys = 0
    shard_paths = [artifact[0] for artifact in shard_artifacts]
    index_paths = [artifact[1] for artifact in shard_artifacts]
    for shard_path in shard_paths:
        if not os.path.exists(shard_path):
            raise RuntimeError(f"missing shard output: {shard_path}")
        shard_samples = _load_cache_as_stable_map(shard_path, args.scienceqa_split)
        for key, value in shard_samples.items():
            if key in merged:
                duplicate_keys += 1
            merged[key] = value
    _dump_struct_cache(final_cache_path, merged, args, budget)
    if int(args.keep_shards) != 1:
        for shard_path in shard_paths:
            try:
                os.remove(shard_path)
            except FileNotFoundError:
                pass
        for index_path in index_paths:
            try:
                os.remove(index_path)
            except FileNotFoundError:
                pass
    return {
        "merged_entries": len(merged),
        "duplicate_keys": duplicate_keys,
        "shard_count": len(shard_paths),
    }


def _spawn_sharded_workers(
    args: argparse.Namespace,
    worker_task: str,
    final_cache_path: str,
    assigned_indices_per_shard: Optional[List[List[int]]] = None,
) -> List[Tuple[str, str]]:
    num_workers = max(1, int(args.num_workers))
    source_cache_path = args.source_cache_path or final_cache_path
    shard_paths = [_resolve_shard_output_path(final_cache_path, shard_id) for shard_id in range(num_workers)]
    index_paths = [_resolve_shard_index_path(final_cache_path, shard_id) for shard_id in range(num_workers)]
    if assigned_indices_per_shard is None:
        assigned_indices_per_shard = [[] for _ in range(num_workers)]
    if len(assigned_indices_per_shard) != num_workers:
        raise ValueError("assigned_indices_per_shard length must match num_workers")
    procs = []
    for shard_id, shard_path in enumerate(shard_paths):
        index_path = index_paths[shard_id]
        _dump_index_list(index_path, assigned_indices_per_shard[shard_id])
        cmd = [
            sys.executable,
            os.path.abspath(__file__),
            "--task", worker_task,
            "--scienceqa_path", args.scienceqa_path,
            "--scienceqa_split", args.scienceqa_split,
            "--train_num", str(args.train_num),
            "--scienceqa_seed", str(args.scienceqa_seed),
            "--max_samples", str(args.max_samples),
            "--data_dir", args.data_dir,
            "--teacher_cache_path", final_cache_path,
            "--source_cache_path", source_cache_path,
            "--victim_model", args.victim_model,
            "--teacher_lang", args.teacher_lang,
            "--teacher_api_base", args.teacher_api_base,
            "--teacher_api_key", args.teacher_api_key,
            "--collect_teacher_data", str(args.collect_teacher_data),
            "--strict_teacher_distill", str(args.strict_teacher_distill),
            "--teacher_observed_max_tokens", str(args.teacher_observed_max_tokens),
            "--teacher_context_max_tokens", str(args.teacher_context_max_tokens),
            "--teacher_reasoning_max_tokens", str(args.teacher_reasoning_max_tokens),
            "--teacher_answer_max_tokens", str(args.teacher_answer_max_tokens),
            "--teacher_max_new_tokens_total", str(args.teacher_max_new_tokens_total),
            "--max_retries", str(args.max_retries),
            "--num_workers", str(num_workers),
            "--save_every", str(args.save_every),
            "--sleep_sec", str(args.sleep_sec),
            "--shard_id", str(shard_id),
            "--shard_output_path", shard_path,
            "--index_file_path", index_path,
            "--keep_shards", str(args.keep_shards),
        ]
        procs.append((shard_id, shard_path, index_path, subprocess.Popen(cmd)))

    failed = []
    for shard_id, shard_path, index_path, proc in procs:
        ret = proc.wait()
        if ret != 0:
            failed.append((shard_id, ret, shard_path))
    if failed:
        raise RuntimeError(f"worker shards failed: {failed}")
    return [(shard_paths[i], index_paths[i]) for i in range(num_workers)]


def _find_missing_sample_indices(
    samples: List[dict],
    cache_data: Dict[str, dict],
    budget: dict,
    split: str,
) -> List[int]:
    missing_indices: List[int] = []
    for sample_idx, sample in enumerate(samples):
        key = _sample_key(sample, split=split)
        ann = _normalize_struct_payload(cache_data.get(key), budget, sample=sample)
        issues = _semantic_issue_flags(ann, sample)
        if ann is None or issues:
            missing_indices.append(sample_idx)
    return missing_indices


def _split_missing_indices(missing_indices: List[int], num_workers: int) -> List[List[int]]:
    shards: List[List[int]] = [[] for _ in range(num_workers)]
    for rank, sample_idx in enumerate(missing_indices):
        shards[rank % num_workers].append(sample_idx)
    return shards


def _finalize_and_validate(args: argparse.Namespace, final_cache_path: str, budget: dict) -> None:
    cache_data = _load_cache_as_stable_map(final_cache_path, args.scienceqa_split)
    samples = _build_scienceqa_samples(
        scienceqa_path=args.scienceqa_path,
        split=args.scienceqa_split,
        train_num=int(args.train_num),
        seed=int(args.scienceqa_seed),
    )
    if int(args.max_samples) > 0 and len(samples) > int(args.max_samples):
        samples = samples[: int(args.max_samples)]

    missing = 0
    for sample in samples:
        key = _sample_key(sample, split=args.scienceqa_split)
        ann = _normalize_struct_payload(cache_data.get(key), budget, sample=sample)
        issues = _semantic_issue_flags(ann, sample)
        if ann is None or issues:
            missing += 1

    print(
        f"collect done: cache={final_cache_path}, final_entries={len(cache_data)}, "
        f"expected_samples={len(samples)}, missing={missing}"
    )
    if int(args.strict_teacher_distill) == 1 and missing > 0:
        raise RuntimeError(f"strict mode enabled, but still missing {missing} structured annotations")


def collect_scienceqa_struct_annotations(args: argparse.Namespace):
    budget = _build_budget(args)
    final_cache_path = _resolve_cache_path(args)
    os.makedirs(os.path.dirname(final_cache_path) or ".", exist_ok=True)
    source_cache_path = args.source_cache_path or final_cache_path
    source_cache = _load_cache_as_stable_map(source_cache_path, args.scienceqa_split)
    samples = _build_scienceqa_samples(
        scienceqa_path=args.scienceqa_path,
        split=args.scienceqa_split,
        train_num=int(args.train_num),
        seed=int(args.scienceqa_seed),
    )
    if int(args.max_samples) > 0 and len(samples) > int(args.max_samples):
        samples = samples[: int(args.max_samples)]

    missing_indices = _find_missing_sample_indices(samples, source_cache, budget, args.scienceqa_split)
    num_workers = max(1, int(args.num_workers))
    assigned_indices_per_shard = _split_missing_indices(missing_indices, num_workers)
    print(
        f"collect coordinator num_workers={num_workers} output={final_cache_path} "
        f"source_cache_entries={len(source_cache)} missing_indices={len(missing_indices)}"
    )
    if len(missing_indices) == 0:
        _dump_struct_cache(final_cache_path, source_cache, args, budget)
        print("collect coordinator: no missing indices, skip worker spawn")
        _finalize_and_validate(args, final_cache_path, budget)
        return

    shard_artifacts = _spawn_sharded_workers(
        args,
        "collect_scienceqa_struct_worker",
        final_cache_path,
        assigned_indices_per_shard=assigned_indices_per_shard,
    )
    merge_stats = _merge_shard_outputs(args, final_cache_path, budget, shard_artifacts)
    print(f"merge_stats={merge_stats}")
    _finalize_and_validate(args, final_cache_path, budget)


def repair_scienceqa_struct_cache(args: argparse.Namespace):
    budget = _build_budget(args)
    final_cache_path = _resolve_cache_path(args)
    if not os.path.exists(final_cache_path):
        raise FileNotFoundError(f"repair cache not found: {final_cache_path}")
    args.source_cache_path = final_cache_path
    source_cache = _load_cache_as_stable_map(final_cache_path, args.scienceqa_split)
    samples = _build_scienceqa_samples(
        scienceqa_path=args.scienceqa_path,
        split=args.scienceqa_split,
        train_num=int(args.train_num),
        seed=int(args.scienceqa_seed),
    )
    if int(args.max_samples) > 0 and len(samples) > int(args.max_samples):
        samples = samples[: int(args.max_samples)]
    missing_indices = _find_missing_sample_indices(samples, source_cache, budget, args.scienceqa_split)
    num_workers = max(1, int(args.num_workers))
    assigned_indices_per_shard = _split_missing_indices(missing_indices, num_workers)
    print(
        f"repair coordinator num_workers={num_workers} cache={final_cache_path} "
        f"source_cache_entries={len(source_cache)} missing_indices={len(missing_indices)}"
    )
    if len(missing_indices) == 0:
        _dump_struct_cache(final_cache_path, source_cache, args, budget)
        print("repair coordinator: no missing indices, skip worker spawn")
        _finalize_and_validate(args, final_cache_path, budget)
        return
    shard_artifacts = _spawn_sharded_workers(
        args,
        "repair_scienceqa_struct_worker",
        final_cache_path,
        assigned_indices_per_shard=assigned_indices_per_shard,
    )
    merge_stats = _merge_shard_outputs(args, final_cache_path, budget, shard_artifacts)
    print(f"repair_merge_stats={merge_stats}")
    _finalize_and_validate(args, final_cache_path, budget)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VQ-LoRD data collector")
    parser.add_argument(
        "--task",
        type=str,
        default="collect_scienceqa_struct",
        choices=[
            "collect_scienceqa_struct",
            "collect_scienceqa_struct_worker",
            "repair_scienceqa_struct_cache",
            "repair_scienceqa_struct_worker",
        ],
    )
    parser.add_argument("--scienceqa_path", type=str, required=True)
    parser.add_argument("--scienceqa_split", type=str, default="train")
    parser.add_argument("--train_num", type=int, default=0, help="0 means full split")
    parser.add_argument("--scienceqa_seed", type=int, default=20240306)
    parser.add_argument("--max_samples", type=int, default=0, help="debug cap, 0 means all")

    parser.add_argument("--data_dir", type=str, default="./vq_lord_data")
    parser.add_argument("--teacher_cache_path", type=str, default="")
    parser.add_argument("--source_cache_path", type=str, default="")
    parser.add_argument("--victim_model", type=str, default="gpt-4o")
    parser.add_argument("--teacher_lang", type=str, default="en", choices=["zh", "en"])
    parser.add_argument("--teacher_api_base", type=str, default=os.environ.get("OPENAI_API_BASE", ""))
    parser.add_argument("--teacher_api_key", type=str, default=os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument("--collect_teacher_data", type=int, default=1)
    parser.add_argument("--strict_teacher_distill", type=int, default=1)

    parser.add_argument("--teacher_observed_max_tokens", type=int, default=256)
    parser.add_argument("--teacher_context_max_tokens", type=int, default=192)
    parser.add_argument("--teacher_reasoning_max_tokens", type=int, default=256)
    parser.add_argument("--teacher_answer_max_tokens", type=int, default=64)
    parser.add_argument("--teacher_max_new_tokens_total", type=int, default=768)

    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--shard_output_path", type=str, default="")
    parser.add_argument("--index_file_path", type=str, default="")
    parser.add_argument("--keep_shards", type=int, default=0)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--sleep_sec", type=float, default=0.0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.task == "collect_scienceqa_struct":
        collect_scienceqa_struct_annotations(args)
    elif args.task == "collect_scienceqa_struct_worker":
        _run_scienceqa_struct_worker(args)
    elif args.task == "repair_scienceqa_struct_cache":
        repair_scienceqa_struct_cache(args)
    elif args.task == "repair_scienceqa_struct_worker":
        _run_scienceqa_struct_worker(args)
    else:
        raise ValueError(f"unsupported task: {args.task}")
