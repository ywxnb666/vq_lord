from __future__ import annotations

import base64
import time
from io import BytesIO
from typing import Any, Dict, Optional

import requests

from .common import TeacherClient, ensure_rgb, parse_optional_bool, resize_image_max_side


class RightCodesTeacherClient(TeacherClient):
    """OpenAI-compatible client variant using input_image/input_text content items."""

    def query_image(
        self,
        image,
        prompt: str,
        max_tokens: int,
        image_detail: Optional[str] = None,
        image_max_side: Optional[int] = None,
    ) -> str:
        image = ensure_rgb(image)
        image = resize_image_max_side(image, image_max_side or self.image_max_side)
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        request_kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_image", "image_url": f"data:image/png;base64,{image_b64}"},
                        {"type": "input_text", "text": prompt},
                    ],
                }
            ],
            "max_tokens": int(max_tokens),
        }
        last_err = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(**request_kwargs)
                return response.choices[0].message.content or ""
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                if attempt < self.max_retries - 1:
                    time.sleep(2 * attempt + 1)
        raise RuntimeError(f"Teacher API failed after {self.max_retries} attempts: {last_err}")


class GeminiNativeClient(TeacherClient):
    """Gemini generateContent client with the same query_image interface."""

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str],
        model: str,
        max_retries: int = 2,
        http_timeout: float = 120.0,
        enable_thinking: Optional[bool] = None,
        image_detail: str = "auto",
        image_max_side: Optional[int] = None,
    ) -> None:
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            model=model,
            max_retries=max_retries,
            http_timeout=http_timeout,
            enable_thinking=enable_thinking,
            image_detail=image_detail,
            image_max_side=image_max_side,
        )
        self.native_api_key = api_key
        self.native_base_url = str(base_url or "").rstrip("/")
        self.native_timeout = float(http_timeout)

    def query_image(
        self,
        image,
        prompt: str,
        max_tokens: int,
        image_detail: Optional[str] = None,
        image_max_side: Optional[int] = None,
    ) -> str:
        image = ensure_rgb(image)
        image = resize_image_max_side(image, image_max_side or self.image_max_side)
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        base = self.native_base_url
        if not base.endswith("/v1"):
            base = f"{base}/v1"
        url = f"{base}/models/{self.model}:generateContent"
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": prompt},
                        {"inline_data": {"mime_type": "image/png", "data": image_b64}},
                    ],
                }
            ],
            "generationConfig": {
                "temperature": 0,
                "maxOutputTokens": max(4096, int(max_tokens) * 4),
                "responseMimeType": "application/json",
            },
        }
        headers = {"x-goog-api-key": self.native_api_key, "Content-Type": "application/json"}
        last_err = None
        for attempt in range(self.max_retries):
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=self.native_timeout)
                if response.status_code >= 400:
                    raise RuntimeError(response.text[:1000])
                data = response.json()
                text_parts = []
                fallback_parts = []
                for candidate in data.get("candidates", []) or []:
                    for part in ((candidate.get("content") or {}).get("parts") or []):
                        text = str(part.get("text") or "")
                        if not text:
                            continue
                        if part.get("thought") is True:
                            fallback_parts.append(text)
                        else:
                            text_parts.append(text)
                return "\n".join(text_parts or fallback_parts).strip()
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                if attempt < self.max_retries - 1:
                    time.sleep(2 * attempt + 1)
        raise RuntimeError(f"Teacher API failed after {self.max_retries} attempts: {last_err}")


def build_teacher_client(provider: Dict[str, Any]) -> TeacherClient:
    provider_type = str(provider.get("type") or "openai_compatible").strip().lower()
    model = str(provider.get("model") or provider.get("victim_model") or "").strip()
    api_key = str(provider.get("api_key") or "").strip()
    base_url = str(provider.get("base_url") or provider.get("api_base") or "").strip() or None
    max_retries = int(provider.get("max_retries", 2))
    http_timeout = float(provider.get("http_timeout", 120.0))
    enable_thinking = parse_optional_bool(provider.get("enable_thinking", ""))

    if not model:
        raise ValueError("provider.model is required")
    if provider_type in {"openai", "openai_compatible", "compatible"}:
        client_cls = TeacherClient
    elif provider_type in {"rightcodes", "right_codes"}:
        client_cls = RightCodesTeacherClient
    elif provider_type in {"gemini", "gemini_native"}:
        client_cls = GeminiNativeClient
    else:
        raise ValueError(f"Unsupported provider.type: {provider_type}")

    return client_cls(
        api_key=api_key,
        base_url=base_url,
        model=model,
        max_retries=max_retries,
        http_timeout=http_timeout,
        enable_thinking=enable_thinking,
    )
