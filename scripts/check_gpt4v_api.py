#!/usr/bin/env python3
"""
快速验证 GPT-4V API 是否可用。

用法示例：
  OPENAI_API_KEY=xxx OPENAI_BASE_URL=https://sg.uiuiapi.com/v1 \
  python scripts/check_gpt4v_api.py --model gpt-4-vision-preview

也可显式传参：
  python scripts/check_gpt4v_api.py \
    --api-key xxx \
    --base-url https://sg.uiuiapi.com/v1 \
    --model gpt-4-vision-preview
"""

import argparse
import os
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检查 GPT-4V API 连通性与可用性")
    parser.add_argument("--api-key", type=str, default=os.environ.get("OPENAI_API_KEY", ""), help="API key")
    parser.add_argument("--base-url", type=str, default=os.environ.get("OPENAI_BASE_URL", os.environ.get("OPENAI_API_BASE", "https://sg.uiuiapi.com/v1")), help="API base url")
    parser.add_argument("--model", type=str, default="gpt-4-vision-preview", help="模型名")
    parser.add_argument("--timeout", type=float, default=60.0, help="HTTP 超时时间（秒）")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.api_key:
        print("[FAIL] 未提供 API key。请设置 OPENAI_API_KEY 或传 --api-key")
        return 2

    try:
        import httpx
        from openai import OpenAI as oa
    except Exception as e:
        print(f"[FAIL] 依赖缺失或导入失败: {e}")
        print("请安装: pip install openai httpx")
        return 2

    http_client = httpx.Client(base_url=args.base_url, timeout=args.timeout)
    client = oa(
        api_key=args.api_key,
        base_url=args.base_url,
        http_client=http_client,
    )

    # 1x1 PNG（红点）base64，避免依赖 PIL
    tiny_png_b64 = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8"
        "/x8AAusB9Y9f6fQAAAAASUVORK5CYII="
    )

    try:
        resp = client.chat.completions.create(
            model=args.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{tiny_png_b64}",
                                "detail": "low",
                            },
                        },
                        {
                            "type": "text",
                            "text": "请回复: API_OK。",
                        },
                    ],
                }
            ],
            max_tokens=16,
        )

        text = (resp.choices[0].message.content or "").strip()
        print("[OK] GPT-4V API 可用")
        print(f"base_url={args.base_url}")
        print(f"model={args.model}")
        print(f"response={text[:200]}")
        return 0

    except Exception as e:
        print("[FAIL] GPT-4V API 调用失败")
        print(f"base_url={args.base_url}")
        print(f"model={args.model}")
        print(f"error={type(e).__name__}: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
