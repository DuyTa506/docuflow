#!/usr/bin/env python3
"""Benchmark DeepSeek-OCR-2 via local vLLM (same request shape as serving/logic.py)."""
from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from openai import AsyncOpenAI

from config.settings import settings
from core.constants import DEFAULT_OCR_PARAMS
from utils.image_utils import render_pdf_page_to_base64


async def ocr_page(
    client: AsyncOpenAI,
    *,
    pdf_path: str,
    page_num: int,
    stream: bool,
) -> dict:
    img_b64 = render_pdf_page_to_base64(
        pdf_path,
        page_num,
        target_dpi=DEFAULT_OCR_PARAMS["target_dpi"],
        max_size=DEFAULT_OCR_PARAMS["max_image_size"],
    )
    prompt = settings.ocr_prompt
    extra = {
        "skip_special_tokens": False,
        "logits_processors": [
            {
                "qualname": "vllm.model_executor.models.deepseek_ocr:NGramPerReqLogitsProcessor",
                "kwargs": {
                    "ngram_size": 20,
                    "window_size": 50,
                    "whitelist_token_ids": [128821, 128822],
                },
            }
        ],
    }
    t0 = time.perf_counter()
    ttft = None
    text = ""
    usage_tokens = 0

    if stream:
        stream_resp = await client.chat.completions.create(
            model=settings.vllm_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                        },
                    ],
                }
            ],
            max_tokens=DEFAULT_OCR_PARAMS["max_tokens"],
            temperature=DEFAULT_OCR_PARAMS["temperature"],
            extra_body=extra,
            stream=True,
        )
        async for chunk in stream_resp:
            if chunk.choices and chunk.choices[0].delta.content:
                if ttft is None:
                    ttft = time.perf_counter() - t0
                text += chunk.choices[0].delta.content
        total = time.perf_counter() - t0
        usage_tokens = max(len(text.split()), len(text) // 4)
    else:
        resp = await client.chat.completions.create(
            model=settings.vllm_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                        },
                    ],
                }
            ],
            max_tokens=DEFAULT_OCR_PARAMS["max_tokens"],
            temperature=DEFAULT_OCR_PARAMS["temperature"],
            extra_body=extra,
            stream=False,
        )
        total = time.perf_counter() - t0
        text = resp.choices[0].message.content or ""
        if resp.usage and resp.usage.completion_tokens:
            usage_tokens = resp.usage.completion_tokens
        else:
            usage_tokens = max(len(text.split()), len(text) // 4)

    decode = max(total - (ttft or total), 1e-6)
    return {
        "page": page_num,
        "chars": len(text),
        "tokens_est": usage_tokens,
        "total_s": total,
        "ttft_s": ttft if ttft is not None else total,
        "decode_s": decode,
        "tok_per_s": usage_tokens / decode if usage_tokens else 0.0,
    }


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pdf", required=True)
    p.add_argument("--pages", default="1,2,3")
    p.add_argument("--warmup", action="store_true", default=True)
    p.add_argument("--no-stream", action="store_true")
    args = p.parse_args()
    pages = [int(x) for x in args.pages.split(",")]

    client = AsyncOpenAI(
        api_key=settings.vllm_api_key,
        base_url=settings.vllm_server_url.rstrip("/"),
    )

    if args.warmup:
        print("warmup page 1…", flush=True)
        await ocr_page(client, pdf_path=args.pdf, page_num=pages[0], stream=not args.no_stream)

    results = []
    for pg in pages:
        print(f"bench page {pg}…", flush=True)
        results.append(
            await ocr_page(
                client, pdf_path=args.pdf, page_num=pg, stream=not args.no_stream
            )
        )

    print("\n=== OCR benchmark ===")
    print(f"pdf: {args.pdf}")
    print(f"model: {settings.vllm_model}")
    print(f"max_image_size: {DEFAULT_OCR_PARAMS['max_image_size']}")
    for r in results:
        print(
            f"  page {r['page']}: total={r['total_s']:.2f}s ttft={r['ttft_s']:.2f}s "
            f"decode={r['decode_s']:.2f}s ~{r['tok_per_s']:.1f} tok/s chars={r['chars']}"
        )
    totals = [r["total_s"] for r in results]
    ttfts = [r["ttft_s"] for r in results]
    tps = [r["tok_per_s"] for r in results if r["tok_per_s"]]
    print(
        f"avg total={statistics.mean(totals):.2f}s "
        f"avg ttft={statistics.mean(ttfts):.2f}s "
        f"avg tok/s={statistics.mean(tps):.1f}" if tps else ""
    )


if __name__ == "__main__":
    asyncio.run(main())
