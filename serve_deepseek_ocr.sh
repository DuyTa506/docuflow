#!/usr/bin/env bash
set -euo pipefail

vllm serve "deepseek-ai/DeepSeek-OCR-2" \
  --logits-processors "vllm.model_executor.models.deepseek_ocr:NGramPerReqLogitsProcessor" \
  --api-key 05062001 \
  --gpu-memory-utilization 0.3 \
