#!/usr/bin/env bash
set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vllm-blackwell

vllm serve "deepseek-ai/DeepSeek-OCR-2" \
  --disable-custom-all-reduce \
  --enforce-eager \
  --logits-processors "vllm.model_executor.models.deepseek_ocr:NGramPerReqLogitsProcessor" \
  --api-key 05062001 \
  --gpu-memory-utilization 0.4 \
  --max-model-len 8192