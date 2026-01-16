#!/usr/bin/env bash
set -euo pipefail

vllm serve "deepseek-ai/DeepSeek-OCR" \
  --logits-processors "vllm.model_executor.models.deepseek_ocr:NGramPerReqLogitsProcessor" \
  --no-enable-prefix-caching \
  --mm-processor-cache-gb 0 \
  --calculate-kv-scales \
  --tensor-parallel-size 1 \
  --api-key 123 \
  --gpu-memory-utilization 0.3 \
  --max-log-len 100000 \
  --disable-log-requests \
  --chat-template-content-format string \
  --disable-custom-all-reduce \
  --served-model-name "ocr" \
  --dtype bfloat16 \
  --max-num-seqs 100 \
  --enable-chunked-prefill \
  --max-num-batched-tokens 1280 \
  --max-model-len 8192
