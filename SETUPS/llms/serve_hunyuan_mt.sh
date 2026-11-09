#!/bin/bash
# Hunyuan-MT-7B (Tencent, BF16) qua vLLM, API tương thích OpenAI ở cổng 5013.
# Chỉ dùng để đo (Nội dung 3.3). Cần dừng model đang chiếm GPU ở :5011 trước;
# DeepSeek-OCR (vLLM :8000, ~10,8 GB) vẫn chạy song song.
#
# Tham số lấy mẫu mặc định lấy từ generation_config.json của model
# (top_k 20, top_p 0.6, repetition_penalty 1.05, temperature 0.7).
set -euo pipefail
cd "$(dirname "$0")"
KEY=$(grep -h '^OPENAI_API_KEY=' ../../.env | head -1 | cut -d= -f2-)
# Như scripts/vllm_ocr_config.sh: tránh FlashInfer JIT trên sm_120 (báo lỗi
# "FlashInfer requires GPUs with sm75 or higher" khi khởi động).
export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
exec ~/anaconda3/envs/vllm-blackwell/bin/vllm serve models/tencent_Hunyuan-MT-7B \
  --served-model-name hunyuan-mt-7b \
  --port 5013 \
  --api-key "${KEY:-changeme}" \
  --dtype bfloat16 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.58 \
  --max-num-seqs 8 \
  --trust-remote-code
