#!/usr/bin/env bash
# Shared vLLM OCR launch config for NVIDIA Blackwell (sm_120) + CUDA 12.9 wheels.
# Sourced by serve_deepseek_ocr.sh and start.sh.

# DocuFlow OCR settings (DOCUFLOW_ prefix — not vLLM built-in env vars)
export DOCUFLOW_VLLM_OCR_MODEL="${DOCUFLOW_VLLM_OCR_MODEL:-deepseek-ai/DeepSeek-OCR-2}"
export DOCUFLOW_VLLM_OCR_API_KEY="${DOCUFLOW_VLLM_OCR_API_KEY:-05062001}"

# Share GPU with llama.cpp (~14 GB). 0.40 ≈ 13 GB vLLM on a 32 GB card.
export DOCUFLOW_VLLM_GPU_MEMORY_UTIL="${DOCUFLOW_VLLM_GPU_MEMORY_UTIL:-0.40}"
export DOCUFLOW_VLLM_MAX_MODEL_LEN="${DOCUFLOW_VLLM_MAX_MODEL_LEN:-8192}"
export DOCUFLOW_VLLM_MAX_NUM_SEQS="${DOCUFLOW_VLLM_MAX_NUM_SEQS:-1}"

# Runtime hints for Blackwell / CUDA 12.9+
export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}"

# Avoid FlashInfer JIT on sm_120 when system nvcc is 12.8 (use PyTorch/vLLM fallback sampler).
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

# Build vLLM serve argument list (array). Caller: vllm serve "${VLLM_OCR_SERVE_ARGS[@]}"
vllm_ocr_build_serve_args() {
  VLLM_OCR_SERVE_ARGS=(
    "$DOCUFLOW_VLLM_OCR_MODEL"
    --dtype bfloat16
    --trust-remote-code
    --logits-processors "vllm.model_executor.models.deepseek_ocr:NGramPerReqLogitsProcessor"
    --api-key "$DOCUFLOW_VLLM_OCR_API_KEY"
    --gpu-memory-utilization "$DOCUFLOW_VLLM_GPU_MEMORY_UTIL"
    --max-model-len "$DOCUFLOW_VLLM_MAX_MODEL_LEN"
    --max-num-seqs "$DOCUFLOW_VLLM_MAX_NUM_SEQS"
    # FlashInfer CUTLASS MoE JIT needs nvcc >= 12.9 for sm_120; Triton works with cu129 wheels + nvcc 12.8.
    --moe-backend triton
  )
}
