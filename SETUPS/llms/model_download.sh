#!/usr/bin/env bash
set -euo pipefail

MODELS_DIR="./models"
mkdir -p "$MODELS_DIR"

download_if_needed() {
  local url="$1"
  local dest="$2"
  local file="$3"

  mkdir -p "$dest"

  if [ -f "$dest/$file" ]; then
    echo "[SKIP] $file already exists."
  else
    echo "[DOWN] $file ..."
    curl -fL -C - --retry 5 --retry-delay 3 -o "$dest/$file" "$url"
    echo "[DONE] $file"
  fi
}

# Model đang phục vụ pipeline (service `gemma-4-26b` trong docker-compose.yml).
# Vừa VRAM còn lại cạnh vLLM OCR nên chạy trọn GPU — đo được nhanh ~3x so với
# Qwen 35B phải đẩy 14 lớp MoE xuống CPU, chất lượng ngang nhau.
echo "=========================================="
echo "Downloading Gemma 4 26B A4B (QAT q4_0) — model chính"
echo "=========================================="
download_if_needed \
  "https://huggingface.co/google/gemma-4-26B-A4B-it-qat-q4_0-gguf/resolve/main/gemma-4-26B_q4_0-it.gguf" \
  "$MODELS_DIR/google_gemma-4-26B-A4B-it-qat-q4_0-GGUF" \
  "gemma-4-26B_q4_0-it.gguf"

# Chỉ cần khi muốn đối chiếu: `docker compose --profile qwen up -d`.
# Bỏ qua bằng SKIP_QWEN=1 nếu không định so sánh (tiết kiệm 20 GB).
if [ "${SKIP_QWEN:-0}" != "1" ]; then
  echo "=========================================="
  echo "Downloading Qwen3.6-35B-A3B Q4_K_M — chỉ để đối chiếu (SKIP_QWEN=1 để bỏ)"
  echo "=========================================="
  download_if_needed \
    "https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/resolve/main/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf" \
    "$MODELS_DIR/unsloth_Qwen3.5-35B-A3B-Q4_K_M-GGUF" \
    "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
fi



echo
echo "=========================================="
echo "Download complete!"
echo "=========================================="
find "$MODELS_DIR" -maxdepth 2 -type f -name "*.gguf" -exec ls -lh {} \;