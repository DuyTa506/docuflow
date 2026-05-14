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

echo "=========================================="
echo "Downloading Qwen3.5-9B Q8_0"
echo "=========================================="
download_if_needed \
  "https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-Q8_0.gguf" \
  "$MODELS_DIR/unsloth_Qwen3.5-9B-Q8_0-GGUF" \
  "Qwen3.5-9B-Q8_0.gguf"

echo "=========================================="
echo "Downloading Qwen3.6-35B-A3B Q4_K_M"
echo "=========================================="
download_if_needed \
  "https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/resolve/main/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf" \
  "$MODELS_DIR/unsloth_Qwen3.5-35B-A3B-Q4_K_M-GGUF" \
  "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"



echo
echo "=========================================="
echo "Download complete!"
echo "=========================================="
find "$MODELS_DIR" -maxdepth 2 -type f -name "*.gguf" -exec ls -lh {} \;