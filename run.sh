#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -n "${USE_LOCAL_PYTHON_PACKAGES:-}" ]]; then
  export PYTHONPATH="${SCRIPT_DIR}/.python_packages${PYTHONPATH:+:${PYTHONPATH}}"
fi

IMAGE_PATH="./example_data/images/Image 4.png"
IMAGE_STEM="$(basename "${IMAGE_PATH}")"
IMAGE_STEM="${IMAGE_STEM%.*}"
EXP_NAME="${EXP_NAME:-${IMAGE_STEM}}"
IMG_SIZE="${IMG_SIZE:-300}"
DEVICE="${DEVICE:-auto}"
PARAM_LIMITS=(0.2 0.2 0.4 0.5)

python run_llmind.py \
  --image_path "${IMAGE_PATH}" \
  --percentage 0.01 \
  --mobius_layers 1 \
  --epochs 20 \
  --lr 1e-3 \
  --device "${DEVICE}" \
  --scorer lhuman \
  --z_dim 64 \
  --hidden 512 \
  --exp_name "${EXP_NAME}" \
  --log_every 1 \
  --log_dir "./logs" \
  --json_path "./example_data/info.json" \
  --vlm_model "qwen" \
  --img_size "${IMG_SIZE}" \
  --param_limits "${PARAM_LIMITS[@]}"
