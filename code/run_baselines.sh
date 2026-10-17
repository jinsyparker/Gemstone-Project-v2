#!/usr/bin/env bash
# Train the five baselines sequentially with the one frozen recipe in config.yaml (resumable).
# Usage: bash code/run_baselines.sh
set -u
cd "$(dirname "$0")/.."
for m in MobileNetV2 ResNet50 DenseNet121 EfficientNetV2-S ConvNeXt-Tiny; do
  echo "=== $m $(date)"
  .venv/bin/python code/train.py --model "$m" --resume
done
echo "=== all baselines done $(date)"
