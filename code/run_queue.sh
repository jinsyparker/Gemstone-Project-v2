#!/usr/bin/env bash
# Train every run of the final analyses that is not trained yet (seeds 43 and 44, the ablation), one after another.
# Resumable: re-running continues unfinished runs and skips finished ones.
#   bash code/run_queue.sh
set -u
cd "$(dirname "$0")/.."
for r in $(.venv/bin/python code/runs.py list --todo); do
  echo "=== $r $(date)"
  .venv/bin/python code/train.py --run "$r" --resume || echo "!!! $r failed $(date)"
done
echo "=== queue done $(date)"
