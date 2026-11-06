#!/usr/bin/env bash
# Stop the training queue or the test pass cleanly if the Mac is discharging at or below a battery threshold, so an abrupt shutdown cannot corrupt a
# checkpoint. Training resumes from the last finished epoch with: bash code/run_queue.sh; the test pass skips finished checkpoints when re-run
#   bash code/battery_guard.sh [percent, default 10]
LIMIT=${1:-10}
while true; do
  out=$(pmset -g batt)
  pct=$(echo "$out" | grep -o '[0-9]*%' | head -1 | tr -d '%')
  if echo "$out" | grep -q discharging && [ -n "$pct" ] && [ "$pct" -le "$LIMIT" ]; then
    echo "$(date) battery ${pct}% and discharging: stopping the training queue" >> results/logs/battery_guard.log
    pkill -f run_queue.sh; pkill -f "train.py --run"; pkill -f "final_eval.py"
    exit 0
  fi
  sleep 60
done
