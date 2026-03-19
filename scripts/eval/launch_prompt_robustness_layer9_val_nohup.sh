#!/usr/bin/env bash
set -eo pipefail

ROOT="/home/iibrohimm/project/next_step"
LOG_DIR="$ROOT/thinkdet/logs"
OUT_DIR="$ROOT/thinkdet/results/eval"
LOG_FILE="$LOG_DIR/prompt_robustness_layer9_refcoco_val_8gpu.log"
PID_FILE="$LOG_DIR/prompt_robustness_layer9_refcoco_val_8gpu.pid"
OUT_JSON="$OUT_DIR/prompt_robustness_layer9_refcoco_val.json"

source /home/iibrohimm/miniconda3/etc/profile.d/conda.sh
conda activate open_led

mkdir -p "$LOG_DIR" "$OUT_DIR"

setsid nohup torchrun \
  --nproc_per_node=8 \
  --master_port=29549 \
  "$ROOT/thinkdet/scripts/eval/eval_prompt_robustness.py" \
  --thinkdet_checkpoint "$ROOT/thinkdet/checkpoints/refcoco/layer9/thinkdet_refcoco_epoch2.pth" \
  --dataset_name refcoco \
  --split val \
  --max_samples 0 \
  --output "$OUT_JSON" \
  > "$LOG_FILE" 2>&1 < /dev/null &

echo "$!" > "$PID_FILE"
echo "pid=$(cat "$PID_FILE")"
echo "log=$LOG_FILE"
echo "out=$OUT_JSON"
