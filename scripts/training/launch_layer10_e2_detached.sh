#!/usr/bin/env bash
set -eo pipefail

ROOT="/home/iibrohimm/project/next_step"
LOG_DIR="$ROOT/thinkdet/logs"
CKPT_DIR="$ROOT/thinkdet/checkpoints/stage1/layer10_e2"
LOG_FILE="$LOG_DIR/stage1_layer10_e2_8gpu.log"
PID_FILE="$LOG_DIR/stage1_layer10_e2_8gpu.pid"

source /home/iibrohimm/miniconda3/etc/profile.d/conda.sh
conda activate open_led

mkdir -p "$LOG_DIR" "$CKPT_DIR"

# Detach from terminal/session so training survives disconnects.
setsid nohup torchrun \
  --nproc_per_node=8 \
  --master_port=29546 \
  "$ROOT/thinkdet/scripts/training/train_stage1.py" \
  --layer_setup layer10 \
  --epochs 2 \
  --output_dir "$CKPT_DIR" \
  > "$LOG_FILE" 2>&1 < /dev/null &

echo "$!" > "$PID_FILE"
echo "pid=$(cat "$PID_FILE")"
echo "log=$LOG_FILE"
