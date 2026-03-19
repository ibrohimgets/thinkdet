#!/usr/bin/env bash
set -eo pipefail

ROOT="/home/iibrohimm/project/next_step"
LOG_DIR="$ROOT/thinkdet/logs"
CKPT_DIR="$ROOT/thinkdet/checkpoints/stage1/layer9_e12_fix"
LOG_FILE="$LOG_DIR/stage1_layer9_e12_fix_8gpu.log"
PID_FILE="$LOG_DIR/stage1_layer9_e12_fix_8gpu.pid"

source /home/iibrohimm/miniconda3/etc/profile.d/conda.sh
conda activate open_led

mkdir -p "$LOG_DIR" "$CKPT_DIR"

# Detached run (survives terminal disconnects).
setsid nohup torchrun \
  --nproc_per_node=8 \
  --master_port=29549 \
  "$ROOT/thinkdet/scripts/training/train_stage1.py" \
  --layer_setup layer9 \
  --epochs 12 \
  --disable_uncertainty \
  --gate_warmup_steps 500 \
  --gate_warmup_value 1.0 \
  --gate_unfreeze_value 0.7 \
  --output_dir "$CKPT_DIR" \
  > "$LOG_FILE" 2>&1 < /dev/null &

echo "$!" > "$PID_FILE"
echo "pid=$(cat "$PID_FILE")"
echo "log=$LOG_FILE"
