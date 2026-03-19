#!/usr/bin/env bash
set -eo pipefail

ROOT="/home/iibrohimm/project/next_step"
LOG_DIR="$ROOT/thinkdet/logs"
LOG_FILE="$LOG_DIR/refcoco_layer9_e2_8gpu.log"
PID_FILE="$LOG_DIR/refcoco_layer9_e2_8gpu.pid"

source /home/iibrohimm/miniconda3/etc/profile.d/conda.sh
conda activate open_led

mkdir -p "$LOG_DIR" "$ROOT/thinkdet/checkpoints/refcoco"

# Detached run (survives terminal disconnects).
setsid nohup torchrun \
  --nproc_per_node=8 \
  --master_port=29548 \
  "$ROOT/thinkdet/scripts/training/train_refcoco.py" \
  --layer_setup layer9 \
  --epochs 2 \
  --adapters_only \
  --per_gpu_batch 2 \
  --coco_checkpoint "$ROOT/thinkdet/checkpoints/stage1/layer9_e2/thinkdet_stage1_epoch2.pth" \
  > "$LOG_FILE" 2>&1 < /dev/null &

echo "$!" > "$PID_FILE"
echo "pid=$(cat "$PID_FILE")"
echo "log=$LOG_FILE"
