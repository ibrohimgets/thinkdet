#!/usr/bin/env bash
set -eo pipefail

ROOT="/home/iibrohimm/project/next_step"
LOG_DIR="$ROOT/thinkdet/logs"
OUT_ROOT="$ROOT/thinkdet/checkpoints/stage2"
mkdir -p "$LOG_DIR" "$OUT_ROOT"

WORLD_SIZE="${WORLD_SIZE:-8}"
MASTER_PORT="${MASTER_PORT:-29690}"
EPOCHS="${EPOCHS:-4}"
LAYER_SETUP="${LAYER_SETUP:-layer9}"
GATE_VALUE="${GATE_VALUE:-0.25}"
KD_WEIGHT="${KD_WEIGHT:-0.10}"
LR_HEADS="${LR_HEADS:-5e-5}"
RUN_TAG="${RUN_TAG:-stage2_heads_kd_${LAYER_SETUP}_g${GATE_VALUE}_kd${KD_WEIGHT}_e${EPOCHS}_$(date +%Y%m%d_%H%M%S)}"

STAGE1_CKPT="${STAGE1_CKPT:-$ROOT/thinkdet/checkpoints/stage1/layer9_e12_fix/thinkdet_stage1_epoch12.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-$OUT_ROOT/heads_kd/${LAYER_SETUP}_g${GATE_VALUE}_kd${KD_WEIGHT}_e${EPOCHS}}"

LOG_PATH="$LOG_DIR/${RUN_TAG}.log"
PID_PATH="$LOG_DIR/${RUN_TAG}.pid"

source /home/iibrohimm/miniconda3/etc/profile.d/conda.sh
conda activate open_led

CMD=(torchrun --nproc_per_node="$WORLD_SIZE" --master_port="$MASTER_PORT"
  "$ROOT/thinkdet/scripts/training/train_stage2_heads_kd.py"
  --stage1_ckpt "$STAGE1_CKPT"
  --layer_setup "$LAYER_SETUP"
  --epochs "$EPOCHS"
  --gate_value "$GATE_VALUE"
  --kd_weight "$KD_WEIGHT"
  --lr_heads "$LR_HEADS"
  --output_dir "$OUTPUT_DIR"
  --freeze_adapters
)

echo "[launch] world_size=$WORLD_SIZE"
echo "[launch] stage1_ckpt=$STAGE1_CKPT"
echo "[launch] output_dir=$OUTPUT_DIR"
echo "[launch] log=$LOG_PATH"

setsid nohup "${CMD[@]}" > "$LOG_PATH" 2>&1 < /dev/null &
echo $! > "$PID_PATH"
echo "[launch] pid=$(cat "$PID_PATH")"
