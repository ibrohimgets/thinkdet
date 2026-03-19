#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/iibrohimm/project/next_step"
LOG_DIR="$ROOT/thinkdet/logs"
OUT_DIR="$ROOT/thinkdet/results/eval"
mkdir -p "$LOG_DIR" "$OUT_DIR"

MODE="${1:-trained}"
CHECKPOINT="${2:-$ROOT/thinkdet/checkpoints/stage1/layer9_e12_fix/thinkdet_stage1_epoch12.pth}"
WORLD_SIZE="${WORLD_SIZE:-8}"
MASTER_PORT="${MASTER_PORT:-29645}"
RUN_TAG="${RUN_TAG:-stage1_${MODE}_$(date +%Y%m%d_%H%M%S)}"

LOG_PATH="$LOG_DIR/${RUN_TAG}.log"
PID_PATH="$LOG_DIR/${RUN_TAG}.pid"
OUT_PATH="$OUT_DIR/${RUN_TAG}.json"

source /home/iibrohimm/miniconda3/etc/profile.d/conda.sh
conda activate open_led

CMD=(torchrun --nproc_per_node="$WORLD_SIZE" --master_port="$MASTER_PORT"
  "$ROOT/thinkdet/scripts/eval/eval_stage1_checkpoint.py"
  --mode "$MODE"
  --output "$OUT_PATH"
  --max_allowed_errors 0
)

if [[ "$MODE" != "baseline" ]]; then
  CMD+=(--checkpoint "$CHECKPOINT")
fi

echo "[launch] mode=$MODE world_size=$WORLD_SIZE"
echo "[launch] log=$LOG_PATH"
echo "[launch] out=$OUT_PATH"

setsid nohup "${CMD[@]}" > "$LOG_PATH" 2>&1 < /dev/null &
echo $! > "$PID_PATH"

echo "[launch] pid=$(cat "$PID_PATH")"
