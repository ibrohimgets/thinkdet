#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/iibrohimm/project/next_step"
TRAIN_SCRIPT="$ROOT/thinkdet/scripts/training/train_stage1.py"

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
OUT_ROOT="${OUT_ROOT:-$ROOT/thinkdet/checkpoints/stage1_layer_sweep}"
INCLUDE_OPTIONAL_911="${INCLUDE_OPTIONAL_911:-0}"

if [[ ! -f "$TRAIN_SCRIPT" ]]; then
  echo "[error] training script not found: $TRAIN_SCRIPT" >&2
  exit 1
fi

setups=("layer9" "layer10" "fusion_9_10")
if [[ "$INCLUDE_OPTIONAL_911" == "1" ]]; then
  setups+=("fusion_9_10_11")
fi

mkdir -p "$OUT_ROOT"

echo "[info] Running stage1 layer sweep"
echo "[info] NPROC_PER_NODE=$NPROC_PER_NODE"
echo "[info] OUT_ROOT=$OUT_ROOT"
echo "[info] setups=${setups[*]}"

timestamp="$(date +%Y%m%d_%H%M%S)"
for setup in "${setups[@]}"; do
  run_out="$OUT_ROOT/${setup}_${timestamp}"
  mkdir -p "$run_out"
  log_file="$run_out/train.log"

  echo ""
  echo "[run] setup=$setup"
  echo "[run] output=$run_out"

  torchrun --nproc_per_node="$NPROC_PER_NODE" \
    "$TRAIN_SCRIPT" \
    --layer_setup "$setup" \
    --output_dir "$run_out" \
    "$@" 2>&1 | tee "$log_file"
done

echo ""
echo "[done] Sweep finished. Outputs in: $OUT_ROOT"
