#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/iibrohimm/project/next_step"
PYTHON_BIN="/home/iibrohimm/miniconda3/envs/open_led/bin/python"
RESULTS_DIR="$ROOT/thinkdet/results/layer_ablation_dev320"
LOG_DIR="$ROOT/thinkdet/logs/layer_ablation_dev320"

mkdir -p "$RESULTS_DIR" "$LOG_DIR"

GROUP_SPECS=(
  "0:1 2 3 4:layers_01_04"
  "2:5 6 7 8:layers_05_08"
  "3:9 10 11 12:layers_09_12"
  "4:13 14 15 16:layers_13_16"
  "5:17 18 19 20:layers_17_20"
  "6:21 22 23 24:layers_21_24"
)

run_group() {
  local gpu="$1"
  local layers="$2"
  local tag="$3"
  local out="$RESULTS_DIR/${tag}.json"
  local log="$LOG_DIR/${tag}.log"

  echo "[launch] gpu=${gpu} layers=${layers} output=${out}"
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES="$gpu" \
    "$PYTHON_BIN" -u "$ROOT/thinkdet/scripts/eval/eval_affordance_dev_layer_sweep.py" \
      --layers $layers \
      --split dev \
      --top_k 5 \
      --device cuda:0 \
      --output "$out" \
      >"$log" 2>&1
}

pids=()
tags=()
for spec in "${GROUP_SPECS[@]}"; do
  IFS=":" read -r gpu layers tag <<< "$spec"
  run_group "$gpu" "$layers" "$tag" &
  pids+=("$!")
  tags+=("$tag")
done

fail=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    echo "[error] ${tags[$i]} failed; see $LOG_DIR/${tags[$i]}.log" >&2
    fail=1
  fi
done

if [[ "$fail" != "0" ]]; then
  exit 1
fi

echo "[done] outputs in $RESULTS_DIR"
