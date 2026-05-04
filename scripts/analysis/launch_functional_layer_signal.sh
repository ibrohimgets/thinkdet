#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/iibrohimm/project/next_step"
PYTHON_BIN="${PYTHON_BIN:-/home/iibrohimm/miniconda3/envs/open_led/bin/python}"
SCRIPT="$ROOT/thinkdet/scripts/analysis/functional_layer_signal.py"
SUMMARIZE="$ROOT/thinkdet/scripts/analysis/summarize_functional_layer_signal.py"

BENCHMARK="${BENCHMARK:-$ROOT/thinkdet/data/benchmarks/affordance_coco_val_heldout_v2.json}"
SPLIT="${SPLIT:-dev}"
LAYERS="${LAYERS:-}"
GPU_IDS="${GPU_IDS:-0,2,3,4,5,6}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
NEUTRAL_PROMPT="${NEUTRAL_PROMPT:-something .}"
RUN_TAG="${RUN_TAG:-functional_layer_signal_$(date +%Y%m%d_%H%M%S)}"
RESULTS_DIR="${RESULTS_DIR:-$ROOT/thinkdet/results/functional_layer_signal/$RUN_TAG}"
LOG_DIR="${LOG_DIR:-$ROOT/thinkdet/logs/functional_layer_signal/$RUN_TAG}"

IFS=',' read -r -a GPU_ARRAY <<< "$GPU_IDS"
NUM_SHARDS="${NUM_SHARDS:-${#GPU_ARRAY[@]}}"

mkdir -p "$RESULTS_DIR/shards" "$LOG_DIR"

echo "[launch] tag=$RUN_TAG"
echo "[launch] benchmark=$BENCHMARK split=$SPLIT"
echo "[launch] gpu_ids=$GPU_IDS num_shards=$NUM_SHARDS"
echo "[launch] output=$RESULTS_DIR"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

pids=()
for shard in $(seq 0 $((NUM_SHARDS - 1))); do
  gpu="${GPU_ARRAY[$((shard % ${#GPU_ARRAY[@]}))]}"
  out="$RESULTS_DIR/shards/shard_${shard}.json"
  log="$LOG_DIR/shard_${shard}.log"
  echo "[run] shard=$shard gpu=$gpu output=$out"

  args=(
    "$SCRIPT"
    --benchmark "$BENCHMARK"
    --split "$SPLIT"
    --num_shards "$NUM_SHARDS"
    --shard_index "$shard"
    --neutral_prompt "$NEUTRAL_PROMPT"
    --device cuda:0
    --output "$out"
    --log_every 10
  )
  if [[ "$MAX_SAMPLES" != "0" ]]; then
    args+=(--max_samples "$MAX_SAMPLES")
  fi
  if [[ -n "$LAYERS" ]]; then
    # shellcheck disable=SC2206
    layer_array=($LAYERS)
    args+=(--layers "${layer_array[@]}")
  fi

  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u "${args[@]}" >"$log" 2>&1 &
  pids+=("$!")
done

fail=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    echo "[error] shard $i failed; see $LOG_DIR/shard_${i}.log" >&2
    fail=1
  fi
done
if [[ "$fail" != "0" ]]; then
  exit 1
fi

"$PYTHON_BIN" "$SUMMARIZE" "$RESULTS_DIR/shards" --output "$RESULTS_DIR/functional_layer_signal.json"

echo "[done] results=$RESULTS_DIR"
