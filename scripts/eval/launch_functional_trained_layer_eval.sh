#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/iibrohimm/project/next_step"
PYTHON_BIN="${PYTHON_BIN:-/home/iibrohimm/miniconda3/envs/open_led/bin/python}"
EVAL_SCRIPT="$ROOT/thinkdet/scripts/eval/eval_affordance_unified.py"
SUMMARY_SCRIPT="$ROOT/thinkdet/scripts/eval/summarize_functional_layer_results.py"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <trained_sweep_dir>"
  echo "Example: $0 $ROOT/thinkdet/checkpoints/unified_layer_sweeps/functional_mllm_layer_sweep_YYYYMMDD_HHMMSS"
  exit 1
fi

SWEEP_DIR="$1"
if [[ ! -d "$SWEEP_DIR" ]]; then
  echo "[error] sweep dir not found: $SWEEP_DIR" >&2
  exit 1
fi

BENCHMARK="${BENCHMARK:-$ROOT/thinkdet/data/benchmarks/affordance_coco_val_heldout_v2.json}"
SPLIT="${SPLIT:-dev}"
TOP_K="${TOP_K:-5}"
EPOCH="${EPOCH:-5}"
EVAL_GPU="${EVAL_GPU:-0}"
LOG_EVERY="${LOG_EVERY:-20}"
RESULTS_ROOT="${RESULTS_ROOT:-$ROOT/thinkdet/results/functional_trained_layer_sweeps}"
RUN_TAG="${RUN_TAG:-$(basename "$SWEEP_DIR")}"
RESULTS_DIR="$RESULTS_ROOT/$RUN_TAG"
LOG_DIR="$ROOT/thinkdet/logs/${RUN_TAG}_eval"

if [[ -n "${LAYERS:-}" ]]; then
  layers="$LAYERS"
else
  layers=""
  for layer_dir in "$SWEEP_DIR"/layer*; do
    [[ -d "$layer_dir" ]] || continue
    layer="${layer_dir##*/layer}"
    layers="$layers $layer"
  done
fi

if [[ -z "${layers// }" ]]; then
  echo "[error] no layer directories found in $SWEEP_DIR" >&2
  exit 1
fi

mkdir -p "$RESULTS_DIR" "$LOG_DIR"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

echo "[eval] sweep_dir=$SWEEP_DIR"
echo "[eval] layers=$layers"
echo "[eval] benchmark=$BENCHMARK split=$SPLIT top_k=$TOP_K"
echo "[eval] results_dir=$RESULTS_DIR"

for layer in $layers; do
  ckpt="$SWEEP_DIR/layer${layer}/thinkdet_unified_epoch${EPOCH}.pth"
  if [[ ! -f "$ckpt" ]]; then
    echo "[warn] missing checkpoint for layer $layer: $ckpt" >&2
    continue
  fi

  out="$RESULTS_DIR/layer${layer}.json"
  log="$LOG_DIR/layer${layer}.log"
  echo "[run] layer=$layer ckpt=$ckpt output=$out"
  CUDA_VISIBLE_DEVICES="$EVAL_GPU" "$PYTHON_BIN" -u "$EVAL_SCRIPT" \
    --benchmark "$BENCHMARK" \
    --split "$SPLIT" \
    --top_k "$TOP_K" \
    --log_every "$LOG_EVERY" \
    --device cuda:0 \
    --thinkdet_checkpoint "$ckpt" \
    --extract_layer "$layer" \
    --output "$out" \
    >"$log" 2>&1
done

"$PYTHON_BIN" "$SUMMARY_SCRIPT" "$RESULTS_DIR"
echo "[done] trained-layer functional eval results: $RESULTS_DIR"
