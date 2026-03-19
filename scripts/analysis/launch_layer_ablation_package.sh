#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/iibrohimm/project/next_step"
CONDA_ROOT="/home/iibrohimm/miniconda3"
ENV_PREFIX="$CONDA_ROOT/envs/open_led"
PYTHON_BIN="$ENV_PREFIX/bin/python"
RESULTS_DIR="$ROOT/thinkdet/results/layer_ablation"
LOG_DIR="$ROOT/thinkdet/logs/layer_ablation"

if [[ $# -gt 0 ]]; then
  LAYERS=("$@")
else
  LAYERS=(0 4 8 9 10 13 20 27)
fi
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
BOOTSTRAP_ITERS="${BOOTSTRAP_ITERS:-2000}"
FORCE_RERUN="${FORCE_RERUN:-0}"

IFS=',' read -r -a GPU_ARRAY <<< "$GPU_IDS"
NUM_GPUS="${#GPU_ARRAY[@]}"

mkdir -p "$RESULTS_DIR" "$LOG_DIR"

BASE_CKPT="${BASE_CKPT:-$ROOT/thinkdet/checkpoints/unified/layer9_kd0p05_l1_1e-4_20260224_135540/thinkdet_unified_epoch5.pth}"

eval_layer() {
  local layer="$1"
  local gpu="$2"
  local layer_log="$LOG_DIR/layer${layer}.log"
  local -a common_args=(
    --thinkdet_checkpoint "$BASE_CKPT"
    --extract_layer "$layer"
    --device cuda:0
  )

  if [[ "$FORCE_RERUN" == "1" ]]; then
    rm -f \
      "$RESULTS_DIR/affordance_layer${layer}.json" \
      "$RESULTS_DIR/flickr_val_layer${layer}.json" \
      "$RESULTS_DIR/flickr_test_layer${layer}.json" \
      "$RESULTS_DIR/refcocog_val_layer${layer}.json"
  fi

  if [[ -f "$RESULTS_DIR/affordance_layer${layer}.json" && \
        -f "$RESULTS_DIR/flickr_val_layer${layer}.json" && \
        -f "$RESULTS_DIR/flickr_test_layer${layer}.json" && \
        -f "$RESULTS_DIR/refcocog_val_layer${layer}.json" ]]; then
    echo "[skip-eval] layer ${layer} already complete"
    return 0
  fi

  echo "[eval] layer ${layer} on GPU ${gpu}" | tee "$layer_log"
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u "$ROOT/thinkdet/scripts/eval/eval_affordance_unified.py" \
    "${common_args[@]}" \
    --split test \
    --output "$RESULTS_DIR/affordance_layer${layer}.json" \
    >>"$layer_log" 2>&1

  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u "$ROOT/thinkdet/scripts/eval/eval_flickr_grounding.py" \
    "${common_args[@]}" \
    --split val \
    --output "$RESULTS_DIR/flickr_val_layer${layer}.json" \
    >>"$layer_log" 2>&1

  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u "$ROOT/thinkdet/scripts/eval/eval_flickr_grounding.py" \
    "${common_args[@]}" \
    --split test \
    --output "$RESULTS_DIR/flickr_test_layer${layer}.json" \
    >>"$layer_log" 2>&1

  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u "$ROOT/thinkdet/scripts/eval/eval_refexp_grounding.py" \
    "${common_args[@]}" \
    --dataset_name refcocog \
    --split val \
    --output "$RESULTS_DIR/refcocog_val_layer${layer}.json" \
    >>"$layer_log" 2>&1
}

if [[ ! -f "$BASE_CKPT" ]]; then
  echo "[error] missing base checkpoint: $BASE_CKPT" >&2
  exit 1
fi

echo "[info] no-retrain layer ablation"
echo "[info] base_checkpoint=$BASE_CKPT"
echo "[info] layers=${LAYERS[*]}"
echo "[info] gpu_ids=$GPU_IDS"

PIDS=()
PID_LAYERS=()
for idx in "${!LAYERS[@]}"; do
  layer="${LAYERS[$idx]}"
  gpu="${GPU_ARRAY[$((idx % NUM_GPUS))]}"
  (
    eval_layer "$layer" "$gpu"
  ) &
  PIDS+=("$!")
  PID_LAYERS+=("$layer")
done

FAIL=0
for idx in "${!PIDS[@]}"; do
  if ! wait "${PIDS[$idx]}"; then
    echo "[error] layer ${PID_LAYERS[$idx]} failed" >&2
    FAIL=1
  fi
done

if [[ "$FAIL" != "0" ]]; then
  exit 1
fi

exec "$PYTHON_BIN" -u "$ROOT/thinkdet/scripts/analysis/summarize_layer_ablation.py" \
  --results_dir "$RESULTS_DIR" \
  --layers "${LAYERS[@]}" \
  --bootstrap_iters "$BOOTSTRAP_ITERS"
