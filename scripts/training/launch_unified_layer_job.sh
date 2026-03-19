#!/usr/bin/env bash
set -eo pipefail

ROOT="/home/iibrohimm/project/next_step"
CONDA_ROOT="/home/iibrohimm/miniconda3"
ENV_PREFIX="$CONDA_ROOT/envs/open_led"

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <layer_or_layers> <gpu_ids_csv> [epochs] [val_every_ep] [job_tag]"
  echo "Examples:"
  echo "  $0 10 0,1 2 2"
  echo "  $0 9,10 2,3 2 2 fusion_9_10_pilot"
  exit 1
fi

LAYER_SPEC="$1"
GPU_IDS="$2"
EPOCHS="${3:-2}"
VAL_EVERY_EP="${4:-2}"
JOB_TAG="${5:-}"

IFS=',' read -r -a GPU_ARRAY <<< "$GPU_IDS"
NUM_GPUS="${#GPU_ARRAY[@]}"
if [[ "$NUM_GPUS" -lt 1 ]]; then
  echo "No GPUs provided."
  exit 1
fi

IFS=',' read -r -a LAYER_ARRAY <<< "$LAYER_SPEC"
if [[ "${#LAYER_ARRAY[@]}" -eq 1 ]]; then
  LAYER_TAG="layer${LAYER_ARRAY[0]}"
else
  LAYER_TAG="layers_$(IFS=_; echo "${LAYER_ARRAY[*]}")"
fi

if [[ -z "$JOB_TAG" ]]; then
  JOB_TAG="${LAYER_TAG}_pilot_e${EPOCHS}_g${NUM_GPUS}_$(date +%Y%m%d_%H%M%S)"
fi

OUT_DIR="$ROOT/thinkdet/checkpoints/unified/$JOB_TAG"
LOG_DIR="$ROOT/thinkdet/logs/$JOB_TAG"
mkdir -p "$OUT_DIR" "$LOG_DIR"

MASTER_PORT=$(python - <<'PY'
import random
print(random.randint(20000, 45000))
PY
)

ARGS=(
  "$ROOT/thinkdet/scripts/training/train_unified.py"
  --epochs "$EPOCHS"
  --val_every_ep "$VAL_EVERY_EP"
  --output_dir "$OUT_DIR"
)

if [[ -n "${EXTRA_TRAIN_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARGS_ARRAY=(${EXTRA_TRAIN_ARGS})
  ARGS+=("${EXTRA_ARGS_ARRAY[@]}")
fi

if [[ "${#LAYER_ARRAY[@]}" -eq 1 ]]; then
  ARGS+=(--extract_layer "${LAYER_ARRAY[0]}")
else
  ARGS+=(--extract_layers "${LAYER_ARRAY[@]}")
fi

{
  echo "[INFO] job_tag=$JOB_TAG"
  echo "[INFO] gpu_ids=$GPU_IDS num_gpus=$NUM_GPUS"
  echo "[INFO] layer_spec=$LAYER_SPEC"
  echo "[INFO] out_dir=$OUT_DIR"
  echo "[INFO] master_port=$MASTER_PORT"
  echo "[INFO] launch_time=$(date -Is)"
} | tee "$LOG_DIR/launch_meta.txt"

export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export OMP_NUM_THREADS=1

source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate open_led

exec torchrun \
  --nproc_per_node "$NUM_GPUS" \
  --master_port "$MASTER_PORT" \
  "${ARGS[@]}" \
  >"$LOG_DIR/train.log" 2>&1
