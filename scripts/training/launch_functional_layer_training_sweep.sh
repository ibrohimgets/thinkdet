#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/iibrohimm/project/next_step"
CONDA_ROOT="/home/iibrohimm/miniconda3"
ENV_NAME="open_led"
TRAIN_SCRIPT="$ROOT/thinkdet/scripts/training/train_unified.py"

# LED-style coarse probe plus our current default layer 9.
# Override for a dense sweep, for example:
#   LAYERS="1 2 3 ... 24" bash scripts/training/launch_functional_layer_training_sweep.sh
LAYERS="${LAYERS:-2 4 8 9 24}"

CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3,4,5,6}"
IFS=',' read -r -a GPU_ARRAY <<< "$CUDA_DEVICES"
NPROC_PER_NODE="${NPROC_PER_NODE:-${#GPU_ARRAY[@]}}"

EPOCHS="${EPOCHS:-5}"
LR_ADAPTERS="${LR_ADAPTERS:-2e-4}"
LR_HEADS="${LR_HEADS:-1e-5}"
LAMBDA_KD="${LAMBDA_KD:-0.05}"
LAMBDA_GATE_L1="${LAMBDA_GATE_L1:-1e-4}"
TMA_M="${TMA_M:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
VAL_EVERY_EP="${VAL_EVERY_EP:-0}"
SEED="${SEED:-1337}"
INJECTION_LAYERS="${INJECTION_LAYERS:-1 3 5}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

SWEEP_TAG="${SWEEP_TAG:-functional_mllm_layer_sweep_$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-$ROOT/thinkdet/checkpoints/unified_layer_sweeps/$SWEEP_TAG}"
LOG_ROOT="${LOG_ROOT:-$ROOT/thinkdet/logs/$SWEEP_TAG}"

mkdir -p "$OUT_ROOT" "$LOG_ROOT"

echo "[sweep] tag=$SWEEP_TAG"
echo "[sweep] layers=$LAYERS"
echo "[sweep] cuda_devices=$CUDA_DEVICES nproc_per_node=$NPROC_PER_NODE"
echo "[sweep] out_root=$OUT_ROOT"
echo "[sweep] recipe: epochs=$EPOCHS lr_adapters=$LR_ADAPTERS lr_heads=$LR_HEADS lambda_kd=$LAMBDA_KD lambda_gate_l1=$LAMBDA_GATE_L1 seed=$SEED"

set +u
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
set -u

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
export OMP_NUM_THREADS=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

for layer in $LAYERS; do
  run_name="layer${layer}"
  out_dir="$OUT_ROOT/$run_name"
  log_path="$LOG_ROOT/${run_name}.log"
  ckpt_path="$out_dir/thinkdet_unified_epoch${EPOCHS}.pth"

  if [[ "$SKIP_EXISTING" == "1" && -f "$ckpt_path" ]]; then
    echo "[skip] $run_name already has $ckpt_path"
    continue
  fi

  mkdir -p "$out_dir"
  master_port=$((20000 + RANDOM % 20000))

  echo "[run] layer=$layer output=$out_dir log=$log_path master_port=$master_port"
  torchrun \
    --nproc_per_node="$NPROC_PER_NODE" \
    --master_port="$master_port" \
    "$TRAIN_SCRIPT" \
      --extract_layer "$layer" \
      --epochs "$EPOCHS" \
      --lr_adapters "$LR_ADAPTERS" \
      --lr_heads "$LR_HEADS" \
      --lambda_kd "$LAMBDA_KD" \
      --lambda_gate_l1 "$LAMBDA_GATE_L1" \
      --num_select 300 \
      --tma_m "$TMA_M" \
      --num_workers "$NUM_WORKERS" \
      --val_every_ep "$VAL_EVERY_EP" \
      --seed "$SEED" \
      --injection_layers $INJECTION_LAYERS \
      --output_dir "$out_dir" \
      >"$log_path" 2>&1
done

echo "[done] trained-layer sweep outputs: $OUT_ROOT"
