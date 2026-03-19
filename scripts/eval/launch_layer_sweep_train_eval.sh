#!/bin/bash
# Train ThinkDet with 4 different early InternVL layers (0, 1, 2, 3) in parallel.
# Each training job uses 2 GPUs (DDP), so all 8 GPUs are utilized.
# After training, evaluate each on RefCOCO val.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/../training/train_unified.py"
EVAL_SCRIPT="${SCRIPT_DIR}/eval_refcoco.py"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

LAYERS=(0 1 2 3)
GPUS_PER_JOB=2

echo "=========================================="
echo "Early Layer Sweep: Training layers ${LAYERS[@]}"
echo "Each job uses ${GPUS_PER_JOB} GPUs"
echo "Timestamp: ${TIMESTAMP}"
echo "=========================================="

PIDS=()
CKPT_DIRS=()

for i in "${!LAYERS[@]}"; do
    LAYER=${LAYERS[$i]}
    GPU_START=$((i * GPUS_PER_JOB))
    GPU_END=$((GPU_START + GPUS_PER_JOB - 1))
    GPU_LIST=$(seq -s, $GPU_START $GPU_END)
    OUTPUT_DIR="/home/iibrohimm/project/next_step/thinkdet/checkpoints/unified/layer${LAYER}_sweep_${TIMESTAMP}"
    CKPT_DIRS+=("${OUTPUT_DIR}")

    echo ""
    echo "[Layer ${LAYER}] GPUs ${GPU_LIST} -> ${OUTPUT_DIR}"

    CUDA_VISIBLE_DEVICES=${GPU_LIST} \
    torchrun \
        --nproc_per_node=${GPUS_PER_JOB} \
        --master_port=$((29500 + LAYER)) \
        "${TRAIN_SCRIPT}" \
        --extract_layer ${LAYER} \
        --epochs 5 \
        --output_dir "${OUTPUT_DIR}" \
        2>&1 | sed "s/^/[L${LAYER}] /" &

    PIDS+=($!)
done

echo ""
echo "Waiting for all ${#LAYERS[@]} training jobs to finish..."
echo "(This may take several hours)"

FAILED=0
for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}"; then
        echo "[Layer ${LAYERS[$i]}] TRAINING DONE"
    else
        echo "[Layer ${LAYERS[$i]}] TRAINING FAILED"
        FAILED=1
    fi
done

if [ "${FAILED}" -eq 1 ]; then
    echo "WARNING: One or more training jobs failed."
fi

echo ""
echo "=========================================="
echo "Training complete. Checkpoints:"
for i in "${!LAYERS[@]}"; do
    echo "  Layer ${LAYERS[$i]}: ${CKPT_DIRS[$i]}"
done
echo "=========================================="

# Now evaluate each on RefCOCO val
echo ""
echo "Starting RefCOCO val evaluations..."

EVAL_PIDS=()
for i in "${!LAYERS[@]}"; do
    LAYER=${LAYERS[$i]}
    CKPT_DIR="${CKPT_DIRS[$i]}"
    # Find the best (last) epoch checkpoint
    BEST_CKPT=$(ls -t "${CKPT_DIR}"/thinkdet_unified_epoch*.pth 2>/dev/null | head -1)
    if [ -z "${BEST_CKPT}" ]; then
        echo "[Layer ${LAYER}] No checkpoint found, skipping eval"
        continue
    fi
    EVAL_OUT="/home/iibrohimm/project/next_step/thinkdet/results/eval/refcoco_eval_layer${LAYER}_sweep_${TIMESTAMP}.json"

    echo "[Layer ${LAYER}] Evaluating ${BEST_CKPT}"

    CUDA_VISIBLE_DEVICES=${i} \
    python "${EVAL_SCRIPT}" \
        --thinkdet_checkpoint "${BEST_CKPT}" \
        --device "cuda:0" \
        --output "${EVAL_OUT}" \
        --conf_thresh 0.05 \
        2>&1 | sed "s/^/[EVAL-L${LAYER}] /" &

    EVAL_PIDS+=($!)
done

for i in "${!EVAL_PIDS[@]}"; do
    wait "${EVAL_PIDS[$i]}" || true
done

echo ""
echo "=========================================="
echo "ALL DONE - Layer sweep complete"
echo "=========================================="
