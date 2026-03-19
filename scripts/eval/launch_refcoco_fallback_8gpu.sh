#!/bin/bash
# Extracted from eval_refcoco.py usage
# Launch ThinkDet with full LLM fallback on RefCOCO val across 8 GPUs

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EVAL_SCRIPT="${SCRIPT_DIR}/eval_refcoco.py"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

CHECKPOINT="/home/iibrohimm/project/next_step/thinkdet/checkpoints/unified/layer9_kd0p05_l1_1e-4_20260224_135540/thinkdet_unified_epoch5.pth"
OUT_DIR="/home/iibrohimm/project/next_step/thinkdet/results/eval"
OUT_FILE="${OUT_DIR}/refcoco_eval_unified_fallback_${TIMESTAMP}.json"

echo "=========================================================="
echo "RefCOCO eval with FULL FALLBACK on 8 GPUs"
echo "Checkpoint: ${CHECKPOINT}"
echo "Output: ${OUT_FILE}"
echo "=========================================================="

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun \
    --nproc_per_node=8 \
    --master_port=29505 \
    "${EVAL_SCRIPT}" \
    --dataset_name "refcoco" \
    --split "val" \
    --thinkdet_checkpoint "${CHECKPOINT}" \
    --distributed \
    --enable_fallback \
    --prompt_refine \
    --log_every 50 \
    --output "${OUT_FILE}"

echo ""
echo "Done. Results saved to:"
echo "  ${OUT_FILE}"
ls -lh "${OUT_FILE}"*
