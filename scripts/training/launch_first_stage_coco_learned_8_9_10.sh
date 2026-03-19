#!/usr/bin/env bash

set -euo pipefail

cd /home/iibrohimm/project/next_step

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTDIR="thinkdet/checkpoints/first_stage_coco/learned_8_9_10_joint_heads_${TIMESTAMP}"
LOG_FILE="/home/iibrohimm/project/next_step/thinkdet/logs/first_stage_coco_learned_8_9_10_${TIMESTAMP}.log"

mkdir -p "$(dirname "$LOG_FILE")" "$OUTDIR"

echo "==================================================="
echo "  ThinkDet First-Stage COCO Training"
echo "  Variant: learned_8_9_10"
echo "  Frozen:  InternVL + GroundingDINO backbones"
echo "  Train:   adapter + gates + detection heads"
echo "  Output:  $OUTDIR"
echo "  Log:     $LOG_FILE"
echo "==================================================="

HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
OMP_NUM_THREADS=1 \
conda run -p /home/iibrohimm/miniconda3/envs/open_led --no-capture-output \
torchrun \
    --nproc_per_node=7 \
    --master_port=29610 \
    thinkdet/scripts/training/train_unified.py \
        --epochs 5 \
        --lr_adapters 2e-4 \
        --lr_heads 1e-5 \
        --lambda_kd 0.05 \
        --lambda_gate_l1 1e-4 \
        --num_select 300 \
        --tma_m 8 \
        --num_workers 4 \
        --val_every_ep 0 \
        --extract_layers 8 9 10 \
        --layer_fusion learned \
        --injection_layers 1 3 5 \
        --output_dir "$OUTDIR" \
2>&1 | tee "$LOG_FILE"

echo ""
echo "Training complete. Checkpoints at: $OUTDIR"
echo "Log saved to: $LOG_FILE"
