#!/usr/bin/env bash
# ============================================================
#  ThinkDet — Unified Training Launch Script
#  Replaces the old stage1 + stage2 split entirely.
#
#  Uses:
#    - Residual fusion only (no concat)
#    - Pre-adapter preservation KD (single forward)
#    - Joint adapter + head training from epoch 1
#    - Training-time eval disabled (run eval later on 1 GPU)
#
# Usage:
#   bash thinkdet/scripts/training/launch_unified.sh
#   # or with nohup:
#   nohup bash thinkdet/scripts/training/launch_unified.sh > /tmp/unified_train.log 2>&1 &
# ============================================================

set -e
cd /home/iibrohimm/project/next_step

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTDIR="thinkdet/checkpoints/unified/layer9_kd0p05_l1_1e-4_${TIMESTAMP}"
LOG_FILE="/tmp/thinkdet_unified_${TIMESTAMP}.log"

echo "==================================================="
echo "  ThinkDet Unified Training"
echo "  Output:  $OUTDIR"
echo "  Log:     $LOG_FILE"
echo "  Eval:    disabled during training (--val_every_ep 0)"
echo "==================================================="

conda run -p /home/iibrohimm/miniconda3/envs/open_led --no-capture-output \
torchrun \
    --nproc_per_node=7 \
    --master_port=29600 \
    thinkdet/scripts/training/train_unified.py \
        --epochs         5         \
        --lr_adapters    2e-4      \
        --lr_heads       1e-5      \
        --lambda_kd      0.05      \
        --lambda_gate_l1 1e-4      \
        --num_select     300       \
        --tma_m          8         \
        --num_workers    4         \
        --val_every_ep   0         \
        --injection_layers 1 3 5   \
        --output_dir     "$OUTDIR" \
2>&1 | tee "$LOG_FILE"

echo ""
echo "Training complete. Checkpoints at: $OUTDIR"
echo "Log saved to: $LOG_FILE"
