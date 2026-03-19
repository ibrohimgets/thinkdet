#!/usr/bin/env bash
# RefCOCO Stage 2 TMA Training — 8 GPUs
# Fine-tune Stage 1 TMA (COCO-trained) on RefCOCO/+/g
#
# Usage:
#   bash launch_refcoco_stage2_tma.sh

set -e

ROOT="/home/iibrohimm/project/next_step"
SCRIPT="$ROOT/thinkdet/scripts/training/train_refcoco_stage2_tma.py"

CKPT="$ROOT/thinkdet/checkpoints/stage1_tma/layer9_tma_m8_full_2ep_7gpu_20260218_152649/thinkdet_tma_stage1_epoch2.pth"

NGPU=8

echo "=============================================="
echo "  RefCOCO Stage 2 TMA Training"
echo "  GPUs: $NGPU"
echo "  Stage1 ckpt: $(basename $CKPT)"
echo "  Epochs: 3"
echo "  Effective batch: 2 * 4 * $NGPU = $((2 * 4 * NGPU))"
echo "=============================================="
echo ""

torchrun \
    --nproc_per_node=$NGPU \
    --master_port=29512 \
    "$SCRIPT" \
    --stage1_ckpt "$CKPT" \
    --epochs 3

echo ""
echo "=============================================="
echo "  Training complete!"
echo "=============================================="
