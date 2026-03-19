#!/usr/bin/env bash
# Full RefCOCO eval: Baseline vs TMA Stage2 (COCO-trained)
# All samples, all 8 splits, 8 GPUs via torchrun.
#
# Usage:
#   bash launch_refcoco_full.sh

set -e

ROOT="/home/iibrohimm/project/next_step"
SCRIPT="$ROOT/thinkdet/scripts/eval/eval_refcoco.py"
OUTDIR="$ROOT/thinkdet/results/eval"
TS=$(date +%Y%m%d_%H%M%S)

CKPT="$ROOT/thinkdet/checkpoints/refcoco_stage2_tma_8gpu_20260220_094146/thinkdet_refcoco_stage2_tma_epoch2.pth"

THRESH=0.05
IOU=0.5
NGPU=8

echo "=============================================="
echo "  RefCOCO FULL Eval  (baseline vs TMA)"
echo "  GPUs: $NGPU   samples: ALL"
echo "  conf_thresh: $THRESH  iou_thresh: $IOU"
echo "  checkpoint: $(basename $CKPT)"
echo "  timestamp: $TS"
echo "=============================================="
echo ""

run_split() {
    local DATASET=$1
    local SPLIT=$2
    local SPLIT_BY=$3
    local OUT="$OUTDIR/refcoco_full_${DATASET}_${SPLIT}_${TS}.json"

    echo "--- $DATASET / $SPLIT_BY / $SPLIT ---"
    torchrun \
        --nproc_per_node=$NGPU \
        --master_port=29511 \
        "$SCRIPT" \
        --distributed \
        --dataset_name "$DATASET" \
        --split_by    "$SPLIT_BY" \
        --split       "$SPLIT" \
        --max_samples 0 \
        --conf_thresh $THRESH \
        --iou_thresh  $IOU \
        --log_every   500 \
        --thinkdet_checkpoint "$CKPT" \
        --output "$OUT"
    echo ""
}

run_split refcoco  val   unc
run_split refcoco  testA unc
run_split refcoco  testB unc

run_split refcoco+ val   unc
run_split refcoco+ testA unc
run_split refcoco+ testB unc

run_split refcocog val  umd
run_split refcocog test umd

echo "=============================================="
echo "  All splits done."
echo "  Results: $OUTDIR/refcoco_full_*_${TS}.json"
echo "=============================================="
