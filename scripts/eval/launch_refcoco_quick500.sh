#!/usr/bin/env bash
# Quick 500-sample RefCOCO eval: Baseline vs TMA Stage2 (COCO-trained)
# Runs all 3 val splits on 8 GPUs. Uses conf_thresh=0.05 for fair TMA comparison.
# Top1 accuracy is threshold-FREE (argmax top-1 checked against GT) — threshold
# only affects no_det_rate which is a diagnostic, not the headline number.
#
# Usage:
#   bash launch_refcoco_quick500.sh
#
# Results saved to:
#   thinkdet/results/eval/refcoco_quick500_*.json

set -e

ROOT="/home/iibrohimm/project/next_step"
SCRIPT="$ROOT/thinkdet/scripts/eval/eval_refcoco.py"
OUTDIR="$ROOT/thinkdet/results/eval"
TS=$(date +%Y%m%d_%H%M%S)

CKPT="$ROOT/thinkdet/checkpoints/stage2_tma/layer9_tma_m8_8gpu_20260219_010134/thinkdet_tma_stage2_epoch2.pth"

N=500          # samples per split
THRESH=0.05    # low threshold — fair for TMA which has calibrated-low scores
IOU=0.5        # standard REC IoU threshold
NGPU=8

echo "=============================================="
echo "  RefCOCO Quick-500 Eval  (baseline vs TMA)"
echo "  GPUs: $NGPU   samples/split: $N"
echo "  conf_thresh: $THRESH  iou_thresh: $IOU"
echo "  checkpoint: $(basename $CKPT)"
echo "=============================================="
echo ""

run_split() {
    local DATASET=$1
    local SPLIT=$2
    local SPLIT_BY=$3
    local OUT="$OUTDIR/refcoco_quick500_${DATASET}_${SPLIT}_${TS}.json"

    echo "--- $DATASET / $SPLIT_BY / $SPLIT ---"
    torchrun \
        --nproc_per_node=$NGPU \
        --master_port=29510 \
        "$SCRIPT" \
        --distributed \
        --dataset_name "$DATASET" \
        --split_by    "$SPLIT_BY" \
        --split       "$SPLIT" \
        --max_samples $N \
        --conf_thresh $THRESH \
        --iou_thresh  $IOU \
        --log_every   50 \
        --thinkdet_checkpoint "$CKPT" \
        --output "$OUT"
    echo ""
}

# ── refcoco (standard REC) ──
run_split refcoco  val   unc
run_split refcoco  testA unc
run_split refcoco  testB unc

# ── refcoco+ (no absolute location words) ──
run_split refcoco+ val   unc
run_split refcoco+ testA unc
run_split refcoco+ testB unc

# ── refcocog (longer, paragraph-style expressions) ──
run_split refcocog val  umd
run_split refcocog test umd

echo "=============================================="
echo "  All splits done. Results in:"
echo "  $OUTDIR/refcoco_quick500_*_${TS}.json"
echo "=============================================="
