#!/usr/bin/env bash
set -eo pipefail

ROOT="/home/iibrohimm/project/next_step"
CONDA_ROOT="/home/iibrohimm/miniconda3"
RUN_ID="${1:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="$ROOT/thinkdet/logs/layer_selection_$RUN_ID"

mkdir -p "$LOG_DIR"
CONDA_RUN=("$CONDA_ROOT/bin/conda" "run" "-p" "$CONDA_ROOT/envs/open_led")

echo "[INFO] Logs -> $LOG_DIR"

"${CONDA_RUN[@]}" python -u "$ROOT/thinkdet/scripts/analysis/analyze_internvl_layer_similarity.py" \
  --source affordance \
  --affordance_split all \
  --max_samples 256 \
  --output_dir "$ROOT/thinkdet/results/layer_similarity" \
  >"$LOG_DIR/layer_similarity.log" 2>&1

"${CONDA_RUN[@]}" python -u "$ROOT/thinkdet/scripts/analysis/select_layer_shortlist.py" \
  --layer_groups "$ROOT/thinkdet/results/layer_similarity/layer_groups.json" \
  --output "$ROOT/thinkdet/results/layer_selection/shortlist.json" \
  >"$LOG_DIR/shortlist.log" 2>&1

echo "[DONE] First pass finished"
