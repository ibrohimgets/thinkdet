# ThinkDet Eval Reliability Fix Plan

Date: 2026-02-17  
Owner: ThinkDet Stage-1 evaluation pipeline

## 1. Problem We Observed

The previously reported near-zero checkpoint result (`mAP ~0.001`) was an evaluation artifact, not the true model performance.

- Mixed baseline+trained evaluation in one process can trigger OOM/memory-state failures on the second phase.
- Current eval path can silently swallow exceptions and continue, which can corrupt metrics.
- Clean trained-only eval shows real performance (`mAP ~0.397`), not collapse.

## 2. Root Cause

In `thinkdet/scripts/training/train_stage1.py` evaluation logic:

- `except Exception: continue` drops failed samples silently.
- Failed samples are excluded from prediction generation, which can make results look catastrophically bad.

Also, running baseline then trained in the same long process increases memory-fragmentation risk and can create false failures.

## 3. Fix Strategy (Priority Order)

## P0: Evaluation protocol hardening (must do first)

1. Never run baseline and trained in the same process for official reporting.
2. Run each as its own fresh job:
   - `baseline-only`
   - `trained-only`
   - optional `trained-gate0-only` sanity
3. Require runtime error accounting in every eval JSON:
   - `processed_images_sum`
   - `errors_sum`
   - `error_rate_pct`
   - `error_types`
4. Reject eval results if `errors_sum > 0`.

## P1: Fail-fast behavior in official eval script

1. Replace broad silent catch with explicit handling:
   - log full exception type
   - increment error counters
   - abort run if errors exceed threshold (default: 0 for official runs)
2. Print and save image ids that failed.

## P2: Baseline equivalence sanity check

1. Add a required check before publishing:
   - `trained_gate0-only` must match `baseline-only` within tiny tolerance.
2. If gate0 does not match baseline, eval environment is invalid and report is blocked.

## P3: Publish only clean deltas

For stage-1 checkpoint comparisons, publish:

1. `delta = trained-only - baseline-only`
2. Both runs must have `errors_sum = 0`
3. Same:
   - dataset/split
   - `conf_thresh`
   - `max_dets`
   - GPU count
   - script version

## 4. Canonical Eval Outputs to Keep

Use and keep these files for this checkpoint:

- `thinkdet/results/eval/stage1_layer9_e12_fix_trained_only_coco_val_8gpu.json`
- `thinkdet/results/eval/stage1_layer9_e12_fix_trained_gate0_only_coco_val_8gpu.json`
- `thinkdet/results/eval/repro_mixed_eval_artifact_subset400.json` (artifact proof, not benchmark)

Do not use as official result:

- `thinkdet/results/eval/stage1_layer9_e12_fix_vs_baseline_coco_val_8gpu.json`

## 5. Acceptance Criteria

A run is valid only if all are true:

1. `errors_sum == 0`
2. `num_images == 4952` on COCO val2017
3. `trained_gate0-only` ~= `baseline-only` (metric drift near 0)
4. Reported checkpoint delta comes from separate clean jobs, not mixed phase runs.

## 6. Next Execution Plan

1. Keep current model weights unchanged.
2. Standardize official evaluator to trained-only and baseline-only isolated jobs.
3. Add runtime error checks and fail-fast gating to the evaluator.
4. Recompute and publish final stage-1 table using clean deltas only.

