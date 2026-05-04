# Affordance Fallback Recovery Tables

These tables compare the corrected Stage-2 ThinkDet base detector against the
same Stage-2 checkpoint with LLM yes/no reranking enabled.

Source files:

- `results/eval/affordance_benchmark_eval_v1_test_corrected_20260308.json`
- `results/eval/affordance_llm_rerank_test_8gpu_merged.json`

Important:

- This is a **fallback** comparison, not a pure base-detector comparison.
- The second table is a **weak-case slice** defined post hoc as the lowest 25%
  of Stage-2 baseline detector confidence on the held-out affordance test set.

## Table A. Full Held-Out Affordance Test

| affordance | Stage-2 | Stage-2 + rerank | delta (pp) | relative |
|---|---:|---:|---:|---:|
| overall | 17.77% | 20.31% | +2.54 | +14.3% |
| carry_in | 20.31% | 18.75% | -1.56 | -7.7% |
| cut_with | 6.25% | 7.81% | +1.56 | +25.0% |
| drink_from | 39.06% | 50.00% | +10.94 | +28.0% |
| eat_with | 4.69% | 3.12% | -1.56 | -33.3% |
| read | 18.75% | 21.88% | +3.12 | +16.7% |
| ride | 32.81% | 32.81% | +0.00 | +0.0% |
| sit_on | 4.69% | 7.81% | +3.12 | +66.7% |
| talk_on | 15.62% | 20.31% | +4.69 | +30.0% |

## Table B. Weak-Case Recovery Slice

Definition:

- select the lowest 25% of Stage-2 baseline detector-confidence cases
- `n = 128` out of `512` held-out test samples

| affordance | n | Stage-2 | Stage-2 + rerank | delta (pp) | relative |
|---|---:|---:|---:|---:|---:|
| overall | 128 | 11.72% | 17.19% | +5.47 | +46.7% |
| carry_in | 24 | 25.00% | 20.83% | -4.17 | -16.7% |
| cut_with | 24 | 8.33% | 8.33% | +0.00 | +0.0% |
| drink_from | 16 | 25.00% | 43.75% | +18.75 | +75.0% |
| eat_with | 16 | 0.00% | 0.00% | +0.00 | --- |
| read | 8 | 12.50% | 25.00% | +12.50 | +100.0% |
| ride | 1 | 0.00% | 0.00% | +0.00 | --- |
| sit_on | 16 | 6.25% | 12.50% | +6.25 | +100.0% |
| talk_on | 23 | 4.35% | 17.39% | +13.04 | +300.0% |

## Recommended Use

Use Table A if you need the fairest single held-out benchmark summary.

Use Table B only if you explicitly frame fallback as a recovery mechanism for
weak detector cases, not as a general overall model improvement.
