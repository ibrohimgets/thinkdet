# Held-Out Affordance Benchmark Results

- benchmark: `/home/iibrohimm/project/next_step/thinkdet/data/benchmarks/affordance_coco_val_heldout_v1.json`
- split: `test`
- n_samples: 64
- top_k: 5

## Overall

| model | mean_best_iou_top1 | hit@0.5_top1 | hit@0.5_topk | hit@0.75_top1 | precision@0.5_top1 | precision@0.5_topk | recall@0.5_top1 | recall@0.5_topk |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.2360 | 0.2188 | 0.4219 | 0.2031 | 0.2188 | 0.1500 | 0.2188 | 0.4219 |
| stage1 | 0.2356 | 0.2188 | 0.4219 | 0.2031 | 0.2188 | 0.1531 | 0.2188 | 0.4219 |
| stage2 | 0.2412 | 0.2188 | 0.4219 | 0.1719 | 0.2188 | 0.1562 | 0.2188 | 0.4219 |

## Per-Affordance Hit@0.5 (Top1)

| affordance | baseline | stage1 | stage2 |
|---|---:|---:|---:|
| carry_in | 0.2500 | 0.2500 | 0.2500 |
| cut_with | 0.0000 | 0.0000 | 0.0000 |
| drink_from | 0.2500 | 0.2500 | 0.2500 |
| eat_with | 0.0000 | 0.0000 | 0.0000 |
| read | 0.2500 | 0.2500 | 0.2500 |
| ride | 0.5000 | 0.5000 | 0.5000 |
| sit_on | 0.1250 | 0.1250 | 0.1250 |
| talk_on | 0.3750 | 0.3750 | 0.3750 |
