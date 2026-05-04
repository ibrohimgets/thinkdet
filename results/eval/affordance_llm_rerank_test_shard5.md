# Held-Out Affordance Benchmark Results

- benchmark: `/home/iibrohimm/project/next_step/thinkdet/data/benchmarks/affordance_coco_val_heldout_v1.json`
- split: `test`
- n_samples: 64
- top_k: 5

## Overall

| model | mean_best_iou_top1 | hit@0.5_top1 | hit@0.5_topk | hit@0.75_top1 | precision@0.5_top1 | precision@0.5_topk | recall@0.5_top1 | recall@0.5_topk |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.2549 | 0.2188 | 0.4844 | 0.1562 | 0.2188 | 0.1469 | 0.2188 | 0.4844 |
| stage1 | 0.2559 | 0.2188 | 0.4062 | 0.1562 | 0.2188 | 0.1281 | 0.2188 | 0.4062 |
| stage2 | 0.2495 | 0.2188 | 0.4219 | 0.1562 | 0.2188 | 0.1344 | 0.2188 | 0.4219 |

## Per-Affordance Hit@0.5 (Top1)

| affordance | baseline | stage1 | stage2 |
|---|---:|---:|---:|
| carry_in | 0.2500 | 0.2500 | 0.2500 |
| cut_with | 0.1250 | 0.1250 | 0.1250 |
| drink_from | 0.7500 | 0.7500 | 0.7500 |
| eat_with | 0.0000 | 0.0000 | 0.0000 |
| read | 0.2500 | 0.2500 | 0.2500 |
| ride | 0.1250 | 0.1250 | 0.1250 |
| sit_on | 0.2500 | 0.2500 | 0.2500 |
| talk_on | 0.0000 | 0.0000 | 0.0000 |
