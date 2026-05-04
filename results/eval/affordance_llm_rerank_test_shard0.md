# Held-Out Affordance Benchmark Results

- benchmark: `/home/iibrohimm/project/next_step/thinkdet/data/benchmarks/affordance_coco_val_heldout_v1.json`
- split: `test`
- n_samples: 64
- top_k: 5

## Overall

| model | mean_best_iou_top1 | hit@0.5_top1 | hit@0.5_topk | hit@0.75_top1 | precision@0.5_top1 | precision@0.5_topk | recall@0.5_top1 | recall@0.5_topk |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.1801 | 0.1406 | 0.3281 | 0.1250 | 0.1406 | 0.1063 | 0.1406 | 0.3281 |
| stage1 | 0.1904 | 0.1562 | 0.3125 | 0.1406 | 0.1562 | 0.1063 | 0.1562 | 0.3125 |
| stage2 | 0.1783 | 0.1406 | 0.3125 | 0.1250 | 0.1406 | 0.1156 | 0.1406 | 0.3125 |

## Per-Affordance Hit@0.5 (Top1)

| affordance | baseline | stage1 | stage2 |
|---|---:|---:|---:|
| carry_in | 0.1250 | 0.1250 | 0.1250 |
| cut_with | 0.1250 | 0.1250 | 0.1250 |
| drink_from | 0.5000 | 0.5000 | 0.5000 |
| eat_with | 0.0000 | 0.0000 | 0.0000 |
| read | 0.1250 | 0.2500 | 0.1250 |
| ride | 0.1250 | 0.1250 | 0.1250 |
| sit_on | 0.0000 | 0.0000 | 0.0000 |
| talk_on | 0.1250 | 0.1250 | 0.1250 |
