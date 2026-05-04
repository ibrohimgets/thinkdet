# Held-Out Affordance Benchmark Results

- benchmark: `/home/iibrohimm/project/next_step/thinkdet/data/benchmarks/affordance_coco_val_heldout_v1.json`
- split: `test`
- n_samples: 64
- top_k: 5

## Overall

| model | mean_best_iou_top1 | hit@0.5_top1 | hit@0.5_topk | hit@0.75_top1 | precision@0.5_top1 | precision@0.5_topk | recall@0.5_top1 | recall@0.5_topk |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.2002 | 0.2031 | 0.3750 | 0.1250 | 0.2031 | 0.1156 | 0.2031 | 0.3750 |
| stage1 | 0.2185 | 0.2344 | 0.3750 | 0.1406 | 0.2344 | 0.1188 | 0.2344 | 0.3750 |
| stage2 | 0.2262 | 0.2344 | 0.3750 | 0.1406 | 0.2344 | 0.1219 | 0.2344 | 0.3750 |

## Per-Affordance Hit@0.5 (Top1)

| affordance | baseline | stage1 | stage2 |
|---|---:|---:|---:|
| carry_in | 0.1250 | 0.2500 | 0.2500 |
| cut_with | 0.1250 | 0.1250 | 0.1250 |
| drink_from | 0.5000 | 0.6250 | 0.6250 |
| eat_with | 0.1250 | 0.1250 | 0.1250 |
| read | 0.1250 | 0.1250 | 0.1250 |
| ride | 0.3750 | 0.3750 | 0.3750 |
| sit_on | 0.1250 | 0.1250 | 0.1250 |
| talk_on | 0.1250 | 0.1250 | 0.1250 |
