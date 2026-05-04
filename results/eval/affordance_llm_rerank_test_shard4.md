# Held-Out Affordance Benchmark Results

- benchmark: `/home/iibrohimm/project/next_step/thinkdet/data/benchmarks/affordance_coco_val_heldout_v1.json`
- split: `test`
- n_samples: 64
- top_k: 5

## Overall

| model | mean_best_iou_top1 | hit@0.5_top1 | hit@0.5_topk | hit@0.75_top1 | precision@0.5_top1 | precision@0.5_topk | recall@0.5_top1 | recall@0.5_topk |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.2007 | 0.1875 | 0.4375 | 0.0938 | 0.1875 | 0.1531 | 0.1875 | 0.4375 |
| stage1 | 0.2051 | 0.1875 | 0.4375 | 0.0938 | 0.1875 | 0.1500 | 0.1875 | 0.4375 |
| stage2 | 0.2166 | 0.2188 | 0.4531 | 0.1250 | 0.2188 | 0.1531 | 0.2188 | 0.4531 |

## Per-Affordance Hit@0.5 (Top1)

| affordance | baseline | stage1 | stage2 |
|---|---:|---:|---:|
| carry_in | 0.1250 | 0.1250 | 0.1250 |
| cut_with | 0.1250 | 0.1250 | 0.1250 |
| drink_from | 0.5000 | 0.5000 | 0.5000 |
| eat_with | 0.0000 | 0.0000 | 0.0000 |
| read | 0.1250 | 0.1250 | 0.2500 |
| ride | 0.3750 | 0.3750 | 0.5000 |
| sit_on | 0.1250 | 0.1250 | 0.1250 |
| talk_on | 0.1250 | 0.1250 | 0.1250 |
