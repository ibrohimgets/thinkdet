# Held-Out Affordance Benchmark Results

- benchmark: `/home/iibrohimm/project/next_step/thinkdet/data/benchmarks/affordance_coco_val_heldout_v1.json`
- split: `test`
- n_samples: 512
- top_k: 5

## Overall

| model | mean_best_iou_top1 | hit@0.5_top1 | hit@0.5_topk | hit@0.75_top1 | precision@0.5_top1 | precision@0.5_topk | recall@0.5_top1 | recall@0.5_topk |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.1868 | 0.1699 | 0.4004 | 0.1250 | 0.1699 | 0.1285 | 0.1699 | 0.4004 |
| stage1 | 0.2021 | 0.1934 | 0.4004 | 0.1289 | 0.1934 | 0.1387 | 0.1934 | 0.4004 |
| stage2 | 0.2019 | 0.1953 | 0.3867 | 0.1309 | 0.1953 | 0.1395 | 0.1953 | 0.3867 |

## Per-Affordance Hit@0.5 (Top1)

| affordance | baseline | stage1 | stage2 |
|---|---:|---:|---:|
| carry_in | 0.1562 | 0.2344 | 0.2500 |
| cut_with | 0.0625 | 0.0625 | 0.0625 |
| drink_from | 0.3750 | 0.4375 | 0.3438 |
| eat_with | 0.0469 | 0.0469 | 0.0625 |
| read | 0.1875 | 0.2188 | 0.2031 |
| ride | 0.3125 | 0.3125 | 0.3125 |
| sit_on | 0.0469 | 0.0156 | 0.0156 |
| talk_on | 0.1719 | 0.2188 | 0.3125 |
