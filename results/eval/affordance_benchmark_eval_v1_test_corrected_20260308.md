# Held-Out Affordance Benchmark Results

- benchmark: `/home/iibrohimm/project/next_step/thinkdet/data/benchmarks/affordance_coco_val_heldout_v1.json`
- split: `test`
- n_samples: 512
- top_k: 5

## Overall

| model | mean_best_iou_top1 | hit@0.5_top1 | hit@0.5_topk | hit@0.75_top1 | precision@0.5_top1 | precision@0.5_topk | recall@0.5_top1 | recall@0.5_topk |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.1868 | 0.1699 | 0.4043 | 0.1230 | 0.1699 | 0.1285 | 0.1699 | 0.4043 |
| stage1 | 0.1898 | 0.1758 | 0.3965 | 0.1270 | 0.1758 | 0.1273 | 0.1758 | 0.3965 |
| stage2 | 0.1893 | 0.1777 | 0.4004 | 0.1289 | 0.1777 | 0.1336 | 0.1777 | 0.4004 |

## Per-Affordance Hit@0.5 (Top1)

| affordance | baseline | stage1 | stage2 |
|---|---:|---:|---:|
| carry_in | 0.1562 | 0.1719 | 0.2031 |
| cut_with | 0.0625 | 0.0781 | 0.0625 |
| drink_from | 0.3750 | 0.3906 | 0.3906 |
| eat_with | 0.0469 | 0.0469 | 0.0469 |
| read | 0.1875 | 0.1875 | 0.1875 |
| ride | 0.3125 | 0.3125 | 0.3281 |
| sit_on | 0.0469 | 0.0469 | 0.0469 |
| talk_on | 0.1719 | 0.1719 | 0.1562 |
