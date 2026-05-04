# Held-Out Affordance Benchmark Results (8-GPU Sharded Merge)

- benchmark: `/home/iibrohimm/project/next_step/thinkdet/data/benchmarks/affordance_coco_val_heldout_v1.json`
- split: `test`
- n_samples: 512
- top_k: 5
- llm_rerank: {'enabled': True, 'weight': 0.2, 'max_new_tokens': 6, 'temperature': 0.0}

## Overall

| model | mean_best_iou_top1 | hit@0.5_top1 | hit@0.5_topk | hit@0.75_top1 | precision@0.5_top1 | precision@0.5_topk | recall@0.5_top1 | recall@0.5_topk |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.2057 | 0.1895 | 0.4004 | 0.1387 | 0.1895 | 0.1285 | 0.1895 | 0.4004 |
| stage1 | 0.2154 | 0.2012 | 0.3867 | 0.1484 | 0.2012 | 0.1273 | 0.2012 | 0.3867 |
| stage2 | 0.2155 | 0.2031 | 0.3906 | 0.1465 | 0.2031 | 0.1301 | 0.2031 | 0.3906 |

## Per-Affordance Hit@0.5 (Top1)

| affordance | baseline | stage1 | stage2 |
|---|---:|---:|---:|
| carry_in | 0.1562 | 0.1875 | 0.1875 |
| cut_with | 0.0781 | 0.0781 | 0.0781 |
| drink_from | 0.4531 | 0.5000 | 0.5000 |
| eat_with | 0.0312 | 0.0312 | 0.0312 |
| read | 0.2031 | 0.2188 | 0.2188 |
| ride | 0.3125 | 0.3125 | 0.3281 |
| sit_on | 0.0781 | 0.0781 | 0.0781 |
| talk_on | 0.2031 | 0.2031 | 0.2031 |
