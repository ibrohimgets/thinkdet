# Commonsense Unified Fallback Results

- benchmark: `/home/iibrohimm/project/next_step/thinkdet/data/benchmarks/commonsense_refcoco_objects_coco_val_v1.json`
- split: `test`
- n_samples: 1
- top_k: 5

## Overall

| model | hit@0.5_top1 | hit@0.5_topk | mean_best_iou_top1 | mean_best_iou_topk |
|---|---:|---:|---:|---:|
| thinkdet | 1.0000 | 1.0000 | 0.9675 | 0.9675 |
| thinkdet_fallback | 1.0000 | 1.0000 | 0.6344 | 0.9675 |

## Fallback Stage Counts

- primary: 0
- llm_feedback: 1
- prompt_refine: 0

## Top Objects By Fallback Hit@0.5 Top1

| object | hit@0.5_top1 | mean_best_iou_top1 | n |
|---|---:|---:|---:|
| person | 1.0000 | 0.6344 | 1 |

## Bottom Objects By Fallback Hit@0.5 Top1

| object | hit@0.5_top1 | mean_best_iou_top1 | n |
|---|---:|---:|---:|
| person | 1.0000 | 0.6344 | 1 |
