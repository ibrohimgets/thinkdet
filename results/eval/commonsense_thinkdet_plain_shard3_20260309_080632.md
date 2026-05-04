# Commonsense Unified Fallback Results

- benchmark: `/home/iibrohimm/project/next_step/thinkdet/data/benchmarks/commonsense_refcoco_objects_coco_val_v1.json`
- split: `test`
- n_samples: 98
- top_k: 5

## Overall

| model | hit@0.5_top1 | hit@0.5_topk | mean_best_iou_top1 | mean_best_iou_topk |
|---|---:|---:|---:|---:|
| thinkdet | 0.4082 | 0.6735 | 0.3962 | 0.6162 |
| thinkdet_fallback | 0.4082 | 0.6735 | 0.3962 | 0.6162 |

## Fallback Stage Counts

- primary: 98
- llm_feedback: 0
- prompt_refine: 0

## Top Objects By Fallback Hit@0.5 Top1

| object | hit@0.5_top1 | mean_best_iou_top1 | n |
|---|---:|---:|---:|
| airplane | 1.0000 | 0.8651 | 2 |
| banana | 1.0000 | 0.5763 | 1 |
| bear | 1.0000 | 0.9128 | 2 |
| bench | 1.0000 | 0.9149 | 2 |
| bird | 1.0000 | 0.7690 | 1 |
| bottle | 1.0000 | 0.9181 | 1 |
| cake | 1.0000 | 0.8700 | 2 |
| couch | 1.0000 | 0.7728 | 1 |
| dining table | 1.0000 | 0.9328 | 1 |
| elephant | 1.0000 | 0.9324 | 1 |
| fire hydrant | 1.0000 | 0.7783 | 1 |
| fork | 1.0000 | 0.6676 | 1 |
| frisbee | 1.0000 | 0.8974 | 1 |
| giraffe | 1.0000 | 0.9198 | 1 |
| horse | 1.0000 | 0.6528 | 1 |
| hot dog | 1.0000 | 0.9756 | 1 |
| knife | 1.0000 | 0.9239 | 2 |
| person | 1.0000 | 0.9697 | 1 |
| remote | 1.0000 | 0.8488 | 2 |
| sandwich | 1.0000 | 0.9205 | 2 |

## Bottom Objects By Fallback Hit@0.5 Top1

| object | hit@0.5_top1 | mean_best_iou_top1 | n |
|---|---:|---:|---:|
| parking meter | 0.0000 | 0.0000 | 1 |
| pizza | 0.0000 | 0.0560 | 1 |
| potted plant | 0.0000 | 0.0220 | 1 |
| refrigerator | 0.0000 | 0.1062 | 1 |
| scissors | 0.0000 | 0.0000 | 1 |
| sheep | 0.0000 | 0.4563 | 1 |
| sink | 0.0000 | 0.0310 | 1 |
| skateboard | 0.0000 | 0.0075 | 1 |
| skis | 0.0000 | 0.1983 | 1 |
| snowboard | 0.0000 | 0.1787 | 2 |
| sports ball | 0.0000 | 0.0000 | 1 |
| stop sign | 0.0000 | 0.0970 | 2 |
| suitcase | 0.0000 | 0.0645 | 1 |
| surfboard | 0.0000 | 0.1062 | 1 |
| teddy bear | 0.0000 | 0.0000 | 1 |
| tie | 0.0000 | 0.1328 | 1 |
| toilet | 0.0000 | 0.2600 | 1 |
| truck | 0.0000 | 0.0015 | 1 |
| tv | 0.0000 | 0.0503 | 1 |
| zebra | 0.0000 | 0.0784 | 1 |
