# Commonsense Unified Fallback Results

- benchmark: `/home/iibrohimm/project/next_step/thinkdet/data/benchmarks/commonsense_refcoco_objects_coco_val_v1.json`
- split: `test`
- n_samples: 98
- top_k: 5

## Overall

| model | hit@0.5_top1 | hit@0.5_topk | mean_best_iou_top1 | mean_best_iou_topk |
|---|---:|---:|---:|---:|
| thinkdet | 0.3673 | 0.6327 | 0.3670 | 0.5872 |
| thinkdet_fallback | 0.3673 | 0.6327 | 0.3670 | 0.5872 |

## Fallback Stage Counts

- primary: 98
- llm_feedback: 0
- prompt_refine: 0

## Top Objects By Fallback Hit@0.5 Top1

| object | hit@0.5_top1 | mean_best_iou_top1 | n |
|---|---:|---:|---:|
| baseball bat | 1.0000 | 0.8332 | 2 |
| bear | 1.0000 | 0.9388 | 1 |
| bed | 1.0000 | 0.9790 | 2 |
| broccoli | 1.0000 | 0.9623 | 1 |
| bus | 1.0000 | 0.9795 | 1 |
| cake | 1.0000 | 0.8761 | 1 |
| cat | 1.0000 | 0.8811 | 1 |
| chair | 1.0000 | 0.9412 | 1 |
| cow | 1.0000 | 0.9667 | 1 |
| dining table | 1.0000 | 0.9084 | 1 |
| dog | 1.0000 | 0.7236 | 1 |
| elephant | 1.0000 | 0.9560 | 2 |
| fire hydrant | 1.0000 | 0.9352 | 1 |
| horse | 1.0000 | 0.6243 | 2 |
| person | 1.0000 | 0.9732 | 2 |
| pizza | 1.0000 | 0.9641 | 1 |
| remote | 1.0000 | 0.7836 | 1 |
| sandwich | 1.0000 | 0.9775 | 1 |
| surfboard | 1.0000 | 0.9401 | 1 |
| train | 1.0000 | 0.9520 | 1 |

## Bottom Objects By Fallback Hit@0.5 Top1

| object | hit@0.5_top1 | mean_best_iou_top1 | n |
|---|---:|---:|---:|
| parking meter | 0.0000 | 0.1072 | 1 |
| potted plant | 0.0000 | 0.0436 | 1 |
| refrigerator | 0.0000 | 0.0000 | 1 |
| scissors | 0.0000 | 0.0322 | 1 |
| sheep | 0.0000 | 0.4774 | 1 |
| sink | 0.0000 | 0.0000 | 1 |
| skateboard | 0.0000 | 0.0678 | 1 |
| skis | 0.0000 | 0.0263 | 2 |
| snowboard | 0.0000 | 0.0321 | 1 |
| spoon | 0.0000 | 0.0000 | 1 |
| stop sign | 0.0000 | 0.0000 | 1 |
| suitcase | 0.0000 | 0.0105 | 1 |
| tennis racket | 0.0000 | 0.0625 | 2 |
| tie | 0.0000 | 0.1335 | 1 |
| toilet | 0.0000 | 0.0000 | 1 |
| toothbrush | 0.0000 | 0.0020 | 1 |
| traffic light | 0.0000 | 0.0022 | 1 |
| truck | 0.0000 | 0.0015 | 1 |
| umbrella | 0.0000 | 0.0000 | 1 |
| vase | 0.0000 | 0.0000 | 1 |
