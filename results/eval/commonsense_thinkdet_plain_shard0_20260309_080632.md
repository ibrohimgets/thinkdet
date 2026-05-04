# Commonsense Unified Fallback Results

- benchmark: `/home/iibrohimm/project/next_step/thinkdet/data/benchmarks/commonsense_refcoco_objects_coco_val_v1.json`
- split: `test`
- n_samples: 98
- top_k: 5

## Overall

| model | hit@0.5_top1 | hit@0.5_topk | mean_best_iou_top1 | mean_best_iou_topk |
|---|---:|---:|---:|---:|
| thinkdet | 0.4082 | 0.6633 | 0.3860 | 0.6167 |
| thinkdet_fallback | 0.4082 | 0.6633 | 0.3860 | 0.6167 |

## Fallback Stage Counts

- primary: 98
- llm_feedback: 0
- prompt_refine: 0

## Top Objects By Fallback Hit@0.5 Top1

| object | hit@0.5_top1 | mean_best_iou_top1 | n |
|---|---:|---:|---:|
| airplane | 1.0000 | 0.7242 | 1 |
| baseball bat | 1.0000 | 0.8179 | 2 |
| bear | 1.0000 | 0.9380 | 1 |
| broccoli | 1.0000 | 0.9715 | 1 |
| bus | 1.0000 | 0.9893 | 1 |
| cake | 1.0000 | 0.9019 | 1 |
| carrot | 1.0000 | 0.9488 | 1 |
| cat | 1.0000 | 0.8731 | 1 |
| cell phone | 1.0000 | 0.8160 | 1 |
| clock | 1.0000 | 0.5696 | 1 |
| couch | 1.0000 | 0.9435 | 1 |
| dining table | 1.0000 | 0.8842 | 1 |
| dog | 1.0000 | 0.7366 | 1 |
| elephant | 1.0000 | 0.9532 | 2 |
| fire hydrant | 1.0000 | 0.9514 | 1 |
| giraffe | 1.0000 | 0.7983 | 1 |
| horse | 1.0000 | 0.6216 | 2 |
| knife | 1.0000 | 0.9636 | 1 |
| person | 1.0000 | 0.9694 | 2 |
| pizza | 1.0000 | 0.9691 | 1 |

## Bottom Objects By Fallback Hit@0.5 Top1

| object | hit@0.5_top1 | mean_best_iou_top1 | n |
|---|---:|---:|---:|
| orange | 0.0000 | 0.0000 | 1 |
| oven | 0.0000 | 0.0054 | 2 |
| parking meter | 0.0000 | 0.0000 | 1 |
| potted plant | 0.0000 | 0.0000 | 1 |
| refrigerator | 0.0000 | 0.0000 | 1 |
| scissors | 0.0000 | 0.0000 | 1 |
| sheep | 0.0000 | 0.3947 | 1 |
| sink | 0.0000 | 0.0000 | 1 |
| skis | 0.0000 | 0.0288 | 2 |
| snowboard | 0.0000 | 0.0327 | 1 |
| stop sign | 0.0000 | 0.0000 | 1 |
| tie | 0.0000 | 0.0673 | 1 |
| toilet | 0.0000 | 0.0000 | 1 |
| traffic light | 0.0000 | 0.0000 | 1 |
| train | 0.0000 | 0.0013 | 1 |
| truck | 0.0000 | 0.0000 | 1 |
| tv | 0.0000 | 0.1654 | 2 |
| umbrella | 0.0000 | 0.3711 | 1 |
| vase | 0.0000 | 0.0000 | 1 |
| zebra | 0.0000 | 0.0000 | 1 |
