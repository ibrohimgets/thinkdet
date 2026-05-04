# Commonsense Unified Fallback Results

- benchmark: `/home/iibrohimm/project/next_step/thinkdet/data/benchmarks/commonsense_refcoco_objects_coco_val_v1.json`
- split: `test`
- n_samples: 97
- top_k: 5

## Overall

| model | hit@0.5_top1 | hit@0.5_topk | mean_best_iou_top1 | mean_best_iou_topk |
|---|---:|---:|---:|---:|
| thinkdet | 0.4433 | 0.6907 | 0.4166 | 0.6245 |
| thinkdet_fallback | 0.4433 | 0.6907 | 0.4166 | 0.6245 |

## Fallback Stage Counts

- primary: 97
- llm_feedback: 0
- prompt_refine: 0

## Top Objects By Fallback Hit@0.5 Top1

| object | hit@0.5_top1 | mean_best_iou_top1 | n |
|---|---:|---:|---:|
| airplane | 1.0000 | 0.7900 | 1 |
| backpack | 1.0000 | 0.7045 | 1 |
| banana | 1.0000 | 0.9391 | 1 |
| baseball bat | 1.0000 | 0.9126 | 1 |
| bear | 1.0000 | 0.9357 | 1 |
| bed | 1.0000 | 0.9741 | 1 |
| bench | 1.0000 | 0.9453 | 1 |
| cake | 1.0000 | 0.8802 | 1 |
| carrot | 1.0000 | 0.9580 | 1 |
| cat | 1.0000 | 0.8376 | 2 |
| cell phone | 1.0000 | 0.8272 | 1 |
| clock | 1.0000 | 0.6164 | 1 |
| couch | 1.0000 | 0.9399 | 1 |
| elephant | 1.0000 | 0.9807 | 1 |
| fire hydrant | 1.0000 | 0.8766 | 2 |
| frisbee | 1.0000 | 0.7632 | 2 |
| horse | 1.0000 | 0.5820 | 1 |
| oven | 1.0000 | 0.6145 | 1 |
| person | 1.0000 | 0.9674 | 1 |
| pizza | 1.0000 | 0.9715 | 2 |

## Bottom Objects By Fallback Hit@0.5 Top1

| object | hit@0.5_top1 | mean_best_iou_top1 | n |
|---|---:|---:|---:|
| motorcycle | 0.0000 | 0.0516 | 1 |
| mouse | 0.0000 | 0.0000 | 1 |
| orange | 0.0000 | 0.0000 | 1 |
| parking meter | 0.0000 | 0.3328 | 2 |
| refrigerator | 0.0000 | 0.0389 | 1 |
| scissors | 0.0000 | 0.0021 | 1 |
| sheep | 0.0000 | 0.3826 | 1 |
| skateboard | 0.0000 | 0.1136 | 1 |
| skis | 0.0000 | 0.0562 | 1 |
| snowboard | 0.0000 | 0.0078 | 1 |
| sports ball | 0.0000 | 0.0000 | 1 |
| stop sign | 0.0000 | 0.0000 | 1 |
| suitcase | 0.0000 | 0.0036 | 2 |
| tennis racket | 0.0000 | 0.0779 | 1 |
| tie | 0.0000 | 0.0863 | 1 |
| train | 0.0000 | 0.0013 | 1 |
| tv | 0.0000 | 0.0000 | 1 |
| umbrella | 0.0000 | 0.2700 | 1 |
| vase | 0.0000 | 0.0000 | 1 |
| wine glass | 0.0000 | 0.0533 | 1 |
