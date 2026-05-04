# Commonsense Unified Fallback Results

- benchmark: `/home/iibrohimm/project/next_step/thinkdet/data/benchmarks/commonsense_refcoco_objects_coco_val_v1.json`
- split: `test`
- n_samples: 97
- top_k: 5

## Overall

| model | hit@0.5_top1 | hit@0.5_topk | mean_best_iou_top1 | mean_best_iou_topk |
|---|---:|---:|---:|---:|
| thinkdet | 0.3711 | 0.6495 | 0.3531 | 0.5988 |
| thinkdet_fallback | 0.3711 | 0.6495 | 0.3531 | 0.5988 |

## Fallback Stage Counts

- primary: 97
- llm_feedback: 0
- prompt_refine: 0

## Top Objects By Fallback Hit@0.5 Top1

| object | hit@0.5_top1 | mean_best_iou_top1 | n |
|---|---:|---:|---:|
| airplane | 1.0000 | 0.9442 | 1 |
| banana | 1.0000 | 0.9436 | 1 |
| baseball bat | 1.0000 | 0.9430 | 1 |
| bear | 1.0000 | 0.8919 | 1 |
| bed | 1.0000 | 0.9701 | 1 |
| bench | 1.0000 | 0.9403 | 1 |
| boat | 1.0000 | 0.6368 | 1 |
| book | 1.0000 | 0.6866 | 1 |
| bowl | 1.0000 | 0.9472 | 1 |
| cake | 1.0000 | 0.8667 | 1 |
| carrot | 1.0000 | 0.9497 | 1 |
| clock | 1.0000 | 0.5539 | 1 |
| couch | 1.0000 | 0.9378 | 1 |
| cow | 1.0000 | 0.9070 | 1 |
| dining table | 1.0000 | 0.9260 | 2 |
| donut | 1.0000 | 0.9408 | 1 |
| elephant | 1.0000 | 0.9786 | 1 |
| fire hydrant | 1.0000 | 0.8526 | 2 |
| handbag | 1.0000 | 0.8330 | 1 |
| person | 1.0000 | 0.9744 | 1 |

## Bottom Objects By Fallback Hit@0.5 Top1

| object | hit@0.5_top1 | mean_best_iou_top1 | n |
|---|---:|---:|---:|
| potted plant | 0.0000 | 0.0211 | 2 |
| refrigerator | 0.0000 | 0.0390 | 1 |
| sheep | 0.0000 | 0.0000 | 1 |
| sink | 0.0000 | 0.0221 | 2 |
| skateboard | 0.0000 | 0.0704 | 1 |
| skis | 0.0000 | 0.0498 | 1 |
| snowboard | 0.0000 | 0.3425 | 1 |
| spoon | 0.0000 | 0.0993 | 1 |
| sports ball | 0.0000 | 0.0000 | 1 |
| stop sign | 0.0000 | 0.1907 | 1 |
| suitcase | 0.0000 | 0.0510 | 2 |
| surfboard | 0.0000 | 0.0536 | 2 |
| tennis racket | 0.0000 | 0.0736 | 1 |
| tie | 0.0000 | 0.0674 | 1 |
| toilet | 0.0000 | 0.0204 | 2 |
| train | 0.0000 | 0.0026 | 1 |
| tv | 0.0000 | 0.0000 | 1 |
| umbrella | 0.0000 | 0.0000 | 1 |
| vase | 0.0000 | 0.0000 | 1 |
| wine glass | 0.0000 | 0.0506 | 1 |
