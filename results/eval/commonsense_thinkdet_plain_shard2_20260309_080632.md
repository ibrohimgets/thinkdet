# Commonsense Unified Fallback Results

- benchmark: `/home/iibrohimm/project/next_step/thinkdet/data/benchmarks/commonsense_refcoco_objects_coco_val_v1.json`
- split: `test`
- n_samples: 98
- top_k: 5

## Overall

| model | hit@0.5_top1 | hit@0.5_topk | mean_best_iou_top1 | mean_best_iou_topk |
|---|---:|---:|---:|---:|
| thinkdet | 0.4388 | 0.6837 | 0.4025 | 0.6203 |
| thinkdet_fallback | 0.4388 | 0.6837 | 0.4025 | 0.6203 |

## Fallback Stage Counts

- primary: 98
- llm_feedback: 0
- prompt_refine: 0

## Top Objects By Fallback Hit@0.5 Top1

| object | hit@0.5_top1 | mean_best_iou_top1 | n |
|---|---:|---:|---:|
| airplane | 1.0000 | 0.8333 | 2 |
| baseball bat | 1.0000 | 0.6919 | 1 |
| bear | 1.0000 | 0.9134 | 2 |
| bench | 1.0000 | 0.9422 | 2 |
| bird | 1.0000 | 0.7764 | 1 |
| book | 1.0000 | 0.5027 | 1 |
| broccoli | 1.0000 | 0.9704 | 1 |
| bus | 1.0000 | 0.6976 | 1 |
| cake | 1.0000 | 0.8687 | 2 |
| cat | 1.0000 | 0.8730 | 1 |
| clock | 1.0000 | 0.7740 | 2 |
| couch | 1.0000 | 0.7726 | 1 |
| donut | 1.0000 | 0.9401 | 2 |
| elephant | 1.0000 | 0.9142 | 1 |
| fire hydrant | 1.0000 | 0.9398 | 1 |
| giraffe | 1.0000 | 0.9347 | 1 |
| horse | 1.0000 | 0.6493 | 1 |
| hot dog | 1.0000 | 0.9745 | 1 |
| person | 1.0000 | 0.9661 | 1 |
| pizza | 1.0000 | 0.9640 | 1 |

## Bottom Objects By Fallback Hit@0.5 Top1

| object | hit@0.5_top1 | mean_best_iou_top1 | n |
|---|---:|---:|---:|
| mouse | 0.0000 | 0.0259 | 1 |
| orange | 0.0000 | 0.0260 | 1 |
| oven | 0.0000 | 0.0110 | 1 |
| parking meter | 0.0000 | 0.0000 | 1 |
| potted plant | 0.0000 | 0.0729 | 1 |
| refrigerator | 0.0000 | 0.0000 | 1 |
| scissors | 0.0000 | 0.0358 | 1 |
| sink | 0.0000 | 0.0000 | 1 |
| skateboard | 0.0000 | 0.0679 | 1 |
| skis | 0.0000 | 0.0000 | 1 |
| snowboard | 0.0000 | 0.0071 | 2 |
| sports ball | 0.0000 | 0.1491 | 1 |
| suitcase | 0.0000 | 0.0048 | 1 |
| teddy bear | 0.0000 | 0.0000 | 1 |
| tie | 0.0000 | 0.1318 | 1 |
| traffic light | 0.0000 | 0.0028 | 1 |
| train | 0.0000 | 0.2707 | 1 |
| truck | 0.0000 | 0.0014 | 1 |
| umbrella | 0.0000 | 0.1956 | 2 |
| vase | 0.0000 | 0.0008 | 1 |
