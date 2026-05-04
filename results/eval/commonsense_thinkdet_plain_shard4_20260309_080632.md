# Commonsense Unified Fallback Results

- benchmark: `/home/iibrohimm/project/next_step/thinkdet/data/benchmarks/commonsense_refcoco_objects_coco_val_v1.json`
- split: `test`
- n_samples: 97
- top_k: 5

## Overall

| model | hit@0.5_top1 | hit@0.5_topk | mean_best_iou_top1 | mean_best_iou_topk |
|---|---:|---:|---:|---:|
| thinkdet | 0.3918 | 0.6186 | 0.3768 | 0.5775 |
| thinkdet_fallback | 0.3918 | 0.6186 | 0.3768 | 0.5775 |

## Fallback Stage Counts

- primary: 97
- llm_feedback: 0
- prompt_refine: 0

## Top Objects By Fallback Hit@0.5 Top1

| object | hit@0.5_top1 | mean_best_iou_top1 | n |
|---|---:|---:|---:|
| airplane | 1.0000 | 0.9522 | 1 |
| banana | 1.0000 | 0.7467 | 2 |
| bed | 1.0000 | 0.9466 | 1 |
| bird | 1.0000 | 0.8038 | 1 |
| bowl | 1.0000 | 0.9445 | 1 |
| cat | 1.0000 | 0.7884 | 1 |
| chair | 1.0000 | 0.9403 | 1 |
| cow | 1.0000 | 0.9394 | 1 |
| dining table | 1.0000 | 0.5705 | 1 |
| dog | 1.0000 | 0.8533 | 1 |
| donut | 1.0000 | 0.9446 | 1 |
| elephant | 1.0000 | 0.9250 | 1 |
| fire hydrant | 1.0000 | 0.7505 | 1 |
| fork | 1.0000 | 0.6699 | 1 |
| frisbee | 1.0000 | 0.8888 | 1 |
| handbag | 1.0000 | 0.8338 | 1 |
| horse | 1.0000 | 0.9491 | 1 |
| hot dog | 1.0000 | 0.9812 | 1 |
| kite | 1.0000 | 0.7958 | 1 |
| person | 1.0000 | 0.9748 | 1 |

## Bottom Objects By Fallback Hit@0.5 Top1

| object | hit@0.5_top1 | mean_best_iou_top1 | n |
|---|---:|---:|---:|
| mouse | 0.0000 | 0.0000 | 2 |
| orange | 0.0000 | 0.0024 | 2 |
| oven | 0.0000 | 0.0109 | 1 |
| parking meter | 0.0000 | 0.0000 | 1 |
| potted plant | 0.0000 | 0.2888 | 1 |
| refrigerator | 0.0000 | 0.0445 | 2 |
| scissors | 0.0000 | 0.0065 | 2 |
| sheep | 0.0000 | 0.4332 | 2 |
| skateboard | 0.0000 | 0.1100 | 2 |
| skis | 0.0000 | 0.0000 | 1 |
| snowboard | 0.0000 | 0.0582 | 1 |
| spoon | 0.0000 | 0.0688 | 1 |
| suitcase | 0.0000 | 0.0000 | 1 |
| surfboard | 0.0000 | 0.0211 | 1 |
| teddy bear | 0.0000 | 0.0000 | 1 |
| tie | 0.0000 | 0.2188 | 2 |
| toilet | 0.0000 | 0.1916 | 1 |
| toothbrush | 0.0000 | 0.0000 | 1 |
| umbrella | 0.0000 | 0.0000 | 1 |
| vase | 0.0000 | 0.0000 | 2 |
