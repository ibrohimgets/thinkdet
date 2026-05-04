# Commonsense Unified Fallback Results

- benchmark: `/home/iibrohimm/project/next_step/thinkdet/data/benchmarks/commonsense_refcoco_objects_coco_val_v1.json`
- split: `test`
- n_samples: 97
- top_k: 5

## Overall

| model | hit@0.5_top1 | hit@0.5_topk | mean_best_iou_top1 | mean_best_iou_topk |
|---|---:|---:|---:|---:|
| thinkdet | 0.4227 | 0.6186 | 0.3977 | 0.5871 |
| thinkdet_fallback | 0.4227 | 0.6186 | 0.3977 | 0.5871 |

## Fallback Stage Counts

- primary: 97
- llm_feedback: 0
- prompt_refine: 0

## Top Objects By Fallback Hit@0.5 Top1

| object | hit@0.5_top1 | mean_best_iou_top1 | n |
|---|---:|---:|---:|
| airplane | 1.0000 | 0.9099 | 1 |
| baseball bat | 1.0000 | 0.9382 | 1 |
| bear | 1.0000 | 0.8828 | 1 |
| bed | 1.0000 | 0.8104 | 1 |
| bench | 1.0000 | 0.9434 | 1 |
| book | 1.0000 | 0.6529 | 1 |
| bowl | 1.0000 | 0.9516 | 1 |
| cake | 1.0000 | 0.8725 | 1 |
| cat | 1.0000 | 0.7999 | 1 |
| clock | 1.0000 | 0.9783 | 1 |
| couch | 1.0000 | 0.8544 | 2 |
| cow | 1.0000 | 0.8723 | 1 |
| dining table | 1.0000 | 0.7172 | 1 |
| dog | 1.0000 | 0.8035 | 1 |
| donut | 1.0000 | 0.9456 | 1 |
| elephant | 1.0000 | 0.9811 | 1 |
| fire hydrant | 1.0000 | 0.5902 | 1 |
| frisbee | 1.0000 | 0.8943 | 1 |
| giraffe | 1.0000 | 0.8711 | 2 |
| kite | 1.0000 | 0.8368 | 1 |

## Bottom Objects By Fallback Hit@0.5 Top1

| object | hit@0.5_top1 | mean_best_iou_top1 | n |
|---|---:|---:|---:|
| motorcycle | 0.0000 | 0.2009 | 1 |
| mouse | 0.0000 | 0.0000 | 2 |
| orange | 0.0000 | 0.0024 | 2 |
| parking meter | 0.0000 | 0.4447 | 1 |
| potted plant | 0.0000 | 0.0172 | 1 |
| remote | 0.0000 | 0.0201 | 1 |
| scissors | 0.0000 | 0.0172 | 2 |
| sheep | 0.0000 | 0.2543 | 2 |
| sink | 0.0000 | 0.0382 | 1 |
| skis | 0.0000 | 0.0096 | 1 |
| snowboard | 0.0000 | 0.0064 | 1 |
| surfboard | 0.0000 | 0.0486 | 1 |
| tennis racket | 0.0000 | 0.0000 | 1 |
| tie | 0.0000 | 0.0585 | 2 |
| toilet | 0.0000 | 0.1880 | 1 |
| traffic light | 0.0000 | 0.0013 | 1 |
| truck | 0.0000 | 0.0027 | 2 |
| tv | 0.0000 | 0.0000 | 1 |
| vase | 0.0000 | 0.0017 | 2 |
| zebra | 0.0000 | 0.0520 | 1 |
