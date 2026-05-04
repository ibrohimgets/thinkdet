# Official Layer-14 Clamped Adapter With Ranking Loss

- Result JSON: results/overfit_sanity/official_layer14_100train_50unseen_clamped_rank/tiny_overfit_layer14.json
- Best checkpoint: results/overfit_sanity/official_layer14_100train_50unseen_clamped_rank/tiny_overfit_layer14_best.pth
- Final checkpoint: results/overfit_sanity/official_layer14_100train_50unseen_clamped_rank/tiny_overfit_layer14_final.pth
- Best step: 20
- Benchmark: data/benchmarks/affordance_coco_val_three_prompts_test.json
- Official InternVL extraction: True
- lambda_rank: 0.5, rank_margin: 0.5
- lambda_delta: 0.01, gate_after_delta: True
- Train/unseen overlap: 0

## Overall Metrics

| Split | Model | n | Top-1 Hit@0.5 | Top-5 Hit@0.5 | mean Top-1 IoU | mean Top-5 IoU |
|---|---|---:|---:|---:|---:|---:|
| Train | Baseline | 100 | 0.1700 | 0.4600 | 0.1822 | 0.4377 |
| Train | Trained | 100 | 0.1900 | 0.4700 | 0.1927 | 0.4452 |
| Unseen | Baseline | 50 | 0.1000 | 0.2600 | 0.1146 | 0.2959 |
| Unseen | Trained | 50 | 0.1200 | 0.2800 | 0.1298 | 0.3052 |

## Per-Prompt Metrics

| Split | Prompt | Model | n | Top-1 Hit@0.5 | Top-5 Hit@0.5 | mean Top-1 IoU | mean Top-5 IoU |
|---|---|---|---:|---:|---:|---:|---:|
| train | something to cut with . | Baseline | 30 | 0.0667 | 0.3667 | 0.0910 | 0.3671 |
| train | something to cut with . | Trained | 30 | 0.0667 | 0.4000 | 0.0964 | 0.3881 |
| train | something to drink from . | Baseline | 31 | 0.4194 | 0.9032 | 0.3789 | 0.7665 |
| train | something to drink from . | Trained | 31 | 0.4839 | 0.9032 | 0.4162 | 0.7735 |
| train | something to sit on . | Baseline | 39 | 0.0513 | 0.1795 | 0.0960 | 0.2306 |
| train | something to sit on . | Trained | 39 | 0.0513 | 0.1795 | 0.0891 | 0.2281 |
| unseen | something to cut with . | Baseline | 22 | 0.0455 | 0.0909 | 0.0606 | 0.1415 |
| unseen | something to cut with . | Trained | 22 | 0.0455 | 0.0909 | 0.0507 | 0.1502 |
| unseen | something to drink from . | Baseline | 15 | 0.2667 | 0.4667 | 0.2652 | 0.5327 |
| unseen | something to drink from . | Trained | 15 | 0.3333 | 0.6667 | 0.3226 | 0.6539 |
| unseen | something to sit on . | Baseline | 13 | 0.0000 | 0.3077 | 0.0322 | 0.2839 |
| unseen | something to sit on . | Trained | 13 | 0.0000 | 0.1538 | 0.0412 | 0.1652 |

## Checkpoint Selection

| Step | Top-1 Hit@0.5 | Top-5 Hit@0.5 | mean Top-1 IoU | mean Top-5 IoU | Best |
|---:|---:|---:|---:|---:|---|
| 20 | 0.1200 | 0.2800 | 0.1298 | 0.3052 | True |
| 40 | 0.0800 | 0.2400 | 0.0891 | 0.2739 | False |
| 60 | 0.1200 | 0.2600 | 0.1278 | 0.2845 | False |
| 80 | 0.1200 | 0.2200 | 0.1297 | 0.2494 | False |
| 100 | 0.1200 | 0.2200 | 0.1269 | 0.2675 | False |

## Gates And Delta Clamp

| Eval split | Decoder layer | gate | delta/memory | delta_norm | memory_norm |
|---|---:|---:|---:|---:|---:|
| train | 1 | 0.090371 | 0.192964 | 2.5540 | 13.2357 |
| train | 3 | 0.095939 | 0.285397 | 3.7774 | 13.2357 |
| train | 5 | 0.104329 | 0.500000 | 6.6179 | 13.2357 |
| unseen | 1 | 0.090371 | 0.193011 | 2.5544 | 13.2345 |
| unseen | 3 | 0.095939 | 0.285438 | 3.7776 | 13.2345 |
| unseen | 5 | 0.104329 | 0.500000 | 6.6173 | 13.2345 |

## Prediction Change

| Split | top1 box changed rate | top1 hit changed count | mean abs top1 score delta |
|---|---:|---:|---:|
| train | 0.1800 | 2 | 0.078152 |
| unseen | 0.1800 | 3 | 0.073185 |

## Success Checks

- unseen_top1_improves: True
- unseen_top5_holds_or_improves: True
- unseen_mean_top5_iou_not_collapsed: True
- train_delta_memory_le_0p5: True
- unseen_delta_memory_le_0p5: True
- train_unseen_overlap_zero: True
- prediction_change_rate_controlled_threshold: 0.5
- train_prediction_change_rate_controlled: True
- unseen_prediction_change_rate_controlled: True

