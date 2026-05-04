# Official Layer-14 Clamped Adapter: 100 Train / 50 Unseen

- Result JSON: results/overfit_sanity/official_layer14_100train_50unseen_clamped/tiny_overfit_layer14.json
- Checkpoint: results/overfit_sanity/official_layer14_100train_50unseen_clamped/tiny_overfit_layer14.pth
- Benchmark: data/benchmarks/affordance_coco_val_three_prompts_test.json
- Official InternVL extraction: True
- Layer setup: [14], fusion: last
- Gate after delta: True, lambda_delta: 0.01
- Train/unseen overlap: 0

## Overall Metrics

| Split | Model | n | Top-1 Hit@0.5 | Top-5 Hit@0.5 | mean Top-1 IoU | mean Top-5 IoU |
|---|---|---:|---:|---:|---:|---:|
| Train | Baseline | 100 | 0.1700 | 0.4600 | 0.1822 | 0.4377 |
| Train | Trained | 100 | 0.1900 | 0.4600 | 0.1992 | 0.4436 |
| Unseen | Baseline | 50 | 0.1000 | 0.2600 | 0.1146 | 0.2959 |
| Unseen | Trained | 50 | 0.0800 | 0.3400 | 0.0916 | 0.3448 |

## Per-Prompt Metrics

| Split | Prompt | Model | n | Top-1 Hit@0.5 | Top-5 Hit@0.5 | mean Top-1 IoU | mean Top-5 IoU |
|---|---|---|---:|---:|---:|---:|---:|
| train | something to cut with . | Baseline | 30 | 0.0667 | 0.3667 | 0.0910 | 0.3671 |
| train | something to cut with . | Trained | 30 | 0.0667 | 0.3333 | 0.1038 | 0.3464 |
| train | something to drink from . | Baseline | 31 | 0.4194 | 0.9032 | 0.3789 | 0.7665 |
| train | something to drink from . | Trained | 31 | 0.5161 | 0.9355 | 0.4527 | 0.8091 |
| train | something to sit on . | Baseline | 39 | 0.0513 | 0.1795 | 0.0960 | 0.2306 |
| train | something to sit on . | Trained | 39 | 0.0256 | 0.1795 | 0.0709 | 0.2279 |
| unseen | something to cut with . | Baseline | 22 | 0.0455 | 0.0909 | 0.0606 | 0.1415 |
| unseen | something to cut with . | Trained | 22 | 0.0000 | 0.1364 | 0.0222 | 0.1736 |
| unseen | something to drink from . | Baseline | 15 | 0.2667 | 0.4667 | 0.2652 | 0.5327 |
| unseen | something to drink from . | Trained | 15 | 0.2667 | 0.6667 | 0.2462 | 0.6563 |
| unseen | something to sit on . | Baseline | 13 | 0.0000 | 0.3077 | 0.0322 | 0.2839 |
| unseen | something to sit on . | Trained | 13 | 0.0000 | 0.3077 | 0.0307 | 0.2751 |

## Gates And Delta Clamp

| Eval split | Decoder layer | gate | delta/memory | delta_norm | memory_norm |
|---|---:|---:|---:|---:|---:|
| train | 1 | 0.086036 | 0.500000 | 6.6179 | 13.2357 |
| train | 3 | 0.090344 | 0.500000 | 6.6179 | 13.2357 |
| train | 5 | 0.092371 | 0.411486 | 5.4463 | 13.2357 |
| unseen | 1 | 0.086036 | 0.500000 | 6.6173 | 13.2345 |
| unseen | 3 | 0.090344 | 0.500000 | 6.6173 | 13.2345 |
| unseen | 5 | 0.092371 | 0.411528 | 5.4464 | 13.2345 |

## Prediction Change

| Split | top1 box changed rate | top1 hit changed count | mean abs top1 score delta |
|---|---:|---:|---:|
| train | 0.2100 | 6 | 0.023416 |
| unseen | 0.1600 | 1 | 0.021678 |

## Success Checks

- train_top5_improves: False
- unseen_top5_improves: True
- train_mean_top5_iou_not_collapsed: True
- unseen_mean_top5_iou_not_collapsed: True
- train_delta_memory_le_0p5: True
- unseen_delta_memory_le_0p5: True
- train_unseen_overlap_zero: True

