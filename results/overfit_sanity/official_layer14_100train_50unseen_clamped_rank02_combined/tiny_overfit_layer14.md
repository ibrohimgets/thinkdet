# Official Layer-14 Clamped Adapter: lambda_rank=0.2 Combined Selection

- Result JSON: results/overfit_sanity/official_layer14_100train_50unseen_clamped_rank02_combined/tiny_overfit_layer14.json
- Best checkpoint: results/overfit_sanity/official_layer14_100train_50unseen_clamped_rank02_combined/tiny_overfit_layer14_best.pth
- Final checkpoint: results/overfit_sanity/official_layer14_100train_50unseen_clamped_rank02_combined/tiny_overfit_layer14_final.pth
- Best step: 80
- Selection metric: combined_top1
- Benchmark: data/benchmarks/affordance_coco_val_three_prompts_test.json
- Official InternVL extraction: True
- lambda_rank: 0.2, rank_margin: 0.5
- lambda_delta: 0.01, gate_after_delta: True
- Train/unseen overlap: 0

## Overall Metrics

| Split | Model | n | Top-1 Hit@0.5 | Top-5 Hit@0.5 | mean Top-1 IoU | mean Top-5 IoU |
|---|---|---:|---:|---:|---:|---:|
| Train | DINO baseline | 100 | 0.1700 | 0.4600 | 0.1822 | 0.4377 |
| Train | Trained | 100 | 0.1700 | 0.4800 | 0.1866 | 0.4600 |
| Unseen | DINO baseline | 50 | 0.1000 | 0.2600 | 0.1146 | 0.2959 |
| Unseen | Trained | 50 | 0.1000 | 0.3200 | 0.1084 | 0.3396 |

## Per-Prompt Top-1 / Top-5

| Split | Prompt | Baseline Top-1 | Baseline Top-5 | Trained Top-1 | Trained Top-5 |
|---|---|---:|---:|---:|---:|
| train | something to cut with . | 0.0667 | 0.3667 | 0.0333 | 0.4000 |
| train | something to drink from . | 0.4194 | 0.9032 | 0.4194 | 0.9355 |
| train | something to sit on . | 0.0513 | 0.1795 | 0.0769 | 0.1795 |
| unseen | something to cut with . | 0.0455 | 0.0909 | 0.0000 | 0.1364 |
| unseen | something to drink from . | 0.2667 | 0.4667 | 0.3333 | 0.6000 |
| unseen | something to sit on . | 0.0000 | 0.3077 | 0.0000 | 0.3077 |

## Checkpoint Selection

| Step | Top-1 | Top-5 | Combined | mean Top-1 IoU | mean Top-5 IoU | Best |
|---:|---:|---:|---:|---:|---:|---|
| 20 | 0.1000 | 0.2800 | 0.3800 | 0.1177 | 0.3015 | True |
| 40 | 0.1000 | 0.2600 | 0.3600 | 0.1100 | 0.3017 | False |
| 60 | 0.1000 | 0.2600 | 0.3600 | 0.1086 | 0.3086 | False |
| 80 | 0.1000 | 0.3200 | 0.4200 | 0.1084 | 0.3396 | True |
| 100 | 0.1200 | 0.2800 | 0.4000 | 0.1373 | 0.3272 | False |

## Gates And Delta Clamp

| Eval split | Decoder layer | gate | delta/memory | delta_norm | memory_norm |
|---|---:|---:|---:|---:|---:|
| train | 1 | 0.101544 | 0.500000 | 6.6179 | 13.2357 |
| train | 3 | 0.106850 | 0.500000 | 6.6179 | 13.2357 |
| train | 5 | 0.110506 | 0.500000 | 6.6179 | 13.2357 |
| unseen | 1 | 0.101544 | 0.500000 | 6.6173 | 13.2345 |
| unseen | 3 | 0.106850 | 0.500000 | 6.6173 | 13.2345 |
| unseen | 5 | 0.110506 | 0.500000 | 6.6173 | 13.2345 |

## Prediction Change

| Split | top1 box changed rate | top1 hit changed count | mean abs top1 score delta |
|---|---:|---:|---:|
| train | 0.2300 | 4 | 0.054546 |
| unseen | 0.2200 | 2 | 0.051955 |

## Comparison

| Run | best step | lambda_rank | selection | unseen Top-1 | unseen Top-5 | combined | mean Top-1 IoU | mean Top-5 IoU | pred change |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| clamped_no_rank |  | 0.0 | none | 0.0800 | 0.3400 | 0.4200 | 0.0916 | 0.3448 | 0.1600 |
| clamped_rank_0p5 | 20 | 0.5 | none | 0.1200 | 0.2800 | 0.4000 | 0.1298 | 0.3052 | 0.1800 |
| clamped_rank_0p2_combined | 80 | 0.2 | combined_top1 | 0.1000 | 0.3200 | 0.4200 | 0.1084 | 0.3396 | 0.2200 |

## Success Checks

- unseen_top1_ge_baseline: True
- unseen_top5_above_baseline: True
- unseen_mean_top5_iou_not_collapsed: True
- train_delta_memory_le_0p5: True
- unseen_delta_memory_le_0p5: True
- train_unseen_overlap_zero: True
- prediction_change_rate_controlled_threshold: 0.5
- train_prediction_change_rate_controlled: True
- unseen_prediction_change_rate_controlled: True

