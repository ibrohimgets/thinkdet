# Final 707-Sample Layer-14 Rank 0.2 Run

- Result JSON: results/overfit_sanity/official_layer14_707train_test707_rank02_combined/tiny_overfit_layer14.json
- Best checkpoint: results/overfit_sanity/official_layer14_707train_test707_rank02_combined/tiny_overfit_layer14_best.pth
- Final checkpoint: results/overfit_sanity/official_layer14_707train_test707_rank02_combined/tiny_overfit_layer14_final.pth
- Best step: 707
- Benchmark: data/benchmarks/affordance_coco_val_promptvar_weak5k_v2.json
- Selection metric: combined_top1
- Train/unseen image overlap: 0
- Win table: results/overfit_sanity/official_layer14_707train_test707_rank02_combined/win_table.csv
- Failure table: results/overfit_sanity/official_layer14_707train_test707_rank02_combined/failure_table.csv
- Qualitative directory: results/overfit_sanity/official_layer14_707train_test707_rank02_combined/qualitative

## Overall Metrics

| Split | Model | n | Top-1 Hit@0.5 | Top-5 Hit@0.5 | mean Top-1 IoU | mean Top-5 IoU |
|---|---|---:|---:|---:|---:|---:|
| train | DINO baseline | 707 | 0.2815 | 0.5219 | 0.2811 | 0.4918 |
| train | ThinkDet | 707 | 0.2984 | 0.5460 | 0.2873 | 0.5039 |
| unseen | DINO baseline | 707 | 0.3013 | 0.5446 | 0.3014 | 0.5151 |
| unseen | ThinkDet | 707 | 0.3168 | 0.5587 | 0.3058 | 0.5221 |

## Checkpoint Selection

| Step | Top-1 | Top-5 | Combined | mean Top-1 IoU | mean Top-5 IoU | Best |
|---:|---:|---:|---:|---:|---:|---|
| 200 | 0.3140 | 0.5601 | 0.8741 | 0.3108 | 0.5286 | True |
| 400 | 0.3211 | 0.5460 | 0.8670 | 0.3120 | 0.5172 | False |
| 600 | 0.3140 | 0.5587 | 0.8727 | 0.3075 | 0.5283 | False |
| 707 | 0.3168 | 0.5587 | 0.8755 | 0.3058 | 0.5221 | True |

## Gates And Delta Clamp

| Eval split | Decoder layer | gate | delta/memory | delta_norm | memory_norm |
|---|---:|---:|---:|---:|---:|
| train | 1 | 0.111184 | 0.500000 | 7.0220 | 14.0440 |
| train | 3 | 0.100060 | 0.500000 | 7.0220 | 14.0440 |
| train | 5 | 0.110943 | 0.500000 | 7.0220 | 14.0440 |
| unseen | 1 | 0.111184 | 0.500000 | 7.0485 | 14.0970 |
| unseen | 3 | 0.100060 | 0.500000 | 7.0485 | 14.0970 |
| unseen | 5 | 0.110943 | 0.500000 | 7.0485 | 14.0970 |

## Prediction Change

| Split | top1 box changed rate | top1 hit changed count | mean abs top1 score delta |
|---|---:|---:|---:|
| train | 0.6308 | 114 | 0.106661 |
| unseen | 0.5658 | 113 | 0.108902 |

## Per-Affordance Unseen Breakdown

| Affordance | n | DINO Top-1 | DINO Top-5 | ThinkDet Top-1 | ThinkDet Top-5 |
|---|---:|---:|---:|---:|---:|
| carry_in | 51 | 0.3137 | 0.6275 | 0.4706 | 0.6471 |
| control_with | 18 | 0.4444 | 0.7778 | 0.4444 | 0.6667 |
| cook_with | 16 | 0.3750 | 0.5625 | 0.1875 | 0.4375 |
| cut_with | 14 | 0.2857 | 0.3571 | 0.2857 | 0.4286 |
| drink_from | 34 | 0.4706 | 0.8529 | 0.4706 | 0.8235 |
| eat | 50 | 0.3200 | 0.5000 | 0.3600 | 0.6800 |
| eat_from | 36 | 0.3056 | 0.5000 | 0.2778 | 0.4722 |
| eat_with | 26 | 0.0769 | 0.2692 | 0.0769 | 0.4231 |
| play_with | 81 | 0.1235 | 0.4198 | 0.2840 | 0.6049 |
| read | 19 | 0.2632 | 0.5263 | 0.2632 | 0.6316 |
| ride | 60 | 0.3500 | 0.6167 | 0.3333 | 0.5833 |
| shelter_under | 45 | 0.1333 | 0.3778 | 0.1556 | 0.2889 |
| sit_on | 50 | 0.2400 | 0.5600 | 0.2400 | 0.4800 |
| sleep_on | 23 | 0.3043 | 0.5217 | 0.2609 | 0.4348 |
| talk_on | 30 | 0.5667 | 0.6667 | 0.5000 | 0.7333 |
| tell_time | 30 | 0.3333 | 0.6333 | 0.6333 | 0.7333 |
| travel_in | 67 | 0.3134 | 0.4627 | 0.2239 | 0.4179 |
| type_on | 22 | 0.4091 | 0.6818 | 0.2273 | 0.5909 |
| wash_at | 17 | 0.2941 | 0.4706 | 0.2353 | 0.4118 |
| watch | 18 | 0.6111 | 0.8333 | 0.4444 | 0.6667 |

## Per-Prompt Unseen Breakdown

| Prompt | n | DINO Top-1 | DINO Top-5 | ThinkDet Top-1 | ThinkDet Top-5 |
|---|---:|---:|---:|---:|---:|
| a bag for carrying things . | 11 | 0.5455 | 0.8182 | 0.5455 | 0.7273 |
| a basin for cleaning things . | 5 | 0.4000 | 0.8000 | 0.4000 | 0.6000 |
| a container for drinking . | 8 | 0.6250 | 0.8750 | 0.3750 | 0.7500 |
| a conveyance for travel . | 14 | 0.0714 | 0.4286 | 0.0000 | 0.1429 |
| a cutting tool . | 5 | 0.6000 | 0.6000 | 0.6000 | 1.0000 |
| a device controller . | 5 | 0.8000 | 1.0000 | 0.8000 | 1.0000 |
| a device to enter text on . | 5 | 0.4000 | 0.6000 | 0.0000 | 0.6000 |
| a device to talk on . | 6 | 0.3333 | 0.5000 | 0.1667 | 0.5000 |
| a display to look at . | 4 | 0.5000 | 0.7500 | 0.5000 | 0.7500 |
| a phone for talking . | 10 | 0.7000 | 0.8000 | 0.7000 | 0.9000 |
| a place to sit . | 8 | 0.1250 | 0.3750 | 0.1250 | 0.2500 |
| a place to sleep . | 9 | 0.2222 | 0.2222 | 0.2222 | 0.2222 |
| a place to wash hands or dishes . | 4 | 0.2500 | 0.2500 | 0.0000 | 0.5000 |
| a screen for watching . | 4 | 1.0000 | 1.0000 | 0.7500 | 1.0000 |
| a surface or container for eating . | 9 | 0.6667 | 0.7778 | 0.5556 | 0.6667 |
| a thing for athletic play . | 14 | 0.1429 | 0.5714 | 0.5000 | 0.7857 |
| a thing for eating food with . | 2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| a thing for keyboard input . | 8 | 0.7500 | 1.0000 | 0.6250 | 1.0000 |
| a thing for preparing hot food . | 6 | 0.1667 | 0.3333 | 0.0000 | 0.3333 |
| a thing for protection overhead . | 12 | 0.3333 | 0.6667 | 0.4167 | 0.7500 |
| a thing for riding . | 9 | 0.2222 | 0.6667 | 0.1111 | 0.2222 |
| a thing to carry items in . | 11 | 0.1818 | 0.5455 | 0.3636 | 0.4545 |
| a thing to cut with . | 2 | 0.0000 | 0.5000 | 0.5000 | 0.5000 |
| a thing to drink out of . | 8 | 0.3750 | 0.7500 | 0.5000 | 0.8750 |
| a thing to eat food from . | 7 | 0.0000 | 0.1429 | 0.0000 | 0.2857 |
| a thing to read . | 4 | 0.2500 | 0.2500 | 0.2500 | 0.5000 |
| a thing used for sitting . | 10 | 0.2000 | 0.7000 | 0.3000 | 0.7000 |
| a thing used to know the time . | 4 | 0.0000 | 0.7500 | 0.7500 | 0.7500 |
| a time-telling object . | 4 | 0.2500 | 0.2500 | 0.5000 | 0.7500 |
| a utensil to eat with . | 3 | 0.6667 | 1.0000 | 0.3333 | 1.0000 |
| a vehicle for transportation . | 7 | 0.7143 | 0.8571 | 0.7143 | 0.8571 |
| a vehicle or animal to ride . | 16 | 0.5000 | 0.6875 | 0.5625 | 0.7500 |
| an appliance for cooking . | 5 | 0.8000 | 1.0000 | 0.6000 | 0.8000 |
| an edible thing . | 5 | 0.2000 | 0.6000 | 0.8000 | 1.0000 |
| an input device for controlling something . | 2 | 0.5000 | 1.0000 | 0.0000 | 0.5000 |
| equipment for playing sports . | 14 | 0.1429 | 0.5000 | 0.2857 | 0.7857 |
| food to eat . | 16 | 0.2500 | 0.5625 | 0.2500 | 0.8750 |
| furniture for sleeping . | 1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| furniture to sit on . | 14 | 0.4286 | 0.7857 | 0.5000 | 0.7857 |
| reading material . | 3 | 0.3333 | 1.0000 | 0.6667 | 1.0000 |
| something food can be eaten from . | 12 | 0.1667 | 0.2500 | 0.0833 | 0.1667 |
| something for a meal or snack . | 14 | 0.5714 | 0.6429 | 0.6429 | 0.7143 |
| something people can ride inside . | 22 | 0.2273 | 0.3182 | 0.0455 | 0.2727 |
| something that shows the time . | 8 | 0.5000 | 0.7500 | 0.7500 | 0.8750 |
| something to carry things in . | 7 | 0.2857 | 0.7143 | 0.4286 | 0.7143 |
| something to control a device with . | 5 | 0.4000 | 0.8000 | 0.6000 | 0.8000 |
| something to cook with . | 2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| something to cut with . | 1 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| something to drink from . | 6 | 0.1667 | 1.0000 | 0.5000 | 0.8333 |
| something to eat . | 6 | 0.0000 | 0.1667 | 0.0000 | 0.0000 |
| something to eat from . | 4 | 0.0000 | 0.7500 | 0.2500 | 0.7500 |
| something to eat with . | 10 | 0.0000 | 0.1000 | 0.1000 | 0.4000 |
| something to hold over yourself in bad weather . | 13 | 0.1538 | 0.4615 | 0.1538 | 0.2308 |
| something to play a sport with . | 21 | 0.0000 | 0.1429 | 0.0476 | 0.3333 |
| something to read . | 6 | 0.3333 | 0.3333 | 0.1667 | 0.5000 |
| something to ride . | 11 | 0.0909 | 0.4545 | 0.1818 | 0.6364 |
| something to ride on . | 14 | 0.5000 | 0.5714 | 0.3571 | 0.6429 |
| something to shelter under . | 6 | 0.0000 | 0.1667 | 0.0000 | 0.0000 |
| something to sit on . | 9 | 0.2222 | 0.3333 | 0.0000 | 0.1111 |
| something to sleep on . | 5 | 0.2000 | 0.4000 | 0.0000 | 0.4000 |
| something to talk on . | 4 | 0.5000 | 0.7500 | 0.5000 | 0.7500 |
| something to tell time with . | 5 | 0.2000 | 0.4000 | 0.4000 | 0.4000 |
| something to travel in . | 14 | 0.5714 | 0.7143 | 0.5000 | 0.7857 |
| something to type on . | 2 | 0.0000 | 0.5000 | 0.0000 | 0.0000 |
| something to watch . | 2 | 0.0000 | 0.5000 | 0.0000 | 0.0000 |
| something used as a bed or resting place . | 4 | 0.7500 | 1.0000 | 0.7500 | 1.0000 |
| something used for cover from rain . | 5 | 0.0000 | 0.2000 | 0.0000 | 0.2000 |
| something used for cutting . | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| something used for drinking . | 5 | 0.6000 | 0.8000 | 0.4000 | 0.8000 |
| something used for eating . | 6 | 0.0000 | 0.3333 | 0.0000 | 0.3333 |
| something used for reading . | 4 | 0.2500 | 0.7500 | 0.2500 | 0.7500 |
| something used for typing . | 3 | 0.3333 | 0.3333 | 0.0000 | 0.3333 |
| something used for viewing shows . | 3 | 0.6667 | 1.0000 | 0.6667 | 1.0000 |
| something used for washing . | 1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| something used in a game . | 14 | 0.2143 | 0.5000 | 0.3571 | 0.6429 |
| something used to carry things . | 7 | 0.2857 | 0.5714 | 0.7143 | 0.7143 |
| something used to control a computer or screen . | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| something used to move from place to place . | 10 | 0.2000 | 0.2000 | 0.2000 | 0.3000 |
| something used to serve food for eating . | 4 | 0.7500 | 1.0000 | 0.7500 | 1.0000 |
| something used to talk on . | 6 | 0.6667 | 0.6667 | 0.5000 | 0.8333 |
| something you can carry things in . | 15 | 0.2667 | 0.5333 | 0.4000 | 0.6667 |
| something you can drink from . | 7 | 0.5714 | 0.8571 | 0.5714 | 0.8571 |
| something you can eat . | 9 | 0.3333 | 0.3333 | 0.1111 | 0.5556 |
| something you can lie on to rest . | 4 | 0.0000 | 0.7500 | 0.0000 | 0.2500 |
| something you can read . | 2 | 0.0000 | 0.5000 | 0.0000 | 0.5000 |
| something you can read time from . | 9 | 0.4444 | 0.7778 | 0.6667 | 0.7778 |
| something you can ride . | 10 | 0.3000 | 0.7000 | 0.3000 | 0.5000 |
| something you can sit on . | 9 | 0.1111 | 0.4444 | 0.1111 | 0.3333 |
| something you can stand under for shade . | 9 | 0.0000 | 0.1111 | 0.0000 | 0.0000 |
| something you can talk on . | 4 | 0.5000 | 0.5000 | 0.5000 | 0.5000 |
| something you can type with . | 4 | 0.0000 | 0.5000 | 0.0000 | 0.2500 |
| something you can use to cut . | 2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| something you can use to eat . | 5 | 0.0000 | 0.2000 | 0.0000 | 0.4000 |
| something you can use to heat food . | 3 | 0.3333 | 0.6667 | 0.0000 | 0.3333 |
| something you can use to point or select . | 5 | 0.2000 | 0.6000 | 0.2000 | 0.4000 |
| something you can watch video on . | 5 | 0.6000 | 0.8000 | 0.2000 | 0.4000 |
| somewhere to wash things . | 4 | 0.2500 | 0.2500 | 0.2500 | 0.2500 |
| somewhere water is used for cleaning . | 3 | 0.0000 | 0.3333 | 0.0000 | 0.0000 |
| sports equipment to play with . | 18 | 0.1667 | 0.5000 | 0.3333 | 0.6111 |

## Win/Failure Tables

- wins: 115 rows, counts={'top1_win': 62, 'top5_win': 53}
- failures: 92 rows, counts={'top1_failure': 51, 'top5_failure': 41}

## Qualitative Images

- results/overfit_sanity/official_layer14_707train_test707_rank02_combined/qualitative/qual_01_top1_win_386352.jpg
- results/overfit_sanity/official_layer14_707train_test707_rank02_combined/qualitative/qual_02_top1_win_25603.jpg
- results/overfit_sanity/official_layer14_707train_test707_rank02_combined/qualitative/qual_03_top1_win_480275.jpg
- results/overfit_sanity/official_layer14_707train_test707_rank02_combined/qualitative/qual_04_top1_win_313454.jpg
- results/overfit_sanity/official_layer14_707train_test707_rank02_combined/qualitative/qual_05_top1_win_377723.jpg

## Success Checks

- unseen_top1_improves: True
- unseen_top5_improves: True
- unseen_mean_top5_iou_not_collapsed: True
- train_delta_memory_le_0p5: True
- unseen_delta_memory_le_0p5: True
- train_unseen_overlap_zero: True
- prediction_change_rate_controlled_threshold: 0.5
- train_prediction_change_rate_controlled: False
- unseen_prediction_change_rate_controlled: False

