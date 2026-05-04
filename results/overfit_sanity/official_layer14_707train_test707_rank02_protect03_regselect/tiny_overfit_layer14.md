# Final 707-Sample Layer-14 Protected Hard-Negative Run

- Result JSON: results/overfit_sanity/official_layer14_707train_test707_rank02_protect03_regselect/tiny_overfit_layer14.json
- Analysis JSON: results/overfit_sanity/official_layer14_707train_test707_rank02_protect03_regselect/analysis_summary.json
- Best checkpoint: results/overfit_sanity/official_layer14_707train_test707_rank02_protect03_regselect/tiny_overfit_layer14_best.pth
- Final checkpoint: results/overfit_sanity/official_layer14_707train_test707_rank02_protect03_regselect/tiny_overfit_layer14_final.pth
- Best step: 200
- Benchmark: data/benchmarks/affordance_coco_val_promptvar_weak5k_v2.json
- Selection metric: combined_regression
- Train/unseen image overlap: 0
- Same train images as previous final: True
- Same unseen images as previous final: True
- Win table: results/overfit_sanity/official_layer14_707train_test707_rank02_protect03_regselect/win_table.csv
- Failure table: results/overfit_sanity/official_layer14_707train_test707_rank02_protect03_regselect/failure_table.csv
- Qualitative wins: results/overfit_sanity/official_layer14_707train_test707_rank02_protect03_regselect/qualitative
- Qualitative failures: results/overfit_sanity/official_layer14_707train_test707_rank02_protect03_regselect/qualitative_failures

## Overall Metrics

| Split | Model | n | Top-1 Hit@0.5 | Top-5 Hit@0.5 | mean Top-1 IoU | mean Top-5 IoU |
| --- | --- | --- | --- | --- | --- | --- |
| train | DINO baseline | 707 | 0.2815 | 0.5219 | 0.2811 | 0.4918 |
| train | ThinkDet | 707 | 0.3083 | 0.5615 | 0.2987 | 0.5265 |
| unseen | DINO baseline | 707 | 0.3013 | 0.5446 | 0.3014 | 0.5151 |
| unseen | ThinkDet | 707 | 0.3267 | 0.5785 | 0.3222 | 0.5418 |

## Previous Final Comparison

| Run | Top-1 | Top-5 | Top-1 gain vs DINO | Top-5 gain vs DINO | mean Top-1 IoU | mean Top-5 IoU | prediction change | regression count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| previous rank0.2 combined | 0.3168 | 0.5587 | 1.56 pp | 1.41 pp | 0.3058 | 0.5221 | 56.58% | 100 |
| protected hard-negative | 0.3267 | 0.5785 | 2.55 pp | 3.39 pp | 0.3222 | 0.5418 | 34.94% | 29 |

## Checkpoint Selection

| Step | Top-1 | Top-5 | Top-1+Top-5 | Regressions | score | Best |
| --- | --- | --- | --- | --- | --- | --- |
| 200 | 0.3267 | 0.5785 | 0.9052 | 29 | 0.8642 | True |
| 400 | 0.3239 | 0.5530 | 0.8769 | 24 | 0.8430 | False |
| 600 | 0.3465 | 0.5474 | 0.8939 | 50 | 0.8232 | False |
| 707 | 0.3380 | 0.5431 | 0.8812 | 50 | 0.8105 | False |

## Regression And Movement

| Metric | Current | Previous final |
| --- | --- | --- |
| Top-1 regressions | 13 | 51 |
| Top-5 regressions | 16 | 49 |
| Regression count | 29 | 100 |
| Upward movements | 71 | 115 |
| Downward movements | 29 | 92 |

| DINO bucket -> ThinkDet bucket | missing_top5 | top5_only | top1 |
| --- | --- | --- | --- |
| missing_top5 | 282 | 40 | 0 |
| top5_only | 16 | 125 | 31 |
| top1 | 0 | 13 | 200 |

## Wins Vs Failures

| Table | total | top1 | top5 |
| --- | --- | --- | --- |
| wins | 71 | 31 | 40 |
| failures | 29 | 13 | 16 |

## Gates And Delta Clamp

| Eval split | Decoder layer | gate | delta/memory | delta_norm | memory_norm |
| --- | --- | --- | --- | --- | --- |
| train | 1 | 0.110505 | 0.500000 | 7.0220 | 14.0440 |
| train | 3 | 0.100757 | 0.500000 | 7.0220 | 14.0440 |
| train | 5 | 0.103541 | 0.500000 | 7.0220 | 14.0440 |
| unseen | 1 | 0.110505 | 0.500000 | 7.0485 | 14.0970 |
| unseen | 3 | 0.100757 | 0.500000 | 7.0485 | 14.0970 |
| unseen | 5 | 0.103541 | 0.500000 | 7.0485 | 14.0970 |

## Prediction Change

| Split | top1 box changed rate | top1 hit changed count | mean abs top1 score delta |
| --- | --- | --- | --- |
| train | 39.75% | 49 | 0.0794 |
| unseen | 34.94% | 44 | 0.0810 |

## Per-Affordance Unseen Breakdown

| Affordance | n | DINO Top-1 | DINO Top-5 | ThinkDet Top-1 | ThinkDet Top-5 | Delta Top-1 | Delta Top-5 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| carry_in | 51 | 0.3137 | 0.6275 | 0.3725 | 0.7059 | 5.9 pp | 7.8 pp |
| control_with | 18 | 0.4444 | 0.7778 | 0.5000 | 0.7222 | 5.6 pp | -5.6 pp |
| cook_with | 16 | 0.3750 | 0.5625 | 0.3125 | 0.5000 | -6.2 pp | -6.2 pp |
| cut_with | 14 | 0.2857 | 0.3571 | 0.2857 | 0.4286 | 0.0 pp | 7.1 pp |
| drink_from | 34 | 0.4706 | 0.8529 | 0.5588 | 0.8824 | 8.8 pp | 2.9 pp |
| eat | 50 | 0.3200 | 0.5000 | 0.3600 | 0.5400 | 4.0 pp | 4.0 pp |
| eat_from | 36 | 0.3056 | 0.5000 | 0.3056 | 0.5000 | 0.0 pp | 0.0 pp |
| eat_with | 26 | 0.0769 | 0.2692 | 0.1154 | 0.2308 | 3.8 pp | -3.8 pp |
| play_with | 81 | 0.1235 | 0.4198 | 0.1852 | 0.6173 | 6.2 pp | 19.8 pp |
| read | 19 | 0.2632 | 0.5263 | 0.3158 | 0.5789 | 5.3 pp | 5.3 pp |
| ride | 60 | 0.3500 | 0.6167 | 0.3333 | 0.6167 | -1.7 pp | 0.0 pp |
| shelter_under | 45 | 0.1333 | 0.3778 | 0.1556 | 0.3111 | 2.2 pp | -6.7 pp |
| sit_on | 50 | 0.2400 | 0.5600 | 0.2400 | 0.5000 | 0.0 pp | -6.0 pp |
| sleep_on | 23 | 0.3043 | 0.5217 | 0.3043 | 0.4783 | 0.0 pp | -4.3 pp |
| talk_on | 30 | 0.5667 | 0.6667 | 0.6333 | 0.7333 | 6.7 pp | 6.7 pp |
| tell_time | 30 | 0.3333 | 0.6333 | 0.4667 | 0.7333 | 13.3 pp | 10.0 pp |
| travel_in | 67 | 0.3134 | 0.4627 | 0.3134 | 0.4776 | 0.0 pp | 1.5 pp |
| type_on | 22 | 0.4091 | 0.6818 | 0.3636 | 0.7273 | -4.5 pp | 4.5 pp |
| wash_at | 17 | 0.2941 | 0.4706 | 0.2353 | 0.5882 | -5.9 pp | 11.8 pp |
| watch | 18 | 0.6111 | 0.8333 | 0.5556 | 0.8333 | -5.6 pp | 0.0 pp |

## Per-Prompt Unseen Breakdown

| Prompt | n | DINO Top-1 | DINO Top-5 | ThinkDet Top-1 | ThinkDet Top-5 | Delta Top-1 | Delta Top-5 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| a bag for carrying things . | 11 | 0.5455 | 0.8182 | 0.4545 | 0.8182 | -9.1 pp | 0.0 pp |
| a basin for cleaning things . | 5 | 0.4000 | 0.8000 | 0.4000 | 0.8000 | 0.0 pp | 0.0 pp |
| a container for drinking . | 8 | 0.6250 | 0.8750 | 0.6250 | 0.8750 | 0.0 pp | 0.0 pp |
| a conveyance for travel . | 14 | 0.0714 | 0.4286 | 0.0714 | 0.2857 | 0.0 pp | -14.3 pp |
| a cutting tool . | 5 | 0.6000 | 0.6000 | 0.6000 | 0.8000 | 0.0 pp | 20.0 pp |
| a device controller . | 5 | 0.8000 | 1.0000 | 1.0000 | 1.0000 | 20.0 pp | 0.0 pp |
| a device to enter text on . | 5 | 0.4000 | 0.6000 | 0.4000 | 0.6000 | 0.0 pp | 0.0 pp |
| a device to talk on . | 6 | 0.3333 | 0.5000 | 0.5000 | 0.5000 | 16.7 pp | 0.0 pp |
| a display to look at . | 4 | 0.5000 | 0.7500 | 0.5000 | 0.7500 | 0.0 pp | 0.0 pp |
| a phone for talking . | 10 | 0.7000 | 0.8000 | 0.8000 | 0.9000 | 10.0 pp | 10.0 pp |
| a place to sit . | 8 | 0.1250 | 0.3750 | 0.1250 | 0.3750 | 0.0 pp | 0.0 pp |
| a place to sleep . | 9 | 0.2222 | 0.2222 | 0.2222 | 0.2222 | 0.0 pp | 0.0 pp |
| a place to wash hands or dishes . | 4 | 0.2500 | 0.2500 | 0.0000 | 0.5000 | -25.0 pp | 25.0 pp |
| a screen for watching . | 4 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0 pp | 0.0 pp |
| a surface or container for eating . | 9 | 0.6667 | 0.7778 | 0.6667 | 0.7778 | 0.0 pp | 0.0 pp |
| a thing for athletic play . | 14 | 0.1429 | 0.5714 | 0.2143 | 0.7857 | 7.1 pp | 21.4 pp |
| a thing for eating food with . | 2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0 pp | 0.0 pp |
| a thing for keyboard input . | 8 | 0.7500 | 1.0000 | 0.7500 | 1.0000 | 0.0 pp | 0.0 pp |
| a thing for preparing hot food . | 6 | 0.1667 | 0.3333 | 0.0000 | 0.3333 | -16.7 pp | 0.0 pp |
| a thing for protection overhead . | 12 | 0.3333 | 0.6667 | 0.4167 | 0.6667 | 8.3 pp | 0.0 pp |
| a thing for riding . | 9 | 0.2222 | 0.6667 | 0.2222 | 0.4444 | 0.0 pp | -22.2 pp |
| a thing to carry items in . | 11 | 0.1818 | 0.5455 | 0.2727 | 0.5455 | 9.1 pp | 0.0 pp |
| a thing to cut with . | 2 | 0.0000 | 0.5000 | 0.5000 | 0.5000 | 50.0 pp | 0.0 pp |
| a thing to drink out of . | 8 | 0.3750 | 0.7500 | 0.3750 | 0.8750 | 0.0 pp | 12.5 pp |
| a thing to eat food from . | 7 | 0.0000 | 0.1429 | 0.0000 | 0.1429 | 0.0 pp | 0.0 pp |
| a thing to read . | 4 | 0.2500 | 0.2500 | 0.2500 | 0.2500 | 0.0 pp | 0.0 pp |
| a thing used for sitting . | 10 | 0.2000 | 0.7000 | 0.2000 | 0.6000 | 0.0 pp | -10.0 pp |
| a thing used to know the time . | 4 | 0.0000 | 0.7500 | 0.7500 | 0.7500 | 75.0 pp | 0.0 pp |
| a time-telling object . | 4 | 0.2500 | 0.2500 | 0.2500 | 0.5000 | 0.0 pp | 25.0 pp |
| a utensil to eat with . | 3 | 0.6667 | 1.0000 | 0.6667 | 1.0000 | 0.0 pp | 0.0 pp |
| a vehicle for transportation . | 7 | 0.7143 | 0.8571 | 0.7143 | 0.8571 | 0.0 pp | 0.0 pp |
| a vehicle or animal to ride . | 16 | 0.5000 | 0.6875 | 0.5000 | 0.7500 | 0.0 pp | 6.2 pp |
| an appliance for cooking . | 5 | 0.8000 | 1.0000 | 0.8000 | 1.0000 | 0.0 pp | 0.0 pp |
| an edible thing . | 5 | 0.2000 | 0.6000 | 0.4000 | 0.6000 | 20.0 pp | 0.0 pp |
| an input device for controlling something . | 2 | 0.5000 | 1.0000 | 0.5000 | 0.5000 | 0.0 pp | -50.0 pp |
| equipment for playing sports . | 14 | 0.1429 | 0.5000 | 0.2143 | 0.7143 | 7.1 pp | 21.4 pp |
| food to eat . | 16 | 0.2500 | 0.5625 | 0.2500 | 0.6875 | 0.0 pp | 12.5 pp |
| furniture for sleeping . | 1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0 pp | 0.0 pp |
| furniture to sit on . | 14 | 0.4286 | 0.7857 | 0.5000 | 0.7857 | 7.1 pp | 0.0 pp |
| reading material . | 3 | 0.3333 | 1.0000 | 0.6667 | 1.0000 | 33.3 pp | 0.0 pp |
| something food can be eaten from . | 12 | 0.1667 | 0.2500 | 0.1667 | 0.2500 | 0.0 pp | 0.0 pp |
| something for a meal or snack . | 14 | 0.5714 | 0.6429 | 0.6429 | 0.6429 | 7.1 pp | 0.0 pp |
| something people can ride inside . | 22 | 0.2273 | 0.3182 | 0.1818 | 0.4091 | -4.5 pp | 9.1 pp |
| something that shows the time . | 8 | 0.5000 | 0.7500 | 0.6250 | 0.8750 | 12.5 pp | 12.5 pp |
| something to carry things in . | 7 | 0.2857 | 0.7143 | 0.2857 | 0.7143 | 0.0 pp | 0.0 pp |
| something to control a device with . | 5 | 0.4000 | 0.8000 | 0.4000 | 0.8000 | 0.0 pp | 0.0 pp |
| something to cook with . | 2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0 pp | 0.0 pp |
| something to cut with . | 1 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | -100.0 pp | 0.0 pp |
| something to drink from . | 6 | 0.1667 | 1.0000 | 0.5000 | 1.0000 | 33.3 pp | 0.0 pp |
| something to eat . | 6 | 0.0000 | 0.1667 | 0.0000 | 0.1667 | 0.0 pp | 0.0 pp |
| something to eat from . | 4 | 0.0000 | 0.7500 | 0.0000 | 0.7500 | 0.0 pp | 0.0 pp |
| something to eat with . | 10 | 0.0000 | 0.1000 | 0.1000 | 0.1000 | 10.0 pp | 0.0 pp |
| something to hold over yourself in bad weather . | 13 | 0.1538 | 0.4615 | 0.1538 | 0.3846 | 0.0 pp | -7.7 pp |
| something to play a sport with . | 21 | 0.0000 | 0.1429 | 0.0000 | 0.3810 | 0.0 pp | 23.8 pp |
| something to read . | 6 | 0.3333 | 0.3333 | 0.3333 | 0.3333 | 0.0 pp | 0.0 pp |
| something to ride . | 11 | 0.0909 | 0.4545 | 0.0909 | 0.4545 | 0.0 pp | 0.0 pp |
| something to ride on . | 14 | 0.5000 | 0.5714 | 0.5000 | 0.6429 | 0.0 pp | 7.1 pp |
| something to shelter under . | 6 | 0.0000 | 0.1667 | 0.0000 | 0.0000 | 0.0 pp | -16.7 pp |
| something to sit on . | 9 | 0.2222 | 0.3333 | 0.1111 | 0.3333 | -11.1 pp | 0.0 pp |
| something to sleep on . | 5 | 0.2000 | 0.4000 | 0.2000 | 0.4000 | 0.0 pp | 0.0 pp |
| something to talk on . | 4 | 0.5000 | 0.7500 | 0.5000 | 0.7500 | 0.0 pp | 0.0 pp |
| something to tell time with . | 5 | 0.2000 | 0.4000 | 0.2000 | 0.6000 | 0.0 pp | 20.0 pp |
| something to travel in . | 14 | 0.5714 | 0.7143 | 0.6429 | 0.7143 | 7.1 pp | 0.0 pp |
| something to type on . | 2 | 0.0000 | 0.5000 | 0.0000 | 0.5000 | 0.0 pp | 0.0 pp |
| something to watch . | 2 | 0.0000 | 0.5000 | 0.0000 | 0.5000 | 0.0 pp | 0.0 pp |
| something used as a bed or resting place . | 4 | 0.7500 | 1.0000 | 0.7500 | 1.0000 | 0.0 pp | 0.0 pp |
| something used for cover from rain . | 5 | 0.0000 | 0.2000 | 0.0000 | 0.2000 | 0.0 pp | 0.0 pp |
| something used for cutting . | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0 pp | 0.0 pp |
| something used for drinking . | 5 | 0.6000 | 0.8000 | 0.6000 | 0.8000 | 0.0 pp | 0.0 pp |
| something used for eating . | 6 | 0.0000 | 0.3333 | 0.0000 | 0.1667 | 0.0 pp | -16.7 pp |
| something used for reading . | 4 | 0.2500 | 0.7500 | 0.2500 | 0.7500 | 0.0 pp | 0.0 pp |
| something used for typing . | 3 | 0.3333 | 0.3333 | 0.0000 | 0.3333 | -33.3 pp | 0.0 pp |
| something used for viewing shows . | 3 | 0.6667 | 1.0000 | 0.6667 | 1.0000 | 0.0 pp | 0.0 pp |
| something used for washing . | 1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0 pp | 0.0 pp |
| something used in a game . | 14 | 0.2143 | 0.5000 | 0.2857 | 0.7857 | 7.1 pp | 28.6 pp |
| something used to carry things . | 7 | 0.2857 | 0.5714 | 0.5714 | 0.7143 | 28.6 pp | 14.3 pp |
| something used to control a computer or screen . | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0 pp | 0.0 pp |
| something used to move from place to place . | 10 | 0.2000 | 0.2000 | 0.2000 | 0.3000 | 0.0 pp | 10.0 pp |
| something used to serve food for eating . | 4 | 0.7500 | 1.0000 | 0.7500 | 1.0000 | 0.0 pp | 0.0 pp |
| something used to talk on . | 6 | 0.6667 | 0.6667 | 0.6667 | 0.8333 | 0.0 pp | 16.7 pp |
| something you can carry things in . | 15 | 0.2667 | 0.5333 | 0.3333 | 0.7333 | 6.7 pp | 20.0 pp |
| something you can drink from . | 7 | 0.5714 | 0.8571 | 0.7143 | 0.8571 | 14.3 pp | 0.0 pp |
| something you can eat . | 9 | 0.3333 | 0.3333 | 0.3333 | 0.3333 | 0.0 pp | 0.0 pp |
| something you can lie on to rest . | 4 | 0.0000 | 0.7500 | 0.0000 | 0.5000 | 0.0 pp | -25.0 pp |
| something you can read . | 2 | 0.0000 | 0.5000 | 0.0000 | 1.0000 | 0.0 pp | 50.0 pp |
| something you can read time from . | 9 | 0.4444 | 0.7778 | 0.4444 | 0.7778 | 0.0 pp | 0.0 pp |
| something you can ride . | 10 | 0.3000 | 0.7000 | 0.2000 | 0.7000 | -10.0 pp | 0.0 pp |
| something you can sit on . | 9 | 0.1111 | 0.4444 | 0.1111 | 0.2222 | 0.0 pp | -22.2 pp |
| something you can stand under for shade . | 9 | 0.0000 | 0.1111 | 0.0000 | 0.0000 | 0.0 pp | -11.1 pp |
| something you can talk on . | 4 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.0 pp | 0.0 pp |
| something you can type with . | 4 | 0.0000 | 0.5000 | 0.0000 | 0.7500 | 0.0 pp | 25.0 pp |
| something you can use to cut . | 2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0 pp | 0.0 pp |
| something you can use to eat . | 5 | 0.0000 | 0.2000 | 0.0000 | 0.2000 | 0.0 pp | 0.0 pp |
| something you can use to heat food . | 3 | 0.3333 | 0.6667 | 0.3333 | 0.3333 | 0.0 pp | -33.3 pp |
| something you can use to point or select . | 5 | 0.2000 | 0.6000 | 0.2000 | 0.6000 | 0.0 pp | 0.0 pp |
| something you can watch video on . | 5 | 0.6000 | 0.8000 | 0.4000 | 0.8000 | -20.0 pp | 0.0 pp |
| somewhere to wash things . | 4 | 0.2500 | 0.2500 | 0.2500 | 0.5000 | 0.0 pp | 25.0 pp |
| somewhere water is used for cleaning . | 3 | 0.0000 | 0.3333 | 0.0000 | 0.3333 | 0.0 pp | 0.0 pp |
| sports equipment to play with . | 18 | 0.1667 | 0.5000 | 0.2778 | 0.5556 | 11.1 pp | 5.6 pp |

## Five Best Wins

| kind | benchmark_id | image_id | affordance | DINO top1 IoU | ThinkDet top1 IoU | DINO top5 IoU | ThinkDet top5 IoU |
| --- | --- | --- | --- | --- | --- | --- | --- |
| top1_win | carry_in_pv2_377723 | 377723 | carry_in | 0.0007 | 0.9490 | 0.8998 | 0.9490 |
| top1_win | read_pv3_25603 | 25603 | read | 0.0190 | 0.9662 | 0.9694 | 0.9662 |
| top1_win | read_pv4_71226 | 71226 | read | 0.0000 | 0.9431 | 0.9205 | 0.9692 |
| top1_win | shelter_under_pv3_313588 | 313588 | shelter_under | 0.0000 | 0.8881 | 0.8900 | 0.8881 |
| top1_win | tell_time_pv2_79588 | 79588 | tell_time | 0.0060 | 0.8903 | 0.8157 | 0.8903 |

## Five Worst Failures

| kind | benchmark_id | image_id | affordance | DINO top1 IoU | ThinkDet top1 IoU | DINO top5 IoU | ThinkDet top5 IoU |
| --- | --- | --- | --- | --- | --- | --- | --- |
| top1_failure | travel_in_pv0_187144 | 187144 | travel_in | 0.9741 | 0.0284 | 0.9741 | 0.9835 |
| top1_failure | sit_on_pv0_548780 | 548780 | sit_on | 0.9528 | 0.0157 | 0.9528 | 0.9655 |
| top1_failure | travel_in_pv2_568439 | 568439 | travel_in | 0.9576 | 0.0755 | 0.9576 | 0.9568 |
| top1_failure | type_on_pv1_479126 | 479126 | type_on | 0.8939 | 0.0129 | 0.8939 | 0.9167 |
| top1_failure | cut_with_pv0_123633 | 123633 | cut_with | 0.7948 | 0.0000 | 0.7948 | 0.7747 |

## Qualitative Images

- results/overfit_sanity/official_layer14_707train_test707_rank02_protect03_regselect/qualitative/qual_01_top1_win_377723.jpg
- results/overfit_sanity/official_layer14_707train_test707_rank02_protect03_regselect/qualitative/qual_02_top1_win_25603.jpg
- results/overfit_sanity/official_layer14_707train_test707_rank02_protect03_regselect/qualitative/qual_03_top1_win_71226.jpg
- results/overfit_sanity/official_layer14_707train_test707_rank02_protect03_regselect/qualitative/qual_04_top1_win_313588.jpg
- results/overfit_sanity/official_layer14_707train_test707_rank02_protect03_regselect/qualitative/qual_05_top1_win_79588.jpg
- results/overfit_sanity/official_layer14_707train_test707_rank02_protect03_regselect/qualitative_failures/failure_01_top1_failure_187144.jpg
- results/overfit_sanity/official_layer14_707train_test707_rank02_protect03_regselect/qualitative_failures/failure_02_top1_failure_548780.jpg
- results/overfit_sanity/official_layer14_707train_test707_rank02_protect03_regselect/qualitative_failures/failure_03_top1_failure_568439.jpg
- results/overfit_sanity/official_layer14_707train_test707_rank02_protect03_regselect/qualitative_failures/failure_04_top1_failure_479126.jpg
- results/overfit_sanity/official_layer14_707train_test707_rank02_protect03_regselect/qualitative_failures/failure_05_top1_failure_123633.jpg

## Success Checks

| Check | Passed | Value |
| --- | --- | --- |
| Top-1 gain beats previous +1.56 pp | True | 2.55 pp vs 1.56 pp |
| Top-5 holds/improves over previous | True | 57.85% vs 55.87% |
| Failures/regressions decrease clearly | True | 29 vs 100 |
| Train delta/memory <= 0.5x | True | 0.500000 |
| Unseen delta/memory <= 0.5x | True | 0.500000 |
| Prediction change controlled vs previous | True | 34.94% vs 56.58% |
| Train/test image overlap = 0 | True | 0 |
