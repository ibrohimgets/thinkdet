# Affordance Qualitative Wins (Baseline vs Unified)

- benchmark: `/home/iibrohimm/project/next_step/thinkdet/data/benchmarks/affordance_coco_val_heldout_v1.json`
- split: `test`
- baseline_eval_json: `/home/iibrohimm/project/next_step/thinkdet/results/eval/affordance_benchmark_eval_v1_test.json`
- unified_ckpt: `/home/iibrohimm/project/next_step/thinkdet/checkpoints/unified/layer9_kd0p05_l1_1e-4_20260224_135540/thinkdet_unified_epoch5.pth`
- table_image: `/home/iibrohimm/project/next_step/thinkdet/results/visual_compare/affordance_win_table_unified_20260226_073759/affordance_win_table_unified_epoch5.jpg`
- selection: `{'n_rows': 512, 'n_strict_candidates': 9, 'n_selected': 8, 'strict_thresholds': {'min_unified_top1_iou': 0.5, 'max_baseline_top1_iou': 0.49, 'min_gap': 0.05}}`

| rank | affordance | image_id | prompt | baseline IoU@top1 | unified IoU@top1 | gap | case_image |
|---:|---|---:|---|---:|---:|---:|---|
| 1 | carry_in | 190923 | something to carry things in . | 0.413 | 0.917 | 0.504 | `cases/case01_carry_in_190923.jpg` |
| 2 | drink_from | 548339 | something to drink from . | 0.000 | 0.760 | 0.760 | `cases/case02_drink_from_548339.jpg` |
| 3 | eat_with | 562059 | something to eat with . | 0.102 | 0.948 | 0.846 | `cases/case03_eat_with_562059.jpg` |
| 4 | talk_on | 579655 | something to talk on . | 0.034 | 0.870 | 0.836 | `cases/case04_talk_on_579655.jpg` |
| 5 | talk_on | 537991 | something to talk on . | 0.020 | 0.856 | 0.836 | `cases/case05_talk_on_537991.jpg` |
| 6 | talk_on | 156076 | something to talk on . | 0.000 | 0.701 | 0.701 | `cases/case06_talk_on_156076.jpg` |
| 7 | talk_on | 398377 | something to talk on . | 0.002 | 0.628 | 0.626 | `cases/case07_talk_on_398377.jpg` |
| 8 | talk_on | 250127 | something to talk on . | 0.002 | 0.586 | 0.584 | `cases/case08_talk_on_250127.jpg` |
