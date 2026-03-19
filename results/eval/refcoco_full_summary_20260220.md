# RefCOCO Full Evaluation Summary (Stage‑2 TMA vs Baseline)

Checkpoint: `/home/iibrohimm/project/next_step/thinkdet/checkpoints/stage2_tma/layer9_tma_m8_8gpu_20260219_010134/thinkdet_tma_stage2_epoch2.pth`

Metric: Top‑1 accuracy (IoU ≥ 0.5).  conf_thresh=0.05.

| Dataset | Split | Samples | Baseline Top‑1 Acc | Stage‑2 TMA Top‑1 Acc | Δ (TMA − Base) |
|---|---|---:|---:|---:|---:|
| refcoco | val   | 10834 | 0.5062 | 0.4861 | -0.0201 |
| refcoco | testA | 5657  | 0.5643 | 0.5183 | -0.0460 |
| refcoco | testB | 5095  | 0.4553 | 0.4522 | -0.0031 |
| refcoco+ | val   | 10758 | 0.5133 | 0.4718 | -0.0415 |
| refcoco+ | testA | 5726  | 0.5582 | 0.4883 | -0.0699 |
| refcoco+ | testB | 4889  | 0.4623 | 0.4600 | -0.0023 |
| refcocog | val  | 4896  | 0.6050 | 0.5758 | -0.0292 |
| refcocog | test | 9602  | 0.6050 | 0.5772 | -0.0278 |

Sources:
- `/home/iibrohimm/project/next_step/thinkdet/results/eval/refcoco_full_refcoco_val_20260220_064614.json`
- `/home/iibrohimm/project/next_step/thinkdet/results/eval/refcoco_full_refcoco_testA_20260220_064614.json`
- `/home/iibrohimm/project/next_step/thinkdet/results/eval/refcoco_full_refcoco_testB_20260220_064614.json`
- `/home/iibrohimm/project/next_step/thinkdet/results/eval/refcoco_full_refcoco+_val_20260220_064614.json`
- `/home/iibrohimm/project/next_step/thinkdet/results/eval/refcoco_full_refcoco+_testA_20260220_064614.json`
- `/home/iibrohimm/project/next_step/thinkdet/results/eval/refcoco_full_refcoco+_testB_20260220_064614.json`
- `/home/iibrohimm/project/next_step/thinkdet/results/eval/refcoco_full_refcocog_val_20260220_064614.json`
- `/home/iibrohimm/project/next_step/thinkdet/results/eval/refcoco_full_refcocog_test_20260220_064614.json`
