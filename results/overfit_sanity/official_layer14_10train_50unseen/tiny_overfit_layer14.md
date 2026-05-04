# Tiny Official Layer-14 Adapter Sanity

Benchmark: `data/benchmarks/affordance_coco_val_three_prompts_test.json`
Official InternVL extraction: `True`
Layer setup: `[14]`
Train/unseen: `10` / `50`
Steps: `120`, lr: `0.001`, alpha_init: `0.1`, lambda_delta: `0.01`

| Split | N | Top-1 Hit@0.5 | Top-5 Hit@0.5 | Mean Top-1 IoU | Mean Top-5 IoU |
|---|---:|---:|---:|---:|---:|
| train baseline | 10 | 20.0% | 40.0% | 0.227 | 0.481 |
| train trained | 10 | 20.0% | 20.0% | 0.229 | 0.283 |
| unseen baseline | 50 | 12.0% | 38.0% | 0.140 | 0.363 |
| unseen trained | 50 | 18.0% | 30.0% | 0.186 | 0.281 |

## Prediction Change

- train top1 box changed: 100.0%
- train hit changed count: 0
- unseen top1 box changed: 88.0%
- unseen hit changed count: 7

## Delta / Memory

| Eval | DINO layer | Gate | Delta/Memory | Delta Norm | Memory Norm |
|---|---:|---:|---:|---:|---:|
| train | 1 | 0.1008 | 0.760 | 10.060 | 13.232 |
| train | 3 | 0.1109 | 3.125 | 41.346 | 13.232 |
| train | 5 | 0.0818 | 0.447 | 5.915 | 13.232 |
| unseen | 1 | 0.1008 | 0.760 | 10.060 | 13.233 |
| unseen | 3 | 0.1109 | 3.124 | 41.346 | 13.233 |
| unseen | 5 | 0.0818 | 0.447 | 5.915 | 13.233 |

## Conclusion

The adapter produced nonzero memory deltas and changed predictions, but this tiny run did not improve Top-5 grounding. It is evidence that the official layer-14 path can affect DINO, not evidence of a useful detector improvement yet.
