# Official InternVL-Style Layer Probe

Benchmark: `/home/iibrohimm/project/next_step/thinkdet/data/benchmarks/affordance_coco_val_three_prompts_test_balanced30.json`
Samples: 30
Layers: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]
Rank key: `top5_token_hit`

| Layer | Top-1 | Top-5 | Top-16 | Mass Lift | GT-BG z | Mean Score | N |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 10.00% | 30.00% | 43.33% | 0.951 | -0.052 | 0.261797 | 30 |
| 1 | 16.67% | 40.00% | 60.00% | 0.992 | 0.014 | 0.374351 | 30 |
| 2 | 10.00% | 33.33% | 53.33% | 0.964 | -0.091 | 0.459176 | 30 |
| 3 | 16.67% | 33.33% | 60.00% | 1.021 | 0.038 | 0.470860 | 30 |
| 4 | 16.67% | 40.00% | 60.00% | 1.132 | 0.290 | 0.395521 | 30 |
| 5 | 13.33% | 43.33% | 66.67% | 1.217 | 0.408 | 0.325135 | 30 |
| 6 | 16.67% | 30.00% | 66.67% | 1.221 | 0.375 | 0.315571 | 30 |
| 7 | 20.00% | 40.00% | 70.00% | 1.250 | 0.381 | 0.304775 | 30 |
| 8 | 20.00% | 50.00% | 76.67% | 1.364 | 0.544 | 0.344299 | 30 |
| 9 | 13.33% | 50.00% | 80.00% | 1.317 | 0.403 | 0.357826 | 30 |
| 10 | 20.00% | 50.00% | 76.67% | 1.372 | 0.417 | 0.390581 | 30 |
| 11 | 13.33% | 50.00% | 76.67% | 1.426 | 0.493 | 0.393315 | 30 |
| 12 | 13.33% | 50.00% | 73.33% | 1.313 | 0.338 | 0.431967 | 30 |
| 13 | 16.67% | 53.33% | 73.33% | 1.256 | 0.284 | 0.463700 | 30 |
| 14 | 20.00% | 53.33% | 76.67% | 1.292 | 0.385 | 0.447703 | 30 |
| 15 | 20.00% | 36.67% | 70.00% | 0.990 | -0.123 | 0.456578 | 30 |
| 16 | 16.67% | 36.67% | 70.00% | 0.930 | -0.300 | 0.415923 | 30 |
| 17 | 13.33% | 40.00% | 66.67% | 0.908 | -0.327 | 0.429364 | 30 |
| 18 | 13.33% | 30.00% | 66.67% | 0.850 | -0.371 | 0.464415 | 30 |
| 19 | 13.33% | 30.00% | 60.00% | 0.923 | -0.211 | 0.499918 | 30 |
| 20 | 10.00% | 26.67% | 56.67% | 0.880 | -0.246 | 0.519643 | 30 |
| 21 | 10.00% | 23.33% | 53.33% | 0.891 | -0.226 | 0.585265 | 30 |
| 22 | 0.00% | 33.33% | 50.00% | 0.867 | -0.161 | 0.621584 | 30 |
| 23 | 6.67% | 33.33% | 53.33% | 0.932 | -0.075 | 0.648589 | 30 |
| 24 | 6.67% | 23.33% | 50.00% | 0.943 | -0.024 | 0.701303 | 30 |
| 25 | 6.67% | 20.00% | 50.00% | 1.025 | 0.111 | 0.751510 | 30 |
| 26 | 6.67% | 20.00% | 50.00% | 0.999 | 0.076 | 0.782726 | 30 |
| 27 | 13.33% | 23.33% | 40.00% | 0.639 | -0.133 | 0.202920 | 30 |

Best layer: **14** (top5_token_hit=0.5333, mass_lift=1.292, gt_bg_z=0.385).
Layer 14 strongest: **yes**.
Layer 14 summary: Top-5=53.33%, mass_lift=1.292, gt_bg_z=0.385.

## Metric Definitions

- Top-k token hit: whether any of the top-k image tokens by query-image cosine score overlaps a GT target box.
- Mass lift: softmax score mass on GT-overlapping tokens divided by the GT token fraction; 1.0 is uniform.
- GT-BG z: mean raw cosine score on GT tokens minus background tokens, divided by the std over all image tokens.

## Limitations

- This is representation probing only; it does not train or inject into the detector.
- It uses one 448x448 resized image, not InternVL dynamic tiling, so box-to-token mapping stays 16x16.
- In causal Qwen, image-token states do not attend to later prompt words; prompt signal is measured through same-layer prompt-token representations compared to image tokens.
- GT boxes are projected to a coarse 16x16 visual-token grid, so small objects can be noisy.
