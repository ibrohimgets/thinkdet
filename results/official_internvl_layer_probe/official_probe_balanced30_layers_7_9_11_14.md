# Official InternVL-Style Layer Probe

Benchmark: `/home/iibrohimm/project/next_step/thinkdet/data/benchmarks/affordance_coco_val_three_prompts_test_balanced30.json`
Samples: 30
Layers: [7, 9, 11, 14]
Rank key: `top5_token_hit`

| Layer | Top-1 | Top-5 | Top-16 | Mass Lift | GT-BG z | Mean Score | N |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 7 | 20.00% | 40.00% | 70.00% | 1.250 | 0.381 | 0.304775 | 30 |
| 9 | 13.33% | 50.00% | 80.00% | 1.317 | 0.403 | 0.357826 | 30 |
| 11 | 13.33% | 50.00% | 76.67% | 1.426 | 0.493 | 0.393315 | 30 |
| 14 | 20.00% | 53.33% | 76.67% | 1.292 | 0.385 | 0.447703 | 30 |

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
