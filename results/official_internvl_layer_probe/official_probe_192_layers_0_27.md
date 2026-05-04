# Official InternVL-Style Layer Probe

Benchmark: `data/benchmarks/affordance_coco_val_three_prompts_test.json`
Samples: 191
Layers: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]
Rank key: `top5_token_hit`

| Layer | Top-1 | Top-5 | Top-16 | Mass Lift | GT-BG z | Mean Score | N |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 12.57% | 32.98% | 59.16% | 1.022 | 0.088 | 0.260061 | 191 |
| 1 | 13.09% | 35.08% | 61.78% | 1.037 | 0.118 | 0.373763 | 191 |
| 2 | 10.47% | 38.22% | 60.73% | 0.997 | 0.006 | 0.457955 | 191 |
| 3 | 11.52% | 35.08% | 63.87% | 1.043 | 0.115 | 0.470419 | 191 |
| 4 | 17.28% | 44.50% | 67.54% | 1.125 | 0.307 | 0.395071 | 191 |
| 5 | 17.80% | 49.74% | 73.82% | 1.199 | 0.421 | 0.325335 | 191 |
| 6 | 16.23% | 48.17% | 74.35% | 1.219 | 0.421 | 0.315972 | 191 |
| 7 | 15.71% | 50.79% | 75.92% | 1.233 | 0.392 | 0.306396 | 191 |
| 8 | 16.75% | 53.40% | 79.06% | 1.301 | 0.480 | 0.345498 | 191 |
| 9 | 13.61% | 51.83% | 77.49% | 1.271 | 0.390 | 0.358705 | 191 |
| 10 | 17.28% | 53.93% | 77.49% | 1.347 | 0.453 | 0.390029 | 191 |
| 11 | 14.66% | 53.40% | 76.44% | 1.385 | 0.499 | 0.393932 | 191 |
| 12 | 10.47% | 50.79% | 75.39% | 1.279 | 0.358 | 0.432389 | 191 |
| 13 | 12.04% | 49.21% | 74.35% | 1.242 | 0.318 | 0.462962 | 191 |
| 14 | 14.66% | 55.50% | 76.96% | 1.285 | 0.398 | 0.446572 | 191 |
| 15 | 12.57% | 43.98% | 73.30% | 1.039 | -0.034 | 0.456442 | 191 |
| 16 | 12.04% | 42.41% | 70.68% | 0.980 | -0.186 | 0.414709 | 191 |
| 17 | 12.57% | 46.60% | 73.82% | 0.994 | -0.175 | 0.429224 | 191 |
| 18 | 10.47% | 42.41% | 71.73% | 0.936 | -0.233 | 0.464224 | 191 |
| 19 | 15.18% | 39.79% | 69.63% | 1.001 | -0.108 | 0.500057 | 191 |
| 20 | 9.95% | 34.03% | 68.06% | 0.965 | -0.156 | 0.519792 | 191 |
| 21 | 10.99% | 37.70% | 66.49% | 0.982 | -0.136 | 0.585520 | 191 |
| 22 | 8.90% | 38.74% | 63.35% | 0.965 | -0.116 | 0.620994 | 191 |
| 23 | 9.42% | 35.08% | 61.26% | 1.001 | -0.062 | 0.649035 | 191 |
| 24 | 11.52% | 32.46% | 58.64% | 0.998 | -0.028 | 0.702139 | 191 |
| 25 | 10.99% | 30.89% | 59.16% | 1.042 | 0.089 | 0.751975 | 191 |
| 26 | 12.04% | 29.84% | 56.54% | 0.998 | 0.019 | 0.783214 | 191 |
| 27 | 10.47% | 32.98% | 51.83% | 0.804 | -0.051 | 0.197419 | 191 |

Best layer: **14** (top5_token_hit=0.5550, mass_lift=1.285, gt_bg_z=0.398).
Layer 14 strongest: **yes**.
Layer 14 summary: Top-5=55.50%, mass_lift=1.285, gt_bg_z=0.398.

## Metric Definitions

- Top-k token hit: whether any of the top-k image tokens by query-image cosine score overlaps a GT target box.
- Mass lift: softmax score mass on GT-overlapping tokens divided by the GT token fraction; 1.0 is uniform.
- GT-BG z: mean raw cosine score on GT tokens minus background tokens, divided by the std over all image tokens.

## Limitations

- This is representation probing only; it does not train or inject into the detector.
- It uses one 448x448 resized image, not InternVL dynamic tiling, so box-to-token mapping stays 16x16.
- In causal Qwen, image-token states do not attend to later prompt words; prompt signal is measured through same-layer prompt-token representations compared to image tokens.
- GT boxes are projected to a coarse 16x16 visual-token grid, so small objects can be noisy.
