# Paper Layer Match Figure Summary

## Protocol

- Measured no-retrain extraction-layer override sweep on the unified epoch-5 checkpoint.
- Layers tested: `0, 4, 8, 9, 10, 13, 20, 27`.
- Dashed baselines come from separately saved downstream eval summaries.
- `Trained fused 8-9-10` references are reported in the table below but are not mixed into the plotted curve.

## Outputs

- PNG: `/home/iibrohimm/project/next_step/thinkdet/results/layer_ablation/paper_layer_match_figure.png`
- PDF: `/home/iibrohimm/project/next_step/thinkdet/results/layer_ablation/paper_layer_match_figure.pdf`

## Table

| Dataset | Baseline | Best single layer | Layer 9 | Delta best-baseline | Trained fused 8-9-10 |
|---|---:|---:|---:|---:|---:|
| Flickr30k Entities Val | 77.25% | L0: 77.29% | 77.29% | 0.04% | 77.74% |
| Affordance Held-Out Test | 16.99% | L0: 18.75% | 18.75% | 1.76% | 20.12% |
