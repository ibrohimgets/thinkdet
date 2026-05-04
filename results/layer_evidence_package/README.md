# Layer Evidence Plots

This folder collects the available layer-selection evidence figures for the
ThinkDet thesis.

## Existing probe / analysis plots

- `01_flickr30k_val_probe.png`
  - Flickr30k validation probe across all InternVL layers.
- `02_flickr30k_test_probe.png`
  - Flickr30k test probe across all InternVL layers.
- `03_flickr_val_layer9_vs_layer10.png`
  - Focused Flickr validation comparison for layer 9 vs layer 10.
- `04_refcocog_val_probe.png`
  - RefCOCOg validation probe across all InternVL layers.
- `05_layer_similarity_heatmap.png`
  - InternVL layer similarity heatmap used for grouping / shortlist analysis.
- `06_layer_sensitivity_summary.png`
  - Prompt-sensitivity probe summary across all InternVL layers.

## New downstream partial ablation plots

- `07_affordance_no_retrain_curve.png`
  - No-retrain affordance benchmark layer curve for tested layers only.
- `08_affordance_bucket_partial.png`
  - No-retrain affordance bucket curves for tested layers only.
- `09_flickr_val_no_retrain_curve.png`
  - No-retrain Flickr30k validation layer curve for tested layers only.

## Interpretation

- Probe plots are useful for showing how the preferred layer changes by proxy.
- The no-retrain downstream plots are useful for showing that the fixed trained
  checkpoint was largely layer-insensitive on the saved affordance and Flickr
  validation results.
