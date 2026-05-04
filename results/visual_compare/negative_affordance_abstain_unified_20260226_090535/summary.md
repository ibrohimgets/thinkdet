# Negative Affordance Abstention Case (Baseline vs Unified)

- prompt: `something to talk on .`
- target_categories: `['cell phone']`
- target_absent_definition: all target_categories absent from sample.category_histogram (COCO annotations)
- num_candidates_scanned: 420
- selected_threshold: 0.25
- selection_mode: `threshold_match`
- output_image: `/home/iibrohimm/project/next_step/thinkdet/results/visual_compare/negative_affordance_abstain_unified_20260226_090535/negative_talk_on_img284623.jpg`

| image_id | source_affordance | baseline_score | unified_score | gap |
|---:|---|---:|---:|---:|
| 284623 | drink_from | 0.360 | 0.244 | 0.115 |
