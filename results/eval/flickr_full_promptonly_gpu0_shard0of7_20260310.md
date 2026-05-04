# Grounding Benchmark Comparison

- benchmark: `/home/iibrohimm/project/next_step/thinkdet/data/benchmarks/flickr30k_entities_val_grounding_full_v1.json`
- benchmark_name: `flickr30k_entities_val_grounding_full_v1`
- split: `all`
- n_samples: 1661
- top_k: 5

## Overall

| model | hit@0.5_top1 | hit@0.5_topk | mean_best_iou_top1 | mean_best_iou_topk |
|---|---:|---:|---:|---:|
| baseline | 0.7676 | 0.9386 | 0.7162 | 0.8552 |
| thinkdet | 0.7694 | 0.9404 | 0.7157 | 0.8574 |
| thinkdet_fallback | 0.7694 | 0.9404 | 0.7157 | 0.8574 |

## Fallback Stage Counts

- primary: 1661
- llm_feedback: 0
- prompt_refine: 0
