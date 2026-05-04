# Grounding Benchmark Comparison

- benchmark: `/home/iibrohimm/project/next_step/thinkdet/data/benchmarks/flickr30k_entities_val_grounding_10k_v1.json`
- benchmark_name: `flickr30k_entities_val_grounding_10k_v1`
- split: `all`
- n_samples: 1429
- top_k: 5

## Overall

| model | hit@0.5_top1 | hit@0.5_topk | mean_best_iou_top1 | mean_best_iou_topk |
|---|---:|---:|---:|---:|
| baseline | 0.7719 | 0.9293 | 0.7236 | 0.8600 |
| thinkdet | 0.7810 | 0.9286 | 0.7239 | 0.8581 |
| thinkdet_fallback | 0.7712 | 0.9286 | 0.7170 | 0.8581 |

## Fallback Stage Counts

- primary: 1240
- llm_feedback: 189
- prompt_refine: 0
