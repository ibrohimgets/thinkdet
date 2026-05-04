# Grounding Benchmark Comparison

- benchmark: `/home/iibrohimm/project/next_step/thinkdet/data/benchmarks/flickr30k_entities_val_grounding_10k_v1.json`
- benchmark_name: `flickr30k_entities_val_grounding_10k_v1`
- split: `all`
- n_samples: 1429
- top_k: 5

## Overall

| model | hit@0.5_top1 | hit@0.5_topk | mean_best_iou_top1 | mean_best_iou_topk |
|---|---:|---:|---:|---:|
| baseline | 0.7719 | 0.9342 | 0.7295 | 0.8596 |
| thinkdet | 0.7838 | 0.9258 | 0.7290 | 0.8557 |
| thinkdet_fallback | 0.7733 | 0.9258 | 0.7208 | 0.8557 |

## Fallback Stage Counts

- primary: 1251
- llm_feedback: 178
- prompt_refine: 0
