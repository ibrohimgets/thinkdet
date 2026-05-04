# Grounding Benchmark Comparison

- benchmark: `/home/iibrohimm/project/next_step/thinkdet/data/benchmarks/flickr30k_entities_val_grounding_10k_v1.json`
- benchmark_name: `flickr30k_entities_val_grounding_10k_v1`
- split: `all`
- n_samples: 1429
- top_k: 5

## Overall

| model | hit@0.5_top1 | hit@0.5_topk | mean_best_iou_top1 | mean_best_iou_topk |
|---|---:|---:|---:|---:|
| baseline | 0.7817 | 0.9391 | 0.7271 | 0.8580 |
| thinkdet | 0.7796 | 0.9335 | 0.7231 | 0.8565 |
| thinkdet_fallback | 0.7754 | 0.9335 | 0.7207 | 0.8565 |

## Fallback Stage Counts

- primary: 1246
- llm_feedback: 183
- prompt_refine: 0
