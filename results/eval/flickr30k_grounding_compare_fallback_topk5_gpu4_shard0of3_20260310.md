# Grounding Benchmark Comparison

- benchmark: `/home/iibrohimm/project/next_step/thinkdet/data/benchmarks/flickr30k_entities_val_grounding_10k_v1.json`
- benchmark_name: `flickr30k_entities_val_grounding_10k_v1`
- split: `all`
- n_samples: 3334
- top_k: 5

## Overall

| model | hit@0.5_top1 | hit@0.5_topk | mean_best_iou_top1 | mean_best_iou_topk |
|---|---:|---:|---:|---:|
| baseline | 0.7711 | 0.9328 | 0.7206 | 0.8563 |
| thinkdet | 0.7762 | 0.9325 | 0.7222 | 0.8544 |
| thinkdet_fallback | 0.7702 | 0.9325 | 0.7190 | 0.8544 |

## Fallback Stage Counts

- primary: 2848
- llm_feedback: 482
- prompt_refine: 4
