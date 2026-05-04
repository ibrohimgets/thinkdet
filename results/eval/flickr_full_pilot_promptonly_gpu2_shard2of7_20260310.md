# Grounding Benchmark Comparison

- benchmark: `/home/iibrohimm/project/next_step/thinkdet/data/benchmarks/flickr30k_entities_val_grounding_full_v1.json`
- benchmark_name: `flickr30k_entities_val_grounding_full_v1`
- split: `all`
- n_samples: 143
- top_k: 5

## Overall

| model | hit@0.5_top1 | hit@0.5_topk | mean_best_iou_top1 | mean_best_iou_topk |
|---|---:|---:|---:|---:|
| baseline | 0.8322 | 0.8951 | 0.7734 | 0.8532 |
| thinkdet | 0.8462 | 0.9021 | 0.7802 | 0.8478 |
| thinkdet_fallback | 0.8462 | 0.9021 | 0.7802 | 0.8478 |

## Fallback Stage Counts

- primary: 143
- llm_feedback: 0
- prompt_refine: 0
