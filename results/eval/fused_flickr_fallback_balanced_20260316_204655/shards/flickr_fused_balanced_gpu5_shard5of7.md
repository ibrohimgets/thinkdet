# Grounding Benchmark Comparison

- benchmark: `/home/iibrohimm/project/next_step/thinkdet/data/benchmarks/flickr30k_entities_val_grounding_10k_v1.json`
- benchmark_name: `flickr30k_entities_val_grounding_10k_v1`
- split: `all`
- n_samples: 1428
- top_k: 5

## Overall

| model | hit@0.5_top1 | hit@0.5_topk | mean_best_iou_top1 | mean_best_iou_topk |
|---|---:|---:|---:|---:|
| baseline | 0.7745 | 0.9356 | 0.7208 | 0.8572 |
| thinkdet | 0.7773 | 0.9258 | 0.7151 | 0.8505 |
| thinkdet_fallback | 0.7689 | 0.9258 | 0.7114 | 0.8505 |

## Fallback Stage Counts

- primary: 1239
- llm_feedback: 189
- prompt_refine: 0
