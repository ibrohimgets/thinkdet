# Grounding Benchmark Comparison

- benchmark: `/home/iibrohimm/project/next_step/thinkdet/data/benchmarks/flickr30k_entities_val_grounding_full_v1.json`
- benchmark_name: `flickr30k_entities_val_grounding_full_v1`
- split: `all`
- n_samples: 143
- top_k: 5

## Overall

| model | hit@0.5_top1 | hit@0.5_topk | mean_best_iou_top1 | mean_best_iou_topk |
|---|---:|---:|---:|---:|
| baseline | 0.7832 | 0.9301 | 0.7405 | 0.8648 |
| thinkdet | 0.7762 | 0.9301 | 0.7400 | 0.8621 |
| thinkdet_fallback | 0.7762 | 0.9301 | 0.7400 | 0.8621 |

## Fallback Stage Counts

- primary: 143
- llm_feedback: 0
- prompt_refine: 0
