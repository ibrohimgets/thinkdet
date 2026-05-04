# Grounding Benchmark Comparison

- benchmark: `/home/iibrohimm/project/next_step/thinkdet/data/benchmarks/flickr30k_entities_val_grounding_full_v1.json`
- benchmark_name: `flickr30k_entities_val_grounding_full_v1`
- split: `all`
- n_samples: 1661
- top_k: 5

## Overall

| model | hit@0.5_top1 | hit@0.5_topk | mean_best_iou_top1 | mean_best_iou_topk |
|---|---:|---:|---:|---:|
| baseline | 0.7700 | 0.9320 | 0.7203 | 0.8595 |
| thinkdet | 0.7766 | 0.9332 | 0.7222 | 0.8579 |
| thinkdet_fallback | 0.7766 | 0.9332 | 0.7222 | 0.8579 |

## Fallback Stage Counts

- primary: 1661
- llm_feedback: 0
- prompt_refine: 0
