# Grounding Benchmark Comparison

- benchmark: `/home/iibrohimm/project/next_step/thinkdet/data/benchmarks/flickr30k_entities_val_grounding_10k_v1.json`
- benchmark_name: `flickr30k_entities_val_grounding_10k_v1`
- split: `all`
- n_samples: 10000
- top_k: 5

## Overall

| model | hit@0.5_top1 | hit@0.5_topk | mean_best_iou_top1 | mean_best_iou_topk |
|---|---:|---:|---:|---:|
| baseline | 0.7725 | 0.9352 | 0.7221 | 0.8577 |
| thinkdet | 0.7774 | 0.9304 | 0.7204 | 0.8551 |
| thinkdet_fallback | 0.7691 | 0.9304 | 0.7148 | 0.8551 |

## Fallback Stage Counts

- primary: 8696
- llm_feedback: 1304
- prompt_refine: 0
