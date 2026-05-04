# Grounding Benchmark Comparison

- benchmark: `/home/iibrohimm/project/next_step/thinkdet/data/benchmarks/flickr30k_entities_val_grounding_10k_v1.json`
- benchmark_name: `flickr30k_entities_val_grounding_10k_v1`
- split: `all`
- n_samples: 1428
- top_k: 5

## Overall

| model | hit@0.5_top1 | hit@0.5_topk | mean_best_iou_top1 | mean_best_iou_topk |
|---|---:|---:|---:|---:|
| baseline | 0.7703 | 0.9363 | 0.7158 | 0.8548 |
| thinkdet | 0.7675 | 0.9300 | 0.7094 | 0.8505 |
| thinkdet_fallback | 0.7563 | 0.9300 | 0.7005 | 0.8505 |

## Fallback Stage Counts

- primary: 1225
- llm_feedback: 203
- prompt_refine: 0
