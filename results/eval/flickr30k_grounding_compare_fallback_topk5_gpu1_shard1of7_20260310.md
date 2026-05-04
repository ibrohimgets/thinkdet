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
| thinkdet | 0.7754 | 0.9314 | 0.7260 | 0.8579 |
| thinkdet_fallback | 0.7656 | 0.9314 | 0.7201 | 0.8578 |

## Fallback Stage Counts

- primary: 1225
- llm_feedback: 201
- prompt_refine: 3
