# Grounding Benchmark Comparison

- benchmark: `/home/iibrohimm/project/next_step/thinkdet/data/benchmarks/flickr30k_entities_val_grounding_10k_v1.json`
- benchmark_name: `flickr30k_entities_val_grounding_10k_v1`
- split: `all`
- n_samples: 1428
- top_k: 5

## Overall

| model | hit@0.5_top1 | hit@0.5_topk | mean_best_iou_top1 | mean_best_iou_topk |
|---|---:|---:|---:|---:|
| baseline | 0.7689 | 0.9356 | 0.7202 | 0.8592 |
| thinkdet | 0.7717 | 0.9328 | 0.7201 | 0.8574 |
| thinkdet_fallback | 0.7668 | 0.9328 | 0.7162 | 0.8574 |

## Fallback Stage Counts

- primary: 1245
- llm_feedback: 183
- prompt_refine: 0
