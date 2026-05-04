# Grounding Benchmark Comparison

- benchmark: `/home/iibrohimm/project/next_step/thinkdet/data/benchmarks/flickr30k_entities_val_grounding_10k_v1.json`
- benchmark_name: `flickr30k_entities_val_grounding_10k_v1`
- split: `all`
- n_samples: 1429
- top_k: 5

## Overall

| model | hit@0.5_top1 | hit@0.5_topk | mean_best_iou_top1 | mean_best_iou_topk |
|---|---:|---:|---:|---:|
| baseline | 0.7684 | 0.9363 | 0.7177 | 0.8548 |
| thinkdet | 0.7712 | 0.9370 | 0.7170 | 0.8569 |
| thinkdet_fallback | 0.7635 | 0.9377 | 0.7126 | 0.8575 |

## Fallback Stage Counts

- primary: 1221
- llm_feedback: 205
- prompt_refine: 3
