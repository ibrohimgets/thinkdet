# Grounding Benchmark Comparison

- benchmark: `/home/iibrohimm/project/next_step/thinkdet/data/benchmarks/flickr30k_entities_val_grounding_full_v1.json`
- benchmark_name: `flickr30k_entities_val_grounding_full_v1`
- split: `all`
- n_samples: 1660
- top_k: 5

## Overall

| model | hit@0.5_top1 | hit@0.5_topk | mean_best_iou_top1 | mean_best_iou_topk |
|---|---:|---:|---:|---:|
| baseline | 0.7711 | 0.9373 | 0.7181 | 0.8574 |
| thinkdet | 0.7741 | 0.9361 | 0.7183 | 0.8553 |
| thinkdet_fallback | 0.7741 | 0.9361 | 0.7183 | 0.8553 |

## Fallback Stage Counts

- primary: 1660
- llm_feedback: 0
- prompt_refine: 0
