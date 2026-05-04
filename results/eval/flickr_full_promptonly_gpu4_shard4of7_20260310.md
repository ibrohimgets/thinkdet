# Grounding Benchmark Comparison

- benchmark: `/home/iibrohimm/project/next_step/thinkdet/data/benchmarks/flickr30k_entities_val_grounding_full_v1.json`
- benchmark_name: `flickr30k_entities_val_grounding_full_v1`
- split: `all`
- n_samples: 1660
- top_k: 5

## Overall

| model | hit@0.5_top1 | hit@0.5_topk | mean_best_iou_top1 | mean_best_iou_topk |
|---|---:|---:|---:|---:|
| baseline | 0.7627 | 0.9331 | 0.7097 | 0.8517 |
| thinkdet | 0.7663 | 0.9307 | 0.7093 | 0.8499 |
| thinkdet_fallback | 0.7663 | 0.9307 | 0.7093 | 0.8499 |

## Fallback Stage Counts

- primary: 1660
- llm_feedback: 0
- prompt_refine: 0
