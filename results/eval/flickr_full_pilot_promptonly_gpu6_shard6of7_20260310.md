# Grounding Benchmark Comparison

- benchmark: `/home/iibrohimm/project/next_step/thinkdet/data/benchmarks/flickr30k_entities_val_grounding_full_v1.json`
- benchmark_name: `flickr30k_entities_val_grounding_full_v1`
- split: `all`
- n_samples: 142
- top_k: 5

## Overall

| model | hit@0.5_top1 | hit@0.5_topk | mean_best_iou_top1 | mean_best_iou_topk |
|---|---:|---:|---:|---:|
| baseline | 0.7817 | 0.9507 | 0.7264 | 0.8679 |
| thinkdet | 0.7958 | 0.9507 | 0.7291 | 0.8694 |
| thinkdet_fallback | 0.7958 | 0.9507 | 0.7291 | 0.8694 |

## Fallback Stage Counts

- primary: 142
- llm_feedback: 0
- prompt_refine: 0
