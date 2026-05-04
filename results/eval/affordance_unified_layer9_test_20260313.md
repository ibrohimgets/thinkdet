# Grounding Benchmark Comparison

- benchmark: `/home/iibrohimm/project/next_step/thinkdet/data/benchmarks/affordance_coco_val_heldout_v1.json`
- benchmark_name: `affordance_coco_val_heldout_v1`
- split: `test`
- n_samples: 512
- top_k: 5

## Overall

| model | hit@0.5_top1 | hit@0.5_topk | mean_best_iou_top1 | mean_best_iou_topk |
|---|---:|---:|---:|---:|
| baseline | 0.1699 | 0.4082 | 0.1866 | 0.3933 |
| thinkdet | 0.1875 | 0.4258 | 0.1970 | 0.4017 |
| thinkdet_fallback | 0.1875 | 0.4258 | 0.1970 | 0.4017 |

## Fallback Stage Counts

- primary: 512
- llm_feedback: 0
- prompt_refine: 0
