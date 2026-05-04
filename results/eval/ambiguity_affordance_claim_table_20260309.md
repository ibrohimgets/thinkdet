# Ambiguity-Heavy Affordance Grounding Table

This table is designed to support the specific thesis claim that ThinkDet helps
on ambiguity-heavy, semantically under-specified functional prompts.

Benchmark:

- held-out affordance benchmark
- source: COCO val2017
- test split: `512` queries
- `8` affordances, `64` test queries each
- all prompts are functional / semantically under-specified, for example:
  - `something to carry things in .`
  - `something to drink from .`
  - `something to talk on .`
  - `something to ride .`

Sources:

- corrected baseline / Stage-2:
  [affordance_benchmark_eval_v1_test_corrected_20260308.json](/home/iibrohimm/project/next_step/thinkdet/results/eval/affordance_benchmark_eval_v1_test_corrected_20260308.json)
- unified detector:
  [affordance_layer9.json](/home/iibrohimm/project/next_step/thinkdet/results/layer_ablation/affordance_layer9.json)
- full pipeline with reranking:
  [affordance_llm_rerank_test_8gpu_merged.json](/home/iibrohimm/project/next_step/thinkdet/results/eval/affordance_llm_rerank_test_8gpu_merged.json)

## Main Table

| affordance | baseline | ThinkDet stage2 | ThinkDet unified | full pipeline | best delta vs baseline |
|---|---:|---:|---:|---:|---:|
| carry_in | 15.62% | 20.31% | **21.88%** | 18.75% | **+6.25pp** |
| cut_with | 6.25% | 6.25% | 6.25% | **7.81%** | **+1.56pp** |
| drink_from | 37.50% | 39.06% | 40.62% | **50.00%** | **+12.50pp** |
| eat_with | **4.69%** | **4.69%** | **4.69%** | 3.12% | +0.00pp |
| read | 18.75% | 18.75% | 18.75% | **21.88%** | **+3.12pp** |
| ride | 31.25% | **32.81%** | **32.81%** | **32.81%** | **+1.56pp** |
| sit_on | 4.69% | 4.69% | 1.56% | **7.81%** | **+3.12pp** |
| talk_on | 17.19% | 15.62% | **23.44%** | 20.31% | **+6.25pp** |
| **overall** | 16.99% | 17.77% | 18.75% | **20.31%** | **+3.32pp** |

## Reading The Table

- `ThinkDet stage2` is the clean corrected base-detector comparison.
- `ThinkDet unified` is the later unified residual detector checkpoint.
- `full pipeline` is the strongest system result and includes the LLM-guided reranking stage.

## Safe Thesis Claim

The held-out affordance benchmark is entirely composed of semantically
under-specified functional prompts. On this benchmark, ThinkDet improves over
the corrected baseline overall, and the largest gains appear on affordances
such as `carry_in`, `talk_on`, and especially `drink_from`.
