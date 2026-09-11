# ThinkDet

ThinkDet investigates whether query-conditioned features from a multimodal
language model can improve a frozen open-vocabulary detector on ambiguous,
functional requests.

The adapter keeps GroundingDINO and InternVL frozen, compresses selected
InternVL hidden states into a small set of summary tokens, and injects them into
GroundingDINO through a gated residual path.

## Result at a glance

| Evaluation | GroundingDINO baseline | ThinkDet | Interpretation |
| --- | ---: | ---: | --- |
| Corrected held-out affordance benchmark, top-1 Hit@0.5 | 16.99% | 17.77% | Small gain on a narrow functional-query stress test |

The broader evaluation is mixed: early checkpoints regress on standard COCO AP
and RefCOCO-family evaluation, while the unified residual run mostly recovers
the baseline. This repository intentionally reports both positive and negative
results.

## Why this work is useful

- Demonstrates a real MLLM-to-detector adapter rather than prompt-only wrapping.
- Includes corrected evaluation artifacts and diagnostics for ranking failures.
- Provides reusable training, inference, and benchmark scripts.
- Shows how to build baseline-safe residual adaptation and how to audit claims
  when an evaluation bug changes the conclusion.

**Status:** research prototype. It is evidence of multimodal-model engineering
and evaluation discipline, not a production detector or a general improvement
over GroundingDINO.

## Active Architecture

The default code path now matches the architecture diagram:

- InternVL3.5 receives the image and text query jointly.
- Intermediate visual-position hidden states are extracted as `H_vlm`
  with shape `[B, 256, D]`.
- A trainable TMA block maps `D -> 256`, pools with `M=8` learnable
  cross-attention queries, and produces `aug_tokens`.
- Decoder layers `[1, 3, 5]` use baseline-safe residual text-memory fusion.
- The adapter gate starts closed (`alpha_init=0.0`) and the residual branch
  is zero-initialized, so new runs begin equivalent to frozen GroundingDINO.
- If confidence is weak, inference can route to InternVL evidence-check
  reranking and optional prompt refinement.

## Current Status

- This is a research prototype, not a production detector.
- The repo has one interesting positive result:
  a small gain on a corrected held-out affordance benchmark built from COCO val
  (`16.99% -> 17.77%` hit@0.5 top-1 for baseline vs Stage 2).
- The repo also has clear negative results:
  the early Stage 1 and Stage 2 checkpoints regress on standard COCO AP and
  RefCOCO-family referring expression evaluation.
- The unified residual run mostly recovers baseline COCO AP, but it does not
  establish a broad win across standard benchmarks.

## What The Evidence Supports

- Adapter mechanism is real and implemented in the active codepath.
- Query-conditioned InternVL features can produce small gains on a narrow
  functional-query stress test.
- Historical artifacts suggested calibration gains, but the corrected
  query-conditioned scorer rerun does not preserve that result.

## What The Evidence Does Not Support

- ThinkDet is not a general improvement to GroundingDINO.
- ThinkDet is not a proven reasoning detector.
- ThinkDet is not close to SOTA on the custom affordance benchmark.

## Active Codepath

- `models/arch.py`
- `models/projector.py`
- `models/cross_attention.py`
- `models/decoder_layer.py`
- `scripts/training/train_stage1_tma.py`
- `scripts/training/train_stage2_tma.py`
- `scripts/training/train_unified.py`
- `scripts/eval/eval_official_protocol.py`
- `scripts/eval/eval_affordance_benchmark.py`

## Thesis Framing

If this repo is used for a thesis, the safest claim is:

- ThinkDet is a lightweight MLLM-to-detector adapter for ambiguity-heavy
  grounding.
- It can help on some affordance-style prompts.
- The gains are narrow and sensitive to scoring, ranking, and fallback policy.

The thesis story should use this order:

1. corrected query-conditioned base scoring
2. ThinkDet as the primary detector
3. LLM fallback as a secondary recovery policy

The LLM fallback should be described as:

- evidence-check candidate reranking first
- prompt refinement second
- semantic-preservation guard enabled for refinement

Do not present fallback as proof that the base detector is intelligent.
Treat it as an inference-time recovery mechanism.

## Key Artifacts

- Corrected affordance benchmark summary:
  `results/eval/affordance_benchmark_eval_v1_test_corrected_20260308.md`
- Historical affordance benchmark summary:
  `results/eval/affordance_benchmark_eval_v1_test.md`
- Official COCO protocol eval:
  `results/eval/official_protocol_eval.json`
- RefCOCO family summary:
  `results/eval/refcoco_full_summary_20260220.md`
- Thesis readiness plan:
  `docs/thesis_readiness_plan.md`

Important:

- The historical affordance summary above predates a later query-conditioned
  scoring fix in `scripts/eval/eval_affordance_benchmark.py`.
- The corrected rerun is now saved in
  `results/eval/affordance_benchmark_eval_v1_test_corrected_20260308.json`
  and `.md`.
- A concrete ranking-failure diagnostic is saved under
  `results/inference/diagnostic_topk_carry_in_99114_20260307_112733.json`
  and the corrected post-fix ranking for that case is in
  `results/inference/postfix_topk_carry_in_99114_20260307_114742.json`.

## Repo Hygiene

- Broken legacy one-off scripts and tests have been removed.
- Historical docs remain under `docs/` and `paper/`, but the current repo
  position is the conservative one described here.
