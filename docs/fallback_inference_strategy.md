# ThinkDet Inference Fallback Strategy

## Goal

Add an inference-time recovery path without extra training.

This path is only meaningful after the base detector uses the corrected
query-conditioned scorer. Fallback should not be used to hide a bad ranking
rule in the primary path.

This fallback is LLM-only:
- no baseline routing
- no hand-written prompt rewrite

## Current Routing Order

1. Run the primary ThinkDet prediction.
2. Score its reliability from:
   - top-1 score
   - top-1 vs top-2 margin
   - optional gate threshold
3. If the result looks weak, ask InternVL to rerank detector candidates with a short evidence check and final yes/no decision.
4. If it still looks weak, ask InternVL to rewrite the query into clearer grounding prompts and retry ThinkDet.
5. Accept a refined query only if it clears a semantic-preservation guard.

## Decision Rule

The policy is heuristic, but the fallback stages themselves are LLM-based.

Primary is treated as weak if any of these hold:
- `top1 < fallback_min_top1`
- `margin < fallback_min_margin`
- `gate_mean < fallback_min_gate` when gate checking is enabled

LLM evidence-check reranking replaces the primary result only when:
- the primary result is weak, and
- feedback improves reliability by `feedback_improve_margin`

LLM prompt refinement replaces the selected result only when:
- the selected result is still weak, and
- the refined prompt improves reliability by `refine_improve_margin`
- the refined prompt preserves the meaning of the original query strongly enough
  to pass the semantic guard

## Semantic Guard

Prompt refinement is allowed, but it is not allowed to win purely by drifting
to a different object meaning.

The current shared fallback policy applies a lightweight semantic-preservation
check before accepting a refined query:

- compare content terms from the original query and refined query
- require at least one shared normalized content term by default
- optionally require explicit anchor terms when the caller provides them

This guard is intentionally simple. Its purpose is not to prove semantic
equivalence; its purpose is to block obvious drift in thesis-facing results.

## Where It Is Wired

- Shared policy utilities:
  - `thinkdet/inference/fallback.py`
- Prompt-robustness eval:
  - `thinkdet/scripts/eval/eval_prompt_robustness.py`
- RefCOCO eval:
  - `thinkdet/scripts/eval/eval_refcoco.py`
- Shared InternVL evidence-check reranker:
  - reused by `thinkdet/scripts/eval/eval_affordance_benchmark.py`

## What This Does Not Claim

- It does not prove ThinkDet is better.
- It does not solve the layer-selection question.
- It does not replace a learned confidence model.
- It does not prove the refined prompt is a logically perfect paraphrase.

It only gives us an LLM-based second chance:
- verify detector candidates with the MLLM evidence-check reranker
- refine the query with the MLLM
- keep the best-scoring pass that still respects the semantic guard
