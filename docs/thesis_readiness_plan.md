# ThinkDet Thesis Readiness Plan

## Thesis Claim

The thesis should make a narrow and defensible claim:

> ThinkDet is a lightweight MLLM-guided adapter for ambiguity-heavy
> open-vocabulary grounding. It can help some affordance-style prompts, but the
> gains are narrow and sensitive to scoring, ranking, and inference policy.

Do not claim:

- ThinkDet is a general improvement to GroundingDINO
- layer 9 is proven best
- fallback proves the base detector is intelligent
- prompt refinement is always semantically faithful

---

## Canonical Thesis Story

The project should be presented in this order:

1. **Base detector with corrected query-conditioned scoring**
2. **ThinkDet adapter as the primary model**
3. **LLM fallback as a secondary recovery policy**

That order matters.

If fallback is introduced before the corrected scorer, the thesis risks
confusing a ranking bug with a reasoning gain.

---

## Core Method Sections

The thesis can be structured around four technical pieces:

1. **Adapter mechanism**
   - frozen InternVL
   - frozen GroundingDINO
   - Text Memory Augmenter
   - low trainable parameter count

2. **Affordance benchmark**
   - held-out functional-query grounding stress test
   - calibration-aware evaluation

3. **Failure analysis**
   - layer-selection uncertainty
   - ranking/scoring failures
   - benchmark transfer failures on COCO and RefCOCO

4. **Inference-time fallback**
   - LLM feedback over top-k candidates
   - LLM prompt refinement
   - semantic-preservation guard

---

## Corrected Rerun Status

The affordance benchmark rerun with corrected query-conditioned scoring is now
available and should be treated as the canonical thesis artifact:

- `thinkdet/results/eval/affordance_benchmark_eval_v1_test_corrected_20260308.json`
- `thinkdet/results/eval/affordance_benchmark_eval_v1_test_corrected_20260308.md`

Key corrected result:

- baseline `hit@0.5_top1 = 16.99%`
- Stage 2 `hit@0.5_top1 = 17.77%`
- net gain `+0.78pp`

Important:

- the older affordance tables remain useful only as historical provenance
- the corrected rerun does not preserve the earlier strong calibration-gain story

Recommended thesis table set:

- baseline
- ThinkDet Stage 1
- ThinkDet Stage 2
- optional unified checkpoint as the practical best variant

Keep the evaluation protocol fixed across all of them.

---

## Fallback Positioning

Fallback is allowed and can be important to the thesis, but it must be framed
correctly.

### Primary policy

- run ThinkDet normally
- if confidence or margin is weak, try LLM feedback
- if still weak, try LLM prompt refinement
- accept refined output only if semantic-preservation guard passes

### What to emphasize

- fallback is a recovery mechanism
- refinement currently appears stronger than yes/no feedback
- fallback should be evaluated separately from the base model

### What to avoid

- counting fallback wins as proof that the base model reasoned correctly
- hiding semantic drift in refined prompts

---

## Required Thesis Diagnostics

At least one chapter or section should explicitly show:

1. **Ranking failure vs. representation failure**
   - the correct box was already in top-k
   - the base ranker chose the wrong box

2. **Effect of corrected scoring**
   - the same prompt before and after the scoring fix

3. **Fallback success and failure**
   - one clean success
   - one failure or drift example

Suggested repo artifacts:

- `thinkdet/results/inference/diagnostic_topk_carry_in_99114_20260307_112733.json`
- `thinkdet/results/inference/postfix_topk_carry_in_99114_20260307_114742.json`
- `thinkdet/results/inference/llm_fallback_focus_carry_in_99114_20260307_084927.json`

---

## Minimum Thesis Experiment Package

If time is tight, do only this:

1. Rerun affordance benchmark with corrected scorer
2. Report historical COCO / RefCOCO regressions as limitations
3. Add one ranking diagnostic case study
4. Add one fallback case study with semantic-guard discussion

That is enough for a rigorous mixed-results thesis.

---

## Nice-To-Have Ablations

If there is time, add:

- fallback on vs off
- feedback only vs refinement only vs both
- semantic guard on vs off
- layer 9 default vs one alternative layer

These are useful, but not required for the thesis to be legitimate.

---

## Writing Rules

Use these rules consistently:

- separate **method claim** from **fallback claim**
- separate **historical artifact** from **final rerun number**
- report negative transfer honestly
- describe layer 9 as the current default, not a solved result
- call the affordance benchmark a stress test, not a broad reasoning benchmark

---

## Thesis-Ready Definition

The project is thesis-ready when all of the following are true:

- the affordance benchmark has been rerun with the corrected scorer
- the thesis numbers match the code currently in the repo
- fallback is presented as secondary inference policy
- semantic drift in prompt refinement is acknowledged and guarded
- COCO / RefCOCO regressions are kept in the main narrative

Until then, the project is a promising research prototype, not a locked final
thesis result.
