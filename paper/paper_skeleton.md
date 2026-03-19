# ThinkDet: Honest Research Summary

## What ThinkDet Actually Is

ThinkDet is a parameter-efficient adapter experiment built on top of:

- frozen GroundingDINO
- frozen InternVL
- a small Text Memory Augmenter (TMA)

The active idea is simple:

1. run InternVL on image + query
2. extract query-conditioned hidden states
3. pool them into `M=8` summary tokens
4. fuse those tokens into GroundingDINO decoder text memory

The current active implementation lives in:

- `thinkdet/models/arch.py`
- `thinkdet/models/projector.py`
- `thinkdet/models/cross_attention.py`
- `thinkdet/models/decoder_layer.py`
- `thinkdet/scripts/training/train_stage1_tma.py`
- `thinkdet/scripts/training/train_stage2_tma.py`
- `thinkdet/scripts/training/train_unified.py`

## What The Repo Evidence Supports

### Narrow Positive Result

On the held-out affordance benchmark (`512` test queries, `8` prompt types),
ThinkDet shows a small improvement over the baseline:

- baseline `hit@0.5_top1 = 16.99%`
- Stage 1 `hit@0.5_top1 = 19.34%`
- Stage 2 `hit@0.5_top1 = 19.53%`

The biggest gains are concentrated in:

- `talk_on`: `17.19% -> 31.25%`
- `carry_in`: `15.62% -> 25.00%`

Calibration on that same custom benchmark also improves:

- `ECE@10`: `0.1478 -> 0.0339`
- `Brier`: `0.1641 -> 0.1571`

## What The Repo Evidence Does Not Support

### Not A General Detector Improvement

The early TMA checkpoints regress on standard detection:

- official COCO AP drops from `48.44` to `46.20`
- the main Stage 2 RefCOCO-family checkpoint loses on all `8` reported splits

The later unified residual run mostly recovers baseline COCO AP (`48.03`),
but it still does not establish a broad win on standard benchmarks.

### Not A Proven Reasoning System

The custom affordance benchmark is useful, but it is still a small,
hand-authored prompt-to-COCO-category stress test. The benchmark builder maps
prompts like:

- `talk_on -> cell phone`
- `read -> book`
- `carry_in -> backpack | handbag | suitcase`

That means the strongest current claim is:

> query-conditioned VLM features can help on a narrow functional-query stress test

That is much weaker than:

> ThinkDet solves reasoning-guided open-vocabulary detection

### Not SOTA Even On The Custom Benchmark

This repo already includes a stronger comparison on the same benchmark:

- ThinkDet Stage 2: `19.53%`
- OWL-ViT: `25.20%`

So the result is interesting, but it is not dominant.

## Claims We Can Make

- The adapter mechanism is implemented and testable.
- The repo contains a real positive result on a narrow custom benchmark.
- The repo also contains strong negative transfer evidence on COCO and RefCOCO.
- The project is worth presenting as an honest research prototype.

## Claims We Should Not Make

- ThinkDet is a general improvement over GroundingDINO.
- ThinkDet is a paper-ready reasoning detector.
- The custom affordance benchmark proves broad commonsense grounding.
- The repo currently demonstrates production or benchmark-grade reliability.

## Suggested Framing

If this project is kept as a research repo, the honest framing is:

> ThinkDet is an experimental VLM-to-detector adapter that shows small gains on
> a custom functional-query benchmark, while failing to transfer cleanly to
> standard detection and referring-expression benchmarks.

That framing matches the saved artifacts and keeps the interesting part without
turning the repo into marketing fiction.
