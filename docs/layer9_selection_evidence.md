# Layer 9 Selection Evidence

## Question

Why was layer 9 chosen, and how strong is that choice after looking beyond
Flickr30k?

## What We Actually Did

Layer 9 was selected from an **inference-only layer probe**, not from a full
downstream training sweep.

The probe used natural-language grounding prompts and scored each InternVL layer
with a weighted composite:

- 40% disambiguation margin
- 20% paraphrase stability
- 20% token diversity
- 15% cross-image spread
- 5% certainty rate

This is implemented in `thinkdet/tests/ablation/probe_reasoning_layers.py`.

## Which Prompts Were Used

Two prompt families were used in the existing repo artifacts:

1. **Flickr30k**
   - phrase-level expressions from caption annotations
   - merged across captions by phrase id so one reference can keep multiple
     paraphrastic expressions

2. **RefCOCOg val**
   - all valid referring-expression sentences per reference

These are natural-language grounding prompts, not the held-out affordance
prompts used in the final ThinkDet benchmark.

## Existing Results

| Probe dataset | Best layer | Layer 9 score | Key comparison | Interpretation |
|---|---:|---:|---|---|
| Flickr30k val | 9 | 0.735185 | L9 vs L10 = 0.735185 vs 0.718519 | Validation probe favored layer 9 |
| Flickr30k test | 10 | 0.640741 | L10 vs L9 = 0.650000 vs 0.640741 | Small reversal on test |
| RefCOCOg val | 0 | 0.614815 | L0 vs L9 = 0.972222 vs 0.614815 | Strong early-layer preference |

## What This Means

The honest interpretation is:

- Layer 9 is a **validation-selected default** from Flickr30k val.
- Layer 9 is **not** a cross-dataset optimum.
- RefCOCOg suggests that the preferred layer depends strongly on task style.

So the thesis-safe wording should be:

> Layer 9 was the best validation-time default under the early Flickr30k probe,
> but it is not proven to be the best layer across datasets or prompt types.

## What We Should Claim

Safe claim:

- “Layer 9 was chosen by validation protocol as the default layer for the main
  ThinkDet runs.”

Unsafe claim:

- “Layer 9 is the best InternVL layer for ambiguous prompts.”

## Best Next Step

If we want a stronger argument, the next dataset that matters is not another
proxy alone, but the actual **held-out affordance benchmark**, because that is
the thesis target task.
