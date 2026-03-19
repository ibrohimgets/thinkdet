# ThinkDet v2: Text-Conditioned Reasoning with InternVL 3.5-1B

## Problem Statement

The current ThinkDet architecture does **not** enable reasoning.
The LLM (Qwen2-0.5B) receives only image tokens — it never sees the text prompt.
This means `H_vlm = f(image)` is query-independent: the same features are produced
regardless of what the user asks. The system is a visual feature enhancer, not a reasoning detector.

## Goal

Make the MLLM actually "think" about the task by seeing **both image and text jointly**,
producing text-conditioned reasoning features that guide detection.

---

## InternVL 3.5-1B Component Specs

| Component | Details |
|---|---|
| **InternViT-300M** | hidden=1024, 24 layers, patch=14, image=448 |
| **MLP Connector** | Pixel shuffle, downsample_ratio=0.5 |
| **Qwen3-0.6B LLM** | hidden=1024, 28 layers, 16 heads, intermediate=3072 |
| **Visual tokens** | 1024 per patch → 256 after pixel shuffle |
| **Total params** | ~1.1B (0.3B vision + 0.8B language) |

---

## How InternVL Solves the Core Problem

InternVL's native forward pass is exactly the pattern we need:

```
Image → InternViT → pixel_shuffle → [256 visual tokens]
                                          ↓
                              LLM input = [visual_tokens ; text_tokens]
                                          ↓
                              Qwen3 self-attention (joint reasoning)
                                          ↓
                              Output hidden states are TEXT-CONDITIONED
```

The LLM's self-attention naturally cross-attends between image and text.
When you ask "find red cars near pedestrians," the visual token representations
at layer N are **already modulated by the text query** — because self-attention mixes them.

This is fundamentally different from the current Qwen2 setup where vision tokens enter alone.

---

## Conceptual Extraction Strategy

Run InternVL with `[image + text_prompt]` as input, then extract intermediate hidden states
corresponding to the **visual token positions**. These are reasoning features —
visual representations conditioned on the text query through the LLM's self-attention.

```
Input sequence:  [vis_1, vis_2, ..., vis_256, "find", "red", "cars", ...]
                  \_________________________/  \________________________/
                       visual tokens              text query tokens

After N layers of self-attention:

H_vlm = hidden_states[layer_N][:, :256, :]    ← visual positions only
```

**Why this works:** At layer N, each visual token has attended to the text tokens
multiple times. `vis_42` no longer represents just "patch at position 42" — it represents
"patch at position 42, in the context of finding red cars."
The feature is query-dependent by construction.

---

## Critical Design Decisions

### 1. Which Layers to Extract From

| Layer Range | Behavior | Suitability |
|---|---|---|
| Early (0–6) | Mostly local spatial features, minimal text conditioning | Too shallow for reasoning |
| Mid (7–14) | Cross-modal fusion active, visual tokens reflect text semantics | Best balance of spatial detail + reasoning |
| Late (20–28) | Heavy text dominance, spatial structure degrades | Better for captioning, worse for detection |

**Recommendation:** Extract from a mid-layer range (e.g., layer 8–12).
Detection needs spatially-grounded features, not fully-abstract language features.
The exact layer should be tuned empirically.

### 2. Full Model vs. Truncated

Running all 28 layers is expensive. Two options:

- **Truncated (first N layers):** Cheaper. With N=12, ~43% of LLM compute
  but likely captures the bulk of cross-modal fusion.
- **Full model, extract mid-layer:** More expensive. Later layers' gradients
  (if unfrozen) can improve mid-layer representations, but not recommended
  for our setting due to compute cost.

**Recommendation:** Truncated to ~10–14 layers for a detection addon.

### 3. Visual Token Count

InternVL produces 256 tokens per patch (after pixel shuffle).
With dynamic resolution and up to 12 patches + 1 thumbnail, that's up to **3328 tokens**.

For detection, we likely want features from a **single-scale view** (1 patch = 256 tokens)
or a controlled multi-scale (2–3 patches) to keep the adapter's AP computation manageable.

---

## Comparison: Current vs. Proposed

| Aspect | Current (Qwen2) | Proposed (InternVL) |
|---|---|---|
| LLM input | Vision tokens only | Vision + text tokens jointly |
| H_vlm depends on | Image only | Image AND text query |
| Reasoning | None (feature transform) | Real (cross-modal attention) |
| Alignment training | MSE(vision, text) in shared space | May not be needed — InternVL is pre-aligned |
| Projector | Swin→896 dim MLP (trained from scratch) | Not needed — InternViT + pixel_shuffle already in LLM space |
| Adapter input dim | 896 (Qwen2) | 1024 (Qwen3/InternVL) |

---

## Impact on Training Stages

### Current 3-Stage Pipeline

| Stage | Purpose | Trainable |
|---|---|---|
| 1 | Projector alignment (MSE) | MLP projector |
| 2 | Projector fine-tuning (MSE) | MLP projector |
| 3 | Detection fine-tuning | Adapter + projector + heads |

### Proposed Pipeline

Stages 1–2 exist because the Qwen2 projector needs alignment from scratch.
InternVL's MLP connector is **already trained** to project vision into LLM space.

- **Stages 1–2 may collapse into a single short fine-tuning or be eliminated entirely**
- Main training effort shifts to **Stage 3**: teaching the adapter to translate
  InternVL's text-conditioned features into useful detection signals

---

## Open Questions

1. **Text prompt format** — InternVL expects chat-template input
   (`<image>\nFind all red cars.`). The detection query
   (`"red car . pedestrian . traffic light ."`) needs to be reformatted
   as a natural language instruction for the LLM to reason over meaningfully.

2. **Frozen or partially unfrozen InternVL?** — Freezing entirely preserves
   pretrained reasoning. Unfreezing the last few LLM layers could adapt
   reasoning to detection-specific queries but risks catastrophic forgetting.

3. **InternViT vs. Swin-Tiny for GroundingDINO** — The detector still needs
   its own visual backbone. We either keep Swin-Tiny for GroundingDINO separately,
   or explore whether InternViT features could replace it
   (likely not — different resolution/scale trade-offs).

---

## Summary

The fundamental shift: replace **"LLM as blind feature refiner"** with
**"LLM as joint vision-language reasoner, extract its text-conditioned visual representations."**

```
Current:   Image → Swin → MLP → Qwen2(vision_only) → H_vlm(image)
Proposed:  Image → InternViT → pixel_shuffle ─┐
           Text ─────────────────────────────┤
                                              ↓
                                   Qwen3(vision + text) → H_vlm(image, text)
```

---

## References

- [InternVL3.5-1B on HuggingFace](https://huggingface.co/OpenGVLab/InternVL3_5-1B)
- [InternVL3.5 Blog](https://internvl.github.io/blog/2025-08-26-InternVL-3.5/)
- [InternVL3.5 Paper (arXiv:2508.18265)](https://arxiv.org/abs/2508.18265)
- [InternVL GitHub](https://github.com/OpenGVLab/InternVL)
