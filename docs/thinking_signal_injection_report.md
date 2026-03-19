# ThinkDet: Turning LLM Reasoning Into Better Boxes

## Technical Design Report — Thinking Signal Injection & Hallucination Mitigation

**Date:** February 8, 2026  
**Status:** Design Analysis + Actionable Next Steps  
**Scope:** How reasoning signals flow from the MLLM into the detector, where the current design breaks, and what to fix.

---

## 1. The One-Line Goal

> **Turn LLM reasoning into better bounding boxes.**

The MLLM "thinks" about the image + text query. We extract those thinking signals (`H_vlm`) and inject them into GroundingDINO's decoder so the detector produces text-conditioned, reasoning-guided boxes.

---

## 2. Baseline Evidence: What InternVL Can and Cannot Do

We ran InternVL 3.5-1B on 5 COCO images × 10 tasks (50 total inferences). Results from `internvl_baseline_results.json`:

### 2.1 Strengths (What we want to keep)

| Capability | Evidence | Avg Time |
|---|---|---|
| Captioning | Accurate 1-sentence descriptions, all 5 images correct | ~2.4s |
| Object listing | 10–15 objects identified per scene, good coverage | ~10.2s |
| Scene reasoning | Sensible "what happens next" inferences | ~12.8s |
| Attributes | Colors, materials, sizes described accurately | ~13.0s |
| Spatial layout | Foreground/middleground/background decomposition works | ~5.9s |

### 2.2 Critical Failures (What ThinkDet must fix)

#### ❌ Failure 1: Hallucination

**Image 000000000285** (a bear sitting on grass):

```
Prompt:  "Is there a person in this image?"
Response: "Yes, there is a person in this image. The person is located
           in the center of the image, directly above the bear."
```

The model **hallucinates a person** that does not exist. It also counted "1 person" in the counting task. This is not a random error — the MLLM's language prior expects people in photos and confabulates one.

**Same image, counting task:**
```
Response: "Number of people: 1" ← WRONG (0 people in image)
```

#### ❌ Failure 2: Zero Spatial Precision

Grounding responses produce only vague text like *"center-left"*, *"right side of the image"*. **No coordinates, no bounding boxes.** Example:

```
Prompt:  "Describe approximate location using coordinates or regions"
Response: "Television: Positioned on the left side of the image, mounted on a stand.
           Located at the center-left of the image."
```

This is completely unusable for detection. The MLLM has **understanding** but cannot produce **coordinates**.

#### ❌ Failure 3: Verbosity ≠ Precision

Attribute descriptions run 25+ seconds and 300+ words, but spatial information is less precise than a single `[x1, y1, x2, y2]` box. The reasoning is rich but ungrounded.

### 2.3 Diagnosis Summary

| What InternVL has | What InternVL lacks |
|---|---|
| Text-conditioned visual understanding | Precise spatial coordinates |
| Object-level reasoning & relationships | Hallucination control |
| Scene-level inference | Grounded verification |
| Multi-object awareness | Efficient structured output |

**Conclusion:** The MLLM should NOT produce boxes directly. Instead, we extract its **internal reasoning representations** and let GroundingDINO's decoder — which excels at precise localization — use them to produce better boxes.

---

## 3. Current Architecture: How Thinking Signals Flow

### 3.1 Signal Extraction Pipeline

```
Image ──→ Swin-Tiny (frozen, shared) ──→ [B, 768, H, W]
                                              │
                                    MLP Projector (trainable)
                                              │
                                         [B, 256, 896]
                                              │
                                   Truncated Qwen2 (layers 0–2)
                                              │
                                     H_vlm = [B, 256, 896]
                                       "thinking signal"
```

**Current modules (from codebase):**

| Module | File | Input → Output | Params |
|---|---|---|---|
| `VisualProjector` | `projector.py` | `[B, 768, H, W]` → `[B, L, 896]` | ~1.4M |
| `TruncatedQwen2` | `projector.py` | `[B, L, 896]` → `[B, L, 896]` | ~100M (frozen) |
| `ThinkDetCrossAttentionAdapter` | `cross_attention.py` | `(E_D, H_vlm)` → `E_D'` | ~1.1M |
| `ThinkDetDecoderLayer` | `decoder_layer.py` | Full decoder layer with adapter slot | inherits DINO |

### 3.2 Signal Injection Point

The adapter is injected at the **last decoder layer** of GroundingDINO (layer index 5 of 6):

```python
# From arch.py: _inject_thinkdet_adapters()
inject_idx = num_layers - 1  # Last layer only
```

The decoder layer processing order (from `decoder_layer.py`):

```
1. Self-attention on detector queries          ← standard DINO
2. Text cross-attention (optional)             ← standard DINO
3. ★ ThinkDet cross-attention ★               ← INJECTED HERE
4. Visual cross-attention (deformable)         ← standard DINO
5. FFN                                         ← standard DINO
```

### 3.3 Adapter Mechanism: Adaptation Prompt + Gated Cross-Attention

From `cross_attention.py`, the adapter follows Equations 2–5 from the reference paper:

```
Step 1: AP = LayerNorm(Conv2D(Linear(H_vlm)))        — Adaptation Prompt
Step 2: Q  = E_D @ W_Q                                — Query from detector
Step 3: K  = Concat(AP @ W_K_ap, E_D @ W_K_det)      — Key from AP + detector
Step 4: V  = Concat(AP @ W_V_ap, E_D @ W_V_det)      — Value from AP + detector
Step 5: S  = Q @ K^T / √d                             — Raw attention scores
Step 6: S_gated = [tanh(g) · softmax(S_ap),            — Gated AP attention
                   softmax(S_det)]                      — Normal detector attention
Step 7: Output = S_gated @ V + E_D                     — Residual connection
```

**Key design choices:**

| Choice | Implementation | Purpose |
|---|---|---|
| **Zero-init gate** | `self.gate = nn.Parameter(torch.zeros(...))` | Start as identity (tanh(0)=0), adapter contributes nothing initially |
| **Zero-init output proj** | `nn.init.zeros_(self.out_proj.weight)` | Double safety: even if gate opens, output is zero at init |
| **Split softmax** | Separate softmax for AP and detector portions | AP attention doesn't compete with detector self-attention |
| **Per-head gating** | Gate shape `[1, num_heads, 1, 1]` | Different heads can learn different gate openings |
| **Conv2D on H_vlm** | 3×3 conv after reshaping to 16×16 spatial | Preserves spatial locality in thinking signal |

---

## 4. Critical Problems in the Current Design

### 4.1 🔴 Problem 1: H_vlm Is Query-Independent (The Reasoning Gap)

**This is the central architectural flaw.** From `arch.py`:

```python
def extract_h_vlm(self, images):
    swin_features = self._extract_swin_features(images)   # Image only
    projected = self.mlp_projector(swin_features)          # Image only
    H_vlm = self.truncated_llm(projected)                  # Image only — NO TEXT!
    return H_vlm
```

The LLM receives **only vision tokens**. It never sees the text query. Therefore:

```
H_vlm("find red cars", image_A)  ==  H_vlm("find pedestrians", image_A)
```

The "thinking signal" doesn't think about the task. It's a fixed visual feature transform, not reasoning. The entire purpose of using an LLM is wasted.

**Why this causes hallucination:** Without text conditioning, the LLM applies generic priors. It "knows" photos usually contain people, so its features encode person-like patterns even when none exist. When these features are injected into the detector, they can bias queries toward phantom objects.

### 4.2 🔴 Problem 2: Single-Layer Injection Is Too Narrow

The adapter is injected at only 1 of 6 decoder layers:

```python
inject_idx = num_layers - 1  # Only the LAST layer
```

GroundingDINO's decoder performs **iterative refinement** — each layer refines box positions. Injecting reasoning only at the last layer means:

- Layers 0–4: Refine boxes without any reasoning guidance
- Layer 5: Suddenly receives reasoning signal, but it's too late to correct spatial errors from earlier layers

The thinking signal arrives after the detector has already committed to spatial positions.

### 4.3 🟡 Problem 3: No Hallucination Feedback Loop

The adapter has no mechanism to **verify** that its outputs are grounded. The gate can open (let thinking signals through) or close (ignore them), but there's no signal that tells the gate "the MLLM hallucinated a person — close."

Currently, the only safeguard is:
- Zero-init gate (starts closed)
- Gradient-based learning (eventually learns to open/close appropriately)

But there's no **explicit** anti-hallucination mechanism.

### 4.4 🟡 Problem 4: Alignment Stage May Be Unnecessary

Stages 1–2 train the projector with MSE loss to align vision and text in LLM space:

```python
loss = F.mse_loss(H_vlm, H_text)
```

If we switch to InternVL (which is **already** vision-language aligned), these stages become redundant. The projector (Swin → LLM space) would be replaced by InternVL's native pixel-shuffle connector.

---

## 5. Proposed Fix: Text-Conditioned Signal Injection

### 5.1 Architecture v2: InternVL as Reasoning Backbone

Replace the blind Qwen2 pathway with InternVL's native joint reasoning:

```
                    ┌──────────────────────────────────────────────┐
                    │              InternVL 3.5-1B                 │
                    │                                              │
Image ──→ InternViT ──→ pixel_shuffle ──→ [256 visual tokens]     │
                                              │                    │
Text query ─────────────────────────→ [text tokens]                │
                                              │                    │
                              Qwen3 (joint self-attention)         │
                                              │                    │
                              Extract visual positions at layer N  │
                    └──────────────────────────────────────────────┘
                                              │
                                    H_vlm(image, text) = [B, 256, 1024]
                                              │
                              ┌───────────────┼───────────────┐
                              ▼               ▼               ▼
                        Decoder L1       Decoder L3       Decoder L5
                       (early guide)   (mid refine)    (final refine)
```

**Key change:** `H_vlm` is now a function of **both image and text**:

```python
# BEFORE (current):
H_vlm = LLM(vision_tokens)              # query-independent

# AFTER (proposed):
H_vlm = LLM([vision_tokens; text_tokens])  # query-conditioned
```

When you ask "find red cars near pedestrians", the visual token at position 42 no longer represents just "patch 42" — it represents "patch 42, in the context of finding red cars near pedestrians." The feature is query-dependent by construction.

### 5.2 Multi-Layer Injection with Progressive Gating

Instead of single-layer injection, inject at **3 decoder layers** with different roles:

```python
# Proposed injection schedule
injection_layers = {
    1: "coarse_spatial",    # Early: broad spatial guidance ("objects are in left half")
    3: "semantic_refine",   # Mid: category-aware refinement ("that's a car, not a truck")
    5: "final_adjust",      # Late: fine-grained adjustment ("tighten box around wheel")
}
```

Each injection point uses the **same H_vlm** but with a **separate gate per layer**:

```
Layer 1 gate: g₁ = tanh(γ₁) · 0.3    ← Small influence early (spatial hints)
Layer 3 gate: g₃ = tanh(γ₃) · 0.6    ← Medium influence mid (semantic)
Layer 5 gate: g₅ = tanh(γ₅) · 1.0    ← Full influence late (refinement)
```

The progressive scaling (`0.3, 0.6, 1.0`) prevents early layers from being dominated by reasoning signals that haven't been spatially grounded yet.

**Implementation change to `arch.py`:**

```python
# BEFORE:
inject_idx = num_layers - 1  # Only last layer

# AFTER:
inject_indices = [1, 3, 5]   # Early, mid, late
for idx in inject_indices:
    layer = decoder_layers[idx]
    thinkdet_layer = ThinkDetDecoderLayer(
        ...,
        gate_scale=gate_scales[idx],  # Progressive scaling
    )
    thinkdet_layer.load_state_dict(layer.state_dict(), strict=False)
    decoder_layers[idx] = thinkdet_layer
```

### 5.3 Adapter Design Changes

Update the cross-attention adapter for the new signal dimensions:

| Parameter | Current (Qwen2) | Proposed (InternVL) |
|---|---|---|
| `mllm_hidden_dim` | 896 | 1024 |
| `num_vis_tokens` | 256 | 256 (single patch) |
| `vis_spatial_size` | 16×16 | 16×16 |
| Injection layers | 1 (last only) | 3 (layers 1, 3, 5) |
| Gate init | `zeros` | `zeros` (unchanged) |
| Gate scaling | None | Progressive `[0.3, 0.6, 1.0]` |

---

## 6. Hallucination Mitigation Strategy

### 6.1 Why the MLLM Hallucinates

From the baseline results, the hallucination pattern is clear:

```
Bear image → "Is there a person?" → "Yes, there is a person"
Bear image → "How many people?"   → "Number of people: 1"
```

**Root cause:** The MLLM's language model has a strong prior that photos contain people. Without grounding in actual visual evidence, it generates text that satisfies its language statistics rather than visual truth.

When this hallucinated reasoning enters the detector via `H_vlm`, it can:
1. Create phantom object queries (detecting non-existent people)
2. Bias existing queries toward wrong categories
3. Shift attention to empty image regions

### 6.2 Defense Layer 1: Text Conditioning Eliminates Blind Hallucination

The single biggest fix is **text conditioning itself**. When InternVL processes `[image + "find all bears"]`, its self-attention focuses visual tokens on bear-relevant regions. The model is less likely to hallucinate people because the text prompt constrains its reasoning.

Current: `H_vlm = f(image)` → LLM applies generic priors → hallucination  
Proposed: `H_vlm = f(image, "find bears")` → LLM focuses on bears → grounded

### 6.3 Defense Layer 2: Confidence-Gated Injection

Add a **confidence estimator** that predicts whether the MLLM's reasoning is reliable:

```python
class ConfidenceGate(nn.Module):
    """Predicts per-token confidence for H_vlm injection."""
    
    def __init__(self, hidden_dim=1024):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, h_vlm):
        # confidence ∈ [0, 1] per token
        confidence = self.gate(h_vlm)          # [B, 256, 1]
        return h_vlm * confidence              # Suppress low-confidence tokens
```

This is trained end-to-end: if a token's signal leads to false positive detections, the confidence gate learns to suppress it. The gradient flows from the detection loss → through the adapter → back to the confidence gate.

### 6.4 Defense Layer 3: Visual-Semantic Consistency Check

Before injecting `H_vlm` into the decoder, compare it with actual visual features:

```python
def consistency_check(h_vlm, visual_features):
    """
    Suppress H_vlm tokens that are inconsistent with raw visual evidence.
    
    If the MLLM says "person here" but the visual features show "grass",
    the cosine similarity will be low → suppress that token.
    """
    # Project to same space
    v_proj = linear_v(visual_features)     # [B, 256, d]
    h_proj = linear_h(h_vlm)              # [B, 256, d]
    
    # Per-token cosine similarity
    cos_sim = F.cosine_similarity(v_proj, h_proj, dim=-1)  # [B, 256]
    
    # Soft mask: high similarity → trust MLLM, low → suppress
    mask = torch.sigmoid(cos_sim * temperature)  # [B, 256]
    
    return h_vlm * mask.unsqueeze(-1)
```

**Intuition:** If InternVL's reasoning says "person in position 42" but the raw Swin features at position 42 look like grass, the consistency score will be low, and that thinking signal gets suppressed before reaching the detector.

### 6.5 Defense Layer 4: Detection-Level Verification (Post-hoc)

After the detector produces boxes, cross-check high-confidence detections against the MLLM:

```
Detector says: "person at [120, 50, 200, 300] with confidence 0.85"
    → Crop that region
    → Ask InternVL: "Is this a person? Yes or No."
    → If "No" → suppress detection
```

This is computationally expensive and only needed for critical applications (autonomous driving, medical imaging). For standard benchmarks, layers 1–3 should suffice.

### 6.6 Hallucination Defense Summary

| Defense | Layer | When | Cost | Expected Impact |
|---|---|---|---|---|
| Text conditioning | Architecture | Always (built-in) | Free | High — eliminates blind hallucination |
| Confidence gate | Training | Per-token in adapter | ~0.3M params | Medium — learns to suppress unreliable tokens |
| Consistency check | Inference | Before injection | 1 cosine sim + mask | Medium — catches visual-semantic mismatch |
| Post-hoc verification | Inference | After detection | Extra MLLM forward | Low frequency — last resort for critical apps |

---

## 7. Revised Training Pipeline

### 7.1 Why Stages 1–2 Change

**Current pipeline** (3 stages):
1. Projector alignment: MSE(H_vlm, H_text) — teach MLP to map Swin → Qwen2 space
2. Projector fine-tuning: Same loss, lower LR
3. Detection fine-tuning: Adapter + projector + heads

**Problem:** Stages 1–2 exist because we're forcing Swin features through a random MLP into Qwen2. InternVL already has a trained connector (pixel shuffle + MLP) from vision to LLM space. We don't need to train one from scratch.

### 7.2 Proposed Pipeline (2 stages)

| Stage | Goal | Trainable | Loss | Data |
|---|---|---|---|---|
| **Stage A** | Adapter warm-up | Adapter + confidence gate (~1.5M) | Detection loss (focal + L1 + GIoU) | COCO train |
| **Stage B** | Joint fine-tuning | Adapter + confidence gate + detection heads + last 2 InternVL LLM layers (~15M) | Detection loss + consistency loss | COCO train |

**Stage A** keeps InternVL fully frozen and only trains the adapter. The zero-init gate gradually opens as the adapter learns to use the thinking signals productively.

**Stage B** unfreezes the last 2 LLM layers so InternVL can adapt its reasoning to detection-specific queries (e.g., learning that "red car . pedestrian ." means "find these categories" rather than "write a sentence about red cars").

### 7.3 Loss Function

```python
L_total = L_detect + λ_consist * L_consistency

where:
    L_detect    = focal_loss(pred_cls, gt_cls)
                + λ_L1 * L1_loss(pred_box, gt_box)
                + λ_giou * giou_loss(pred_box, gt_box)
    
    L_consistency = 1 - mean(cosine_sim(h_vlm_proj, visual_proj))
```

The consistency loss encourages `H_vlm` to stay correlated with actual visual evidence, providing a gradient signal against hallucinated features.

---

## 8. Implementation Roadmap

### Phase 1: InternVL Integration (Week 1–2)

```
□ Load InternVL 3.5-1B and verify it runs on current hardware
□ Implement text-conditioned H_vlm extraction:
    - Input: image + text prompt → InternVL forward
    - Output: visual hidden states at layer N → H_vlm [B, 256, 1024]
□ Update adapter dimensions: mllm_hidden_dim 896 → 1024
□ Verify H_vlm changes with different text prompts (same image)
□ Benchmark inference time vs. current Qwen2 pathway
```

### Phase 2: Multi-Layer Injection (Week 2–3)

```
□ Modify _inject_thinkdet_adapters() for 3 injection points
□ Implement progressive gate scaling [0.3, 0.6, 1.0]
□ Add per-layer gate monitoring (log gate values during training)
□ Test that zero-init still works (model == GroundingDINO at init)
□ Verify gradient flow through all 3 injection points
```

### Phase 3: Hallucination Defenses (Week 3–4)

```
□ Implement ConfidenceGate module
□ Implement consistency_check function
□ Add consistency loss to training pipeline
□ Create hallucination test suite:
    - Bear image + "find people" → should detect 0 people
    - Empty road + "find cars" → should detect 0 cars
    - 3 cats image + "how many dogs" → should detect 0 dogs
□ Measure false positive rate before/after defenses
```

### Phase 4: Training & Evaluation (Week 4–6)

```
□ Stage A: Train adapter only on COCO train2017
□ Evaluate on COCO val2017 (mAP, mAP@50, mAP@75)
□ Run hallucination test suite — measure FP reduction
□ Stage B: Unfreeze last 2 LLM layers, joint fine-tuning
□ Final evaluation: compare ThinkDet v2 vs. baseline GroundingDINO
□ Ablation: with/without text conditioning, 1 vs. 3 injection layers
```

---

## 9. Expected Outcomes

| Metric | Baseline GroundingDINO | ThinkDet v1 (current) | ThinkDet v2 (proposed) |
|---|---|---|---|
| mAP (COCO val) | ~48.4 | ~48.4 (no real reasoning) | **Target: 50+** |
| False positives on hallucination suite | N/A | Unknown (no text cond.) | **Target: <5%** |
| Query-dependent detection | ❌ | ❌ | ✅ |
| Text-conditioned features | ❌ | ❌ | ✅ |
| Stages needed | 0 | 3 | 2 |

---

## 10. Summary

The core insight is simple:

```
Current:  LLM sees image only    → H_vlm is a fancy feature transform → no reasoning
Proposed: LLM sees image + text  → H_vlm encodes actual thinking     → better boxes
```

Three things make this work:

1. **Text conditioning** — InternVL's self-attention naturally cross-attends between visual and text tokens, producing query-specific features. This is the primary fix.

2. **Multi-layer injection** — Reasoning signals guide detection at multiple stages of refinement, not just the last layer. Progressive gating prevents early interference.

3. **Hallucination defenses** — Confidence gating + consistency checks ensure hallucinated signals (like phantom people in a bear photo) get suppressed before reaching the detector.

The result: an MLLM that actually **thinks** about what to detect, producing reasoning-guided features that a precise detector turns into accurate boxes.

---

## References

- ThinkDet v1 codebase: `thinkdet/models/{arch, cross_attention, decoder_layer, projector}.py`
- InternVL 3.5-1B: [HuggingFace](https://huggingface.co/OpenGVLab/InternVL3_5-1B)
- Baseline results: `thinkdet/tests/mllm_baseline/results/internvl_baseline_results.json`
- Reasoning architecture proposal: `thinkdet/docs/reasoning_architecture_proposal.md`
- GroundingDINO: [GitHub](https://github.com/IDEA-Research/GroundingDINO)
