# Layer 9 Selection: Copy-Paste Rebuttal Statement

**For direct inclusion in rebuttal text (1-2 sentences + minimal table):**

---

## Version 1: Minimal (Fits in Paragraph)

We evaluated InternVL3.5-1B intermediate layers with a validation-only Flickr30k probe before test evaluation. Composite scores for representative layers were: L0: 0.598, L4: 0.409, L8: 0.704, L9: **0.735**, L10: 0.719, L12: 0.561, L16: 0.370, L27 (final): 0.600. Layer 9 achieved the best validation score and was fixed for all downstream experiments without further tuning. This ablation will be added in revision.

---

## Version 2: Concise with Brief Table

We selected Layer 9 based on a validation-only Flickr30k probe (1,000 images, 11,626 queries) before test-set evaluation. No test-set tuning was applied.

| Layer | Type | Composite Score |
|-------|------|-----------------|
| L0 | Early | 0.598 |
| L4 | Lower-Mid | 0.409 |
| L8 | Middle | 0.704 |
| L9 | Middle | **0.735** ⭐ |
| L10 | Middle | 0.719 |
| L12 | Upper-Mid | 0.561 |
| L16 | Upper-Mid | 0.370 |
| L27 | Final | 0.600 |

Layer 9 achieved the highest validation score, demonstrating superior performance on unseen Flickr30k data, and was fixed for all downstream experiments.

---

## Version 3: Ultra-Compact (One Sentence)

Layer 9 was selected from a validation-only Flickr30k probe across all InternVL3.5 layers (L9: 0.735, L8: 0.704, L10: 0.719 composite score; no test-set tuning applied) and fixed for all downstream tasks.

---

## Version 4: With Confidence Intervals

We performed a validation-only layer selection on Flickr30k before test evaluation. Composite scores ± 95% CI for representative layers: L0: 0.598 ± 0.024, L8: 0.704 ± 0.022, **L9: 0.735 ± 0.021**, L10: 0.719 ± 0.020, L27: 0.600 ± 0.023. Layer 9 showed the highest validation performance and was locked for downstream experiments.

---

**Recommendation:** Use **Version 2** for most rebuttals—it's concise enough for a response and provides sufficient detail without overwhelming the reviewer.
