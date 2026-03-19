# ThinkDet System Investigation (Paper-Ready)

Date: 2026-02-19  
Audience: UAI paper drafting (methods + reliability + validity framing)

## 1) Scope and Ground Truth Sources

This audit is based on implementation and run artifacts, not memory:

- Core architecture: `thinkdet/models/arch.py`, `thinkdet/models/projector.py`, `thinkdet/models/cross_attention.py`, `thinkdet/models/decoder_layer.py`
- Stage 1/2 training: `thinkdet/scripts/training/train_stage1_tma.py`, `thinkdet/scripts/training/train_stage2_tma.py`
- Loss/data: `thinkdet/scripts/training/train_stage1.py`, `thinkdet/data/coco_grounding.py`
- Official COCO eval path: `thinkdet/scripts/eval/eval_official_protocol.py`
- Layer probe methodology: `thinkdet/tests/ablation/probe_reasoning_layers.py`
- Layer probe results: `thinkdet/results/layer_probe_flickr_val/layer_probe_results.json`, `thinkdet/results/layer_probe_flickr_test/layer_probe_results.json`
- Main run logs: `thinkdet/logs/stage1_tma_full_7gpu_20260218_152649.log`, `thinkdet/logs/stage2_tma_layer9_m8_e2_7gpu_20260218_234913.log`

## 2) What the Current System Actually Is

Current ThinkDet is a **Text Memory Augmentation (TMA)** system on top of GroundingDINO:

- GroundingDINO backbone: frozen
- InternVL (feature extractor): frozen
- Learned module: small TMA augmenters injected into DINO decoder text memory path
- Injection layers: `[1, 3, 5]`
- No delta injection, no confidence gate, no uncertainty weighting, no KD in current TMA path

This is explicit in `thinkdet/models/arch.py` and `thinkdet/models/cross_attention.py`.

## 3) Architecture: End-to-End Dataflow

### 3.1 Forward path

1. Input image is processed twice:
- InternVL view: `448x448` path for `H_vlm`
- DINO view: short side 800, max 1333

2. InternVL extraction:
- Run joint image+text path
- Extract visual-token hidden states from selected layer(s)
- Current training uses layer 9 (`extract_layers=[9]`)

3. TMA augmenter per injected decoder layer:
- Project `H_vlm` from InternVL dim (1024) to DINO dim (256)
- M learnable query tokens (`M=8`) cross-attend over InternVL visual tokens
- Output scaled by learnable `alpha` (init 0.1)

4. Decoder integration:
- Before each wrapped DINO decoder layer executes, prepend augmented tokens to `memory_text`
- DINO's native text cross-attention then attends over original text tokens + TMA tokens

### 3.2 Mathematical sketch

For each injected layer:

- `K,V = LN(W_proj * H_vlm)`
- `Q = learned_queries`
- `A = MHA(Q, K, V)`  (shape `[B, M, 256]`)
- `A_scaled = alpha * A`
- `memory_text' = concat(memory_text, A_scaled)`

## 4) Training System (How It Was Built Operationally)

## Stage 1 (TMA-only warmup)

From `train_stage1_tma.py`:

- Dataset: COCO `train2017` (`117,266` images observed in logs)
- GPUs: 7
- Per-GPU batch: 1
- Grad accumulation: 8
- Effective batch: 56
- AMP: fp16
- Epochs: 2
- LR (TMA): `2e-4`
- Grad clip: `1.0`
- `OMP_NUM_THREADS=1`, `cudnn.benchmark=True`
- Gradient health check at step 100
- Alpha logging every 200 steps for layers 1/3/5

Observed in log (`stage1_tma_full_7gpu_20260218_152649.log`):

- Trainable params: `1,583,619 / 1,235,321,093` (0.13%)
- Gradient check passed ("All augmenter params have healthy gradients")
- Alpha stayed small and stable:
  - L1: `0.0997 -> 0.0879`
  - L3: `0.0999 -> 0.0962`
  - L5: `0.0988 -> 0.0968`
- Checkpoints saved at step 2000, epoch1, step4000, epoch2

## Stage 2 (TMA + detection heads)

From `train_stage2_tma.py`:

- Loads Stage-1 checkpoint
- Unfreezes:
  - TMA modules
  - `class_embed`, `bbox_embed`, `enc_output`, `enc_score_head`, `enc_bbox_head`
- Keeps backbone and InternVL frozen
- Epochs: 2
- LR: `5e-5` for TMA and heads

Observed in log (`stage2_tma_layer9_m8_e2_7gpu_20260218_234913.log`):

- Trainable params: `1,716,231 / 1,235,321,093` (0.14%)
- Gradient check summary: `TMA=27 ok, Heads=6 ok, Issues=0`
- Alpha remained stable near Stage-1 values
- Epoch checkpoints saved

## 5) Dataset and Loss Design

From `thinkdet/data/coco_grounding.py` and `compute_detection_loss` in `train_stage1.py`:

- Query for COCO training uses **fixed 80-class vocabulary string**, not image GT categories.
- This is meant to avoid direct GT category leakage in text query construction.
- Matching/loss:
  - Hungarian matching cost: `5*L1 + 2*GIoU + class_cost`
  - Final loss per matched pair: `5*L1 + 2*GIoU + focal`

## 6) Why Layer 9 Was Chosen (Not 13)

Decision source is the Flickr30k **validation** layer probe (not test set tuning).

Method file: `thinkdet/tests/ablation/probe_reasoning_layers.py`  
Output files:
- `thinkdet/results/layer_probe_flickr_val/layer_probe_results.json`
- `thinkdet/results/layer_probe_flickr_test/layer_probe_results.json`

Composite score uses rank-normalized weighted criteria:

- 40% disambiguation margin
- 20% paraphrase stability
- 20% token diversity
- 15% cross-image spread
- 5% certainty rate

### L9 vs L10 evidence

| Split | L9 Composite | L10 Composite | Winner |
|---|---:|---:|---|
| Flickr30k val | 0.735185 | 0.718519 | **L9** |
| Flickr30k test | 0.640741 | 0.650000 | L10 |

Interpretation:

- Layer 9 was selected by validation protocol.
- Test showed a small reversal, but selection should remain val-driven to avoid test leakage.
- This is why the run policy used layer 9.

## 7) Baseline AP Clarification (43 vs 48.5 vs 52.5)

Three different numbers came from different protocols:

1. **Official-style protocol in this repo**  
File: `thinkdet/results/eval/official_protocol_eval.json`
- Baseline AP: `0.4844` (48.44)
- TMA Stage-1 AP: `0.4620`
- Settings: no confidence threshold, top-300 detections, COCO API

2. **Thresholded custom eval (conf=0.3)**
File: `thinkdet/results/eval/tma_full_stage1_epoch2_eval.json`
- Baseline AP: `0.4316` (43.16)
- TMA AP: `0.3996`

3. **README headline 52.5**
- In local GroundingDINO README, the explicit zero-shot Swin-T OGC eval section says expected result is about **48.5** with `demo/test_ap_on_coco.py`.
- The `52.5` line is a broad highlight statement and does not match the concrete Swin-T OGC eval table/recipe in the same README.

Conclusion:

- Your measured baseline `48.44` under official-like protocol is consistent with local GroundingDINO Swin-T OGC expectations.
- The `43.x` result is from a stricter thresholded custom protocol, so not directly comparable to official no-threshold AP.

## 8) Are the Visual "Failure Cases" Fake?

Short answer: **not fake inference, but strongly curated evidence**.

From `visual_comparison_summary.json` files in `results/visual_compare/...`:

- Cases are selected with explicit oracle filters using GT IoU:
  - `min_gap`
  - `max_base_iou`
  - `min_tma_iou`
- Example criteria in one summary:
  - `min_gap=0.3`
  - `max_base_iou=0.35`
  - `min_tma_iou=0.55`
- This is valid for qualitative demonstration, but it is cherry-picked by design.

Implication for paper:

- Present these figures as "illustrative retrieval of large-gap cases under predefined criteria", not as random sample behavior.

## 9) Important Codebase Reality: Legacy vs Current Paths

There are legacy scripts still in repo (`train_stage1.py`, `train_stage2_heads_kd.py`, parts of `eval_stage1_checkpoint.py`) that refer to older APIs (`set_stage_a`, gate/delta concepts).

Current active TMA implementation is in:

- `train_stage1_tma.py`
- `train_stage2_tma.py`
- `models/arch.py` with `set_tma_only` and `set_tma_and_heads`

For paper reproducibility, cite only the current TMA pipeline unless explicitly analyzing the legacy branch.

## 10) Suggested UAI Framing (Honest + Defensible)

Potential core claim:

- "Text-conditioned visual memory augmentation for grounding under ambiguous language, with frozen backbone and low trainable parameter budget."

What you can claim strongly now:

- Stable Stage-1/2 training with very low trainable fraction
- Clear implementation-level mechanism
- Layer-selection protocol (val-driven)
- Existence of targeted qualitative wins on compositional prompts

What should be framed carefully:

- Overall AP gains are not yet positive under current full COCO official evaluation.
- Qualitative win sets are curated by IoU-gap criteria.

## 11) Repro Artifacts to Cite in Paper

- Main Stage-1 checkpoint:
  - `thinkdet/checkpoints/stage1_tma/layer9_tma_m8_full_2ep_7gpu_20260218_152649/thinkdet_tma_stage1_epoch2.pth`
- Main Stage-2 checkpoint:
  - `thinkdet/checkpoints/stage2_tma/layer9_tma_m8_20260218_234913/thinkdet_tma_stage2_epoch2.pth`
- Official protocol result JSON:
  - `thinkdet/results/eval/official_protocol_eval.json`
- Layer probe result JSONs:
  - `thinkdet/results/layer_probe_flickr_val/layer_probe_results.json`
  - `thinkdet/results/layer_probe_flickr_test/layer_probe_results.json`

