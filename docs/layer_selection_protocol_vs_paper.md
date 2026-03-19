# ThinkDet Layer-Selection Plan vs. Paper Method

## Goal

Pick the strongest **single-layer default** for ThinkDet, and determine whether
**multi-layer fusion** beats every single-layer option on the target benchmarks.

This plan is designed to answer a narrower and honest question:

> Which InternVL hidden layer, or small fused layer set, works best for ThinkDet on defined grounding tasks?

It is **not** a plan to prove that one layer is best for every ambiguous prompt.

---

## Why We Need This

The current repo still treats **layer 9** as the default training choice, but the
evidence is mixed:

- Flickr validation proxy prefers layer 9
- Flickr test proxy slightly prefers layer 10
- RefCOCO proxy favors much earlier layers
- The sensitivity probe favors very late layers

That means layer 9 is a reasonable default, but not a settled result.

---

## Current Repo Capabilities

Already available:

- Single-layer extraction via `extract_layer`
- Multi-layer extraction via `extract_layers`
- Simple fusion via `layer_fusion`
- Detector training scripts that already accept these settings
- Probe scripts that can score all InternVL layers cheaply

Relevant codepaths:

- `thinkdet/models/projector.py`
- `thinkdet/models/arch.py`
- `thinkdet/scripts/training/train_stage1_tma.py`
- `thinkdet/scripts/training/train_stage2_tma.py`
- `thinkdet/scripts/training/train_unified.py`
- `thinkdet/scripts/eval/eval_affordance_benchmark.py`
- `thinkdet/scripts/eval/eval_official_protocol.py`
- `thinkdet/tests/ablation/probe_reasoning_layers.py`
- `thinkdet/tests/ablation/probe_layer_sensitivity.py`

What is missing:

- A real **layer-similarity analysis** that partitions InternVL hidden layers into early/mid/late groups
- A formal **shortlist selection rule**
- A single **result table** showing per-bucket winners
- A disciplined rule for when to try fusion

---

## Exact Implementation Plan

## Phase 1: Add Layer-Similarity Analysis

### Objective

Partition InternVL hidden layers into **early / mid / late** groups based on how similar
their extracted visual-token representations are.

### New script

Add:

- `thinkdet/scripts/analysis/analyze_internvl_layer_similarity.py`

### Inputs

- Fixed image/query sample pool
- InternVL model path
- Optional dataset source:
  - COCO grounding samples
  - affordance benchmark samples
  - prompt-robustness samples

### Method

For each image-query pair:

1. Run InternVL once and extract all hidden layers at visual-token positions
2. Pool each layer into one or more summary vectors:
   - mean over visual tokens
   - optional token-wise flattened representation
3. Compute pairwise layer similarity across samples

Recommended metrics:

- cosine similarity between pooled layer features
- optional CKA or SVCCA later if needed

### Output

- `thinkdet/results/layer_similarity/layer_similarity_matrix.json`
- `thinkdet/results/layer_similarity/layer_similarity_heatmap.png`
- `thinkdet/results/layer_similarity/layer_groups.json`

### Grouping rule

We should keep the grouping **contiguous** in layer index. The target output is something like:

- early: `0-8`
- mid: `9-17`
- late: `18-27`

Exact boundaries should come from the similarity matrix, not intuition.

---

## Phase 2: Screen All Layers Cheaply

### Objective

Use inference-only probes to find the strongest candidate layers inside each similarity group.

### Reuse existing scripts

- `thinkdet/tests/ablation/probe_reasoning_layers.py`
- `thinkdet/tests/ablation/probe_layer_sensitivity.py`

### Rule

Run both probes across all layers, then keep:

- top 1-2 layers from each group

Example shortlist:

- early: `8`
- mid: `9`, `13`
- late: `20`

This keeps the downstream experiment budget reasonable.

### Deliverable

- `thinkdet/results/layer_selection/shortlist.json`

with:

- similarity group per layer
- probe scores
- shortlist decision

---

## Phase 3: Run Matched Single-Layer Downstream Experiments

### Objective

Stop deciding from proxy metrics alone. Evaluate the shortlisted layers on real tasks.

### Training/eval rule

For every candidate layer:

- use the same model architecture
- use the same training budget
- use the same seed set
- change only:
  - `extract_layer`
  - `extract_layers=[layer]`

### Reused training entrypoints

- `thinkdet/scripts/training/train_stage1_tma.py`
- `thinkdet/scripts/training/train_stage2_tma.py`
- `thinkdet/scripts/training/train_unified.py`

### Reused eval entrypoints

- `thinkdet/scripts/eval/eval_affordance_benchmark.py`
- `thinkdet/scripts/eval/eval_official_protocol.py`
- prompt-robustness eval if still useful

### Task buckets

We should log results by bucket, not just one aggregate score.

Target buckets:

- affordance grounding
- referential grounding
- counting
- position / spatial
- OCR
- compositional / long-query / distractor-heavy prompts

Important:

- If a bucket does not yet exist in the repo, we should not fake it.
- For missing buckets, create a small explicit stress set and mark it as such.

### Output

- one result row per `seed x layer x bucket`
- summary table with:
  - mean
  - std
  - confidence interval if available
  - delta vs baseline

---

## Phase 4: Pick Per-Bucket Winners

### Objective

Determine whether one single layer wins broadly or whether the preferred layer depends on task type.

### Decision logic

If one layer consistently wins the important buckets and stays stable across seeds:

- adopt it as the best single-layer default

If different buckets prefer different layers:

- do not force a global claim
- move to fusion

### Deliverable

- `thinkdet/results/layer_selection/per_bucket_winners.md`

with:

- winner per bucket
- runner-up
- margin
- confidence notes

---

## Phase 5: Test Lightweight Fusion Only If Justified

### Objective

Check whether mixing one representative layer from multiple groups beats every single-layer setup.

### Candidate fusion setups

Only test a small number of setups:

- best early + best mid
- best mid + best late
- best early + best mid + best late

Example:

- `extract_layers=[9, 13, 20]`
- `layer_fusion="mean"`

### Important note

Current code supports:

- multi-layer extraction
- simple `mean` or `last` fusion

It does **not** yet implement a learned per-group projector-plus-sum fusion head. We should start
with the existing lightweight fusion and only add a learned fusion module if the simple version
shows promise.

### Decision rule

Fusion is worth keeping only if it beats:

- the baseline detector
- the best single-layer ThinkDet setup

and does so by more than seed noise.

---

## Success Criteria

We should walk away with one of these conclusions:

1. Layer 9 is the best tested single-layer default
2. Another single layer beats layer 9
3. No single layer is robustly best, but fusion is better
4. Layer choice is not the main bottleneck; training/objective design matters more

Any of these is useful. The experiment is still a success if it kills a weak assumption.

---

## Does This Match the Paper Method?

Short answer:

- **Mostly yes in spirit**
- **Not fully yet in rigor**

---

## Match Table

| Paper method | Planned ThinkDet version | Match? | Notes |
|---|---|---|---|
| Group layers by representation similarity | Add InternVL layer-similarity analysis and contiguous early/mid/late partition | Yes | This is the most important missing piece to copy |
| Compare layers on downstream task families, not one global score | Evaluate per task bucket | Yes | We need real bucket definitions and result tables |
| Report which layer is best for which task | Per-bucket winner table | Yes | This is critical for avoiding fake “one best layer” claims |
| Check robustness across scales / settings | Multiple seeds first, data-scale checks later | Partial | We likely will not match their full scale sweep in the first pass |
| Use architecture-aware conclusions | Analyze InternVL multimodal layers, not CLIP ViT layers | Yes | Same logic, different encoder |
| Validate fusion after discovering complementary layers | Small fusion ablation | Yes | We can do this with current multi-layer support |
| Broad benchmark coverage across many tasks | Narrower ThinkDet task suite | Partial | Our task space is much smaller than theirs |
| Strong evidence for general layer behavior | Evidence limited to ThinkDet and its task buckets | Partial | Honest and acceptable, but narrower |

---

## Where We Still Do Not Match the Paper

Even after implementing this plan, we still will not match the paper in a few ways:

1. They study a larger and more diverse benchmark suite.
2. They analyze multiple model sizes and multiple data scales.
3. Their claims are about broad MLLM layer behavior.
4. Our claims will only be about ThinkDet on our downstream tasks.

That is fine. The right goal is not to copy their paper exactly.
The right goal is to copy the **good scientific structure**:

- similarity grouping
- task-bucket evaluation
- per-bucket winners
- fusion only when justified

---

## Honest Claim We Can Support If This Works

If the study is run cleanly, the strongest claim we can make is:

> For ThinkDet, layer X is the best tested single-layer default on our defined grounding benchmarks, and fusion Y is or is not better.

That is a strong and defensible claim.

What we still cannot honestly claim:

> Layer X is universally best for any ambiguous prompt.

---

## Recommended First Pass

The first pass should be lean:

1. add similarity analysis
2. run both probes on all layers
3. shortlist `1-2` layers per group
4. run single-layer downstream experiments on the shortlist
5. run fusion only if per-bucket winners disagree

This is the minimum version that meaningfully upgrades the evidence over the current layer-9 default.
