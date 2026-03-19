# InternVL 3.5-1B Baseline Test Report

## Setup
- **Model**: OpenGVLab/InternVL3_5-1B (1.1B params)
- **GPU**: NVIDIA RTX 2080 Ti (11.5GB VRAM)
- **VRAM used**: 2.12 GB (float16)
- **Load time**: 45.6s
- **Environment**: conda open_led, transformers 4.57.6, torch 2.3.0+cu121
- **Images**: 5 COCO val2017 images, 10 prompts each = 50 tests total
- **Errors**: 0/50

## Test Images
| Image | Content |
|---|---|
| 000000000139.jpg | Living room with person cooking, TV, fireplace |
| 000000000285.jpg | Brown bear sitting on grass |
| 000000000632.jpg | Bedroom with bed, dresser, bookshelf |
| 000000000724.jpg | Stop sign at intersection with truck |
| 000000000776.jpg | Three teddy bears on a bed |

---

## Results by Capability

### 1. Captioning — STRONG
The model produces accurate, fluent single-sentence descriptions.

| Image | Response |
|---|---|
| 000139 | "a cozy living room with a woman cooking at a dining table, a television, and a fireplace" |
| 000285 | "A brown bear sits on grass, looking directly at the camera with a curious expression" |
| 000724 | "A red stop sign...positioned at an intersection with trees, a vehicle, and a building" |
| 000776 | "Three brown teddy bears are cuddled together on a bed" |

**Verdict**: Solid. Captures main subjects and scene context.

### 2. Object Recognition — STRONG
Lists objects with reasonable completeness. Identifies furniture, electronics, plants, signs, vehicles.

**Verdict**: Good coverage. Occasionally verbose but accurate.

### 3. Counting — MIXED
| Image | Actual | Model Said | Correct? |
|---|---|---|---|
| 000139 | 1 person, 0 vehicles | 1 person, 0 vehicles | YES |
| 000285 | 0 people, 0 vehicles | **1 person**, 0 vehicles | NO — hallucinated a person behind the bear |
| 000632 | 0 people, 0 vehicles | 0 people, 0 vehicles | YES |
| 000724 | 0 people, 1 vehicle | 0 people, 1 vehicle | YES |
| 000776 | 0 people, 0 vehicles | 0 people, 0 vehicles | YES |

**Verdict**: 4/5 correct. One hallucination (bear image — said person exists behind bear).

### 4. Spatial Reasoning — STRONG
Correctly identifies foreground/middleground/background layout in all 5 images. Descriptions match actual spatial arrangement.

Example (000724): "Foreground: stop sign. Middle ground: vehicle and trees. Background: buildings and more trees."

**Verdict**: Genuinely understands depth and spatial arrangement.

### 5. Relational Reasoning — MODERATE
Can describe obvious interactions (person cooking at table, bears cuddling). Struggles with non-obvious relationships — tends to fall back on generic descriptions.

**Verdict**: Identifies clear interactions but doesn't deeply reason about subtle relationships.

### 6. Detection-style Queries — MODERATE
| Image | Question | Response Quality |
|---|---|---|
| 000139 | "Is there a person?" | YES, "center of room, near dining table" — CORRECT |
| 000285 | "Is there a person?" | YES, "center of image, above the bear" — WRONG (hallucinated) |
| 000632 | "Is there a person?" | NO — CORRECT |
| 000724 | "Is there a person?" | NO — CORRECT |
| 000776 | "Is there a person?" | NO, correctly identified teddy bears instead — CORRECT |

**Verdict**: 4/5 correct. Same hallucination on bear image.

### 7. Grounding (Location Description) — WEAK-MODERATE
Uses region descriptions (left/right/center) but lacks precision. Never produces coordinates or bounding boxes. Descriptions are qualitative, not quantitative.

Example (000724): "Stop sign is located in the center of the image, with a truck and building visible in the background."

**Verdict**: Coarse spatial regions only. Not suitable for detection without further processing.

### 8. Scene Reasoning — STRONG
Makes plausible inferences about activities and next actions:
- 000139: "person preparing food or organizing items"
- 000285: "fur appears wet, suggesting it might have been in water"
- 000724: "vehicles are required to halt at this location"

**Verdict**: Genuine scene-level reasoning with causal inferences. This is what we want for ThinkDet.

### 9. Attribute Recognition — STRONG
Identifies colors, materials, sizes accurately:
- "red sign with white lettering, made of metal"
- "brown bear with thick furry coat"
- "blue quilt, wooden bed frame"

**Verdict**: Rich attribute understanding.

### 10. Chain-of-Thought (Step-by-Step) — STRONG
When prompted to think step by step, produces structured analysis:
1. Identifies objects systematically
2. Assigns locations to each
3. Describes relationships
4. Summarizes scene

**Verdict**: The model CAN reason when prompted. CoT improves output quality.

---

## Key Findings

### What InternVL 3.5-1B CAN Do
1. **Scene understanding** — correctly grasps what's happening
2. **Spatial awareness** — foreground/background, left/right
3. **Object recognition** — identifies diverse objects accurately
4. **Attribute binding** — knows which colors/materials belong to which objects
5. **Causal reasoning** — "fur appears wet, suggesting it was in water"
6. **Step-by-step thinking** — CoT prompting works

### What It CANNOT Do
1. **Precise localization** — no bounding boxes, no coordinates, only vague regions
2. **Reliable counting** — hallucinated a person in the bear image (1/5 wrong)
3. **Fine-grained grounding** — "center-left" is as specific as it gets
4. **Consistent hallucination resistance** — confidently described non-existent person

### Critical Insight for ThinkDet
The model performs **reasoning but not detection**. It understands scenes, relationships, and attributes — but cannot localize precisely. This is exactly the gap ThinkDet should fill:

- **InternVL provides**: text-conditioned reasoning features (what to look for, why, relationships)
- **GroundingDINO provides**: precise localization (bounding boxes, confidence scores)
- **ThinkDet adapter bridges**: reasoning features → detection guidance

The reasoning capability IS there. The model genuinely thinks about the scene when given text + image jointly. The current Qwen2 setup wastes this by never showing text to the LLM.

---

## Performance Stats

| Category | Avg Time | Tests |
|---|---|---|
| captioning | 2.4s | 5 |
| counting | 2.6s | 5 |
| detection | 2.4s | 5 |
| spatial | 5.9s | 5 |
| grounding | 10.4s | 5 |
| object_recognition | 10.2s | 5 |
| reasoning | 11.9s | 10 |
| reasoning_cot | 11.2s | 5 |
| attributes | 12.9s | 5 |

Total inference time for 50 queries: ~6 minutes on RTX 2080 Ti.
