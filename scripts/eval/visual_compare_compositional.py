"""
ThinkDet — Compositional Prompt Visual Comparison

Runs baseline (DINO-only) vs Stage 1 TMA on targeted compositional/ambiguous
prompts and generates side-by-side visualizations.

Demonstrates TMA's reasoning behavior, not final AP gains.

Usage:
    python thinkdet/scripts/eval/visual_compare_compositional.py
"""

import os
import sys
import json
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = "/home/iibrohimm/project/next_step"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "GroundingDINO", "GroundingDINO"))

from thinkdet.models.arch import ThinkDetModel, DEFAULT_INJECTION_LAYERS
from thinkdet.data.coco_grounding import COCOGroundingDataset, build_positive_map
from groundingdino.util.inference import load_model as load_gd_model
from groundingdino.util.misc import NestedTensor
from thinkdet.scripts.training.train_stage1 import box_cxcywh_to_xyxy
import torchvision.transforms as T
import torchvision.transforms.functional as TF


GD_CONFIG = f"{ROOT}/GroundingDINO/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GD_WEIGHTS = f"{ROOT}/GroundingDINO/GroundingDINO/weights/groundingdino_swint_ogc.pth"
INTERNVL_PATH = f"{ROOT}/InternVL3_5-1B"
COCO_VAL_IMG = f"{ROOT}/dataSets/coco/val2017"
COCO_VAL_ANN = f"{ROOT}/dataSets/coco/annotations/instances_val2017.json"
STAGE1_CKPT = (
    f"{ROOT}/thinkdet/checkpoints/stage1_tma/"
    "layer9_tma_m8_full_2ep_7gpu_20260218_152649/"
    "thinkdet_tma_stage1_epoch2.pth"
)
OUTPUT_DIR = f"{ROOT}/thinkdet/results/visual_compare/compositional"

# ── Test cases ──────────────────────────────────────────────
# Each: (image_file, list of (prompt, description))
TEST_CASES = [
    {
        "image": "000000372819.jpg",
        "image_id": 372819,
        "description": "4 dogs running in park, 2 people sitting on bench",
        "prompts": [
            ("the largest dog running in the field .", "Spatial+attribute: largest among 4 dogs"),
            ("the dog closest to the people on the bench .", "Spatial reasoning: proximity to people"),
            ("a small white and brown dog .", "Attribute disambiguation: specific dog"),
        ],
    },
    {
        "image": "000000512836.jpg",
        "image_id": 512836,
        "description": "Person with red umbrella in snow, 2 dogs",
        "prompts": [
            ("the person holding a red umbrella .", "Attribute binding: person+umbrella"),
            ("the darker colored dog on the left .", "Attribute+spatial: dark dog, left side"),
            ("a dog walking in the snow near a person .", "Compositional: dog+snow+person"),
        ],
    },
    {
        "image": "000000210032.jpg",
        "image_id": 210032,
        "description": "Birds near food on outdoor table, person in background",
        "prompts": [
            ("the bird closest to the sandwich .", "Spatial reasoning: bird near food"),
            ("a person sitting at an outdoor table near the water .", "Compositional scene understanding"),
            ("the sandwich on the plate .", "Object grounding with context"),
        ],
    },
]


def build_dino_transform():
    """Same DINO transform used in training/eval."""
    normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def transform(image):
        # Resize: short side = 800, max long side = 1333
        w, h = image.size
        target_size = 800
        max_size = 1333
        min_side = min(w, h)
        max_side = max(w, h)
        scale = target_size / min_side
        if scale * max_side > max_size:
            scale = max_size / max_side
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = image.resize((new_w, new_h), Image.BILINEAR)
        img_tensor = TF.to_tensor(image)
        img_tensor = normalize(img_tensor)
        return img_tensor

    return transform


def build_internvl_transform():
    return T.Compose([
        T.Resize((448, 448), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


@torch.no_grad()
def run_single_prompt(model_fn, image_pil, prompt, device,
                      dino_transform, internvl_transform, tokenizer, special_tokens):
    """
    Run a single image + prompt through the model.
    Returns list of (box_xyxy_abs, score, label_text) sorted by score desc.
    """
    img_w, img_h = image_pil.size

    # DINO input
    dino_tensor = dino_transform(image_pil).unsqueeze(0).to(device)
    mask = torch.zeros(1, dino_tensor.shape[2], dino_tensor.shape[3],
                       dtype=torch.bool, device=device)
    dino_nested = NestedTensor(dino_tensor, mask)

    # InternVL input
    internvl_tensor = internvl_transform(image_pil).unsqueeze(0).to(device)

    dino_inputs = {"samples": dino_nested, "captions": [prompt]}

    out = model_fn(internvl_tensor, [prompt], dino_inputs)
    if isinstance(out, tuple):
        outputs = out[0]
    else:
        outputs = out

    logits = outputs["pred_logits"][0].sigmoid()  # [N, 256]
    boxes = outputs["pred_boxes"][0]               # [N, 4]

    # For targeted prompts, just use max logit across tokens as score
    max_scores = logits.max(dim=-1).values         # [N]

    # Top-10 predictions
    topk = min(10, max_scores.shape[0])
    vals, idx = max_scores.topk(topk)

    results = []
    for i in range(topk):
        box = boxes[idx[i]]
        score = vals[i].item()
        # Convert cxcywh -> xyxy absolute
        cx, cy, w, h = box.tolist()
        x1 = (cx - w/2) * img_w
        y1 = (cy - h/2) * img_h
        x2 = (cx + w/2) * img_w
        y2 = (cy + h/2) * img_h
        results.append({
            "box": [x1, y1, x2, y2],
            "score": score,
        })

    return results


def draw_boxes_on_image(image_pil, results, title, max_boxes=5, min_score=0.15):
    """Draw bounding boxes on image with scores."""
    img = image_pil.copy()
    draw = ImageDraw.Draw(img)

    # Try to load a decent font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except:
        font = ImageFont.load_default()
        font_small = font

    colors = [
        (255, 50, 50),    # red
        (50, 200, 50),    # green
        (50, 100, 255),   # blue
        (255, 200, 50),   # yellow
        (200, 50, 255),   # purple
    ]

    drawn = 0
    for i, r in enumerate(results):
        if r["score"] < min_score:
            continue
        if drawn >= max_boxes:
            break
        color = colors[drawn % len(colors)]
        x1, y1, x2, y2 = r["box"]

        # Draw box
        for offset in range(3):  # thick border
            draw.rectangle([x1-offset, y1-offset, x2+offset, y2+offset], outline=color)

        # Draw score label
        label = f"{r['score']:.3f}"
        # Background for text
        tw = len(label) * 9
        draw.rectangle([x1, y1-20, x1+tw+4, y1], fill=color)
        draw.text((x1+2, y1-18), label, fill=(255, 255, 255), font=font_small)

        # Rank number
        rank_label = f"#{drawn+1}"
        draw.rectangle([x1, y2, x1+25, y2+20], fill=color)
        draw.text((x1+3, y2+2), rank_label, fill=(255, 255, 255), font=font_small)

        drawn += 1

    # Title bar
    title_h = 35
    title_img = Image.new("RGB", (img.width, img.height + title_h), (40, 40, 40))
    title_img.paste(img, (0, title_h))
    title_draw = ImageDraw.Draw(title_img)
    title_draw.text((10, 8), title, fill=(255, 255, 255), font=font)

    return title_img


def create_side_by_side(img_baseline, img_tma, prompt_text, description):
    """Create a single comparison image."""
    # Resize both to same height
    target_h = max(img_baseline.height, img_tma.height)
    if img_baseline.height != target_h:
        scale = target_h / img_baseline.height
        img_baseline = img_baseline.resize(
            (int(img_baseline.width * scale), target_h), Image.BILINEAR)
    if img_tma.height != target_h:
        scale = target_h / img_tma.height
        img_tma = img_tma.resize(
            (int(img_tma.width * scale), target_h), Image.BILINEAR)

    gap = 10
    total_w = img_baseline.width + gap + img_tma.width

    # Header for prompt
    header_h = 50
    canvas = Image.new("RGB", (total_w, target_h + header_h), (30, 30, 30))

    # Header text
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
        font_desc = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except:
        font = ImageFont.load_default()
        font_desc = font

    draw = ImageDraw.Draw(canvas)
    draw.text((10, 5), f'Prompt: "{prompt_text}"', fill=(100, 255, 100), font=font)
    draw.text((10, 28), f"({description})", fill=(180, 180, 180), font=font_desc)

    canvas.paste(img_baseline, (0, header_h))
    canvas.paste(img_tma, (img_baseline.width + gap, header_h))

    return canvas


def main():
    device = torch.device("cuda:0")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("  Compositional Prompt Comparison: Baseline vs TMA")
    print("=" * 60)

    # Build transforms
    dino_transform = build_dino_transform()
    internvl_transform = build_internvl_transform()

    # ── Load baseline ──
    print("\nLoading baseline model...")
    gd_base = load_gd_model(GD_CONFIG, GD_WEIGHTS, device="cpu")
    gd_base = gd_base.to(device).eval()

    def baseline_fn(images, queries, dino_inputs):
        return gd_base(**dino_inputs), {}

    # ── Load TMA ──
    print("Loading TMA model...")
    ckpt = torch.load(STAGE1_CKPT, map_location="cpu")
    gd_tma = load_gd_model(GD_CONFIG, GD_WEIGHTS, device="cpu")
    model_tma = ThinkDetModel(
        grounding_dino=gd_tma,
        internvl_path=INTERNVL_PATH,
        extract_layer=9,
        extract_layers=[9],
        injection_layers=DEFAULT_INJECTION_LAYERS,
        tma_m=8,
        tma_n_heads=8,
    )
    model_tma.load_state_dict(ckpt["trainable_state_dict"], strict=False)
    model_tma = model_tma.to(device).eval()

    tokenizer = gd_base.tokenizer
    special_tokens = gd_base.specical_tokens

    # ── Run comparisons ──
    all_results = []

    for tc_idx, tc in enumerate(TEST_CASES):
        img_path = os.path.join(COCO_VAL_IMG, tc["image"])
        image_pil = Image.open(img_path).convert("RGB")

        print(f"\n{'='*50}")
        print(f"Image: {tc['image']} — {tc['description']}")
        print(f"{'='*50}")

        for p_idx, (prompt, desc) in enumerate(tc["prompts"]):
            print(f"\n  Prompt: \"{prompt}\"")
            print(f"  ({desc})")

            # Baseline
            results_base = run_single_prompt(
                baseline_fn, image_pil, prompt, device,
                dino_transform, internvl_transform, tokenizer, special_tokens,
            )

            # TMA
            results_tma = run_single_prompt(
                model_tma, image_pil, prompt, device,
                dino_transform, internvl_transform, tokenizer, special_tokens,
            )

            base_scores = [f"{r['score']:.3f}" for r in results_base[:3]]
            tma_scores = [f"{r['score']:.3f}" for r in results_tma[:3]]
            print(f"  Baseline top-3: {base_scores}")
            print(f"  TMA top-3:      {tma_scores}")

            # Draw
            img_base_drawn = draw_boxes_on_image(
                image_pil, results_base, "BASELINE (DINO only)", max_boxes=3)
            img_tma_drawn = draw_boxes_on_image(
                image_pil, results_tma, "TMA (Stage 1)", max_boxes=3)

            # Side by side
            comparison = create_side_by_side(img_base_drawn, img_tma_drawn, prompt, desc)

            fname = f"compare_{tc_idx+1}_{p_idx+1}_{tc['image'].replace('.jpg', '')}_{desc.split(':')[0].strip().replace(' ', '_').lower()}.jpg"
            out_path = os.path.join(OUTPUT_DIR, fname)
            comparison.save(out_path, quality=95)
            print(f"  Saved: {out_path}")

            all_results.append({
                "image": tc["image"],
                "prompt": prompt,
                "description": desc,
                "baseline_top3": [{"score": r["score"], "box": r["box"]} for r in results_base[:3]],
                "tma_top3": [{"score": r["score"], "box": r["box"]} for r in results_tma[:3]],
                "output_file": fname,
            })

    # Save results JSON
    json_path = os.path.join(OUTPUT_DIR, "comparison_results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n\nResults JSON: {json_path}")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"\nTotal comparisons: {len(all_results)}")


if __name__ == "__main__":
    main()
