"""
ThinkDet — Visual Comparison V2

Strategy: Instead of compositional prompts, focus on scenarios where
InternVL visual context could actually influence detection through
the text-memory pathway:

1. Fine-grained category disambiguation (breeds, food types)
2. Context-dependent grounding (objects that look similar)
3. Score re-ranking across multiple instances

For each test, show top-5 detections ranked by score.
Look for cases where TMA RE-RANKS detections vs baseline.
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
from groundingdino.util.inference import load_model as load_gd_model
from groundingdino.util.misc import NestedTensor
import torchvision.transforms as T
import torchvision.transforms.functional as TF


GD_CONFIG = f"{ROOT}/GroundingDINO/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GD_WEIGHTS = f"{ROOT}/GroundingDINO/GroundingDINO/weights/groundingdino_swint_ogc.pth"
INTERNVL_PATH = f"{ROOT}/InternVL3_5-1B"
COCO_VAL_IMG = f"{ROOT}/dataSets/coco/val2017"
STAGE1_CKPT = (
    f"{ROOT}/thinkdet/checkpoints/stage1_tma/"
    "layer9_tma_m8_full_2ep_7gpu_20260218_152649/"
    "thinkdet_tma_stage1_epoch2.pth"
)
OUTPUT_DIR = f"{ROOT}/thinkdet/results/visual_compare/compositional_v2"


# Images with rich multi-object scenes for re-ranking analysis
TEST_CASES = [
    # Scene with multiple dogs — which dog responds to which prompt?
    {
        "image": "000000372819.jpg",
        "prompts": [
            "dog .",
            "the husky .",
            "a small dog .",
            "the white dog running .",
        ],
    },
    # Snow scene — person+dogs
    {
        "image": "000000512836.jpg",
        "prompts": [
            "dog .",
            "person .",
            "umbrella .",
            "the black dog .",
        ],
    },
    # Bird + food scene
    {
        "image": "000000210032.jpg",
        "prompts": [
            "bird .",
            "sandwich .",
            "person .",
            "the sparrow .",
        ],
    },
    # Let's try more diverse images
    {
        "image": "000000060823.jpg",  # birds + cows
        "prompts": [
            "bird .",
            "cow .",
            "the black bird .",
            "the white cow .",
        ],
    },
    {
        "image": "000000279278.jpg",  # people + bikes + skateboard + dog
        "prompts": [
            "dog .",
            "skateboard .",
            "bicycle .",
            "the person with the dog .",
        ],
    },
]


def build_dino_transform():
    normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    def transform(image):
        w, h = image.size
        scale = 800 / min(w, h)
        if scale * max(w, h) > 1333:
            scale = 1333 / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
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
def run_single(model_fn, image_pil, prompt, device, dino_tf, internvl_tf, top_k=5):
    img_w, img_h = image_pil.size
    dino_tensor = dino_tf(image_pil).unsqueeze(0).to(device)
    mask = torch.zeros(1, dino_tensor.shape[2], dino_tensor.shape[3],
                       dtype=torch.bool, device=device)
    dino_nested = NestedTensor(dino_tensor, mask)
    internvl_tensor = internvl_tf(image_pil).unsqueeze(0).to(device)
    dino_inputs = {"samples": dino_nested, "captions": [prompt]}

    out = model_fn(internvl_tensor, [prompt], dino_inputs)
    outputs = out[0] if isinstance(out, tuple) else out

    logits = outputs["pred_logits"][0].sigmoid()
    boxes = outputs["pred_boxes"][0]
    max_scores = logits.max(dim=-1).values

    vals, idx = max_scores.topk(min(top_k, max_scores.shape[0]))
    results = []
    for i in range(len(vals)):
        cx, cy, w, h = boxes[idx[i]].tolist()
        results.append({
            "box": [(cx - w/2) * img_w, (cy - h/2) * img_h,
                    (cx + w/2) * img_w, (cy + h/2) * img_h],
            "score": vals[i].item(),
            "query_idx": idx[i].item(),
        })
    return results


def draw_detections(image_pil, results, title, max_boxes=5, min_score=0.10):
    img = image_pil.copy()
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except:
        font = ImageFont.load_default()

    colors = [(255, 50, 50), (50, 200, 50), (50, 100, 255), (255, 200, 50), (200, 50, 255)]
    drawn = 0
    for r in results:
        if r["score"] < min_score or drawn >= max_boxes:
            break
        color = colors[drawn % len(colors)]
        x1, y1, x2, y2 = r["box"]
        for off in range(3):
            draw.rectangle([x1-off, y1-off, x2+off, y2+off], outline=color)
        label = f"#{drawn+1} {r['score']:.3f} [q{r['query_idx']}]"
        draw.rectangle([x1, y1-17, x1+len(label)*8, y1], fill=color)
        draw.text((x1+2, y1-15), label, fill=(255, 255, 255), font=font)
        drawn += 1

    title_h = 28
    out = Image.new("RGB", (img.width, img.height + title_h), (40, 40, 40))
    out.paste(img, (0, title_h))
    d = ImageDraw.Draw(out)
    d.text((8, 5), title, fill=(255, 255, 255), font=font)
    return out


def make_comparison(img_base, img_tma, prompt):
    target_h = max(img_base.height, img_tma.height)
    for im in [img_base, img_tma]:
        if im.height != target_h:
            s = target_h / im.height
            im = im.resize((int(im.width * s), target_h), Image.BILINEAR)

    gap = 6
    header_h = 40
    total_w = img_base.width + gap + img_tma.width
    canvas = Image.new("RGB", (total_w, target_h + header_h), (25, 25, 25))

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except:
        font = ImageFont.load_default()

    d = ImageDraw.Draw(canvas)
    d.text((10, 10), f'Prompt: "{prompt}"', fill=(100, 255, 100), font=font)
    canvas.paste(img_base, (0, header_h))
    canvas.paste(img_tma, (img_base.width + gap, header_h))
    return canvas


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    a1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    a2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    return inter / (a1 + a2 - inter + 1e-6)


def analyze_reranking(base_results, tma_results):
    """Check if TMA re-ranked the detections compared to baseline."""
    if len(base_results) < 2 or len(tma_results) < 2:
        return {"reranked": False}

    # Match TMA top-1 to baseline by IoU
    tma_top1_box = tma_results[0]["box"]
    best_iou, best_rank = 0, -1
    for i, br in enumerate(base_results):
        iou = compute_iou(tma_top1_box, br["box"])
        if iou > best_iou:
            best_iou = iou
            best_rank = i

    # Match baseline top-1 to TMA by IoU
    base_top1_box = base_results[0]["box"]
    best_iou2, best_rank2 = 0, -1
    for i, tr in enumerate(tma_results):
        iou = compute_iou(base_top1_box, tr["box"])
        if iou > best_iou2:
            best_iou2 = iou
            best_rank2 = i

    reranked = best_rank > 0  # TMA's top-1 was NOT baseline's top-1
    return {
        "reranked": reranked,
        "tma_top1_was_baseline_rank": best_rank + 1 if best_rank >= 0 else -1,
        "baseline_top1_is_tma_rank": best_rank2 + 1 if best_rank2 >= 0 else -1,
        "match_iou": best_iou,
    }


def main():
    device = torch.device("cuda:0")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    dino_tf = build_dino_transform()
    internvl_tf = build_internvl_transform()

    print("Loading models...")
    gd_base = load_gd_model(GD_CONFIG, GD_WEIGHTS, device="cpu").to(device).eval()
    def baseline_fn(img, q, di): return gd_base(**di), {}

    ckpt = torch.load(STAGE1_CKPT, map_location="cpu")
    gd_tma = load_gd_model(GD_CONFIG, GD_WEIGHTS, device="cpu")
    model_tma = ThinkDetModel(
        grounding_dino=gd_tma, internvl_path=INTERNVL_PATH,
        extract_layer=9, extract_layers=[9],
        injection_layers=DEFAULT_INJECTION_LAYERS, tma_m=8, tma_n_heads=8,
    )
    model_tma.load_state_dict(ckpt["trainable_state_dict"], strict=False)
    model_tma = model_tma.to(device).eval()

    print("Running comparisons...\n")
    all_analysis = []
    rerank_count = 0
    total_prompts = 0

    for tc_idx, tc in enumerate(TEST_CASES):
        img_path = os.path.join(COCO_VAL_IMG, tc["image"])
        image_pil = Image.open(img_path).convert("RGB")
        print(f"Image: {tc['image']}")

        for p_idx, prompt in enumerate(tc["prompts"]):
            total_prompts += 1
            results_base = run_single(baseline_fn, image_pil, prompt, device, dino_tf, internvl_tf)
            results_tma = run_single(model_tma, image_pil, prompt, device, dino_tf, internvl_tf)

            analysis = analyze_reranking(results_base, results_tma)

            base_scores = [f"{r['score']:.3f}" for r in results_base[:3]]
            tma_scores = [f"{r['score']:.3f}" for r in results_tma[:3]]
            base_qidx = [r['query_idx'] for r in results_base[:3]]
            tma_qidx = [r['query_idx'] for r in results_tma[:3]]

            rerank_flag = " *** RE-RANKED ***" if analysis["reranked"] else ""
            print(f"  [{prompt:.<40s}] B:{base_scores} q{base_qidx}  T:{tma_scores} q{tma_qidx}{rerank_flag}")

            if analysis["reranked"]:
                rerank_count += 1

            # Save image for re-ranked cases
            img_base = draw_detections(image_pil, results_base, "BASELINE", max_boxes=5)
            img_tma = draw_detections(image_pil, results_tma, "TMA (Stage 1)", max_boxes=5)
            comp = make_comparison(img_base, img_tma, prompt)

            fname = f"{tc_idx+1}_{p_idx+1}_{tc['image'].replace('.jpg', '')}_{prompt.replace(' ', '_').replace('.', '').strip()}.jpg"
            out_path = os.path.join(OUTPUT_DIR, fname)
            comp.save(out_path, quality=95)

            all_analysis.append({
                "image": tc["image"],
                "prompt": prompt,
                "reranking": analysis,
                "baseline_top3_scores": base_scores,
                "tma_top3_scores": tma_scores,
                "baseline_top3_qidx": base_qidx,
                "tma_top3_qidx": tma_qidx,
                "output_file": fname,
            })

        print()

    print(f"\n{'='*60}")
    print(f"  Re-ranking summary: {rerank_count}/{total_prompts} prompts had different top-1")
    print(f"{'='*60}")

    # Print re-ranked cases
    print("\nRe-ranked cases (TMA chose different top-1 than baseline):")
    for a in all_analysis:
        if a["reranking"]["reranked"]:
            print(f"  {a['image']} | \"{a['prompt']}\"")
            print(f"    TMA top-1 was baseline rank #{a['reranking']['tma_top1_was_baseline_rank']}")
            print(f"    Baseline top-1 became TMA rank #{a['reranking']['baseline_top1_is_tma_rank']}")
            print(f"    IoU match: {a['reranking']['match_iou']:.3f}")

    with open(os.path.join(OUTPUT_DIR, "analysis.json"), "w") as f:
        json.dump(all_analysis, f, indent=2)
    print(f"\nSaved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
