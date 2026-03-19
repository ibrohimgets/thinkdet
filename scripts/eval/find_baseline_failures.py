"""
Aggressive search for cases where baseline FAILS and TMA SUCCEEDS.

Strategy:
  - Scan many COCO val images with targeted prompts
  - Look for cases where TMA picks a DIFFERENT top-1 detection (re-ranking)
  - Among re-ranked cases, check which top-1 is closer to the GT annotation
  - Output only the clear wins for TMA
"""

import os
import sys
import json
import torch
import numpy as np
from PIL import Image
from collections import defaultdict

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
COCO_VAL_ANN = f"{ROOT}/dataSets/coco/annotations/instances_val2017.json"
STAGE1_CKPT = (
    f"{ROOT}/thinkdet/checkpoints/stage1_tma/"
    "layer9_tma_m8_full_2ep_7gpu_20260218_152649/"
    "thinkdet_tma_stage1_epoch2.pth"
)


def build_dino_transform():
    normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    def transform(image):
        w, h = image.size
        scale = 800 / min(w, h)
        if scale * max(w, h) > 1333:
            scale = 1333 / max(w, h)
        image = image.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
        return normalize(TF.to_tensor(image))
    return transform


def build_internvl_transform():
    return T.Compose([
        T.Resize((448, 448), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def compute_iou(b1, b2):
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
    a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
    return inter / (a1 + a2 - inter + 1e-6)


@torch.no_grad()
def run_model(model_fn, image_pil, prompt, device, dino_tf, internvl_tf, top_k=5):
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
            "box": [(cx-w/2)*img_w, (cy-h/2)*img_h, (cx+w/2)*img_w, (cy+h/2)*img_h],
            "score": vals[i].item(),
            "query_idx": idx[i].item(),
        })
    return results


def main():
    device = torch.device("cuda:0")
    dino_tf = build_dino_transform()
    internvl_tf = build_internvl_transform()

    # Load annotations for GT checking
    with open(COCO_VAL_ANN) as f:
        coco_data = json.load(f)
    cat_map = {c["id"]: c["name"] for c in coco_data["categories"]}
    img_info = {i["id"]: i for i in coco_data["images"]}

    # Group GT annotations by image
    img_anns = defaultdict(list)
    for ann in coco_data["annotations"]:
        img_anns[ann["image_id"]].append(ann)

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

    # ── Find candidate images ──
    # Scenes with multiple instances + relational prompts
    from collections import Counter

    candidates = []
    for img_id, anns in img_anns.items():
        cats = [cat_map[a["category_id"]] for a in anns]
        cat_counts = Counter(cats)
        unique = set(cats)

        prompts = []

        # Multiple people + object interaction
        if cat_counts.get("person", 0) >= 2:
            for obj in unique - {"person"}:
                if obj in {"dog", "cat", "horse", "bicycle", "motorcycle",
                           "surfboard", "skateboard", "tennis racket",
                           "umbrella", "backpack", "cell phone", "laptop",
                           "frisbee", "kite", "baseball bat", "snowboard"}:
                    prompts.append(f"the person with the {obj} .")
                    prompts.append(f"the person closest to the {obj} .")

        # Multiple animals — attribute prompts
        for animal in ["dog", "cat", "horse", "cow", "bird", "elephant"]:
            if cat_counts.get(animal, 0) >= 2:
                prompts.append(f"the largest {animal} .")
                prompts.append(f"the smallest {animal} .")

        # Person + multiple objects
        if "person" in unique and len(unique) >= 3:
            for obj in unique - {"person"}:
                if cat_counts.get(obj, 0) == 1:
                    prompts.append(f"the {obj} near the person .")

        if prompts:
            candidates.append({
                "image_id": img_id,
                "file_name": img_info[img_id]["file_name"],
                "prompts": prompts[:6],  # cap at 6 prompts per image
                "anns": anns,
            })

    # Sort by prompt count, take top 100
    candidates.sort(key=lambda x: -len(x["prompts"]))
    candidates = candidates[:100]

    print(f"Scanning {len(candidates)} images with relational prompts...\n")

    wins = []
    total_tested = 0
    total_reranked = 0

    for ci, cand in enumerate(candidates):
        img_path = os.path.join(COCO_VAL_IMG, cand["file_name"])
        if not os.path.exists(img_path):
            continue
        image_pil = Image.open(img_path).convert("RGB")
        img_w, img_h = image_pil.size

        # Get GT boxes for this image
        gt_boxes = {}
        for ann in cand["anns"]:
            cat_name = cat_map[ann["category_id"]]
            x, y, w, h = ann["bbox"]
            gt_box = [x, y, x+w, y+h]
            if cat_name not in gt_boxes:
                gt_boxes[cat_name] = []
            gt_boxes[cat_name].append(gt_box)

        for prompt in cand["prompts"]:
            total_tested += 1
            try:
                res_base = run_model(baseline_fn, image_pil, prompt, device, dino_tf, internvl_tf)
                res_tma = run_model(model_tma, image_pil, prompt, device, dino_tf, internvl_tf)
            except Exception:
                continue

            if len(res_base) < 2 or len(res_tma) < 2:
                continue

            # Check if top-1 boxes are different (IoU < 0.5 = different object)
            iou_top1 = compute_iou(res_base[0]["box"], res_tma[0]["box"])

            if iou_top1 < 0.5:
                total_reranked += 1

                # Find which top-1 is closer to the relevant GT
                # Extract the target object from prompt
                target_cat = None
                for cat in gt_boxes:
                    if cat in prompt:
                        target_cat = cat
                        break

                if target_cat and gt_boxes.get(target_cat):
                    # Compute max IoU of each top-1 with GT boxes for target cat
                    base_max_iou = max(compute_iou(res_base[0]["box"], gt)
                                       for gt in gt_boxes[target_cat])
                    tma_max_iou = max(compute_iou(res_tma[0]["box"], gt)
                                      for gt in gt_boxes[target_cat])

                    # TMA wins if its top-1 has higher IoU with GT
                    if tma_max_iou > base_max_iou + 0.05:
                        wins.append({
                            "image_id": cand["image_id"],
                            "file_name": cand["file_name"],
                            "prompt": prompt,
                            "base_top1_score": res_base[0]["score"],
                            "tma_top1_score": res_tma[0]["score"],
                            "base_top1_box": res_base[0]["box"],
                            "tma_top1_box": res_tma[0]["box"],
                            "base_gt_iou": round(base_max_iou, 3),
                            "tma_gt_iou": round(tma_max_iou, 3),
                            "iou_improvement": round(tma_max_iou - base_max_iou, 3),
                            "target_cat": target_cat,
                            "top1_iou": round(iou_top1, 3),
                            "base_top3": [{"score": r["score"], "box": r["box"]} for r in res_base[:3]],
                            "tma_top3": [{"score": r["score"], "box": r["box"]} for r in res_tma[:3]],
                        })

        if (ci + 1) % 20 == 0:
            print(f"  [{ci+1}/{len(candidates)}] tested={total_tested} reranked={total_reranked} wins={len(wins)}")

    print(f"\n{'='*60}")
    print(f"  RESULTS: {total_tested} prompts tested")
    print(f"  Different top-1 (IoU<0.5): {total_reranked}")
    print(f"  TMA wins (better GT match): {len(wins)}")
    print(f"{'='*60}")

    # Sort wins by IoU improvement
    wins.sort(key=lambda x: -x["iou_improvement"])

    print("\nTop TMA wins:")
    for w in wins[:15]:
        print(f"  {w['file_name']} | \"{w['prompt']}\"")
        print(f"    target={w['target_cat']}  base_iou={w['base_gt_iou']}  tma_iou={w['tma_gt_iou']}  "
              f"improvement=+{w['iou_improvement']}  top1_diff={w['top1_iou']}")

    out_path = os.path.join(ROOT, "thinkdet/results/eval/baseline_failures.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(wins, f, indent=2)
    print(f"\nSaved {len(wins)} wins to {out_path}")


if __name__ == "__main__":
    main()
