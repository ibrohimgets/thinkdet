"""
Find fresh affordance cases where baseline fails and ThinkDet (S1/S2) succeeds.

Outputs a summary JSON compatible with visual_compare_3way.py --source_summary.
"""

import argparse
import datetime
import json
import os
import sys
from collections import Counter, defaultdict

import torch
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF

ROOT = "/home/iibrohimm/project/next_step"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "GroundingDINO", "GroundingDINO"))

from thinkdet.models.arch import ThinkDetModel, DEFAULT_INJECTION_LAYERS
from groundingdino.util.inference import load_model as load_gd_model
from groundingdino.util.misc import NestedTensor


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
STAGE2_CKPT = (
    f"{ROOT}/thinkdet/checkpoints/stage2_tma/"
    "layer9_tma_m8_8gpu_20260219_010134/"
    "thinkdet_tma_stage2_epoch2.pth"
)


# Keep these out so we don't reuse the same showcase images.
HARDCODED_EXCLUDE_IDS = {
    309391, 139099, 372819, 546219, 244833,
    538544, 485014, 538263, 505884, 508429,
    521618, 550311, 570440, 572949, 512836,
    549347, 513643, 498639, 279278,
}


AFFORDANCES = [
    {"name": "drink_from", "prompt": "something to drink from .", "targets": ["cup", "bottle", "wine glass"]},
    {"name": "sit_on", "prompt": "something to sit on .", "targets": ["chair", "bench", "couch", "bed"]},
    {"name": "ride", "prompt": "something to ride .", "targets": ["bicycle", "motorcycle", "horse"]},
    {"name": "cut_with", "prompt": "something to cut with .", "targets": ["knife", "scissors"]},
    {"name": "carry_in", "prompt": "something to carry things in .", "targets": ["backpack", "handbag", "suitcase"]},
    {"name": "talk_on", "prompt": "something to talk on .", "targets": ["cell phone"]},
    {"name": "eat_with", "prompt": "something to eat with .", "targets": ["fork", "spoon", "knife"]},
    {"name": "read", "prompt": "something to read .", "targets": ["book"]},
]


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


def iou_xyxy(b1, b2):
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
    a2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
    return inter / (a1 + a2 - inter + 1e-9)


def xywh_to_xyxy(x, y, w, h):
    return [x, y, x + w, y + h]


def xywh_abs_to_norm_cxcywh(box_xywh, img_w, img_h):
    x, y, w, h = box_xywh
    cx = (x + 0.5 * w) / img_w
    cy = (y + 0.5 * h) / img_h
    return [cx, cy, w / img_w, h / img_h]


def collect_excluded_image_ids(visual_root):
    excluded = set(HARDCODED_EXCLUDE_IDS)
    for root, _, files in os.walk(visual_root):
        for fn in files:
            if not fn.endswith(".json"):
                continue
            path = os.path.join(root, fn)
            try:
                with open(path, "r") as f:
                    payload = json.load(f)
            except Exception:
                continue
            stack = [payload]
            while stack:
                cur = stack.pop()
                if isinstance(cur, dict):
                    for k, v in cur.items():
                        if k == "image_id":
                            try:
                                excluded.add(int(v))
                            except Exception:
                                pass
                        if isinstance(v, (dict, list)):
                            stack.append(v)
                elif isinstance(cur, list):
                    for v in cur:
                        if isinstance(v, (dict, list)):
                            stack.append(v)
    return excluded


@torch.no_grad()
def run_top1(model_fn, image_pil, prompt, device, dino_tf, internvl_tf):
    img_w, img_h = image_pil.size

    dino_tensor = dino_tf(image_pil).unsqueeze(0).to(device)
    mask = torch.zeros(1, dino_tensor.shape[2], dino_tensor.shape[3], dtype=torch.bool, device=device)
    dino_nested = NestedTensor(dino_tensor, mask)
    internvl_tensor = internvl_tf(image_pil).unsqueeze(0).to(device)
    dino_inputs = {"samples": dino_nested, "captions": [prompt]}

    out = model_fn(internvl_tensor, [prompt], dino_inputs)
    outputs = out[0] if isinstance(out, tuple) else out

    logits = outputs["pred_logits"][0].sigmoid()
    boxes = outputs["pred_boxes"][0]  # normalized cxcywh
    max_scores = logits.max(dim=-1).values
    idx = int(max_scores.argmax().item())
    score = float(max_scores[idx].item())
    cx, cy, w, h = [float(v) for v in boxes[idx].tolist()]
    abs_xyxy = [
        (cx - 0.5 * w) * img_w,
        (cy - 0.5 * h) * img_h,
        (cx + 0.5 * w) * img_w,
        (cy + 0.5 * h) * img_h,
    ]
    return {
        "score": score,
        "query_idx": idx,
        "box_norm_cxcywh": [cx, cy, w, h],
        "box_abs_xyxy": abs_xyxy,
    }


def choose_best_gt(gt_boxes_xyxy, pred_abs_xyxy):
    best_iou = -1.0
    best_gt = None
    for gt in gt_boxes_xyxy:
        v = iou_xyxy(pred_abs_xyxy, gt)
        if v > best_iou:
            best_iou = v
            best_gt = gt
    return best_gt, max(best_iou, 0.0)


def build_candidates(coco_data, excluded_ids, max_candidates):
    cat_map = {c["id"]: c["name"] for c in coco_data["categories"]}
    img_info = {i["id"]: i for i in coco_data["images"]}
    img_anns = defaultdict(list)
    for ann in coco_data["annotations"]:
        img_anns[ann["image_id"]].append(ann)

    candidates = []
    for img_id, anns in img_anns.items():
        if img_id in excluded_ids:
            continue
        info = img_info.get(img_id)
        if info is None:
            continue
        cats = [cat_map[a["category_id"]] for a in anns]
        counts = Counter(cats)
        if len(anns) < 5:
            continue

        for aff in AFFORDANCES:
            target_boxes_xyxy = []
            target_count = 0
            for a in anns:
                cat_name = cat_map[a["category_id"]]
                if cat_name in aff["targets"]:
                    x, y, w, h = a["bbox"]
                    target_boxes_xyxy.append(xywh_to_xyxy(x, y, w, h))
                    target_count += 1
            if target_count == 0:
                continue

            # Prefer ambiguous scenes: multiple valid targets or crowded context.
            if target_count < 2 and len(anns) < 8:
                continue

            scene_complexity = len(anns)
            heuristic = (
                3.0 * min(target_count, 3)
                + 0.15 * scene_complexity
                + 0.1 * counts.get("person", 0)
            )
            candidates.append({
                "image_id": img_id,
                "image_file": info["file_name"],
                "width": int(info["width"]),
                "height": int(info["height"]),
                "prompt": aff["prompt"],
                "affordance": aff["name"],
                "target_categories": aff["targets"],
                "target_count": target_count,
                "scene_complexity": scene_complexity,
                "heuristic": heuristic,
                "gt_boxes_xyxy": target_boxes_xyxy,
            })

    # Keep one candidate per (image,prompt) naturally; sort by heuristic.
    candidates.sort(key=lambda x: -x["heuristic"])
    if max_candidates > 0:
        candidates = candidates[:max_candidates]
    return candidates


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max_candidates", type=int, default=160)
    parser.add_argument("--top_k", type=int, default=10, help="Number of final winners to save")
    parser.add_argument("--min_gap", type=float, default=0.35)
    parser.add_argument("--max_base_iou", type=float, default=0.20)
    parser.add_argument("--min_ours_iou", type=float, default=0.55)
    parser.add_argument("--visual_root", type=str, default=f"{ROOT}/thinkdet/results/visual_compare")
    parser.add_argument("--output", type=str, default="")

    parser.add_argument("--gd_config", type=str, default=GD_CONFIG)
    parser.add_argument("--gd_weights", type=str, default=GD_WEIGHTS)
    parser.add_argument("--internvl_path", type=str, default=INTERNVL_PATH)
    parser.add_argument("--stage1_ckpt", type=str, default=STAGE1_CKPT)
    parser.add_argument("--stage2_ckpt", type=str, default=STAGE2_CKPT)
    parser.add_argument("--coco_val_img", type=str, default=COCO_VAL_IMG)
    parser.add_argument("--coco_val_ann", type=str, default=COCO_VAL_ANN)
    return parser.parse_args()


def main():
    args = parse_args()
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = args.output or f"{ROOT}/thinkdet/results/eval/affordance_baseline_failures_{ts}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    device = torch.device(args.device)
    dino_tf = build_dino_transform()
    internvl_tf = build_internvl_transform()

    print("Loading COCO annotations...")
    with open(args.coco_val_ann, "r") as f:
        coco_data = json.load(f)

    excluded_ids = collect_excluded_image_ids(args.visual_root)
    print(f"Excluded image_ids from prior visual outputs: {len(excluded_ids)}")

    candidates = build_candidates(coco_data, excluded_ids, args.max_candidates)
    print(f"Candidates to scan: {len(candidates)}")
    if not candidates:
        print("No candidates found after filtering.")
        return

    # Resolve image paths early and drop missing.
    valid = []
    for c in candidates:
        p = os.path.join(args.coco_val_img, c["image_file"])
        if os.path.exists(p):
            c = dict(c)
            c["image_path"] = p
            valid.append(c)
    candidates = valid
    print(f"Candidates with existing images: {len(candidates)}")
    if not candidates:
        print("No image files found for candidates.")
        return

    preds_base = [None] * len(candidates)
    preds_s1 = [None] * len(candidates)
    preds_s2 = [None] * len(candidates)

    # Pass 1: Baseline
    print("\n[1/3] Loading baseline...")
    gd_base = load_gd_model(args.gd_config, args.gd_weights, device="cpu").to(device).eval()

    def baseline_fn(ivl_imgs, queries, dino_inputs):
        return gd_base(**dino_inputs), {}

    print(f"[1/3] Running baseline on {len(candidates)} candidates...")
    for i, c in enumerate(candidates):
        image_pil = Image.open(c["image_path"]).convert("RGB")
        preds_base[i] = run_top1(baseline_fn, image_pil, c["prompt"], device, dino_tf, internvl_tf)
        if (i + 1) % 25 == 0:
            print(f"  baseline {i+1}/{len(candidates)}")
    del gd_base, baseline_fn
    torch.cuda.empty_cache()

    # Pass 2: Stage 1
    print("\n[2/3] Loading Stage1...")
    ckpt1 = torch.load(args.stage1_ckpt, map_location="cpu")
    gd_s1 = load_gd_model(args.gd_config, args.gd_weights, device="cpu")
    model_s1 = ThinkDetModel(
        grounding_dino=gd_s1,
        internvl_path=args.internvl_path,
        extract_layer=9, extract_layers=[9],
        injection_layers=DEFAULT_INJECTION_LAYERS,
        tma_m=8, tma_n_heads=8,
    )
    model_s1.load_state_dict(ckpt1["trainable_state_dict"], strict=False)
    model_s1 = model_s1.to(device).eval()
    del ckpt1

    print(f"[2/3] Running Stage1 on {len(candidates)} candidates...")
    for i, c in enumerate(candidates):
        image_pil = Image.open(c["image_path"]).convert("RGB")
        preds_s1[i] = run_top1(model_s1, image_pil, c["prompt"], device, dino_tf, internvl_tf)
        if (i + 1) % 25 == 0:
            print(f"  stage1 {i+1}/{len(candidates)}")
    del model_s1
    torch.cuda.empty_cache()

    # Pass 3: Stage 2
    print("\n[3/3] Loading Stage2...")
    ckpt2 = torch.load(args.stage2_ckpt, map_location="cpu")
    inject_layers = ckpt2.get("injection_layers", DEFAULT_INJECTION_LAYERS)
    tma_m = ckpt2.get("tma_m", 8)
    tma_nheads = ckpt2.get("tma_n_heads", 8)
    ext_layers = ckpt2.get("extract_layers", [9])

    gd_s2 = load_gd_model(args.gd_config, args.gd_weights, device="cpu")
    model_s2 = ThinkDetModel(
        grounding_dino=gd_s2,
        internvl_path=args.internvl_path,
        extract_layer=max(ext_layers), extract_layers=ext_layers,
        injection_layers=inject_layers,
        tma_m=tma_m, tma_n_heads=tma_nheads,
    )
    model_s2.load_state_dict(ckpt2["trainable_state_dict"], strict=False)
    model_s2 = model_s2.to(device).eval()
    del ckpt2

    print(f"[3/3] Running Stage2 on {len(candidates)} candidates...")
    for i, c in enumerate(candidates):
        image_pil = Image.open(c["image_path"]).convert("RGB")
        preds_s2[i] = run_top1(model_s2, image_pil, c["prompt"], device, dino_tf, internvl_tf)
        if (i + 1) % 25 == 0:
            print(f"  stage2 {i+1}/{len(candidates)}")
    del model_s2
    torch.cuda.empty_cache()

    # Score and filter
    wins = []
    for i, c in enumerate(candidates):
        gt_boxes = c["gt_boxes_xyxy"]
        base_gt, base_iou = choose_best_gt(gt_boxes, preds_base[i]["box_abs_xyxy"])
        s1_gt, s1_iou = choose_best_gt(gt_boxes, preds_s1[i]["box_abs_xyxy"])
        s2_gt, s2_iou = choose_best_gt(gt_boxes, preds_s2[i]["box_abs_xyxy"])

        ours_iou = s1_iou
        ours_name = "stage1"
        ours_pred = preds_s1[i]
        ours_gt = s1_gt
        if s2_iou > ours_iou:
            ours_iou = s2_iou
            ours_name = "stage2"
            ours_pred = preds_s2[i]
            ours_gt = s2_gt

        gap = ours_iou - base_iou
        if base_iou <= args.max_base_iou and ours_iou >= args.min_ours_iou and gap >= args.min_gap:
            chosen_gt = ours_gt if ours_gt is not None else base_gt
            if chosen_gt is None:
                continue
            x1, y1, x2, y2 = chosen_gt
            gt_xywh = [x1, y1, x2 - x1, y2 - y1]
            gt_norm = xywh_abs_to_norm_cxcywh(gt_xywh, c["width"], c["height"])

            wins.append({
                "idx": i,
                "image_id": c["image_id"],
                "image_file": c["image_file"],
                "prompt": c["prompt"],
                "expression": c["prompt"].strip().rstrip("."),
                "affordance": c["affordance"],
                "target_categories": c["target_categories"],
                "baseline": {
                    "top_iou": base_iou,
                    "top_score": preds_base[i]["score"],
                    "top_box": preds_base[i]["box_norm_cxcywh"],
                },
                "stage1": {
                    "top_iou": s1_iou,
                    "top_score": preds_s1[i]["score"],
                    "top_box": preds_s1[i]["box_norm_cxcywh"],
                },
                "stage2": {
                    "top_iou": s2_iou,
                    "top_score": preds_s2[i]["score"],
                    "top_box": preds_s2[i]["box_norm_cxcywh"],
                },
                "winner_model": ours_name,
                "winner_iou": ours_iou,
                "iou_gap": gap,
                "gt_box": gt_norm,
            })

    wins.sort(key=lambda x: -x["iou_gap"])

    # Keep unique images and diverse affordances
    selected = []
    used_img = set()
    aff_count = Counter()
    for w in wins:
        if w["image_id"] in used_img:
            continue
        # prevent one affordance from taking all slots
        if aff_count[w["affordance"]] >= 2:
            continue
        selected.append(w)
        used_img.add(w["image_id"])
        aff_count[w["affordance"]] += 1
        if len(selected) >= args.top_k:
            break

    for rank, s in enumerate(selected, 1):
        s["rank"] = rank

    payload = {
        "status": "ok",
        "timestamp": ts,
        "dataset": {
            "name": "coco",
            "split": "val2017",
            "evaluated_samples": len(candidates),
        },
        "selection_criteria": {
            "prompt_type": "affordance",
            "min_gap": args.min_gap,
            "max_base_iou": args.max_base_iou,
            "min_ours_iou": args.min_ours_iou,
            "excluded_prev_visual_image_ids": True,
        },
        "num_candidates": len(wins),
        "num_selected": len(selected),
        "selected_cases": selected,
    }

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print("\n" + "=" * 70)
    print("AFFORDANCE FAILURE MINING SUMMARY")
    print("=" * 70)
    print(f"Scanned candidates: {len(candidates)}")
    print(f"Wins (before diversity filter): {len(wins)}")
    print(f"Selected: {len(selected)}")
    for s in selected[:10]:
        print(
            f"  rank{s['rank']} img={s['image_id']} aff={s['affordance']:<10} "
            f"base={s['baseline']['top_iou']:.3f} s1={s['stage1']['top_iou']:.3f} "
            f"s2={s['stage2']['top_iou']:.3f} gap={s['iou_gap']:.3f} "
            f"winner={s['winner_model']}"
        )
        print(f"    prompt: {s['prompt']}")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
