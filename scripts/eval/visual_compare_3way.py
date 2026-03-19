"""
ThinkDet — 3-Way Qualitative Comparison
Baseline (DINO) | Stage 1 TMA | Stage 2 TMA

Uses ambiguous / compositional prompts on COCO val images with
multiple similar objects. Raw top-1 prediction, no confidence filtering.

Usage:
    python thinkdet/scripts/eval/visual_compare_3way.py
    python thinkdet/scripts/eval/visual_compare_3way.py \
        --source_summary /path/to/visual_comparison_summary.json --max_cases 5
"""

import os
import sys
import json
import datetime
import argparse
from collections import Counter

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = "/home/iibrohimm/project/next_step"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "GroundingDINO", "GroundingDINO"))

from pycocotools.coco import COCO
from thinkdet.models.arch import ThinkDetModel, DEFAULT_INJECTION_LAYERS
from groundingdino.util.inference import load_model as load_gd_model
from groundingdino.util.misc import NestedTensor
import torchvision.transforms as T
import torchvision.transforms.functional as TF


# ── Paths ──────────────────────────────────────────────────────────────────
GD_CONFIG = f"{ROOT}/GroundingDINO/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GD_WEIGHTS = f"{ROOT}/GroundingDINO/GroundingDINO/weights/groundingdino_swint_ogc.pth"
INTERNVL_PATH = f"{ROOT}/InternVL3_5-1B"
COCO_VAL_IMG = f"{ROOT}/dataSets/coco/val2017"
COCO_TRAIN_IMG = f"{ROOT}/dataSets/coco/train2017"
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

TS = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
DEFAULT_OUTPUT_DIR = f"{ROOT}/thinkdet/results/visual_compare/3way_s1s2_baseline_{TS}"


# ── Image auto-selection ────────────────────────────────────────────────────
def find_images_with_multiple(coco, cat_name, min_count=3, exclude_ids=None):
    """Return image IDs that have >= min_count instances of cat_name."""
    cat_ids = coco.getCatIds(catNms=[cat_name])
    if not cat_ids:
        return []
    ann_ids = coco.getAnnIds(catIds=cat_ids)
    anns = coco.loadAnns(ann_ids)
    counter = Counter(a["image_id"] for a in anns)
    exclude = exclude_ids or set()
    results = [
        img_id for img_id, cnt in sorted(counter.items(), key=lambda x: -x[1])
        if cnt >= min_count and img_id not in exclude
    ]
    return results


def pick_test_cases(coco):
    """
    Auto-select 5 COCO val images and assign compositional prompts.
    Returns list of dicts: {image_id, image_file, prompt, prompt_type}.
    """
    used = set()
    cases = []

    # 1. Multiple people — ordinal/spatial prompt
    person_imgs = find_images_with_multiple(coco, "person", min_count=4, exclude_ids=used)
    if person_imgs:
        img_id = person_imgs[0]
        used.add(img_id)
        info = coco.loadImgs(img_id)[0]
        cases.append({
            "image_id":   img_id,
            "image_file": info["file_name"],
            "prompt":     "the second person from the right .",
            "prompt_type": "Ordinal+spatial (people)",
        })

    # 2. Multiple people — attribute prompt
    person_imgs2 = find_images_with_multiple(coco, "person", min_count=4, exclude_ids=used)
    if person_imgs2:
        img_id = person_imgs2[0]
        used.add(img_id)
        info = coco.loadImgs(img_id)[0]
        cases.append({
            "image_id":   img_id,
            "image_file": info["file_name"],
            "prompt":     "the man wearing a white shirt .",
            "prompt_type": "Attribute binding (person+clothing)",
        })

    # 3. Multiple dogs — size+spatial
    dog_imgs = find_images_with_multiple(coco, "dog", min_count=2, exclude_ids=used)
    if dog_imgs:
        img_id = dog_imgs[0]
        used.add(img_id)
        info = coco.loadImgs(img_id)[0]
        cases.append({
            "image_id":   img_id,
            "image_file": info["file_name"],
            "prompt":     "the smaller dog on the left .",
            "prompt_type": "Size+spatial (dogs)",
        })

    # 4. Cups / mugs — open-vocabulary function
    cup_imgs = find_images_with_multiple(coco, "cup", min_count=2, exclude_ids=used)
    if cup_imgs:
        img_id = cup_imgs[0]
        used.add(img_id)
        info = coco.loadImgs(img_id)[0]
        cases.append({
            "image_id":   img_id,
            "image_file": info["file_name"],
            "prompt":     "something to drink from .",
            "prompt_type": "Open-vocab function (cup)",
        })

    # 5. Multiple chairs — spatial
    chair_imgs = find_images_with_multiple(coco, "chair", min_count=4, exclude_ids=used)
    if chair_imgs:
        img_id = chair_imgs[0]
        used.add(img_id)
        info = coco.loadImgs(img_id)[0]
        cases.append({
            "image_id":   img_id,
            "image_file": info["file_name"],
            "prompt":     "the empty chair closest to the camera .",
            "prompt_type": "Spatial+attribute (chair)",
        })

    return cases


def load_test_cases_from_summary(summary_path, max_cases=0):
    """
    Load hard cases from previously mined summary JSON.
    Expected schema:
      - selected_cases[].{image_id, prompt, iou_gap, gt_box}
      - selection_criteria
      - dataset
    """
    with open(summary_path, "r") as f:
        payload = json.load(f)

    selected = payload.get("selected_cases", [])
    if max_cases and max_cases > 0:
        selected = selected[:max_cases]

    dataset = payload.get("dataset", {})
    criteria = payload.get("selection_criteria", {})
    cases = []
    for c in selected:
        image_id = int(c["image_id"])
        cases.append({
            "image_id": image_id,
            "image_file": f"{image_id:012d}.jpg",
            "prompt": c.get("prompt", c.get("expression", "")),
            "prompt_type": "Hard-mined baseline-failure",
            "gt_box_norm": c.get("gt_box", None),  # assumed normalized cxcywh
            "source_rank": c.get("rank", None),
            "source_iou_gap": c.get("iou_gap", None),
            "source_baseline_iou": (c.get("baseline", {}) or {}).get("top_iou", None),
            "source_tma_iou": (c.get("tma", {}) or {}).get("top_iou", None),
        })
    return cases, {"dataset": dataset, "selection_criteria": criteria}


def resolve_image_path(image_file, image_dirs):
    for d in image_dirs:
        if not d:
            continue
        p = os.path.join(d, image_file)
        if os.path.exists(p):
            return p
    return None


def cxcywh_norm_to_xyxy_abs(box, img_w, img_h):
    if box is None:
        return None
    cx, cy, bw, bh = box
    x1 = (cx - bw / 2) * img_w
    y1 = (cy - bh / 2) * img_h
    x2 = (cx + bw / 2) * img_w
    y2 = (cy + bh / 2) * img_h
    return [x1, y1, x2, y2]


# ── Transforms ─────────────────────────────────────────────────────────────
def build_dino_transform():
    normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    def transform(image_pil):
        w, h = image_pil.size
        scale = 800 / min(w, h)
        if scale * max(w, h) > 1333:
            scale = 1333 / max(w, h)
        nw, nh = int(w * scale), int(h * scale)
        img = image_pil.resize((nw, nh), Image.BILINEAR)
        t = TF.to_tensor(img)
        return normalize(t)
    return transform


def build_internvl_transform():
    return T.Compose([
        T.Resize((448, 448), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


# ── Inference ───────────────────────────────────────────────────────────────
@torch.no_grad()
def run_inference(model_fn, image_pil, prompt, device, dino_tf, ivl_tf):
    """
    Run one image+prompt through model_fn.
    Returns list of dicts sorted by score descending:
        [{"box": [x1,y1,x2,y2], "score": float}, ...]  (absolute pixel coords)
    """
    img_w, img_h = image_pil.size

    dino_t = dino_tf(image_pil).unsqueeze(0).to(device)
    mask   = torch.zeros(1, dino_t.shape[2], dino_t.shape[3],
                         dtype=torch.bool, device=device)
    nested = NestedTensor(dino_t, mask)
    ivl_t  = ivl_tf(image_pil).unsqueeze(0).to(device)

    dino_inputs = {"samples": nested, "captions": [prompt]}

    out = model_fn(ivl_t, [prompt], dino_inputs)
    outputs = out[0] if isinstance(out, tuple) else out

    logits = outputs["pred_logits"][0].sigmoid()   # [N, 256]
    boxes  = outputs["pred_boxes"][0]               # [N, 4]

    scores = logits.max(dim=-1).values             # [N]
    order  = scores.argsort(descending=True)

    results = []
    for i in order:
        cx, cy, bw, bh = boxes[i].tolist()
        x1 = (cx - bw / 2) * img_w
        y1 = (cy - bh / 2) * img_h
        x2 = (cx + bw / 2) * img_w
        y2 = (cy + bh / 2) * img_h
        results.append({"box": [x1, y1, x2, y2], "score": float(scores[i])})
    return results


# ── Drawing ─────────────────────────────────────────────────────────────────
try:
    FONT_BOLD  = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15)
    FONT_NORM  = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    FONT_SMALL = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
except Exception:
    FONT_BOLD = FONT_NORM = FONT_SMALL = ImageFont.load_default()


def draw_panel(image_pil, results, title, max_show=3, gt_box_xyxy=None):
    """
    Draw detections on image. Top-1 = thick green, 2-3 = thin orange/red.
    Returns PIL image with title bar on top.
    """
    img = image_pil.copy().convert("RGB")
    draw = ImageDraw.Draw(img)

    COLORS = [(0, 220, 60), (255, 160, 20), (220, 40, 40)]  # green, orange, red
    WIDTHS = [4, 2, 1]

    for rank, r in enumerate(results[:max_show]):
        if r["score"] < 0.01:
            continue
        x1, y1, x2, y2 = [max(0, v) for v in r["box"]]
        col = COLORS[rank]
        lw  = WIDTHS[rank]
        for off in range(lw):
            draw.rectangle([x1-off, y1-off, x2+off, y2+off], outline=col)

        # Score label (top-1 only, others de-emphasized)
        label = f"#{rank+1}  {r['score']:.3f}"
        tw = len(label) * 8
        bg_alpha = 220 if rank == 0 else 140
        draw.rectangle([x1, y1 - 18, x1 + tw + 4, y1], fill=(*col, bg_alpha))
        draw.text((x1 + 2, y1 - 17), label, fill=(255, 255, 255), font=FONT_SMALL)

    if gt_box_xyxy is not None:
        x1, y1, x2, y2 = [max(0, v) for v in gt_box_xyxy]
        for off in range(3):
            draw.rectangle([x1 - off, y1 - off, x2 + off, y2 + off], outline=(30, 255, 30))
        gt_label = "GT"
        draw.rectangle([x1, y1 - 16, x1 + 24, y1], fill=(30, 180, 30))
        draw.text((x1 + 3, y1 - 15), gt_label, fill=(255, 255, 255), font=FONT_SMALL)

    # Title bar
    bar_h = 32
    panel = Image.new("RGB", (img.width, img.height + bar_h), (45, 45, 45))
    panel.paste(img, (0, bar_h))
    d = ImageDraw.Draw(panel)
    d.text((8, 7), title, fill=(255, 255, 255), font=FONT_BOLD)
    return panel


def create_3way(panels, prompt, prompt_type, image_id):
    """Join 3 panels side-by-side with a shared header."""
    gap = 8
    h   = max(p.height for p in panels)
    panels_resized = []
    for p in panels:
        if p.height != h:
            sc = h / p.height
            p  = p.resize((int(p.width * sc), h), Image.BILINEAR)
        panels_resized.append(p)

    total_w = sum(p.width for p in panels_resized) + gap * (len(panels) - 1)
    header_h = 52

    canvas = Image.new("RGB", (total_w, h + header_h), (22, 22, 22))
    d = ImageDraw.Draw(canvas)

    # Prompt header
    d.text((10, 6),  f'Prompt: "{prompt}"',      fill=(100, 255, 120), font=FONT_BOLD)
    d.text((10, 28), f"({prompt_type})  img={image_id}", fill=(160, 160, 160), font=FONT_SMALL)

    x = 0
    for p in panels_resized:
        canvas.paste(p, (x, header_h))
        x += p.width + gap

    return canvas


# ── Main ────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gd_config", type=str, default=GD_CONFIG)
    parser.add_argument("--gd_weights", type=str, default=GD_WEIGHTS)
    parser.add_argument("--internvl_path", type=str, default=INTERNVL_PATH)
    parser.add_argument("--stage1_ckpt", type=str, default=STAGE1_CKPT)
    parser.add_argument("--stage2_ckpt", type=str, default=STAGE2_CKPT)

    parser.add_argument("--coco_val_ann", type=str, default=COCO_VAL_ANN)
    parser.add_argument("--coco_val_img", type=str, default=COCO_VAL_IMG)
    parser.add_argument("--coco_train_img", type=str, default=COCO_TRAIN_IMG)
    parser.add_argument("--source_summary", type=str, default=None,
                        help="If set, load hard-mined cases from summary JSON.")
    parser.add_argument("--max_cases", type=int, default=0,
                        help="Only used with --source_summary. 0=all.")

    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max_show", type=int, default=3,
                        help="How many ranked predictions to draw per panel.")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 65)
    print("  3-Way Qualitative Comparison: Baseline | Stage1 | Stage2")
    print("=" * 65)

    source_meta = {}
    if args.source_summary:
        print(f"\nLoading hard-mined cases from summary:\n  {args.source_summary}")
        test_cases, source_meta = load_test_cases_from_summary(
            args.source_summary, max_cases=args.max_cases
        )
        print(f"  Loaded {len(test_cases)} cases from summary")
    else:
        # ── COCO ──
        print("\nLoading COCO annotations...")
        coco = COCO(args.coco_val_ann)
        # ── Auto-pick 5 test cases ──
        print("Selecting test images...")
        test_cases = pick_test_cases(coco)
        print(f"  Found {len(test_cases)} test cases")

    if not test_cases:
        print("No test cases found. Exiting.")
        return

    image_dirs = [args.coco_val_img, args.coco_train_img]
    if args.source_summary:
        # RefCOCO-mined cases usually come from train2017
        image_dirs = [args.coco_train_img, args.coco_val_img]

    resolved_cases = []
    for tc in test_cases:
        img_path = resolve_image_path(tc["image_file"], image_dirs)
        if img_path is None:
            print(f"  [WARN] image not found: {tc['image_file']} (img_id={tc['image_id']})")
            continue
        tc2 = dict(tc)
        tc2["image_path"] = img_path
        resolved_cases.append(tc2)
    test_cases = resolved_cases

    if not test_cases:
        print("No resolvable image paths for selected cases. Exiting.")
        return

    for tc in test_cases:
        print(f"    [{tc['image_id']}] {tc['prompt_type']}")
        print(f"      prompt: {tc['prompt']}")
        if tc.get("source_iou_gap") is not None:
            print(
                f"      source gap={tc['source_iou_gap']:.3f} "
                f"(base={tc['source_baseline_iou']:.3f}, tma={tc['source_tma_iou']:.3f})"
            )

    # ── Transforms ──
    dino_tf = build_dino_transform()
    ivl_tf  = build_internvl_transform()

    def run_all_cases(model_fn, label):
        preds = []
        print(f"\n[{label}] running {len(test_cases)} cases...")
        for idx, tc in enumerate(test_cases):
            img_path = tc["image_path"]
            image_pil = Image.open(img_path).convert("RGB")
            res = run_inference(model_fn, image_pil, tc["prompt"], device, dino_tf, ivl_tf)
            preds.append(res)
            print(f"  [{idx+1}] top1={res[0]['score']:.4f}")
        return preds

    all_preds = {}

    # --- PASS 1: Baseline (load → run → delete) ---
    print("\n[1/3] Loading Baseline (DINO)...")
    gd_base = load_gd_model(args.gd_config, args.gd_weights, device="cpu")
    gd_base = gd_base.to(device).eval()
    def baseline_fn(ivl_imgs, queries, dino_inputs):
        return gd_base(**dino_inputs), {}
    all_preds["baseline"] = run_all_cases(baseline_fn, "PASS 1/3  Baseline")
    del gd_base, baseline_fn
    torch.cuda.empty_cache()

    # --- PASS 2: Stage 1 (load → run → delete) ---
    print("\n[2/3] Loading Stage 1 TMA...")
    ckpt1 = torch.load(args.stage1_ckpt, map_location="cpu")
    inject_layers1 = ckpt1.get("injection_layers", DEFAULT_INJECTION_LAYERS)
    tma_m1 = ckpt1.get("tma_m", 8) or 8
    tma_nheads1 = ckpt1.get("tma_n_heads", 8) or 8
    ext_layers1 = ckpt1.get("extract_layers") or [ckpt1.get("extract_layer", 9)]
    fusion_mode1 = ckpt1.get("fusion_mode", "concat") or "concat"
    layer_fusion1 = ckpt1.get("layer_fusion", "mean") or "mean"
    gd_s1 = load_gd_model(args.gd_config, args.gd_weights, device="cpu")
    model_s1 = ThinkDetModel(
        grounding_dino=gd_s1,
        internvl_path=args.internvl_path,
        extract_layer=max(ext_layers1),
        extract_layers=ext_layers1,
        layer_fusion=layer_fusion1,
        injection_layers=inject_layers1,
        tma_m=tma_m1,
        tma_n_heads=tma_nheads1,
        fusion_mode=fusion_mode1,
    )
    mk, uk = model_s1.load_state_dict(ckpt1["trainable_state_dict"], strict=False)
    print(f"  missing={len(mk)} unexpected={len(uk)}")
    model_s1 = model_s1.to(device).eval()
    del ckpt1
    all_preds["stage1"] = run_all_cases(model_s1, "PASS 2/3  Stage 1")
    del model_s1
    torch.cuda.empty_cache()

    # --- PASS 3: Stage 2 (load → run → delete) ---
    print("\n[3/3] Loading Stage 2 TMA...")
    ckpt2 = torch.load(args.stage2_ckpt, map_location="cpu")
    inject_layers = ckpt2.get("injection_layers", DEFAULT_INJECTION_LAYERS)
    tma_m      = ckpt2.get("tma_m", 8)
    tma_nheads = ckpt2.get("tma_n_heads", 8) or 8
    ext_layers = ckpt2.get("extract_layers") or [ckpt2.get("extract_layer", 9)]
    fusion_mode2 = ckpt2.get("fusion_mode", "concat") or "concat"
    layer_fusion2 = ckpt2.get("layer_fusion", "mean") or "mean"
    gd_s2 = load_gd_model(args.gd_config, args.gd_weights, device="cpu")
    model_s2 = ThinkDetModel(
        grounding_dino=gd_s2,
        internvl_path=args.internvl_path,
        extract_layer=max(ext_layers),
        extract_layers=ext_layers,
        layer_fusion=layer_fusion2,
        injection_layers=inject_layers,
        tma_m=tma_m, tma_n_heads=tma_nheads,
        fusion_mode=fusion_mode2,
    )
    mk, uk = model_s2.load_state_dict(ckpt2["trainable_state_dict"], strict=False)
    print(f"  missing={len(mk)} unexpected={len(uk)}")
    model_s2 = model_s2.to(device).eval()
    del ckpt2
    all_preds["stage2"] = run_all_cases(model_s2, "PASS 3/3  Stage 2")
    del model_s2
    torch.cuda.empty_cache()

    # --- Compose visualizations ---
    print("\nComposing visualizations...")
    summary = []

    for idx, tc in enumerate(test_cases):
        img_path = tc["image_path"]
        image_pil = Image.open(img_path).convert("RGB")
        prompt    = tc["prompt"]
        ptype     = tc["prompt_type"]
        img_id    = tc["image_id"]

        res_base = all_preds["baseline"][idx]
        res_s1   = all_preds["stage1"][idx]
        res_s2   = all_preds["stage2"][idx]

        top1_base = res_base[0]
        top1_s1   = res_s1[0]
        top1_s2   = res_s2[0]

        print(f"\n[{idx+1}] {ptype}")
        print(f"  Baseline: score={top1_base['score']:.4f}  box={[f'{v:.0f}' for v in top1_base['box']]}")
        print(f"  Stage 1:  score={top1_s1['score']:.4f}  box={[f'{v:.0f}' for v in top1_s1['box']]}")
        print(f"  Stage 2:  score={top1_s2['score']:.4f}  box={[f'{v:.0f}' for v in top1_s2['box']]}")

        gt_box_xyxy = None
        if tc.get("gt_box_norm") is not None:
            gt_box_xyxy = cxcywh_norm_to_xyxy_abs(
                tc["gt_box_norm"], image_pil.width, image_pil.height
            )

        panel_base = draw_panel(
            image_pil, res_base, "BASELINE (DINO only)", max_show=args.max_show, gt_box_xyxy=gt_box_xyxy
        )
        panel_s1 = draw_panel(
            image_pil, res_s1, "Stage 1 TMA", max_show=args.max_show, gt_box_xyxy=gt_box_xyxy
        )
        panel_s2 = draw_panel(
            image_pil, res_s2, "Stage 2 TMA", max_show=args.max_show, gt_box_xyxy=gt_box_xyxy
        )

        comparison = create_3way([panel_base, panel_s1, panel_s2], prompt, ptype, img_id)

        slug  = ptype.split("(")[0].strip().lower().replace(" ", "_").replace("+", "_")
        fname = f"case{idx+1}_{img_id}_{slug}.jpg"
        out_path = os.path.join(args.output_dir, fname)
        comparison.save(out_path, quality=95)
        print(f"  -> {out_path}")

        entry = {
            "case":          idx + 1,
            "image_id":      img_id,
            "image_file":    tc["image_file"],
            "prompt":        prompt,
            "prompt_type":   ptype,
            "baseline_top1": top1_base,
            "stage1_top1":   top1_s1,
            "stage2_top1":   top1_s2,
            "output_file":   fname,
        }
        if tc.get("source_rank") is not None:
            entry["source_rank"] = tc["source_rank"]
            entry["source_iou_gap"] = tc["source_iou_gap"]
            entry["source_baseline_iou"] = tc["source_baseline_iou"]
            entry["source_tma_iou"] = tc["source_tma_iou"]
        summary.append(entry)

    # ── Summary JSON ──
    json_path = os.path.join(args.output_dir, "summary.json")
    with open(json_path, "w") as f:
        json.dump({
            "timestamp":   TS,
            "stage1_ckpt": args.stage1_ckpt,
            "stage2_ckpt": args.stage2_ckpt,
            "source_summary": args.source_summary,
            "source_meta": source_meta,
            "cases":       summary,
        }, f, indent=2)

    print("=" * 65)
    print("  RESULTS SUMMARY")
    print("=" * 65)
    print(f"  {'Case':<4} {'Prompt type':<35} {'Base':>7} {'S1':>7} {'S2':>7}")
    print(f"  {'-'*62}")
    for s in summary:
        b = s["baseline_top1"]["score"] if s["baseline_top1"] else 0
        s1 = s["stage1_top1"]["score"]  if s["stage1_top1"]   else 0
        s2 = s["stage2_top1"]["score"]  if s["stage2_top1"]   else 0
        print(f"  {s['case']:<4} {s['prompt_type']:<35} {b:>7.4f} {s1:>7.4f} {s2:>7.4f}")
    print(f"\n  Saved: {args.output_dir}")
    print(f"  JSON:  {json_path}")


if __name__ == "__main__":
    main()
