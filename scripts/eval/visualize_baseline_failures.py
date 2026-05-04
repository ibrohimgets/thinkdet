"""
Visualize baseline failure cases vs Stage-2 ThinkDet predictions.

Uses the pre-computed box data in affordance_baseline_failures_20260219_074604.json
so NO GPU / model loading is needed.

Output:
    thinkdet/results/qualitative/baseline_failures_vis/
        case_<N>_<affordance>.png  — side-by-side: baseline | stage2
"""

import json
import os
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = "/home/iibrohimm/project/next_step"
COCO_VAL = f"{ROOT}/dataSets/coco/val2017"
FAILURES_JSON = (
    f"{ROOT}/thinkdet/results/eval/"
    "affordance_baseline_failures_20260219_074604.json"
)
OUT_DIR = f"{ROOT}/thinkdet/results/qualitative/baseline_failures_vis"


# ── colours ──────────────────────────────────────────────────────────
COL_BASE  = (220, 50,  50)   # red   — baseline (wrong)
COL_OURS  = (50,  200, 80)   # green — stage-2  (correct)
COL_GT    = (255, 200, 0)    # gold  — ground truth
BOX_W     = 3
FONT_SIZE = 22


def load_font(size=FONT_SIZE):
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    ]:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    return ImageFont.load_default()


def cxcywh_to_xyxy(box, W, H):
    """Normalised cxcywh  →  absolute xyxy."""
    cx, cy, bw, bh = box
    x1 = (cx - bw / 2) * W
    y1 = (cy - bh / 2) * H
    x2 = (cx + bw / 2) * W
    y2 = (cy + bh / 2) * H
    return [x1, y1, x2, y2]


def draw_box(draw, box_xyxy, colour, label, font):
    x1, y1, x2, y2 = [int(v) for v in box_xyxy]
    for t in range(BOX_W):
        draw.rectangle([x1 + t, y1 + t, x2 - t, y2 - t], outline=colour)
    # Label background
    tw, th = 0, 0
    try:
        bbox = font.getbbox(label)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except AttributeError:
        tw, th = font.getsize(label)
    pad = 4
    draw.rectangle([x1, y1 - th - pad * 2, x1 + tw + pad * 2, y1], fill=colour)
    draw.text((x1 + pad, y1 - th - pad), label, fill=(255, 255, 255), font=font)


def make_panel(img_pil, case, model_key, colour, label_prefix, font):
    """Return an annotated copy of img_pil for one model."""
    W, H = img_pil.size
    out = img_pil.copy()
    draw = ImageDraw.Draw(out)

    # Ground-truth box
    gt_xyxy = cxcywh_to_xyxy(case["gt_box"], W, H)
    draw_box(draw, gt_xyxy, COL_GT, "GT", font)

    # Model prediction box
    pred = case[model_key]
    pred_xyxy = cxcywh_to_xyxy(pred["top_box"], W, H)
    iou = pred["top_iou"]
    draw_box(draw, pred_xyxy, colour, f"{label_prefix}  IoU={iou:.2f}", font)

    # Header banner
    hdr_h = 36
    banner = Image.new("RGB", (W, hdr_h), colour)
    bd = ImageDraw.Draw(banner)
    hdr_txt = f"{label_prefix}  |  {case['affordance']}  |  \"{case['prompt']}\""
    bd.text((8, 6), hdr_txt, fill=(255, 255, 255), font=font)
    combined = Image.new("RGB", (W, H + hdr_h))
    combined.paste(banner, (0, 0))
    combined.paste(out, (0, hdr_h))
    return combined


def visualize_case(case, idx, font):
    img_path = os.path.join(COCO_VAL, case["image_file"])
    if not os.path.exists(img_path):
        print(f"  [SKIP] image not found: {img_path}")
        return None

    img = Image.open(img_path).convert("RGB")

    base_panel  = make_panel(img, case, "baseline", COL_BASE,  "BASELINE",  font)
    stage2_panel = make_panel(img, case, "stage2",  COL_OURS,  "THINKDET",  font)

    # Side-by-side
    W = base_panel.width + stage2_panel.width + 8
    H = max(base_panel.height, stage2_panel.height)
    canvas = Image.new("RGB", (W, H), (30, 30, 30))
    canvas.paste(base_panel, (0, 0))
    canvas.paste(stage2_panel, (base_panel.width + 8, 0))

    aff = case["affordance"]
    out_name = f"case_{idx+1:02d}_{aff}.png"
    out_path = os.path.join(OUT_DIR, out_name)
    canvas.save(out_path)
    print(f"  Saved → {out_path}")
    return out_path


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    font = load_font()

    with open(FAILURES_JSON) as f:
        data = json.load(f)

    cases = data["selected_cases"]
    print(f"Visualising {len(cases)} baseline-failure cases...")
    saved = []
    for i, case in enumerate(cases):
        path = visualize_case(case, i, font)
        if path:
            saved.append(path)

    print(f"\nDone. {len(saved)} images saved to:\n  {OUT_DIR}")


if __name__ == "__main__":
    main()
