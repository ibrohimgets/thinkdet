"""
Full ThinkDet pipeline on baseline-failure cases — matching the architecture diagram:

  Image + Query
      │
      ▼
  InternVL (frozen)  →  Layer-9 hidden states  →  TMA adapter  →  aug_tokens
                                                                        │
  GroundingDINO  ←─────────────────── inject into layers 1,3,5 ────────┘
      │
      ▼
  Detection Heads  →  top-K predictions + scores
      │
      ├─ confidence > threshold? ──► YES ──► refined box ✓
      │
      └─ NO ──► LLM evidence-check reranker (InternVL crops)
                    │
                    └─► best reranked box

Output:
    thinkdet/results/qualitative/full_pipeline_vis/
        case_<N>_<affordance>.png   — 3-panel: baseline | thinkdet | after-fallback
"""

import json
import os
import sys

import torch
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as T
import torchvision.transforms.functional as TF

ROOT = "/home/iibrohimm/project/next_step"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "GroundingDINO", "GroundingDINO"))

from thinkdet.models.arch import ThinkDetModel, DEFAULT_INJECTION_LAYERS
from thinkdet.inference.fallback import InternVLYesNoReranker
from groundingdino.util.inference import load_model as load_gd_model
from groundingdino.util.misc import NestedTensor

# ── paths ─────────────────────────────────────────────────────────────
GD_CONFIG  = f"{ROOT}/GroundingDINO/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GD_WEIGHTS = f"{ROOT}/GroundingDINO/GroundingDINO/weights/groundingdino_swint_ogc.pth"
INTERNVL_PATH = f"{ROOT}/InternVL3_5-1B"
STAGE2_CKPT = (
    f"{ROOT}/thinkdet/checkpoints/stage2_tma/"
    "layer9_tma_m8_20260218_234913/thinkdet_tma_stage2_epoch2.pth"
)
FAILURES_JSON = (
    f"{ROOT}/thinkdet/results/eval/"
    "affordance_baseline_failures_20260219_074604.json"
)
COCO_VAL = f"{ROOT}/dataSets/coco/val2017"
OUT_DIR   = f"{ROOT}/thinkdet/results/qualitative/full_pipeline_vis"

# ── thresholds (matching the diagram) ─────────────────────────────────
CONF_THRESHOLD = 0.20   # if top-1 score < this → trigger LLM fallback
TOP_K          = 5      # detector candidates passed to reranker
LLM_WEIGHT     = 0.30   # combined = detector_score + weight * llm_score

# ── colours ───────────────────────────────────────────────────────────
COL_BASE    = (220, 50,  50)   # red   — baseline
COL_THINK   = (50,  140, 255)  # blue  — thinkdet (before fallback)
COL_FINAL   = (50,  210, 80)   # green — after fallback (or thinkdet if confident)
COL_GT      = (255, 200, 0)    # gold  — ground truth
BOX_W = 3
FONT_SIZE = 20


# ═══════════════════════════════════════════════════════════════════════
# helpers
# ═══════════════════════════════════════════════════════════════════════

def load_font(size=FONT_SIZE):
    for p in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    ]:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


def build_dino_transform():
    normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    def tf(img):
        w, h = img.size
        scale = 800 / min(w, h)
        if scale * max(w, h) > 1333:
            scale = 1333 / max(w, h)
        nw, nh = int(w * scale), int(h * scale)
        img = img.resize((nw, nh), Image.BILINEAR)
        return normalize(TF.to_tensor(img))
    return tf


def build_internvl_transform():
    return T.Compose([
        T.Resize((448, 448), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def cxcywh_norm_to_xyxy_abs(box, W, H):
    cx, cy, bw, bh = box
    return [
        (cx - bw / 2) * W,
        (cy - bh / 2) * H,
        (cx + bw / 2) * W,
        (cy + bh / 2) * H,
    ]


def iou_xyxy(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0., x2 - x1) * max(0., y2 - y1)
    aa = max(0., a[2]-a[0]) * max(0., a[3]-a[1])
    bb = max(0., b[2]-b[0]) * max(0., b[3]-b[1])
    return inter / (aa + bb - inter + 1e-9)


def draw_box(draw, box, colour, label, font):
    x1, y1, x2, y2 = [int(v) for v in box]
    for t in range(BOX_W):
        draw.rectangle([x1+t, y1+t, x2-t, y2-t], outline=colour)
    try:
        bb = font.getbbox(label)
        tw, th = bb[2]-bb[0], bb[3]-bb[1]
    except AttributeError:
        tw, th = font.getsize(label)
    pad = 4
    draw.rectangle([x1, y1-th-pad*2, x1+tw+pad*2, y1], fill=colour)
    draw.text((x1+pad, y1-th-pad), label, fill=(255,255,255), font=font)


def make_panel(img_pil, gt_box_abs, pred_box_abs, pred_iou,
               colour, title, font):
    W, H = img_pil.size
    out  = img_pil.copy()
    draw = ImageDraw.Draw(out)
    draw_box(draw, gt_box_abs,   COL_GT,  f"GT", font)
    draw_box(draw, pred_box_abs, colour,  f"{title}  IoU={pred_iou:.2f}", font)
    hdr_h = 34
    banner = Image.new("RGB", (W, hdr_h), colour)
    bd = ImageDraw.Draw(banner)
    bd.text((8, 7), title, fill=(255,255,255), font=font)
    canvas = Image.new("RGB", (W, H+hdr_h))
    canvas.paste(banner, (0,0))
    canvas.paste(out,    (0, hdr_h))
    return canvas


# ═══════════════════════════════════════════════════════════════════════
# model inference
# ═══════════════════════════════════════════════════════════════════════

def build_positive_map(tokenizer, special_tokens, query_text, max_len=512):
    tok = tokenizer(query_text, return_tensors="pt")
    ids = tok["input_ids"][0]
    pmap = torch.zeros(1, max_len)
    special_set = set(int(x) for x in special_tokens)
    for pos, tok_id in enumerate(ids.tolist()):
        if pos >= max_len: break
        if tok_id not in special_set:
            pmap[0, pos] = 1.0
    row_sum = pmap.sum(dim=-1, keepdim=True).clamp(min=1e-6)
    return pmap / row_sum


@torch.no_grad()
def run_thinkdet(model, image_pil, prompt, device, dino_tf, ivl_tf,
                 tokenizer, special_tokens, top_k=TOP_K):
    W, H = image_pil.size
    dino_t = dino_tf(image_pil).unsqueeze(0).to(device)
    mask   = torch.zeros(1, dino_t.shape[2], dino_t.shape[3],
                         dtype=torch.bool, device=device)
    nested = NestedTensor(dino_t, mask)
    ivl_t  = ivl_tf(image_pil).unsqueeze(0).to(device)
    query  = prompt.strip()
    if not query.endswith("."):
        query += " ."
    dino_inputs = {"samples": nested, "captions": [query]}

    out = model(ivl_t, [prompt], dino_inputs)
    outputs = out[0] if isinstance(out, tuple) else out

    pmap = build_positive_map(tokenizer, special_tokens, query).to(device)
    logits = outputs["pred_logits"][0].clamp(-50, 50)
    boxes  = outputs["pred_boxes"][0]
    scores = (logits.sigmoid() * pmap[:, :logits.shape[-1]]).sum(dim=-1)

    k = min(top_k, scores.shape[0])
    vals, idx = scores.topk(k)

    preds = []
    for j in range(k):
        cx, cy, bw, bh = boxes[idx[j]].tolist()
        preds.append({
            "score": float(vals[j]),
            "box_abs_xyxy": [
                (cx - bw/2)*W, (cy - bh/2)*H,
                (cx + bw/2)*W, (cy + bh/2)*H,
            ],
        })
    return preds


# ═══════════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    font   = load_font()

    # load failures
    with open(FAILURES_JSON) as f:
        data = json.load(f)
    cases = data["selected_cases"]

    print("=" * 65)
    print("  ThinkDet full pipeline  — baseline failure cases")
    print(f"  {len(cases)} cases  |  device: {device}")
    print("=" * 65)

    # ── load models ──────────────────────────────────────────────────
    print("\n[1/3] Loading baseline GroundingDINO ...")
    gd_base = load_gd_model(GD_CONFIG, GD_WEIGHTS, device="cpu").to(device).eval()
    tokenizer     = gd_base.tokenizer
    special_tokens = list(gd_base.specical_tokens)

    print("[2/3] Loading ThinkDet Stage-2 ...")
    ckpt = torch.load(STAGE2_CKPT, map_location="cpu")
    inj  = ckpt.get("injection_layers", DEFAULT_INJECTION_LAYERS)
    extl = ckpt.get("extract_layers") or [ckpt.get("extract_layer", 9)]
    tma_m = ckpt.get("tma_m", 8)
    tma_nh = ckpt.get("tma_n_heads", 8)
    fm   = ckpt.get("fusion_mode", "concat")

    gd_td = load_gd_model(GD_CONFIG, GD_WEIGHTS, device="cpu")
    model = ThinkDetModel(
        grounding_dino   = gd_td,
        internvl_path    = INTERNVL_PATH,
        extract_layer    = max(extl),
        extract_layers   = extl,
        injection_layers = inj,
        tma_m            = tma_m,
        tma_n_heads      = tma_nh,
        fusion_mode      = fm,
    )
    model.load_state_dict(ckpt["trainable_state_dict"], strict=False)
    model = model.to(device).eval()
    del ckpt

    print("[3/3] Loading InternVL evidence-check reranker ...")
    ivl_tf   = build_internvl_transform()
    reranker = InternVLYesNoReranker(
        internvl_model = model.feature_extractor.internvl,
        tokenizer      = model.feature_extractor.tokenizer,
        device         = device,
        image_transform= ivl_tf,
        weight         = LLM_WEIGHT,
        max_new_tokens = 48,
        temperature    = 0.0,
    )

    dino_tf = build_dino_transform()

    # ── run pipeline on each case ─────────────────────────────────────
    print(f"\nRunning {len(cases)} cases ...\n")
    for i, case in enumerate(cases):
        print(f"  Case {i+1}: [{case['affordance']}] {case['prompt']}")
        img_path = os.path.join(COCO_VAL, case["image_file"])
        if not os.path.exists(img_path):
            print(f"    SKIP — image not found: {img_path}")
            continue

        img = Image.open(img_path).convert("RGB")
        W, H = img.size
        gt_abs = cxcywh_norm_to_xyxy_abs(case["gt_box"], W, H)

        # ── BASELINE (from stored results — no re-inference needed) ──
        base_abs = cxcywh_norm_to_xyxy_abs(case["baseline"]["top_box"], W, H)
        base_iou = case["baseline"]["top_iou"]
        print(f"    baseline IoU = {base_iou:.3f}")

        # ── THINKDET forward pass ─────────────────────────────────────
        preds = run_thinkdet(model, img, case["prompt"], device,
                             dino_tf, ivl_tf, tokenizer, special_tokens)
        td_box  = preds[0]["box_abs_xyxy"]
        td_score = preds[0]["score"]
        td_iou  = iou_xyxy(td_box, gt_abs)
        print(f"    thinkdet  IoU = {td_iou:.3f}  score = {td_score:.3f}", end="")

        # ── CONFIDENCE CHECK → LLM FALLBACK? ─────────────────────────
        if td_score < CONF_THRESHOLD:
            print(f"  → score<{CONF_THRESHOLD}, triggering LLM reranker ...")
            preds_reranked = reranker.rerank_predictions(img, case["prompt"], preds)
            final_box   = preds_reranked[0]["box_abs_xyxy"]
            final_iou   = iou_xyxy(final_box, gt_abs)
            final_label = f"THINKDET+LLM  (reranked)"
            final_colour = COL_FINAL
        else:
            print(f"  → confident, keeping ThinkDet result")
            final_box   = td_box
            final_iou   = td_iou
            final_label = f"THINKDET  (confident)"
            final_colour = COL_THINK

        print(f"    final     IoU = {final_iou:.3f}")

        # ── VISUALISE — 3 panels ──────────────────────────────────────
        p_base  = make_panel(img, gt_abs, base_abs,  base_iou,
                             COL_BASE,  "BASELINE",     font)
        p_think = make_panel(img, gt_abs, td_box,    td_iou,
                             COL_THINK, "THINKDET",     font)
        p_final = make_panel(img, gt_abs, final_box, final_iou,
                             final_colour, final_label, font)

        gap = 6
        W3 = p_base.width + p_think.width + p_final.width + 2*gap
        H3 = max(p_base.height, p_think.height, p_final.height)
        canvas = Image.new("RGB", (W3, H3), (25, 25, 25))
        canvas.paste(p_base,  (0, 0))
        canvas.paste(p_think, (p_base.width + gap, 0))
        canvas.paste(p_final, (p_base.width + p_think.width + 2*gap, 0))

        # ── legend strip ─────────────────────────────────────────────
        leg_h = 28
        leg   = Image.new("RGB", (W3, leg_h), (40, 40, 40))
        ld    = ImageDraw.Draw(leg)
        items = [
            (COL_GT,    "■ Ground truth"),
            (COL_BASE,  "■ Baseline"),
            (COL_THINK, "■ ThinkDet"),
            (COL_FINAL, "■ Final (after fallback)"),
        ]
        x = 10
        for col, txt in items:
            ld.text((x, 6), txt, fill=col, font=font)
            try:
                bb = font.getbbox(txt)
                x += bb[2] - bb[0] + 30
            except Exception:
                x += 200

        full = Image.new("RGB", (W3, H3 + leg_h), (25, 25, 25))
        full.paste(canvas, (0, 0))
        full.paste(leg,    (0, H3))

        fname = f"case_{i+1:02d}_{case['affordance']}.png"
        out_p = os.path.join(OUT_DIR, fname)
        full.save(out_p)
        print(f"    → saved {out_p}\n")

    print(f"\nDone. Images in:\n  {OUT_DIR}")


if __name__ == "__main__":
    main()
