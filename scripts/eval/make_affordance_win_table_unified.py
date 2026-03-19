"""
Build an 8-case qualitative table where a unified ThinkDet checkpoint beats
the baseline on the held-out affordance benchmark.

Selection logic (default):
- one case per affordance when possible
- baseline top1 IoU < 0.5
- unified top1 IoU >= 0.5
- unified top1 IoU > baseline top1 IoU

Rendering:
- Baseline panel shows only baseline top-1 box.
- Unified panel shows only unified top-1 box (selected cases enforce it is correct).
- No extra boxes are drawn in the unified panel.
"""

import argparse
import csv
import datetime
import json
import os
import sys
from collections import defaultdict

import torch
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as T
import torchvision.transforms.functional as TF

ROOT = "/home/iibrohimm/project/next_step"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "GroundingDINO", "GroundingDINO"))

from thinkdet.models.arch import ThinkDetModel, DEFAULT_INJECTION_LAYERS
from groundingdino.util.inference import load_model as load_gd_model
from groundingdino.util.misc import NestedTensor


DEFAULT_BENCH = f"{ROOT}/thinkdet/data/benchmarks/affordance_coco_val_heldout_v1.json"
DEFAULT_BASELINE_EVAL = f"{ROOT}/thinkdet/results/eval/affordance_benchmark_eval_v1_test.json"
DEFAULT_UNIFIED_CKPT = (
    f"{ROOT}/thinkdet/checkpoints/unified/"
    "layer9_kd0p05_l1_1e-4_20260224_135540/"
    "thinkdet_unified_epoch5.pth"
)
GD_CONFIG = f"{ROOT}/GroundingDINO/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GD_WEIGHTS = f"{ROOT}/GroundingDINO/GroundingDINO/weights/groundingdino_swint_ogc.pth"
INTERNVL_PATH = f"{ROOT}/InternVL3_5-1B"
TS = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
DEFAULT_OUTDIR = f"{ROOT}/thinkdet/results/visual_compare/affordance_win_table_unified_{TS}"


try:
    FONT_BOLD = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    FONT_NORM = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    FONT_SMALL = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
except Exception:
    FONT_BOLD = FONT_NORM = FONT_SMALL = ImageFont.load_default()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", type=str, default=DEFAULT_BENCH)
    p.add_argument("--baseline_eval_json", type=str, default=DEFAULT_BASELINE_EVAL)
    p.add_argument("--split", type=str, default="test", choices=["dev", "test", "all"])
    p.add_argument("--num_cases", type=int, default=8)
    p.add_argument("--top_k_scan", type=int, default=1, help="Top-K used for unified scan metrics (default 1).")
    p.add_argument("--top_k_render", type=int, default=5, help="Top-K to fetch for rendering/debug (only top-1 drawn).")
    p.add_argument("--min_unified_top1_iou", type=float, default=0.5)
    p.add_argument("--max_baseline_top1_iou", type=float, default=0.49)
    p.add_argument("--min_gap", type=float, default=0.05)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--gd_config", type=str, default=GD_CONFIG)
    p.add_argument("--gd_weights", type=str, default=GD_WEIGHTS)
    p.add_argument("--internvl_path", type=str, default=INTERNVL_PATH)
    p.add_argument("--unified_ckpt", type=str, default=DEFAULT_UNIFIED_CKPT)
    p.add_argument("--output_dir", type=str, default=DEFAULT_OUTDIR)
    p.add_argument("--max_rows_per_table", type=int, default=8)
    return p.parse_args()


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
    return T.Compose(
        [
            T.Resize((448, 448), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def iou_xyxy(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (aa + bb - inter + 1e-9)


def xywh_to_xyxy(box):
    x, y, w, h = box
    return [x, y, x + w, y + h]


def get_special_tokens(model):
    tokens = getattr(model, "specical_tokens", None)
    if tokens is not None:
        return tokens
    tokens = getattr(model, "special_tokens", None)
    if tokens is not None:
        return tokens
    return model.tokenizer.all_special_ids


def build_positive_map_for_query(tokenizer, special_tokens, query_text, max_text_len=512):
    tokenized = tokenizer(query_text, return_tensors="pt")
    input_ids = tokenized["input_ids"][0]

    positive_map = torch.zeros(1, max_text_len, dtype=torch.float32)
    special_set = set(int(x) for x in special_tokens)

    for pos, tok_id in enumerate(input_ids.tolist()):
        if pos >= max_text_len:
            break
        if tok_id not in special_set:
            positive_map[0, pos] = 1.0

    row_sum = positive_map.sum(dim=-1, keepdim=True).clamp(min=1e-6)
    return positive_map / row_sum


def score_outputs_for_query(outputs, positive_map_norm):
    logits = outputs["pred_logits"][0].clamp(-50, 50)
    boxes = outputs["pred_boxes"][0]
    text_len = logits.shape[-1]
    pmap = positive_map_norm[:, :text_len].to(logits.device)
    probs = logits.sigmoid()
    scores = (probs * pmap).sum(dim=-1)
    return scores, boxes


def resolve_query_scoring_assets(model_fn):
    if hasattr(model_fn, "grounding_dino"):
        dino_model = model_fn.grounding_dino
    else:
        dino_model = model_fn
    if not hasattr(dino_model, "tokenizer"):
        raise AttributeError("Could not resolve GroundingDINO tokenizer for query scoring")
    return dino_model.tokenizer, get_special_tokens(dino_model)


@torch.no_grad()
def run_topk(
    model_fn,
    image_pil,
    prompt,
    device,
    dino_tf,
    ivl_tf,
    top_k=5,
    tokenizer=None,
    special_tokens=None,
):
    img_w, img_h = image_pil.size

    dino_t = dino_tf(image_pil).unsqueeze(0).to(device)
    mask = torch.zeros(1, dino_t.shape[2], dino_t.shape[3], dtype=torch.bool, device=device)
    nested = NestedTensor(dino_t, mask)
    ivl_t = ivl_tf(image_pil).unsqueeze(0).to(device)

    # GroundingDINO parser is safest with trailing delimiter.
    dino_prompt = prompt if prompt.strip().endswith(".") else (prompt.strip() + " .")
    out = model_fn(ivl_t, [prompt], {"samples": nested, "captions": [dino_prompt]})
    outputs = out[0] if isinstance(out, tuple) else out

    if tokenizer is None or special_tokens is None:
        tokenizer, special_tokens = resolve_query_scoring_assets(model_fn)
    positive_map_norm = build_positive_map_for_query(
        tokenizer, special_tokens, dino_prompt, max_text_len=512
    )
    scores, boxes = score_outputs_for_query(outputs, positive_map_norm)
    vals, idx = scores.topk(min(top_k, scores.shape[0]))

    results = []
    for j in range(len(vals)):
        cx, cy, bw, bh = boxes[idx[j]].tolist()
        x1 = (cx - bw / 2) * img_w
        y1 = (cy - bh / 2) * img_h
        x2 = (cx + bw / 2) * img_w
        y2 = (cy + bh / 2) * img_h
        results.append(
            {
                "score": float(vals[j].item()),
                "box_abs_xyxy": [x1, y1, x2, y2],
            }
        )
    return results


def best_iou_for_pred(pred_box, gt_boxes):
    if pred_box is None or not gt_boxes:
        return 0.0
    return max(iou_xyxy(pred_box, g) for g in gt_boxes)


def load_benchmark_samples(path, split):
    with open(path, "r") as f:
        bench = json.load(f)
    samples = bench["samples"]
    if split != "all":
        samples = [s for s in samples if s.get("split") == split]
    return bench, samples


def load_baseline_per_sample(eval_json_path):
    with open(eval_json_path, "r") as f:
        payload = json.load(f)
    res = payload["results"]["baseline"]["per_sample"]
    return {r["benchmark_id"]: r for r in res}, payload


def load_unified_model(args, device):
    ckpt = torch.load(args.unified_ckpt, map_location="cpu")
    inject_layers = ckpt.get("injection_layers", DEFAULT_INJECTION_LAYERS)
    tma_m = ckpt.get("tma_m", 8)
    tma_nheads = ckpt.get("tma_n_heads", 8)
    ext_layers = ckpt.get("extract_layers") or [ckpt.get("extract_layer", 9)]
    fusion_mode = ckpt.get("fusion_mode", "concat")

    gd = load_gd_model(args.gd_config, args.gd_weights, device="cpu")
    model = ThinkDetModel(
        grounding_dino=gd,
        internvl_path=args.internvl_path,
        extract_layer=max(ext_layers),
        extract_layers=ext_layers,
        injection_layers=inject_layers,
        tma_m=tma_m,
        tma_n_heads=tma_nheads,
        fusion_mode=fusion_mode,
    )
    model.load_state_dict(ckpt["trainable_state_dict"], strict=False)
    model = model.to(device).eval()
    return model, {
        "extract_layers": ext_layers,
        "injection_layers": inject_layers,
        "tma_m": tma_m,
        "tma_n_heads": tma_nheads,
        "fusion_mode": fusion_mode,
    }


def load_baseline_model(args, device):
    gd = load_gd_model(args.gd_config, args.gd_weights, device="cpu").to(device).eval()

    def baseline_fn(ivl_imgs, queries, dino_inputs):
        return gd(**dino_inputs), {}

    baseline_fn.tokenizer = gd.tokenizer
    baseline_fn.specical_tokens = get_special_tokens(gd)

    return gd, baseline_fn


def scan_unified(samples, args, device, dino_tf, ivl_tf, out_json):
    print("\n[scan] Loading unified model...")
    model, ckpt_meta = load_unified_model(args, device)
    tokenizer, special_tokens = resolve_query_scoring_assets(model)
    out = {}
    for i, s in enumerate(samples, 1):
        image_pil = Image.open(s["image_path"]).convert("RGB")
        preds = run_topk(
            model,
            image_pil,
            s["prompt"],
            device,
            dino_tf,
            ivl_tf,
            top_k=args.top_k_scan,
            tokenizer=tokenizer,
            special_tokens=special_tokens,
        )
        gt_boxes = [xywh_to_xyxy(t["bbox_xywh"]) for t in s["positive_targets"]]
        top1_box = preds[0]["box_abs_xyxy"] if preds else None
        top1_score = float(preds[0]["score"]) if preds else 0.0
        top1_iou = best_iou_for_pred(top1_box, gt_boxes)
        out[s["benchmark_id"]] = {
            "benchmark_id": s["benchmark_id"],
            "image_id": int(s["image_id"]),
            "affordance_id": s["affordance_id"],
            "prompt": s["prompt"],
            "top1_iou": float(top1_iou),
            "top1_score": top1_score,
            "top1_box_abs_xyxy": top1_box,
        }
        if i % 50 == 0 or i == len(samples):
            print(f"  [scan] {i}/{len(samples)}")
    del model
    torch.cuda.empty_cache()

    payload = {
        "status": "ok",
        "timestamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
        "benchmark": args.benchmark,
        "split": args.split,
        "checkpoint": args.unified_ckpt,
        "ckpt_meta": ckpt_meta,
        "n_samples": len(samples),
        "results": list(out.values()),
    }
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[scan] Saved unified scan: {out_json}")
    return out


def select_cases(samples, baseline_map, unified_map, args):
    rows = []
    for s in samples:
        bid = s["benchmark_id"]
        b = baseline_map.get(bid)
        u = unified_map.get(bid)
        if not b or not u:
            continue
        b_iou = float(b.get("top1_iou", 0.0))
        u_iou = float(u.get("top1_iou", 0.0))
        gap = u_iou - b_iou
        rows.append(
            {
                "sample": s,
                "benchmark_id": bid,
                "image_id": int(s["image_id"]),
                "affordance_id": s["affordance_id"],
                "prompt": s["prompt"],
                "baseline_top1_iou": b_iou,
                "baseline_top1_score": float(b.get("top1_score", 0.0)),
                "unified_top1_iou": u_iou,
                "unified_top1_score": float(u.get("top1_score", 0.0)),
                "iou_gap": gap,
            }
        )

    strict = [
        r
        for r in rows
        if r["unified_top1_iou"] >= args.min_unified_top1_iou
        and r["baseline_top1_iou"] <= args.max_baseline_top1_iou
        and r["iou_gap"] >= args.min_gap
    ]

    strict.sort(key=lambda r: (-r["iou_gap"], -r["unified_top1_iou"], r["image_id"]))
    by_aff = defaultdict(list)
    for r in strict:
        by_aff[r["affordance_id"]].append(r)

    selected = []
    used_bids = set()

    # Prefer one case per affordance.
    for aff in sorted(by_aff.keys()):
        if len(selected) >= args.num_cases:
            break
        cand = by_aff[aff][0]
        selected.append(cand)
        used_bids.add(cand["benchmark_id"])

    # Backfill strongest remaining wins if we still need more.
    if len(selected) < args.num_cases:
        for r in strict:
            if r["benchmark_id"] in used_bids:
                continue
            selected.append(r)
            used_bids.add(r["benchmark_id"])
            if len(selected) >= args.num_cases:
                break

    # Last-resort relaxation if strict count is too low.
    if len(selected) < args.num_cases:
        relaxed = [
            r for r in rows
            if r["unified_top1_iou"] >= args.min_unified_top1_iou
            and r["iou_gap"] > 0
            and r["benchmark_id"] not in used_bids
        ]
        relaxed.sort(key=lambda r: (-r["iou_gap"], -r["unified_top1_iou"], r["image_id"]))
        for r in relaxed:
            selected.append(r)
            used_bids.add(r["benchmark_id"])
            if len(selected) >= args.num_cases:
                break

    selected.sort(key=lambda r: (r["affordance_id"], -r["iou_gap"], r["image_id"]))
    return selected, {
        "n_rows": len(rows),
        "n_strict_candidates": len(strict),
        "n_selected": len(selected),
        "strict_thresholds": {
            "min_unified_top1_iou": args.min_unified_top1_iou,
            "max_baseline_top1_iou": args.max_baseline_top1_iou,
            "min_gap": args.min_gap,
        },
    }


def _draw_single_box_panel(image_pil, box_xyxy, score, title, color):
    img = image_pil.copy().convert("RGB")
    draw = ImageDraw.Draw(img)

    if box_xyxy is not None:
        x1, y1, x2, y2 = box_xyxy
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img.width - 1, x2)
        y2 = min(img.height - 1, y2)
        for off in range(3):
            draw.rectangle([x1 - off, y1 - off, x2 + off, y2 + off], outline=color)
        label = f"{score:.3f}"
        tw = 8 * len(label) + 8
        draw.rectangle([x1, max(0, y1 - 18), x1 + tw, y1], fill=color)
        draw.text((x1 + 4, max(0, y1 - 16)), label, fill=(255, 255, 255), font=FONT_SMALL)

    bar_h = 28
    panel = Image.new("RGB", (img.width, img.height + bar_h), (44, 44, 44))
    panel.paste(img, (0, bar_h))
    d = ImageDraw.Draw(panel)
    d.text((8, 6), title, fill=(255, 255, 255), font=FONT_BOLD)
    return panel


def _scale_to_height(img, target_h):
    if img.height == target_h:
        return img
    scale = target_h / float(img.height)
    return img.resize((max(1, int(img.width * scale)), target_h), Image.BILINEAR)


def _compose_case_row(case_idx, record, image_pil, base_pred, uni_pred):
    panel_base = _draw_single_box_panel(
        image_pil,
        base_pred["box_abs_xyxy"] if base_pred else None,
        (base_pred or {}).get("score", 0.0),
        f"BASELINE (top1 IoU={record['baseline_top1_iou']:.3f})",
        (220, 60, 60),
    )
    panel_uni = _draw_single_box_panel(
        image_pil,
        uni_pred["box_abs_xyxy"] if uni_pred else None,
        (uni_pred or {}).get("score", 0.0),
        f"UNIFIED (top1 IoU={record['unified_top1_iou']:.3f})",
        (20, 220, 60),
    )

    target_h = 260
    panel_base = _scale_to_height(panel_base, target_h)
    panel_uni = _scale_to_height(panel_uni, target_h)

    gap = 6
    header_h = 52
    total_w = panel_base.width + panel_uni.width + gap
    row = Image.new("RGB", (total_w, header_h + target_h), (20, 20, 20))
    d = ImageDraw.Draw(row)

    prompt = record["prompt"]
    aff = record["affordance_id"]
    img_id = record["image_id"]
    d.text(
        (8, 5),
        f"#{case_idx:02d}  {aff}  img={img_id}  gap={record['iou_gap']:.3f}",
        fill=(255, 255, 255),
        font=FONT_BOLD,
    )
    d.text((8, 28), f'Prompt: "{prompt}"', fill=(120, 255, 140), font=FONT_NORM)

    row.paste(panel_base, (0, header_h))
    row.paste(panel_uni, (panel_base.width + gap, header_h))
    return row


def _compose_table(rows, title, subtitle):
    if not rows:
        return Image.new("RGB", (600, 160), (20, 20, 20))
    gap = 8
    header_h = 70
    total_w = max(r.width for r in rows)
    total_h = header_h + sum(r.height for r in rows) + gap * (len(rows) - 1)
    canvas = Image.new("RGB", (total_w, total_h), (16, 16, 16))
    d = ImageDraw.Draw(canvas)
    d.text((10, 8), title, fill=(255, 255, 255), font=FONT_BOLD)
    d.text((10, 32), subtitle, fill=(180, 180, 180), font=FONT_NORM)
    d.text((10, 50), "Unified panel shows only one box (the selected correct top-1 detection).", fill=(130, 255, 150), font=FONT_SMALL)
    y = header_h
    for r in rows:
        canvas.paste(r, (0, y))
        y += r.height + gap
    return canvas


def render_and_save(selected, args, device, dino_tf, ivl_tf, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    case_dir = os.path.join(out_dir, "cases")
    os.makedirs(case_dir, exist_ok=True)

    selected_samples = [r["sample"] for r in selected]
    by_bid = {r["benchmark_id"]: r for r in selected}
    render_preds = {bid: {} for bid in by_bid}

    print("\n[render] Baseline predictions for selected cases...")
    gd_base, baseline_fn = load_baseline_model(args, device)
    baseline_tokenizer, baseline_special_tokens = resolve_query_scoring_assets(baseline_fn)
    for i, s in enumerate(selected_samples, 1):
        image_pil = Image.open(s["image_path"]).convert("RGB")
        preds = run_topk(
            baseline_fn,
            image_pil,
            s["prompt"],
            device,
            dino_tf,
            ivl_tf,
            top_k=args.top_k_render,
            tokenizer=baseline_tokenizer,
            special_tokens=baseline_special_tokens,
        )
        render_preds[s["benchmark_id"]]["baseline"] = preds
        print(f"  [baseline] {i}/{len(selected_samples)} img={s['image_id']}")
    del baseline_fn, gd_base
    torch.cuda.empty_cache()

    print("\n[render] Unified predictions for selected cases...")
    unified_model, ckpt_meta = load_unified_model(args, device)
    unified_tokenizer, unified_special_tokens = resolve_query_scoring_assets(unified_model)
    for i, s in enumerate(selected_samples, 1):
        image_pil = Image.open(s["image_path"]).convert("RGB")
        preds = run_topk(
            unified_model,
            image_pil,
            s["prompt"],
            device,
            dino_tf,
            ivl_tf,
            top_k=args.top_k_render,
            tokenizer=unified_tokenizer,
            special_tokens=unified_special_tokens,
        )
        render_preds[s["benchmark_id"]]["unified"] = preds
        print(f"  [unified] {i}/{len(selected_samples)} img={s['image_id']}")
    del unified_model
    torch.cuda.empty_cache()

    rows = []
    case_summaries = []
    for idx, r in enumerate(selected, 1):
        s = r["sample"]
        image_pil = Image.open(s["image_path"]).convert("RGB")
        b_preds = render_preds[r["benchmark_id"]]["baseline"]
        u_preds = render_preds[r["benchmark_id"]]["unified"]
        b_top1 = b_preds[0] if b_preds else None
        u_top1 = u_preds[0] if u_preds else None

        row_img = _compose_case_row(idx, r, image_pil, b_top1, u_top1)
        case_file = f"case{idx:02d}_{s['affordance_id']}_{int(s['image_id'])}.jpg"
        case_path = os.path.join(case_dir, case_file)
        row_img.save(case_path, quality=95)
        rows.append(row_img)

        case_summaries.append(
            {
                "rank": idx,
                "benchmark_id": r["benchmark_id"],
                "image_id": int(s["image_id"]),
                "image_file": s["image_file"],
                "affordance_id": s["affordance_id"],
                "prompt": s["prompt"],
                "target_categories": s["target_categories"],
                "baseline_top1_iou": r["baseline_top1_iou"],
                "baseline_top1_score": r["baseline_top1_score"],
                "baseline_top1_box_abs_xyxy": (b_top1 or {}).get("box_abs_xyxy"),
                "unified_top1_iou": r["unified_top1_iou"],
                "unified_top1_score": r["unified_top1_score"],
                "unified_top1_box_abs_xyxy": (u_top1 or {}).get("box_abs_xyxy"),
                "iou_gap": r["iou_gap"],
                "case_image": os.path.relpath(case_path, out_dir),
            }
        )

    subtitle = (
        f"split={args.split} | num_cases={len(rows)} | "
        f"thresholds: unified_iou>={args.min_unified_top1_iou}, "
        f"baseline_iou<={args.max_baseline_top1_iou}, gap>={args.min_gap}"
    )
    table_img = _compose_table(
        rows,
        title="Affordance Qualitative Wins: Baseline vs Unified (epoch5)",
        subtitle=subtitle,
    )
    table_path = os.path.join(out_dir, "affordance_win_table_unified_epoch5.jpg")
    table_img.save(table_path, quality=95)

    return case_summaries, table_path, ckpt_meta


def save_tables(out_dir, selected_meta, case_summaries, args, bench, baseline_eval_payload, unified_scan_path, table_path, ckpt_meta):
    summary_json = os.path.join(out_dir, "summary.json")
    summary_md = os.path.join(out_dir, "summary.md")
    summary_csv = os.path.join(out_dir, "summary.csv")

    payload = {
        "status": "ok",
        "timestamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
        "benchmark": args.benchmark,
        "benchmark_name": bench.get("benchmark_name"),
        "split": args.split,
        "baseline_eval_json": args.baseline_eval_json,
        "baseline_eval_timestamp": baseline_eval_payload.get("timestamp"),
        "unified_ckpt": args.unified_ckpt,
        "unified_scan_json": unified_scan_path,
        "output_table_image": table_path,
        "selection_meta": selected_meta,
        "ckpt_meta": ckpt_meta,
        "cases": case_summaries,
    }
    with open(summary_json, "w") as f:
        json.dump(payload, f, indent=2)

    with open(summary_md, "w") as f:
        f.write("# Affordance Qualitative Wins (Baseline vs Unified)\n\n")
        f.write(f"- benchmark: `{args.benchmark}`\n")
        f.write(f"- split: `{args.split}`\n")
        f.write(f"- baseline_eval_json: `{args.baseline_eval_json}`\n")
        f.write(f"- unified_ckpt: `{args.unified_ckpt}`\n")
        f.write(f"- table_image: `{table_path}`\n")
        f.write(f"- selection: `{selected_meta}`\n\n")
        f.write("| rank | affordance | image_id | prompt | baseline IoU@top1 | unified IoU@top1 | gap | case_image |\n")
        f.write("|---:|---|---:|---|---:|---:|---:|---|\n")
        for c in case_summaries:
            f.write(
                f"| {c['rank']} | {c['affordance_id']} | {c['image_id']} | "
                f"{c['prompt']} | {c['baseline_top1_iou']:.3f} | {c['unified_top1_iou']:.3f} | "
                f"{c['iou_gap']:.3f} | `{c['case_image']}` |\n"
            )

    with open(summary_csv, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(
            [
                "rank",
                "benchmark_id",
                "affordance_id",
                "image_id",
                "prompt",
                "baseline_top1_iou",
                "unified_top1_iou",
                "iou_gap",
                "baseline_top1_score",
                "unified_top1_score",
                "case_image",
            ]
        )
        for c in case_summaries:
            wr.writerow(
                [
                    c["rank"],
                    c["benchmark_id"],
                    c["affordance_id"],
                    c["image_id"],
                    c["prompt"],
                    f"{c['baseline_top1_iou']:.6f}",
                    f"{c['unified_top1_iou']:.6f}",
                    f"{c['iou_gap']:.6f}",
                    f"{c['baseline_top1_score']:.6f}",
                    f"{c['unified_top1_score']:.6f}",
                    c["case_image"],
                ]
            )

    return summary_json, summary_md, summary_csv


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("=" * 72)
    print("Affordance win table: baseline vs unified")
    print(f"benchmark: {args.benchmark}")
    print(f"baseline eval: {args.baseline_eval_json}")
    print(f"unified ckpt: {args.unified_ckpt}")
    print(f"device: {device}")
    print(f"output_dir: {args.output_dir}")
    print("=" * 72)

    bench, samples = load_benchmark_samples(args.benchmark, args.split)
    if not samples:
        raise RuntimeError(f"No samples found for split={args.split}")
    baseline_map, baseline_eval_payload = load_baseline_per_sample(args.baseline_eval_json)

    dino_tf = build_dino_transform()
    ivl_tf = build_internvl_transform()

    unified_scan_json = os.path.join(args.output_dir, "unified_scan_top1.json")
    unified_map = scan_unified(samples, args, device, dino_tf, ivl_tf, unified_scan_json)

    selected, selected_meta = select_cases(samples, baseline_map, unified_map, args)
    print(f"\n[select] strict candidates={selected_meta['n_strict_candidates']} selected={len(selected)}")
    if not selected:
        raise RuntimeError("No qualifying cases found with current thresholds.")

    case_summaries, table_path, ckpt_meta = render_and_save(selected, args, device, dino_tf, ivl_tf, args.output_dir)
    summary_json, summary_md, summary_csv = save_tables(
        args.output_dir,
        selected_meta,
        case_summaries,
        args,
        bench,
        baseline_eval_payload,
        unified_scan_json,
        table_path,
        ckpt_meta,
    )

    print("\n" + "=" * 72)
    print("Done")
    print(f"table image: {table_path}")
    print(f"summary json: {summary_json}")
    print(f"summary md:   {summary_md}")
    print(f"summary csv:  {summary_csv}")
    print("=" * 72)


if __name__ == "__main__":
    main()
