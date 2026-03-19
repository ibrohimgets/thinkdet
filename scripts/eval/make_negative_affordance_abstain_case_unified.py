"""
Find and render a negative affordance case where:
- target object for the prompt is absent in the image (per COCO annotations)
- baseline emits a detection above threshold
- unified model stays below threshold ("no detection")

Default negative query:
  talk_on -> "something to talk on ." (target category: cell phone)
"""

import argparse
import csv
import datetime
import json
import os
import sys

from PIL import Image, ImageDraw, ImageFont
import torch

ROOT = "/home/iibrohimm/project/next_step"
sys.path.insert(0, ROOT)

from thinkdet.scripts.eval.build_affordance_benchmark import AFFORDANCES
from thinkdet.scripts.eval.make_affordance_win_table_unified import (
    build_dino_transform,
    build_internvl_transform,
    load_baseline_model,
    load_unified_model,
    load_benchmark_samples,
    run_topk,
)


DEFAULT_BENCH = f"{ROOT}/thinkdet/data/benchmarks/affordance_coco_val_heldout_v1.json"
DEFAULT_UNIFIED_CKPT = (
    f"{ROOT}/thinkdet/checkpoints/unified/"
    "layer9_kd0p05_l1_1e-4_20260224_135540/"
    "thinkdet_unified_epoch5.pth"
)
TS = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
DEFAULT_OUTDIR = f"{ROOT}/thinkdet/results/visual_compare/negative_affordance_abstain_unified_{TS}"


try:
    FONT_BOLD = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    FONT_NORM = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    FONT_SMALL = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
except Exception:
    FONT_BOLD = FONT_NORM = FONT_SMALL = ImageFont.load_default()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", type=str, default=DEFAULT_BENCH)
    p.add_argument("--split", type=str, default="test", choices=["dev", "test", "all"])
    p.add_argument("--affordance_id", type=str, default="talk_on")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--gd_config", type=str, default=f"{ROOT}/GroundingDINO/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    p.add_argument("--gd_weights", type=str, default=f"{ROOT}/GroundingDINO/GroundingDINO/weights/groundingdino_swint_ogc.pth")
    p.add_argument("--internvl_path", type=str, default=f"{ROOT}/InternVL3_5-1B")
    p.add_argument("--unified_ckpt", type=str, default=DEFAULT_UNIFIED_CKPT)
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--score_thresholds", type=float, nargs="*", default=[0.25, 0.22, 0.20, 0.18, 0.15, 0.12])
    p.add_argument("--min_score_gap", type=float, default=0.05)
    p.add_argument("--output_dir", type=str, default=DEFAULT_OUTDIR)
    p.add_argument("--max_candidates", type=int, default=0, help="0=all")
    return p.parse_args()


def get_affordance_cfg(affordance_id):
    for a in AFFORDANCES:
        if a["id"] == affordance_id:
            return a
    raise KeyError(f"Unknown affordance_id={affordance_id}")


def target_absent(sample, target_categories):
    hist = sample.get("category_histogram") or {}
    return all(hist.get(cat, 0) == 0 for cat in target_categories)


def _draw_panel(image_pil, title, pred=None, threshold=None, no_det_text="NO DETECTION"):
    img = image_pil.copy().convert("RGB")
    draw = ImageDraw.Draw(img)

    if pred is not None:
        x1, y1, x2, y2 = pred["box_abs_xyxy"]
        col = (230, 60, 60) if "BASELINE" in title else (20, 220, 60)
        for off in range(3):
            draw.rectangle([x1 - off, y1 - off, x2 + off, y2 + off], outline=col)
        label = f"{pred['score']:.3f}"
        tw = 8 * len(label) + 8
        draw.rectangle([x1, max(0, y1 - 18), x1 + tw, y1], fill=col)
        draw.text((x1 + 4, max(0, y1 - 16)), label, fill=(255, 255, 255), font=FONT_SMALL)
    else:
        # Explicit abstention banner.
        banner = f"{no_det_text}" + (f"  (< {threshold:.2f})" if threshold is not None else "")
        draw.rounded_rectangle([10, 10, min(img.width - 10, 230), 38], radius=6, fill=(40, 120, 40))
        draw.text((16, 18), banner, fill=(255, 255, 255), font=FONT_SMALL)

    bar_h = 28
    panel = Image.new("RGB", (img.width, img.height + bar_h), (44, 44, 44))
    panel.paste(img, (0, bar_h))
    d = ImageDraw.Draw(panel)
    d.text((8, 6), title, fill=(255, 255, 255), font=FONT_BOLD)
    return panel


def _compose_two_panel(image_pil, record, threshold):
    b = record["baseline_top1"]
    u = record["unified_top1"]
    b_draw = b if b["score"] >= threshold else None
    u_draw = u if u["score"] >= threshold else None

    pb = _draw_panel(
        image_pil,
        title=f"BASELINE ({'detect' if b_draw else 'none'} @ {threshold:.2f})",
        pred=b_draw,
        threshold=threshold,
    )
    pu = _draw_panel(
        image_pil,
        title=f"UNIFIED ({'detect' if u_draw else 'none'} @ {threshold:.2f})",
        pred=u_draw,
        threshold=threshold,
    )

    target_h = 320
    if pb.height != target_h:
        s = target_h / pb.height
        pb = pb.resize((int(pb.width * s), target_h), Image.BILINEAR)
    if pu.height != target_h:
        s = target_h / pu.height
        pu = pu.resize((int(pu.width * s), target_h), Image.BILINEAR)

    gap = 8
    header_h = 78
    total_w = pb.width + pu.width + gap
    canvas = Image.new("RGB", (total_w, target_h + header_h), (18, 18, 18))
    d = ImageDraw.Draw(canvas)
    d.text(
        (10, 8),
        f'Negative Affordance Case ({record["affordance_id"]})  img={record["image_id"]}',
        fill=(255, 255, 255),
        font=FONT_BOLD,
    )
    d.text((10, 30), f'Prompt: "{record["prompt"]}"', fill=(120, 255, 140), font=FONT_NORM)
    d.text(
        (10, 49),
        "Target absent by COCO annotations for this prompt's categories.",
        fill=(200, 200, 200),
        font=FONT_SMALL,
    )
    d.text(
        (10, 63),
        f"Threshold={threshold:.2f} | baseline={b['score']:.3f} | unified={u['score']:.3f}",
        fill=(255, 210, 120),
        font=FONT_SMALL,
    )
    canvas.paste(pb, (0, header_h))
    canvas.paste(pu, (pb.width + gap, header_h))
    return canvas


def select_case(rows, thresholds, min_score_gap):
    thresholds = sorted(set(float(t) for t in thresholds), reverse=True)
    rows_sorted = sorted(
        rows,
        key=lambda r: (-(r["baseline_top1"]["score"] - r["unified_top1"]["score"]), -r["baseline_top1"]["score"], r["image_id"]),
    )
    for th in thresholds:
        for r in rows_sorted:
            b = r["baseline_top1"]["score"]
            u = r["unified_top1"]["score"]
            if b >= th and u < th and (b - u) >= min_score_gap:
                return r, th, "threshold_match"
    # Fallback: strongest gap (report that threshold criterion could not be met).
    if rows_sorted:
        return rows_sorted[0], None, "best_gap_fallback"
    return None, None, "none"


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    aff = get_affordance_cfg(args.affordance_id)
    prompt = aff["prompt"]
    target_categories = aff["target_categories"]

    print("=" * 72)
    print("Negative affordance abstention case mining (baseline vs unified)")
    print(f"benchmark: {args.benchmark}")
    print(f"split: {args.split}")
    print(f"affordance: {args.affordance_id}  prompt={prompt}")
    print(f"target_categories: {target_categories}")
    print(f"unified_ckpt: {args.unified_ckpt}")
    print(f"device: {device}")
    print(f"output_dir: {args.output_dir}")
    print("=" * 72)

    bench, samples = load_benchmark_samples(args.benchmark, args.split)
    candidates = [s for s in samples if target_absent(s, target_categories)]
    if args.max_candidates and args.max_candidates > 0:
        candidates = candidates[: args.max_candidates]
    if not candidates:
        raise RuntimeError("No candidate images with target absent.")

    dino_tf = build_dino_transform()
    ivl_tf = build_internvl_transform()

    rows = []

    print(f"\n[baseline] scanning {len(candidates)} target-absent images...")
    gd_base, baseline_fn = load_baseline_model(args, device)
    for i, s in enumerate(candidates, 1):
        image_pil = Image.open(s["image_path"]).convert("RGB")
        preds = run_topk(baseline_fn, image_pil, prompt, device, dino_tf, ivl_tf, top_k=args.top_k)
        rows.append(
            {
                "benchmark_id": s["benchmark_id"],
                "image_id": int(s["image_id"]),
                "image_file": s["image_file"],
                "image_path": s["image_path"],
                "source_affordance_id": s["affordance_id"],
                "prompt": prompt,
                "affordance_id": args.affordance_id,
                "target_categories": target_categories,
                "target_absent_by_coco": True,
                "category_histogram": s.get("category_histogram", {}),
                "baseline_top1": preds[0] if preds else {"score": 0.0, "box_abs_xyxy": None},
            }
        )
        if i % 50 == 0 or i == len(candidates):
            print(f"  [baseline] {i}/{len(candidates)}")
    del baseline_fn, gd_base
    torch.cuda.empty_cache()

    print(f"\n[unified] scanning {len(candidates)} target-absent images...")
    unified_model, ckpt_meta = load_unified_model(args, device)
    for i, r in enumerate(rows, 1):
        image_pil = Image.open(r["image_path"]).convert("RGB")
        preds = run_topk(unified_model, image_pil, prompt, device, dino_tf, ivl_tf, top_k=args.top_k)
        r["unified_top1"] = preds[0] if preds else {"score": 0.0, "box_abs_xyxy": None}
        if i % 50 == 0 or i == len(rows):
            print(f"  [unified] {i}/{len(rows)}")
    del unified_model
    torch.cuda.empty_cache()

    selected, threshold, selection_mode = select_case(rows, args.score_thresholds, args.min_score_gap)
    if selected is None:
        raise RuntimeError("No negative cases found.")

    if threshold is None:
        # Pick a visualization threshold between the two scores if possible.
        b = selected["baseline_top1"]["score"]
        u = selected["unified_top1"]["score"]
        threshold = round((b + u) / 2.0, 3) if b > u else args.score_thresholds[-1]

    image_pil = Image.open(selected["image_path"]).convert("RGB")
    out_img = _compose_two_panel(image_pil, selected, threshold)
    img_name = f"negative_{args.affordance_id}_img{selected['image_id']}.jpg"
    out_img_path = os.path.join(args.output_dir, img_name)
    out_img.save(out_img_path, quality=95)

    # Save all candidate scores for transparency / later selection.
    rows_sorted = sorted(
        rows,
        key=lambda r: (-(r["baseline_top1"]["score"] - r["unified_top1"]["score"]), -r["baseline_top1"]["score"], r["image_id"]),
    )

    csv_path = os.path.join(args.output_dir, "candidates_scores.csv")
    with open(csv_path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow([
            "image_id",
            "image_file",
            "source_affordance_id",
            "prompt_affordance_id",
            "prompt",
            "baseline_top1_score",
            "unified_top1_score",
            "score_gap_baseline_minus_unified",
        ])
        for r in rows_sorted:
            wr.writerow([
                r["image_id"],
                r["image_file"],
                r["source_affordance_id"],
                r["affordance_id"],
                r["prompt"],
                f"{r['baseline_top1']['score']:.6f}",
                f"{r['unified_top1']['score']:.6f}",
                f"{(r['baseline_top1']['score'] - r['unified_top1']['score']):.6f}",
            ])

    summary = {
        "status": "ok",
        "timestamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
        "benchmark": args.benchmark,
        "benchmark_name": bench.get("benchmark_name"),
        "split": args.split,
        "device": str(device),
        "affordance_id": args.affordance_id,
        "prompt": prompt,
        "target_categories": target_categories,
        "target_absence_definition": "all target_categories absent from sample.category_histogram (COCO annotations)",
        "num_candidates_scanned": len(candidates),
        "score_thresholds_tried": args.score_thresholds,
        "min_score_gap": args.min_score_gap,
        "selection_mode": selection_mode,
        "selected_threshold": threshold,
        "unified_ckpt": args.unified_ckpt,
        "ckpt_meta": ckpt_meta,
        "selected_case": {
            "image_id": selected["image_id"],
            "image_file": selected["image_file"],
            "image_path": selected["image_path"],
            "source_affordance_id": selected["source_affordance_id"],
            "prompt_affordance_id": selected["affordance_id"],
            "prompt": selected["prompt"],
            "target_categories": selected["target_categories"],
            "category_histogram": selected["category_histogram"],
            "baseline_top1": selected["baseline_top1"],
            "unified_top1": selected["unified_top1"],
            "score_gap_baseline_minus_unified": selected["baseline_top1"]["score"] - selected["unified_top1"]["score"],
            "target_absent_by_coco": True,
            "output_image": out_img_path,
        },
        "output_image": out_img_path,
        "candidates_csv": csv_path,
    }

    summary_json = os.path.join(args.output_dir, "summary.json")
    summary_md = os.path.join(args.output_dir, "summary.md")
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)
    with open(summary_md, "w") as f:
        sc = summary["selected_case"]
        f.write("# Negative Affordance Abstention Case (Baseline vs Unified)\n\n")
        f.write(f"- prompt: `{prompt}`\n")
        f.write(f"- target_categories: `{target_categories}`\n")
        f.write(f"- target_absent_definition: {summary['target_absence_definition']}\n")
        f.write(f"- num_candidates_scanned: {len(candidates)}\n")
        f.write(f"- selected_threshold: {threshold}\n")
        f.write(f"- selection_mode: `{selection_mode}`\n")
        f.write(f"- output_image: `{out_img_path}`\n\n")
        f.write("| image_id | source_affordance | baseline_score | unified_score | gap |\n")
        f.write("|---:|---|---:|---:|---:|\n")
        f.write(
            f"| {sc['image_id']} | {sc['source_affordance_id']} | "
            f"{sc['baseline_top1']['score']:.3f} | {sc['unified_top1']['score']:.3f} | "
            f"{sc['score_gap_baseline_minus_unified']:.3f} |\n"
        )

    print("\n" + "=" * 72)
    print("Done")
    print(f"output_image: {out_img_path}")
    print(f"summary_json: {summary_json}")
    print(f"summary_md:   {summary_md}")
    print(f"candidates_csv: {csv_path}")
    print(
        f"selected image_id={selected['image_id']} "
        f"baseline_score={selected['baseline_top1']['score']:.3f} "
        f"unified_score={selected['unified_top1']['score']:.3f} "
        f"threshold={threshold:.3f}"
    )
    print("=" * 72)


if __name__ == "__main__":
    main()
