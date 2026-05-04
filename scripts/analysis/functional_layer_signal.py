#!/usr/bin/env python
"""
Probe where InternVL LLM layers carry functional-query localization signal.

This is a no-training diagnostic. For each benchmark sample, it forwards the
same image with:
  1. the functional prompt, for example "something to cut with ."
  2. a neutral prompt, by default "something ."

For each selected LLM layer, it scores the 16x16 visual tokens by how much the
functional prompt changes their hidden state relative to the neutral prompt.
The script then checks whether high-change visual tokens overlap the COCO GT
boxes for the functional target categories.
"""

import argparse
import datetime
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image


ROOT = "/home/iibrohimm/project/next_step"
sys.path.insert(0, ROOT)

from thinkdet.models.projector import InternVLFeatureExtractor, build_internvl_transform


DEFAULT_BENCH = f"{ROOT}/thinkdet/data/benchmarks/affordance_coco_val_heldout_v2.json"
DEFAULT_INTERNVL = f"{ROOT}/InternVL3_5-1B"
DEFAULT_OUT = (
    f"{ROOT}/thinkdet/results/functional_layer_signal/"
    f"functional_layer_signal_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=str, default=DEFAULT_BENCH)
    parser.add_argument("--split", type=str, default="dev", choices=["dev", "test", "all"])
    parser.add_argument("--internvl_path", type=str, default=DEFAULT_INTERNVL)
    parser.add_argument("--layers", type=int, nargs="+", default=None)
    parser.add_argument("--neutral_prompt", type=str, default="something .")
    parser.add_argument("--topk_tokens", type=int, nargs="+", default=[1, 5, 16])
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output", type=str, default=DEFAULT_OUT)
    parser.add_argument("--log_every", type=int, default=10)
    return parser.parse_args()


def read_num_llm_layers(model_path):
    with open(os.path.join(model_path, "config.json"), "r") as f:
        cfg = json.load(f)
    return int(cfg["llm_config"]["num_hidden_layers"])


def xywh_to_xyxy(box):
    x, y, w, h = box
    return [float(x), float(y), float(x + w), float(y + h)]


def token_overlap_mask(image_size, gt_boxes_xyxy, n_tokens):
    img_w, img_h = image_size
    side = int(round(math.sqrt(n_tokens)))
    if side * side != n_tokens:
        raise ValueError(f"Expected square visual-token grid, got {n_tokens} tokens")

    positives = []
    overlap_fracs = []
    cell_w = float(img_w) / side
    cell_h = float(img_h) / side
    cell_area = max(cell_w * cell_h, 1e-9)

    for idx in range(n_tokens):
        row = idx // side
        col = idx % side
        x1 = col * cell_w
        y1 = row * cell_h
        x2 = (col + 1) * cell_w
        y2 = (row + 1) * cell_h

        overlap_area = 0.0
        for gx1, gy1, gx2, gy2 in gt_boxes_xyxy:
            ix1 = max(x1, gx1)
            iy1 = max(y1, gy1)
            ix2 = min(x2, gx2)
            iy2 = min(y2, gy2)
            overlap_area += max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)

        frac = min(1.0, overlap_area / cell_area)
        overlap_fracs.append(frac)
        positives.append(frac > 0.0)

    return (
        torch.tensor(positives, dtype=torch.bool),
        torch.tensor(overlap_fracs, dtype=torch.float32),
    )


def compute_layer_metrics(token_scores, positive_mask, overlap_fracs, topk_tokens):
    scores = token_scores.float().clamp_min(0)
    total = scores.sum().clamp_min(1e-12)
    pos = positive_mask.to(scores.device)
    overlap = overlap_fracs.to(scores.device)

    pos_score = scores[pos].sum() if pos.any() else scores.new_zeros(())
    mass_in_positive_tokens = pos_score / total
    positive_token_frac = pos.float().mean()
    mass_lift = mass_in_positive_tokens / positive_token_frac.clamp_min(1e-6)

    pos_scores = scores[pos]
    neg_scores = scores[~pos]
    if len(pos_scores) and len(neg_scores):
        pooled_std = scores.std(unbiased=False).clamp_min(1e-6)
        gt_bg_z = (pos_scores.mean() - neg_scores.mean()) / pooled_std
    else:
        gt_bg_z = scores.new_zeros(())

    weighted_overlap_mass = (scores * overlap).sum() / total

    order = scores.argsort(descending=True)
    top_hits = {}
    for k in topk_tokens:
        kk = min(int(k), len(order))
        top_hits[f"top{k}_token_hit"] = bool(pos[order[:kk]].any().item())

    out = {
        "mass_in_positive_tokens": float(mass_in_positive_tokens.item()),
        "positive_token_frac": float(positive_token_frac.item()),
        "mass_lift": float(mass_lift.item()),
        "weighted_overlap_mass": float(weighted_overlap_mass.item()),
        "gt_bg_z": float(gt_bg_z.item()),
        "mean_delta": float(scores.mean().item()),
        "max_delta": float(scores.max().item()),
    }
    out.update({k: int(v) for k, v in top_hits.items()})
    return out


def summarize_rows(rows, topk_tokens):
    n = max(len(rows), 1)
    summary = {
        "n_samples": len(rows),
        "mass_in_positive_tokens": sum(r["mass_in_positive_tokens"] for r in rows) / n,
        "positive_token_frac": sum(r["positive_token_frac"] for r in rows) / n,
        "mass_lift": sum(r["mass_lift"] for r in rows) / n,
        "weighted_overlap_mass": sum(r["weighted_overlap_mass"] for r in rows) / n,
        "gt_bg_z": sum(r["gt_bg_z"] for r in rows) / n,
        "mean_delta": sum(r["mean_delta"] for r in rows) / n,
        "max_delta": sum(r["max_delta"] for r in rows) / n,
    }
    for k in topk_tokens:
        key = f"top{k}_token_hit"
        summary[key] = sum(r[key] for r in rows) / n
    return summary


def write_summary_markdown(payload, out_json):
    topk_tokens = payload["topk_tokens"]
    per_layer = payload["per_layer"]
    hit_cols = [f"top{k}_token_hit" for k in topk_tokens]
    rows = []
    for layer, item in sorted(per_layer.items(), key=lambda kv: int(kv[0])):
        summary = item["summary"]
        rows.append((int(layer), summary))

    rank_key = hit_cols[1] if len(hit_cols) > 1 else hit_cols[0]
    ranked = sorted(
        rows,
        key=lambda x: (
            -x[1].get(rank_key, 0.0),
            -x[1].get("mass_lift", 0.0),
            x[0],
        ),
    )
    winner_layer, winner_summary = ranked[0]

    header = [
        "Layer",
        "Top1 Token Hit",
        "Top5 Token Hit",
        "Top16 Token Hit",
        "Mass Lift",
        "GT-BG z",
        "Mean Delta",
        "N",
    ]
    lines = [
        "| " + " | ".join(header) + " |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for layer, summary in rows:
        lines.append(
            f"| {layer} | "
            f"{100 * summary.get('top1_token_hit', 0.0):.2f}% | "
            f"{100 * summary.get('top5_token_hit', 0.0):.2f}% | "
            f"{100 * summary.get('top16_token_hit', 0.0):.2f}% | "
            f"{summary['mass_lift']:.3f} | "
            f"{summary['gt_bg_z']:.3f} | "
            f"{summary['mean_delta']:.6f} | "
            f"{summary['n_samples']} |"
        )
    lines.append("")
    lines.append(
        f"Winner by {rank_key}: layer {winner_layer} "
        f"({100 * winner_summary.get(rank_key, 0.0):.2f}%, "
        f"mass lift {winner_summary['mass_lift']:.3f})."
    )

    md_path = Path(out_json).with_suffix(".md")
    md_path.write_text("\n".join(lines) + "\n")
    return str(md_path), winner_layer


@torch.no_grad()
def main():
    args = parse_args()
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard_index must be in [0, num_shards)")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    num_layers = read_num_llm_layers(args.internvl_path)
    layers = args.layers if args.layers is not None else list(range(num_layers))
    layers = sorted(set(int(x) for x in layers))
    for layer in layers:
        if layer < 0 or layer >= num_layers:
            raise ValueError(f"Layer {layer} outside valid range 0..{num_layers - 1}")

    with open(args.benchmark, "r") as f:
        bench = json.load(f)
    samples = bench["samples"]
    if args.split != "all":
        samples = [s for s in samples if s.get("split") == args.split]
    if args.max_samples > 0:
        samples = samples[: args.max_samples]
    samples = [
        s for idx, s in enumerate(samples)
        if idx % int(args.num_shards) == int(args.shard_index)
    ]

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    extractor = InternVLFeatureExtractor(
        model_path=args.internvl_path,
        extract_layer=max(layers),
        extract_layers=layers,
        layer_fusion="mean",
        freeze=True,
    ).to(device)
    extractor.eval()
    transform = build_internvl_transform()

    rows_by_layer = {layer: [] for layer in layers}
    per_affordance = {layer: defaultdict(list) for layer in layers}

    for i, sample in enumerate(samples, 1):
        image_pil = Image.open(sample["image_path"]).convert("RGB")
        image_t = transform(image_pil).unsqueeze(0).to(device)
        gt_boxes = [xywh_to_xyxy(t["bbox_xywh"]) for t in sample["positive_targets"]]

        selected_functional = extractor.extract_selected_layers(image_t, [sample["prompt"]])
        selected_neutral = extractor.extract_selected_layers(image_t, [args.neutral_prompt])

        n_tokens = selected_functional[layers[0]].shape[1]
        positive_mask, overlap_fracs = token_overlap_mask(image_pil.size, gt_boxes, n_tokens)

        for layer in layers:
            functional_h = selected_functional[layer][0]
            neutral_h = selected_neutral[layer][0]
            token_scores = (functional_h.float() - neutral_h.float()).norm(dim=-1)
            metrics = compute_layer_metrics(
                token_scores,
                positive_mask,
                overlap_fracs,
                args.topk_tokens,
            )
            row = {
                "benchmark_id": sample["benchmark_id"],
                "image_id": int(sample["image_id"]),
                "affordance_id": sample["affordance_id"],
                "prompt": sample["prompt"],
                "layer": int(layer),
                **metrics,
            }
            rows_by_layer[layer].append(row)
            per_affordance[layer][sample["affordance_id"]].append(row)

        if i % max(1, args.log_every) == 0 or i == len(samples):
            print(f"[signal] shard={args.shard_index}/{args.num_shards} {i}/{len(samples)}")

    per_layer = {}
    for layer in layers:
        rows = rows_by_layer[layer]
        per_layer[str(layer)] = {
            "summary": summarize_rows(rows, args.topk_tokens),
            "per_affordance": {
                aff: summarize_rows(aff_rows, args.topk_tokens)
                for aff, aff_rows in sorted(per_affordance[layer].items())
            },
            "per_sample": rows,
        }

    payload = {
        "status": "ok",
        "timestamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
        "benchmark": args.benchmark,
        "split": args.split,
        "neutral_prompt": args.neutral_prompt,
        "internvl_path": args.internvl_path,
        "num_llm_layers": num_layers,
        "layers": layers,
        "topk_tokens": [int(k) for k in args.topk_tokens],
        "num_shards": int(args.num_shards),
        "shard_index": int(args.shard_index),
        "n_samples": len(samples),
        "metric_note": (
            "Token scores are ||H_functional - H_neutral|| over 16x16 visual tokens; "
            "hits indicate whether top changed visual tokens overlap GT target boxes."
        ),
        "per_layer": per_layer,
    }
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)

    md_path, winner = write_summary_markdown(payload, args.output)
    print("Saved:", args.output)
    print("Saved:", md_path)
    print("winner_layer:", winner)


if __name__ == "__main__":
    main()
