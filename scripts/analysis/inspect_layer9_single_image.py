#!/usr/bin/env python3
"""
Inspect InternVL layer-9 query-conditioned visual features on one image.

Outputs:
- raw layer-9 feature tensors per prompt
- per-prompt token-norm heatmaps
- delta heatmaps against a reference prompt
- prompt-level pooled cosine similarity matrix
"""

import argparse
import json
import math
import os
import sys
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = "/home/iibrohimm/project/next_step"
sys.path.insert(0, ROOT)

from thinkdet.models.projector import InternVLFeatureExtractor, build_internvl_transform


DEFAULT_INTERNVL_PATH = f"{ROOT}/InternVL3_5-1B"
DEFAULT_OUTPUT_DIR = f"{ROOT}/thinkdet/results/analysis/layer9_single_image"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument(
        "--prompts",
        type=str,
        nargs="+",
        required=True,
        help="Prompts to compare on the same image.",
    )
    parser.add_argument("--extract_layer", type=int, default=9)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--internvl_path", type=str, default=DEFAULT_INTERNVL_PATH)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--reference_prompt_idx",
        type=int,
        default=0,
        help="Prompt index used as the delta baseline.",
    )
    return parser.parse_args()


def normalize_query(text: str) -> str:
    text = " ".join(str(text).strip().lower().split())
    if not text:
        text = "object"
    if text.endswith("."):
        text = text[:-1].rstrip()
    return f"{text} ."


def sanitize_name(text: str) -> str:
    out = []
    for ch in text.lower():
        if ch.isalnum():
            out.append(ch)
        elif ch in {" ", "-", "_"}:
            out.append("_")
    collapsed = "".join(out).strip("_")
    while "__" in collapsed:
        collapsed = collapsed.replace("__", "_")
    return collapsed[:80] or "prompt"


def infer_grid_size(num_tokens: int):
    side = int(round(math.sqrt(num_tokens)))
    if side * side != num_tokens:
        raise ValueError(f"Cannot infer square grid from {num_tokens} tokens")
    return side


def save_grid_heatmap(grid: np.ndarray, out_path: str, title: str):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 5), dpi=160)
    im = ax.imshow(grid, cmap="magma")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_overlay(image_pil: Image.Image, grid: np.ndarray, out_path: str, title: str):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6), dpi=160)
    ax.imshow(image_pil)
    ax.imshow(
        grid,
        cmap="magma",
        alpha=0.55,
        interpolation="bilinear",
        extent=(0, image_pil.width, image_pil.height, 0),
    )
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def save_similarity_heatmap(labels: List[str], sim: np.ndarray, out_path: str, extract_layer: int):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(max(5, len(labels) * 1.3), 4), dpi=160)
    im = ax.imshow(sim, cmap="viridis", vmin=-1.0, vmax=1.0)
    ax.set_title(f"Prompt-Level Cosine Similarity of Layer-{extract_layer} Features")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    for i in range(sim.shape[0]):
        for j in range(sim.shape[1]):
            ax.text(j, i, f"{sim[i, j]:.2f}", ha="center", va="center", color="white", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def top_tokens(token_norms: np.ndarray, side: int, image_w: int, image_h: int, top_k: int = 10):
    order = np.argsort(-token_norms)[:top_k]
    items = []
    for idx in order.tolist():
        row = int(idx // side)
        col = int(idx % side)
        x = float((col + 0.5) * image_w / side)
        y = float((row + 0.5) * image_h / side)
        items.append(
            {
                "token_index": int(idx),
                "row": row,
                "col": col,
                "norm": float(token_norms[idx]),
                "approx_center_xy": [round(x, 2), round(y, 2)],
            }
        )
    return items


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    image_pil = Image.open(args.image).convert("RGB")
    image_name = os.path.basename(args.image)

    norm_prompts = [normalize_query(p) for p in args.prompts]
    if not 0 <= args.reference_prompt_idx < len(norm_prompts):
        raise ValueError("reference_prompt_idx out of range")

    transform = build_internvl_transform(448)
    pixel_values = transform(image_pil).unsqueeze(0).to(device)

    extractor = InternVLFeatureExtractor(
        model_path=args.internvl_path,
        extract_layer=args.extract_layer,
        extract_layers=[args.extract_layer],
        freeze=True,
        use_flash_attn=False,
    ).to(device).eval()

    print("=" * 68)
    print("InternVL layer feature inspection")
    print(f"image: {args.image}")
    print(f"device: {device}")
    print(f"extract_layer: {args.extract_layer}")
    print(f"prompts: {len(norm_prompts)}")
    print("=" * 68)

    per_prompt: List[Dict] = []
    pooled_features = []
    raw_features = {}

    with torch.no_grad():
        for idx, query_text in enumerate(norm_prompts):
            selected = extractor.extract_selected_layers(pixel_values, [query_text])
            feat = selected[args.extract_layer][0].detach().cpu()  # [N_vis, D]
            raw_features[query_text] = feat
            token_norms = feat.norm(dim=-1).numpy()
            side = infer_grid_size(token_norms.shape[0])
            grid = token_norms.reshape(side, side)
            grid01 = (grid - grid.min()) / max(float(grid.max() - grid.min()), 1e-8)

            slug = f"{idx:02d}_{sanitize_name(query_text)}"
            torch.save(feat, os.path.join(args.output_dir, f"{slug}_layer{args.extract_layer}.pt"))
            save_grid_heatmap(
                grid01,
                os.path.join(args.output_dir, f"{slug}_norm_heatmap.png"),
                title=f"Layer {args.extract_layer} token norms\n{query_text}",
            )
            save_overlay(
                image_pil,
                grid01,
                os.path.join(args.output_dir, f"{slug}_norm_overlay.png"),
                title=f"Layer {args.extract_layer} token norms\n{query_text}",
            )

            pooled = F.normalize(feat.mean(dim=0, keepdim=True).float(), dim=-1)[0]
            pooled_features.append(pooled)
            per_prompt.append(
                {
                    "prompt_index": idx,
                    "prompt_text": query_text,
                    "tensor_path": os.path.join(args.output_dir, f"{slug}_layer{args.extract_layer}.pt"),
                    "grid_side": side,
                    "feature_shape": list(feat.shape),
                    "token_norm_mean": float(token_norms.mean()),
                    "token_norm_std": float(token_norms.std()),
                    "token_norm_min": float(token_norms.min()),
                    "token_norm_max": float(token_norms.max()),
                    "top_tokens": top_tokens(token_norms, side, image_pil.width, image_pil.height),
                }
            )
            print(
                f"[{idx + 1}/{len(norm_prompts)}] {query_text} "
                f"| mean_norm={token_norms.mean():.4f} max_norm={token_norms.max():.4f}"
            )

    pooled_stack = torch.stack(pooled_features, dim=0)
    sim = (pooled_stack @ pooled_stack.T).numpy()
    labels = [f"{i}: {sanitize_name(p)[:18]}" for i, p in enumerate(norm_prompts)]
    sim_path = os.path.join(args.output_dir, "prompt_similarity_heatmap.png")
    save_similarity_heatmap(labels, sim, sim_path, args.extract_layer)

    ref_query = norm_prompts[args.reference_prompt_idx]
    ref_feat = raw_features[ref_query]
    ref_norm = ref_feat.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    ref_unit = ref_feat / ref_norm

    delta_summaries = []
    for idx, query_text in enumerate(norm_prompts):
        if idx == args.reference_prompt_idx:
            continue
        feat = raw_features[query_text]
        feat_unit = feat / feat.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        delta_mag = (feat - ref_feat).norm(dim=-1).numpy()
        cos = (feat_unit * ref_unit).sum(dim=-1).numpy()
        side = infer_grid_size(delta_mag.shape[0])
        delta_grid = delta_mag.reshape(side, side)
        cos_grid = cos.reshape(side, side)
        delta01 = (delta_grid - delta_grid.min()) / max(float(delta_grid.max() - delta_grid.min()), 1e-8)

        slug = f"delta_from_{args.reference_prompt_idx:02d}_to_{idx:02d}_{sanitize_name(query_text)}"
        save_grid_heatmap(
            delta01,
            os.path.join(args.output_dir, f"{slug}_delta_heatmap.png"),
            title=f"Layer {args.extract_layer} delta magnitude\n{ref_query} -> {query_text}",
        )
        save_overlay(
            image_pil,
            delta01,
            os.path.join(args.output_dir, f"{slug}_delta_overlay.png"),
            title=f"Layer {args.extract_layer} delta magnitude\n{ref_query} -> {query_text}",
        )
        save_grid_heatmap(
            cos_grid,
            os.path.join(args.output_dir, f"{slug}_cosine_heatmap.png"),
            title=f"Layer {args.extract_layer} token cosine\n{ref_query} vs {query_text}",
        )
        delta_summaries.append(
            {
                "reference_prompt": ref_query,
                "target_prompt": query_text,
                "delta_norm_mean": float(delta_mag.mean()),
                "delta_norm_max": float(delta_mag.max()),
                "token_cosine_mean": float(cos.mean()),
                "token_cosine_min": float(cos.min()),
                "token_cosine_max": float(cos.max()),
            }
        )

    report = {
        "image_path": args.image,
        "image_name": image_name,
        "extract_layer": args.extract_layer,
        "device": str(device),
        "reference_prompt_index": args.reference_prompt_idx,
        "reference_prompt_text": ref_query,
        "prompts": per_prompt,
        "prompt_similarity_matrix": {
            "labels": labels,
            "matrix": sim.tolist(),
            "heatmap_path": sim_path,
        },
        "delta_summaries": delta_summaries,
    }

    report_path = os.path.join(args.output_dir, "report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print("\nSaved analysis:")
    print(f"- report: {report_path}")
    print(f"- similarity heatmap: {sim_path}")
    print(f"- per-prompt tensors and overlays: {args.output_dir}")


if __name__ == "__main__":
    main()
