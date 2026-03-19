#!/usr/bin/env python3
"""
H_delta diagnostic (no training):
    H_delta = H(image, prompt) - H(image, neutral_prompt)

Measures:
  1) Norm of H_delta
  2) Cosine similarity between deltas of different prompts
  3) Magnitude ratio: ||H_delta|| / ||H_raw||

Default prompts:
  - "the man"
  - "the tall man"
  - "the man on the left"
neutral:
  - "a photo"
"""

import argparse
import json
import os
import sys
from itertools import combinations

import torch
import torch.nn.functional as F
from PIL import Image

ROOT = "/home/iibrohimm/project/next_step"
sys.path.insert(0, ROOT)

from thinkdet.models.projector import InternVLFeatureExtractor, build_internvl_transform


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--layer", type=int, default=9)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument(
        "--image_path",
        type=str,
        default=f"{ROOT}/dataSets/coco/val2017/000000000139.jpg",
        help="Single fixed image path",
    )
    p.add_argument("--neutral_prompt", type=str, default="a photo")
    p.add_argument(
        "--prompts",
        nargs=3,
        default=["the man", "the tall man", "the man on the left"],
    )
    p.add_argument("--output", type=str, default="")
    return p.parse_args()


def _norm_stats(x: torch.Tensor):
    token_norm = x.norm(dim=-1)  # [256]
    fro = x.norm()               # scalar
    return {
        "fro_norm": float(fro.item()),
        "token_norm_mean": float(token_norm.mean().item()),
        "token_norm_std": float(token_norm.std(unbiased=False).item()),
        "token_norm_min": float(token_norm.min().item()),
        "token_norm_max": float(token_norm.max().item()),
    }


def _pair_cos_stats(a: torch.Tensor, b: torch.Tensor):
    token_cos = F.cosine_similarity(a, b, dim=-1)  # [256]
    flat_cos = F.cosine_similarity(a.reshape(1, -1), b.reshape(1, -1), dim=-1)
    return {
        "flat_cosine": float(flat_cos.item()),
        "token_cosine_mean": float(token_cos.mean().item()),
        "token_cosine_std": float(token_cos.std(unbiased=False).item()),
        "token_cosine_min": float(token_cos.min().item()),
        "token_cosine_max": float(token_cos.max().item()),
    }


def main():
    args = parse_args()
    if not os.path.isfile(args.image_path):
        raise FileNotFoundError(f"Image not found: {args.image_path}")

    device = args.device if torch.cuda.is_available() else "cpu"
    model_path = f"{ROOT}/InternVL3_5-1B"

    extractor = InternVLFeatureExtractor(
        model_path=model_path,
        extract_layer=args.layer,
        freeze=True,
        use_flash_attn=False,
    ).to(device).eval()

    transform = build_internvl_transform(448)
    image = Image.open(args.image_path).convert("RGB")
    pixel_values = transform(image).unsqueeze(0).to(device)

    @torch.no_grad()
    def extract(prompt: str):
        return extractor(pixel_values, [prompt])[0].float().cpu()  # [256, D]

    h_neutral = extract(args.neutral_prompt)
    h_raw = {}
    h_delta = {}
    prompt_stats = {}

    for prompt in args.prompts:
        raw = extract(prompt)
        delta = raw - h_neutral
        h_raw[prompt] = raw
        h_delta[prompt] = delta

        raw_stats = _norm_stats(raw)
        delta_stats = _norm_stats(delta)
        ratio = delta_stats["fro_norm"] / max(raw_stats["fro_norm"], 1e-12)

        prompt_stats[prompt] = {
            "raw_norm": raw_stats,
            "delta_norm": delta_stats,
            "delta_over_raw_fro_ratio": float(ratio),
        }

    pairwise = {}
    for p1, p2 in combinations(args.prompts, 2):
        key = f"{p1}  ||  {p2}"
        pairwise[key] = {
            "raw_cosine": _pair_cos_stats(h_raw[p1], h_raw[p2]),
            "delta_cosine": _pair_cos_stats(h_delta[p1], h_delta[p2]),
        }

    result = {
        "layer": int(args.layer),
        "device": device,
        "image_path": args.image_path,
        "neutral_prompt": args.neutral_prompt,
        "prompts": list(args.prompts),
        "prompt_metrics": prompt_stats,
        "pairwise_prompt_metrics": pairwise,
    }

    text = json.dumps(result, indent=2)
    print(text)
    if args.output:
        out_dir = os.path.dirname(args.output)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.output, "w") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
