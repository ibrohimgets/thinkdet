#!/usr/bin/env python3
"""
Analyze similarity across InternVL hidden layers at the visual-token positions.

The goal is to partition the MLLM stack into contiguous early / mid / late
groups using actual ThinkDet-style image + query inputs, rather than intuition.
"""

import argparse
import json
import os
import random
import sys
import time
from typing import Dict, Iterable, List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = "/home/iibrohimm/project/next_step"
sys.path.insert(0, ROOT)

from thinkdet.data.coco_grounding import COCOGroundingDataset
from thinkdet.data.refcoco_grounding import build_refcoco_eval
from thinkdet.models.projector import InternVLFeatureExtractor, build_internvl_transform


DEFAULT_OUTPUT_DIR = f"{ROOT}/thinkdet/results/layer_similarity"
DEFAULT_AFFORDANCE_BENCH = (
    f"{ROOT}/thinkdet/data/benchmarks/affordance_coco_val_heldout_v1.json"
)
DEFAULT_COCO_VAL_IMG = f"{ROOT}/dataSets/coco/val2017"
DEFAULT_COCO_VAL_ANN = f"{ROOT}/dataSets/coco/annotations/instances_val2017.json"
DEFAULT_REFCOCO_ROOT = f"{ROOT}/data/refcoco"
DEFAULT_REFCOCO_IMG = f"{ROOT}/dataSets/coco/train2014"
DEFAULT_INTERNVL_PATH = f"{ROOT}/InternVL3_5-1B"


def normalize_query(text: str) -> str:
    text = " ".join(str(text).strip().lower().split())
    if not text:
        text = "object"
    if not text.endswith("."):
        text = f"{text} ."
    elif not text.endswith(" ."):
        text = text[:-1].rstrip() + " ."
    return text


def rank01(values: List[float]) -> List[float]:
    arr = np.asarray(values, dtype=np.float64)
    out = np.zeros_like(arr, dtype=np.float64)
    finite = np.isfinite(arr)
    if finite.sum() == 0:
        return out.tolist()
    if finite.sum() == 1:
        out[finite] = 1.0
        return out.tolist()

    finite_vals = arr[finite]
    order = np.argsort(finite_vals, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(finite_vals.size, dtype=np.float64)
    out_vals = ranks / max(finite_vals.size - 1, 1)
    out[finite] = out_vals
    return out.tolist()


def mean_offdiag(mat: np.ndarray) -> float:
    if mat.shape[0] <= 1:
        return float(mat.mean()) if mat.size else 0.0
    mask = ~np.eye(mat.shape[0], dtype=bool)
    vals = mat[mask]
    if vals.size == 0:
        return 0.0
    return float(vals.mean())


def sample_affordance_records(args) -> List[Dict]:
    with open(args.affordance_benchmark, "r") as f:
        benchmark = json.load(f)

    split = str(args.affordance_split).lower()
    samples = benchmark.get("samples", [])
    if split != "all":
        samples = [s for s in samples if str(s.get("split", "")).lower() == split]

    rng = random.Random(args.seed)
    rng.shuffle(samples)
    if args.max_samples > 0:
        samples = samples[:args.max_samples]

    transform = build_internvl_transform(448)
    records = []
    for sample in samples:
        records.append({
            "sample_id": sample.get("benchmark_id", f"img_{sample.get('image_id')}"),
            "image_path": sample["image_path"],
            "query_text": normalize_query(sample["prompt"]),
            "pixel_values": transform(Image.open(sample["image_path"]).convert("RGB")),
            "metadata": {
                "image_id": sample.get("image_id"),
                "split": sample.get("split"),
                "affordance_id": sample.get("affordance_id"),
                "target_categories": sample.get("target_categories", []),
            },
        })
    return records


def sample_refcoco_records(args) -> List[Dict]:
    dataset = build_refcoco_eval(
        data_root=args.refcoco_root,
        image_dir=args.refcoco_image_dir,
        dataset_name=args.refcoco_dataset_name,
        split_by=args.refcoco_split_by,
        split=args.refcoco_split,
        internvl_size=448,
    )

    indices = list(range(len(dataset)))
    rng = random.Random(args.seed)
    rng.shuffle(indices)
    if args.max_samples > 0:
        indices = indices[:args.max_samples]

    records = []
    for idx in indices:
        sample = dataset[idx]
        records.append({
            "sample_id": f"{sample['ref_id']}_{sample['image_id']}_{idx}",
            "query_text": normalize_query(sample["query_text"]),
            "pixel_values": sample["internvl_image"],
            "metadata": {
                "ref_id": sample["ref_id"],
                "image_id": sample["image_id"],
                "expression": sample["expression"],
            },
        })
    return records


def sample_coco_records(args) -> List[Dict]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    dataset = COCOGroundingDataset(
        img_dir=args.coco_img_dir,
        ann_file=args.coco_ann_file,
        internvl_size=448,
        query_mode=args.coco_query_mode,
        dynamic_query_ratio=1.0,
        max_query_categories=args.coco_max_query_categories,
    )

    indices = list(range(len(dataset)))
    rng = random.Random(args.seed)
    rng.shuffle(indices)
    if args.max_samples > 0:
        indices = indices[:args.max_samples]

    records = []
    for idx in indices:
        sample = dataset[idx]
        records.append({
            "sample_id": f"coco_{sample['image_id']}_{idx}",
            "query_text": normalize_query(sample["query_text"]),
            "pixel_values": sample["internvl_image"],
            "metadata": {
                "image_id": sample["image_id"],
                "num_boxes": int(sample["boxes"].shape[0]),
                "category_names": sample["category_names"],
            },
        })
    return records


def load_records(args) -> List[Dict]:
    if args.source == "affordance":
        return sample_affordance_records(args)
    if args.source == "refcoco":
        return sample_refcoco_records(args)
    if args.source == "coco":
        return sample_coco_records(args)
    raise ValueError(f"Unsupported source={args.source!r}")


def save_heatmap(sim_matrix: np.ndarray, out_path: str):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[WARN] Skipping heatmap: matplotlib unavailable ({exc})")
        return

    fig, ax = plt.subplots(figsize=(8, 6), dpi=160)
    im = ax.imshow(sim_matrix, cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_title("InternVL Layer Similarity (Visual Tokens)")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Layer")
    ax.set_xticks(range(sim_matrix.shape[0]))
    ax.set_yticks(range(sim_matrix.shape[0]))
    ax.tick_params(axis="x", labelsize=6)
    ax.tick_params(axis="y", labelsize=6)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def pick_boundaries(sim_matrix: np.ndarray, num_groups: int, min_group_size: int):
    num_layers = sim_matrix.shape[0]
    if num_groups < 2 or num_layers <= num_groups:
        return []

    min_size = max(1, min(min_group_size, num_layers // num_groups))
    adj_gap = [float(1.0 - sim_matrix[i, i + 1]) for i in range(num_layers - 1)]

    if num_groups != 3:
        raise ValueError("This script currently supports exactly 3 groups.")

    best = None
    for split1 in range(min_size, num_layers - (2 * min_size) + 1):
        for split2 in range(split1 + min_size, num_layers - min_size + 1):
            group_sizes = (split1, split2 - split1, num_layers - split2)
            if min(group_sizes) < min_size:
                continue

            gap_score = adj_gap[split1 - 1] + adj_gap[split2 - 1]

            groups = [
                sim_matrix[0:split1, 0:split1],
                sim_matrix[split1:split2, split1:split2],
                sim_matrix[split2:num_layers, split2:num_layers],
            ]
            within = np.mean([mean_offdiag(g) for g in groups])
            boundary_cross = float(
                (sim_matrix[split1 - 1, split1] + sim_matrix[split2 - 1, split2]) / 2.0
            )

            score = gap_score + (0.1 * within) - (0.1 * boundary_cross)
            candidate = {
                "splits": [split1, split2],
                "score": round(float(score), 6),
                "gap_score": round(float(gap_score), 6),
                "within_group_similarity": round(float(within), 6),
                "boundary_similarity": round(float(boundary_cross), 6),
                "group_sizes": list(group_sizes),
            }
            if best is None or candidate["score"] > best["score"]:
                best = candidate

    if best is None:
        split1 = num_layers // 3
        split2 = (2 * num_layers) // 3
        best = {
            "splits": [split1, split2],
            "score": 0.0,
            "gap_score": 0.0,
            "within_group_similarity": 0.0,
            "boundary_similarity": 0.0,
            "group_sizes": [split1, split2 - split1, num_layers - split2],
        }
    return best


def build_group_dict(num_layers: int, split_info: Dict) -> Dict:
    split1, split2 = split_info["splits"]
    groups = {
        "early": list(range(0, split1)),
        "mid": list(range(split1, split2)),
        "late": list(range(split2, num_layers)),
    }
    out = {}
    for name, layers in groups.items():
        out[name] = {
            "layers": layers,
            "start": layers[0],
            "end": layers[-1],
            "count": len(layers),
        }
    return out


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["affordance", "refcoco", "coco"], default="affordance")
    parser.add_argument("--max_samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--internvl_path", type=str, default=DEFAULT_INTERNVL_PATH)
    parser.add_argument("--min_group_size", type=int, default=3)

    parser.add_argument("--affordance_benchmark", type=str, default=DEFAULT_AFFORDANCE_BENCH)
    parser.add_argument("--affordance_split", choices=["dev", "test", "all"], default="all")

    parser.add_argument("--refcoco_root", type=str, default=DEFAULT_REFCOCO_ROOT)
    parser.add_argument("--refcoco_image_dir", type=str, default=DEFAULT_REFCOCO_IMG)
    parser.add_argument("--refcoco_dataset_name", type=str, default="refcocog")
    parser.add_argument("--refcoco_split_by", type=str, default="umd")
    parser.add_argument("--refcoco_split", type=str, default="val")

    parser.add_argument("--coco_img_dir", type=str, default=DEFAULT_COCO_VAL_IMG)
    parser.add_argument("--coco_ann_file", type=str, default=DEFAULT_COCO_VAL_ANN)
    parser.add_argument("--coco_query_mode", choices=["fixed", "gt", "mixed"], default="gt")
    parser.add_argument("--coco_max_query_categories", type=int, default=12)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"[INFO] Loading records from source={args.source}")
    records = load_records(args)
    if not records:
        raise RuntimeError("No records loaded for similarity analysis.")

    print(f"[INFO] Loaded {len(records)} records")
    device = torch.device(args.device)

    extractor = InternVLFeatureExtractor(
        model_path=args.internvl_path,
        extract_layer=0,
        freeze=True,
        use_flash_attn=False,
    )
    extractor.extract_layers = list(range(extractor.num_llm_layers))
    extractor.extract_layer = extractor.extract_layers[-1]
    extractor = extractor.to(device).eval()

    num_layers = extractor.num_llm_layers
    sim_sum = torch.zeros(num_layers, num_layers, dtype=torch.float64)
    feature_norms = [[] for _ in range(num_layers)]
    sample_manifest = []

    t0 = time.time()
    with torch.no_grad():
        for idx, record in enumerate(records):
            pixel_values = record["pixel_values"].unsqueeze(0).to(device)
            query_text = record["query_text"]
            selected = extractor.extract_selected_layers(pixel_values, [query_text])

            pooled = []
            for layer_idx in range(num_layers):
                feat = selected[layer_idx].mean(dim=1).float()
                feature_norms[layer_idx].append(float(feat.norm(dim=-1).mean().item()))
                pooled.append(F.normalize(feat, dim=-1)[0].cpu())

            pooled_stack = torch.stack(pooled, dim=0).to(torch.float64)
            sim_sum += pooled_stack @ pooled_stack.T

            sample_manifest.append({
                "sample_id": record["sample_id"],
                "query_text": query_text,
                "metadata": record.get("metadata", {}),
            })

            if (idx + 1) % 25 == 0 or idx == len(records) - 1:
                elapsed = time.time() - t0
                print(
                    f"[INFO] Processed {idx + 1}/{len(records)} samples "
                    f"in {elapsed:.1f}s"
                )

            del selected
            del pixel_values
            if device.type == "cuda":
                torch.cuda.empty_cache()

    sim_matrix = (sim_sum / max(len(records), 1)).numpy()
    split_info = pick_boundaries(
        sim_matrix=sim_matrix,
        num_groups=3,
        min_group_size=args.min_group_size,
    )
    groups = build_group_dict(num_layers, split_info)

    adjacent_gaps = [
        {
            "between_layers": [i, i + 1],
            "distance": round(float(1.0 - sim_matrix[i, i + 1]), 6),
            "similarity": round(float(sim_matrix[i, i + 1]), 6),
        }
        for i in range(num_layers - 1)
    ]
    adjacent_gaps_sorted = sorted(adjacent_gaps, key=lambda x: x["distance"], reverse=True)

    layer_stats = []
    diag_vals = np.diag(sim_matrix)
    diag_ranks = rank01(diag_vals.tolist())
    for layer_idx in range(num_layers):
        layer_stats.append({
            "layer": layer_idx,
            "mean_adjacent_similarity": round(
                float(
                    np.mean([
                        sim_matrix[layer_idx, j]
                        for j in (layer_idx - 1, layer_idx + 1)
                        if 0 <= j < num_layers
                    ])
                ),
                6,
            ) if num_layers > 1 else 1.0,
            "self_similarity": round(float(diag_vals[layer_idx]), 6),
            "self_similarity_rank01": round(float(diag_ranks[layer_idx]), 6),
            "feature_norm_mean": round(float(np.mean(feature_norms[layer_idx])), 6),
        })

    matrix_payload = {
        "config": {
            "source": args.source,
            "max_samples": args.max_samples,
            "seed": args.seed,
            "device": args.device,
            "internvl_path": args.internvl_path,
        },
        "summary": {
            "num_samples": len(records),
            "num_layers": num_layers,
            "grouping_method": "largest_adjacent_similarity_gaps_with_contiguous_groups",
            "split_info": split_info,
            "groups": groups,
        },
        "adjacent_gaps_ranked": adjacent_gaps_sorted,
        "layer_stats": layer_stats,
        "similarity_matrix": [[round(float(v), 6) for v in row] for row in sim_matrix],
        "sample_manifest": sample_manifest,
    }

    groups_payload = {
        "config": matrix_payload["config"],
        "summary": matrix_payload["summary"],
        "groups": groups,
        "selected_boundaries": split_info["splits"],
    }

    matrix_path = os.path.join(args.output_dir, "layer_similarity_matrix.json")
    groups_path = os.path.join(args.output_dir, "layer_groups.json")
    heatmap_path = os.path.join(args.output_dir, "layer_similarity_heatmap.png")

    with open(matrix_path, "w") as f:
        json.dump(matrix_payload, f, indent=2)
    with open(groups_path, "w") as f:
        json.dump(groups_payload, f, indent=2)
    save_heatmap(sim_matrix, heatmap_path)

    print(f"[DONE] Wrote matrix -> {matrix_path}")
    print(f"[DONE] Wrote groups -> {groups_path}")
    print(f"[DONE] Wrote heatmap -> {heatmap_path}")
    print(f"[DONE] Groups: {groups}")


if __name__ == "__main__":
    main()
