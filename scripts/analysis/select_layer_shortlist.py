#!/usr/bin/env python3
"""
Build a candidate layer shortlist by combining:
1. similarity-derived early / mid / late groups
2. existing all-layer probe results

This is a screening step, not a final benchmark decision rule.
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List

import numpy as np

ROOT = "/home/iibrohimm/project/next_step"
sys.path.insert(0, ROOT)


DEFAULT_GROUPS = f"{ROOT}/thinkdet/results/layer_similarity/layer_groups.json"
DEFAULT_REASONING = [
    f"{ROOT}/thinkdet/results/layer_probe_flickr_val/layer_probe_results.json",
    f"{ROOT}/thinkdet/results/layer_probe_flickr_test/layer_probe_results.json",
    f"{ROOT}/thinkdet/results/refcoco layer results/layer_probe_results.json",
]
DEFAULT_SENSITIVITY = f"{ROOT}/thinkdet/results/layer_sensitivity/sensitivity_results.json"
DEFAULT_OUTPUT = f"{ROOT}/thinkdet/results/layer_selection/shortlist.json"


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


def choose_score_key(layer_record: Dict) -> str:
    for key in ("composite_score", "composite_rank_score", "score", "accuracy", "mean_iou"):
        if key in layer_record:
            return key
    raise KeyError(f"Could not find a score key in layer record keys={list(layer_record.keys())}")


def load_probe_file(path: str) -> Dict[int, Dict]:
    with open(path, "r") as f:
        payload = json.load(f)

    layers = payload.get("layers", [])
    if not layers:
        raise ValueError(f"No layers found in {path}")

    score_key = choose_score_key(layers[0])
    raw_scores = [float(rec[score_key]) for rec in layers]
    rank_scores = rank01(raw_scores)

    out = {}
    for rec, rank_score in zip(layers, rank_scores):
        out[int(rec["layer"])] = {
            "raw_score": float(rec[score_key]),
            "rank01": float(rank_score),
            "score_key": score_key,
        }
    return out


def load_groups(path: str):
    with open(path, "r") as f:
        payload = json.load(f)

    groups = payload["groups"]
    layer_to_group = {}
    for group_name, info in groups.items():
        for layer in info["layers"]:
            layer_to_group[int(layer)] = group_name
    return groups, layer_to_group


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer_groups", type=str, default=DEFAULT_GROUPS)
    parser.add_argument("--reasoning_probe", type=str, nargs="+", default=DEFAULT_REASONING)
    parser.add_argument("--sensitivity_probe", type=str, default=DEFAULT_SENSITIVITY)
    parser.add_argument("--top_k_per_group", type=int, default=2)
    parser.add_argument("--reasoning_weight", type=float, default=0.75)
    parser.add_argument("--sensitivity_weight", type=float, default=0.25)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    groups, layer_to_group = load_groups(args.layer_groups)
    reasoning_runs = [load_probe_file(path) for path in args.reasoning_probe]
    sensitivity_scores = load_probe_file(args.sensitivity_probe)

    all_layers = sorted(layer_to_group.keys())
    per_layer = {}
    for layer in all_layers:
        reasoning_rank_vals = []
        reasoning_raw_vals = []
        for run_idx, run_scores in enumerate(reasoning_runs):
            if layer not in run_scores:
                continue
            reasoning_rank_vals.append(float(run_scores[layer]["rank01"]))
            reasoning_raw_vals.append(float(run_scores[layer]["raw_score"]))

        sensitivity_entry = sensitivity_scores.get(layer)
        if sensitivity_entry is None:
            sensitivity_rank = 0.0
            sensitivity_raw = None
        else:
            sensitivity_rank = float(sensitivity_entry["rank01"])
            sensitivity_raw = float(sensitivity_entry["raw_score"])

        reasoning_rank_mean = float(np.mean(reasoning_rank_vals)) if reasoning_rank_vals else 0.0
        reasoning_raw_mean = float(np.mean(reasoning_raw_vals)) if reasoning_raw_vals else 0.0
        combined = (
            args.reasoning_weight * reasoning_rank_mean
            + args.sensitivity_weight * sensitivity_rank
        )
        per_layer[layer] = {
            "layer": layer,
            "group": layer_to_group[layer],
            "reasoning_rank_mean": round(reasoning_rank_mean, 6),
            "reasoning_raw_mean": round(reasoning_raw_mean, 6),
            "sensitivity_rank": round(sensitivity_rank, 6),
            "sensitivity_raw": None if sensitivity_raw is None else round(sensitivity_raw, 6),
            "combined_score": round(float(combined), 6),
            "reasoning_runs_count": len(reasoning_rank_vals),
        }

    group_rankings = defaultdict(list)
    for layer, info in per_layer.items():
        group_rankings[info["group"]].append(info)

    selected_layers = []
    selected_by_group = {}
    for group_name in ("early", "mid", "late"):
        ranked = sorted(
            group_rankings[group_name],
            key=lambda x: (x["combined_score"], x["reasoning_rank_mean"], x["layer"]),
            reverse=True,
        )
        winners = ranked[: args.top_k_per_group]
        selected_by_group[group_name] = winners
        selected_layers.extend([item["layer"] for item in winners])

    payload = {
        "config": {
            "layer_groups": args.layer_groups,
            "reasoning_probe": args.reasoning_probe,
            "sensitivity_probe": args.sensitivity_probe,
            "top_k_per_group": args.top_k_per_group,
            "reasoning_weight": args.reasoning_weight,
            "sensitivity_weight": args.sensitivity_weight,
        },
        "groups": groups,
        "selected_layers": sorted(set(selected_layers)),
        "selected_by_group": selected_by_group,
        "per_layer": [per_layer[layer] for layer in sorted(per_layer.keys())],
    }

    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"[DONE] Wrote shortlist -> {args.output}")
    print(f"[DONE] Selected layers -> {payload['selected_layers']}")
    for group_name, winners in selected_by_group.items():
        pretty = ", ".join(
            f"L{item['layer']} ({item['combined_score']:.3f})" for item in winners
        )
        print(f"  {group_name}: {pretty}")


if __name__ == "__main__":
    main()
