"""
Aggregate per-layer evaluation outputs into the thesis evidence package.

Expected input files under results/layer_ablation:
  - affordance_layer{L}.json
  - flickr_val_layer{L}.json
  - flickr_test_layer{L}.json
  - refcocog_val_layer{L}.json

Outputs:
  - layer_ablation_results.json
  - layer_ablation_summary.md
  - figure_a_layer_vs_score.png
  - figure_b_affordance_buckets.png
"""

import argparse
import json
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_RESULTS_DIR = "/home/iibrohimm/project/next_step/thinkdet/results/layer_ablation"
DEFAULT_LAYERS = [0, 4, 8, 9, 10, 13, 20, 27]
AFFORDANCE_BUCKETS = ["carry_in", "drink_from", "talk_on", "ride", "sit_on"]
DATASETS = {
    "flickr_val": {
        "filename": "flickr_val_layer{layer}.json",
        "success_key": "top1_correct_iou50",
        "summary_path": ("summary", "top1_acc_iou50"),
        "label": "Flickr30k val",
    },
    "flickr_test": {
        "filename": "flickr_test_layer{layer}.json",
        "success_key": "top1_correct_iou50",
        "summary_path": ("summary", "top1_acc_iou50"),
        "label": "Flickr30k test",
    },
    "refcocog_val": {
        "filename": "refcocog_val_layer{layer}.json",
        "success_key": "top1_correct_iou50",
        "summary_path": ("summary", "top1_acc"),
        "label": "RefCOCOg val",
    },
    "affordance": {
        "filename": "affordance_layer{layer}.json",
        "success_key": "top1_correct_iou50",
        "summary_path": ("summary", "hit@0.5_top1"),
        "label": "Affordance",
    },
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--layers", type=int, nargs="+", default=DEFAULT_LAYERS)
    parser.add_argument("--bootstrap_iters", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow_partial", action="store_true")
    return parser.parse_args()


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def bootstrap_ci(binary_values, num_boot=2000, seed=42):
    arr = np.asarray(binary_values, dtype=np.float64)
    if arr.size == 0:
        return {"mean": 0.0, "ci95": [0.0, 0.0]}
    mean = float(arr.mean())
    if arr.size == 1:
        return {"mean": mean, "ci95": [mean, mean]}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(num_boot, arr.size))
    boot = arr[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"mean": mean, "ci95": [float(lo), float(hi)]}


def paired_bootstrap_delta(ref_map, cur_map, num_boot=2000, seed=42):
    common = sorted(set(ref_map) & set(cur_map))
    if not common:
        return {"n": 0, "delta": 0.0, "ci95": [0.0, 0.0]}
    ref = np.asarray([float(ref_map[k]) for k in common], dtype=np.float64)
    cur = np.asarray([float(cur_map[k]) for k in common], dtype=np.float64)
    delta = cur - ref
    mean = float(delta.mean())
    if delta.size == 1:
        return {"n": 1, "delta": mean, "ci95": [mean, mean]}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, delta.size, size=(num_boot, delta.size))
    boot = delta[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"n": int(delta.size), "delta": mean, "ci95": [float(lo), float(hi)]}


def dig(payload, path):
    cur = payload
    for key in path:
        cur = cur[key]
    return cur


def load_dataset_payloads(results_dir, layers, dataset_key, allow_partial=False):
    spec = DATASETS[dataset_key]
    payloads = {}
    missing = []
    for layer in layers:
        path = Path(results_dir) / spec["filename"].format(layer=layer)
        if not path.exists():
            missing.append(str(path))
            continue
        payloads[layer] = read_json(path)
    if missing and not allow_partial:
        raise FileNotFoundError("Missing required ablation files:\n" + "\n".join(missing))
    return payloads


def sample_success_map(payload, success_key):
    return {row["sample_id"]: int(row[success_key]) for row in payload["per_sample"]}


def affordance_bucket_map(payload, bucket):
    return {
        row["benchmark_id"]: int(row["top1_correct_iou50"])
        for row in payload["per_sample"]
        if row["affordance_id"] == bucket
    }


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def infer_protocol(dataset_payloads, available_layers):
    checkpoints = set()
    eval_layers = {}
    checkpoint_layers = {}
    for payloads in dataset_payloads.values():
        for layer in available_layers:
            payload = payloads[layer]
            checkpoints.add(payload["checkpoint"])
            meta = payload.get("ckpt_meta", {})
            eval_layers[layer] = meta.get("extract_layers")
            checkpoint_layers[layer] = meta.get("checkpoint_extract_layers")

    shared_checkpoint = len(checkpoints) == 1
    overrides_active = any(
        eval_layers.get(layer) is not None
        and checkpoint_layers.get(layer) is not None
        and list(eval_layers[layer]) != list(checkpoint_layers[layer])
        for layer in available_layers
    )
    return {
        "mode": "inference_only_layer_override" if shared_checkpoint and overrides_active else "checkpoint_native",
        "shared_checkpoint": next(iter(checkpoints)) if shared_checkpoint else None,
        "eval_extract_layers_by_tested_layer": {
            str(layer): eval_layers.get(layer) for layer in available_layers
        },
        "checkpoint_extract_layers_by_tested_layer": {
            str(layer): checkpoint_layers.get(layer) for layer in available_layers
        },
    }


def make_figure_a(results_dir, layers, dataset_metrics):
    plt.figure(figsize=(9, 5.5))
    for dataset_key in ["flickr_val", "flickr_test", "refcocog_val", "affordance"]:
        y = [100.0 * dataset_metrics[dataset_key]["per_layer"][layer]["mean"] for layer in layers]
        plt.plot(layers, y, marker="o", linewidth=2, label=DATASETS[dataset_key]["label"])
    plt.xlabel("InternVL Layer")
    plt.ylabel("Top-1 Accuracy / Hit@0.5 (%)")
    plt.title("Figure A. Layer vs Score Across Downstream Datasets")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_path = Path(results_dir) / "figure_a_layer_vs_score.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    return str(out_path)


def make_figure_b(results_dir, layers, bucket_metrics):
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.5), sharex=True, sharey=True)
    axes = axes.flatten()
    for ax, bucket in zip(axes, AFFORDANCE_BUCKETS):
        y = [100.0 * bucket_metrics[bucket]["per_layer"][layer]["mean"] for layer in layers]
        ax.plot(layers, y, marker="o", linewidth=2)
        ax.set_title(bucket)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Top-1 Hit@0.5 (%)")
    axes[-1].axis("off")
    fig.suptitle("Figure B. Per-Affordance Bucket Performance by Layer", y=0.98)
    fig.tight_layout()
    out_path = Path(results_dir) / "figure_b_affordance_buckets.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return str(out_path)


def fmt_pct(x):
    return f"{100.0 * float(x):.2f}%"


def fmt_ci(ci):
    return f"[{100.0 * ci[0]:.2f}, {100.0 * ci[1]:.2f}]"


def main():
    args = parse_args()
    ensure_dir(args.results_dir)

    dataset_payloads = {
        key: load_dataset_payloads(args.results_dir, args.layers, key, allow_partial=args.allow_partial)
        for key in DATASETS.keys()
    }
    available_layers = sorted(
        set.intersection(*[
            set(payloads.keys()) for payloads in dataset_payloads.values() if payloads
        ])
    )
    if not available_layers:
        raise RuntimeError("No complete layer set found across datasets.")

    if 9 not in available_layers:
        raise RuntimeError("Layer 9 results are required as the reference layer.")

    dataset_metrics = {}
    for dataset_key, payloads in dataset_payloads.items():
        success_key = DATASETS[dataset_key]["success_key"]
        per_layer = {}
        ref_map = sample_success_map(payloads[9], success_key)
        for layer in available_layers:
            cur_map = sample_success_map(payloads[layer], success_key)
            vals = [cur_map[k] for k in sorted(cur_map)]
            stats = bootstrap_ci(vals, num_boot=args.bootstrap_iters, seed=args.seed + layer)
            stats["summary_value"] = float(dig(payloads[layer], DATASETS[dataset_key]["summary_path"]))
            stats["delta_vs_layer9"] = paired_bootstrap_delta(
                ref_map,
                cur_map,
                num_boot=args.bootstrap_iters,
                seed=args.seed + 1000 + layer,
            )
            per_layer[layer] = stats
        dataset_metrics[dataset_key] = {
            "label": DATASETS[dataset_key]["label"],
            "per_layer": per_layer,
            "checkpoint_by_layer": {
                str(layer): payloads[layer]["checkpoint"] for layer in available_layers
            },
        }

    affordance_payloads = dataset_payloads["affordance"]
    bucket_metrics = {}
    for bucket in AFFORDANCE_BUCKETS:
        per_layer = {}
        ref_map = affordance_bucket_map(affordance_payloads[9], bucket)
        for layer in available_layers:
            cur_map = affordance_bucket_map(affordance_payloads[layer], bucket)
            vals = [cur_map[k] for k in sorted(cur_map)]
            stats = bootstrap_ci(vals, num_boot=args.bootstrap_iters, seed=args.seed + 2000 + layer)
            stats["delta_vs_layer9"] = paired_bootstrap_delta(
                ref_map,
                cur_map,
                num_boot=args.bootstrap_iters,
                seed=args.seed + 3000 + layer,
            )
            per_layer[layer] = stats
        bucket_metrics[bucket] = {"per_layer": per_layer}

    protocol = infer_protocol(dataset_payloads, available_layers)
    fig_a = make_figure_a(args.results_dir, available_layers, dataset_metrics)
    fig_b = make_figure_b(args.results_dir, available_layers, bucket_metrics)

    results_payload = {
        "status": "ok",
        "timestamp": datetime_now(),
        "layers": available_layers,
        "protocol": protocol,
        "datasets": dataset_metrics,
        "affordance_buckets": bucket_metrics,
        "figures": {
            "figure_a": fig_a,
            "figure_b": fig_b,
        },
    }

    json_path = Path(args.results_dir) / "layer_ablation_results.json"
    with open(json_path, "w") as f:
        json.dump(results_payload, f, indent=2)

    md_lines = [
        "# Layer Ablation Summary",
        "",
        "## Protocol",
        "",
        f"- Mode: `{protocol['mode']}`",
    ]
    if protocol["shared_checkpoint"]:
        md_lines.append(f"- Shared checkpoint: `{protocol['shared_checkpoint']}`")
    md_lines.extend([
        "",
        "## Figure Outputs",
        "",
        f"- Figure A: `{fig_a}`",
        f"- Figure B: `{fig_b}`",
        "",
        "## Table 1. Full Numeric Comparison Across Layers",
        "",
        "| layer | Flickr val | Flickr test | RefCOCOg val | Affordance | carry_in | drink_from | talk_on | ride | sit_on |",
        "|---:|---|---|---|---|---|---|---|---|---|",
    ])
    for layer in available_layers:
        md_lines.append(
            "| {layer} | {fv} {fv_ci} | {ft} {ft_ci} | {rv} {rv_ci} | {af} {af_ci} | {carry} | {drink} | {talk} | {ride} | {sit} |".format(
                layer=layer,
                fv=fmt_pct(dataset_metrics["flickr_val"]["per_layer"][layer]["mean"]),
                fv_ci=fmt_ci(dataset_metrics["flickr_val"]["per_layer"][layer]["ci95"]),
                ft=fmt_pct(dataset_metrics["flickr_test"]["per_layer"][layer]["mean"]),
                ft_ci=fmt_ci(dataset_metrics["flickr_test"]["per_layer"][layer]["ci95"]),
                rv=fmt_pct(dataset_metrics["refcocog_val"]["per_layer"][layer]["mean"]),
                rv_ci=fmt_ci(dataset_metrics["refcocog_val"]["per_layer"][layer]["ci95"]),
                af=fmt_pct(dataset_metrics["affordance"]["per_layer"][layer]["mean"]),
                af_ci=fmt_ci(dataset_metrics["affordance"]["per_layer"][layer]["ci95"]),
                carry=fmt_pct(bucket_metrics["carry_in"]["per_layer"][layer]["mean"]),
                drink=fmt_pct(bucket_metrics["drink_from"]["per_layer"][layer]["mean"]),
                talk=fmt_pct(bucket_metrics["talk_on"]["per_layer"][layer]["mean"]),
                ride=fmt_pct(bucket_metrics["ride"]["per_layer"][layer]["mean"]),
                sit=fmt_pct(bucket_metrics["sit_on"]["per_layer"][layer]["mean"]),
            )
        )

    md_lines.extend([
        "",
        "## Delta vs Layer 9",
        "",
        "| layer | Flickr val Δ | Flickr test Δ | RefCOCOg val Δ | Affordance Δ |",
        "|---:|---|---|---|---|",
    ])
    for layer in available_layers:
        md_lines.append(
            "| {layer} | {fv} {fv_ci} | {ft} {ft_ci} | {rv} {rv_ci} | {af} {af_ci} |".format(
                layer=layer,
                fv=fmt_pct(dataset_metrics["flickr_val"]["per_layer"][layer]["delta_vs_layer9"]["delta"]),
                fv_ci=fmt_ci(dataset_metrics["flickr_val"]["per_layer"][layer]["delta_vs_layer9"]["ci95"]),
                ft=fmt_pct(dataset_metrics["flickr_test"]["per_layer"][layer]["delta_vs_layer9"]["delta"]),
                ft_ci=fmt_ci(dataset_metrics["flickr_test"]["per_layer"][layer]["delta_vs_layer9"]["ci95"]),
                rv=fmt_pct(dataset_metrics["refcocog_val"]["per_layer"][layer]["delta_vs_layer9"]["delta"]),
                rv_ci=fmt_ci(dataset_metrics["refcocog_val"]["per_layer"][layer]["delta_vs_layer9"]["ci95"]),
                af=fmt_pct(dataset_metrics["affordance"]["per_layer"][layer]["delta_vs_layer9"]["delta"]),
                af_ci=fmt_ci(dataset_metrics["affordance"]["per_layer"][layer]["delta_vs_layer9"]["ci95"]),
            )
        )
    md_lines.extend([
        "",
        "All confidence intervals are bootstrap 95% intervals. Delta rows use paired bootstrap on shared per-sample outcomes against layer 9.",
    ])

    md_path = Path(args.results_dir) / "layer_ablation_summary.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines) + "\n")

    print("Saved:", json_path)
    print("Saved:", md_path)
    print("Saved:", fig_a)
    print("Saved:", fig_b)


def datetime_now():
    from datetime import datetime

    return datetime.now().strftime("%Y%m%d_%H%M%S")


if __name__ == "__main__":
    main()
