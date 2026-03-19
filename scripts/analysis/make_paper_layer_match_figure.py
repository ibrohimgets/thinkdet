#!/usr/bin/env python3
"""Build a paper-style layer ablation figure from measured downstream evals.

This figure uses the no-retrain extraction-layer override sweep that already
exists under `thinkdet/results/layer_ablation` and pairs it with saved baseline
references from full downstream evals.

It intentionally keeps the plot focused on measured single-layer behavior. Any
separate trained fused checkpoint results are summarized in the markdown output
instead of being mixed into the curve.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt


ROOT = "/home/iibrohimm/project/next_step"
DEFAULT_OUT_DIR = f"{ROOT}/thinkdet/results/layer_ablation"
DEFAULT_LAYERS = [0, 4, 8, 9, 10, 13, 20, 27]

DATASETS = {
    "flickr_val": {
        "title": "Flickr30k Entities Val",
        "filename": "flickr_val_layer{layer}.json",
        "metric_path": ("summary", "top1_acc_iou50"),
        "metric_label": "Hit@0.5 Top-1 (%)",
        "color": "#1f4e79",
        "baseline_path": (
            f"{ROOT}/thinkdet/results/eval/"
            "fused_flickr_fallback_balanced_20260316_204655/"
            "flickr30k_grounding_compare_fused_fallback_balanced_20260316_204655.json"
        ),
        "baseline_metric_path": ("results", "baseline", "overall", "hit@0.5_top1"),
        "baseline_label": "Baseline",
        "trained_reference_path": (
            f"{ROOT}/thinkdet/results/eval/"
            "fused_flickr_fallback_balanced_20260316_204655/"
            "flickr30k_grounding_compare_fused_fallback_balanced_20260316_204655.json"
        ),
        "trained_reference_metric_path": ("results", "thinkdet", "overall", "hit@0.5_top1"),
        "trained_reference_label": "Trained fused 8-9-10",
    },
    "affordance": {
        "title": "Affordance Held-Out Test",
        "filename": "affordance_layer{layer}.json",
        "metric_path": ("summary", "hit@0.5_top1"),
        "metric_label": "Hit@0.5 Top-1 (%)",
        "color": "#8b1e3f",
        "baseline_path": (
            f"{ROOT}/thinkdet/results/eval/"
            "affordance_benchmark_eval_v1_test_corrected_20260308.json"
        ),
        "baseline_metric_path": ("results", "baseline", "overall", "hit@0.5_top1"),
        "baseline_label": "Baseline",
        "trained_reference_path": (
            f"{ROOT}/thinkdet/results/eval/"
            "affordance_refcocoplus_learned_8_9_10_test_20260316.json"
        ),
        "trained_reference_metric_path": ("summary", "hit@0.5_top1"),
        "trained_reference_label": "Trained fused 8-9-10",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default=DEFAULT_OUT_DIR)
    parser.add_argument("--layers", type=int, nargs="+", default=DEFAULT_LAYERS)
    parser.add_argument(
        "--output_stem",
        type=str,
        default="paper_layer_match_figure",
        help="Base filename for the png/pdf/json/md outputs.",
    )
    return parser.parse_args()


def read_json(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def dig(payload: Dict, path: Sequence[str]):
    cur = payload
    for key in path:
        cur = cur[key]
    return cur


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def pct(x: float) -> float:
    return 100.0 * float(x)


def fmt_pct(x: float) -> str:
    return f"{pct(x):.2f}%"


def collect_dataset(results_dir: str, dataset_key: str, layers: Sequence[int]) -> Dict:
    spec = DATASETS[dataset_key]
    points: List[Dict] = []
    for layer in layers:
        path = Path(results_dir) / spec["filename"].format(layer=layer)
        if not path.exists():
            raise FileNotFoundError(f"Missing layer result: {path}")
        payload = read_json(str(path))
        points.append(
            {
                "layer": int(layer),
                "path": str(path),
                "metric": float(dig(payload, spec["metric_path"])),
                "checkpoint": payload.get("checkpoint") or payload.get("thinkdet_checkpoint"),
                "extract_layers": (
                    payload.get("ckpt_meta", {}).get("extract_layers")
                    or payload.get("checkpoint_meta", {}).get("extract_layers")
                ),
            }
        )

    baseline_payload = read_json(spec["baseline_path"])
    baseline = float(dig(baseline_payload, spec["baseline_metric_path"]))

    trained_reference_payload = read_json(spec["trained_reference_path"])
    trained_reference = float(
        dig(trained_reference_payload, spec["trained_reference_metric_path"])
    )

    best = max(points, key=lambda row: row["metric"])
    layer9 = next(row for row in points if int(row["layer"]) == 9)

    return {
        "key": dataset_key,
        "title": spec["title"],
        "metric_label": spec["metric_label"],
        "color": spec["color"],
        "baseline": baseline,
        "baseline_label": spec["baseline_label"],
        "baseline_path": spec["baseline_path"],
        "trained_reference": trained_reference,
        "trained_reference_label": spec["trained_reference_label"],
        "trained_reference_path": spec["trained_reference_path"],
        "points": points,
        "best": best,
        "layer9": layer9,
    }


def annotate_points(ax, xs: Sequence[int], ys: Sequence[float], color: str) -> None:
    y_span = max(ys) - min(ys)
    offset = max(0.25, y_span * 0.12)
    for idx, (x, y) in enumerate(zip(xs, ys)):
        dy = offset if idx % 2 == 0 else -offset
        va = "bottom" if dy > 0 else "top"
        ax.text(
            x,
            y + dy,
            f"{y:.2f}%",
            color=color,
            fontsize=8,
            ha="center",
            va=va,
        )


def plot_panel(ax, dataset: Dict) -> None:
    xs = [row["layer"] for row in dataset["points"]]
    ys = [pct(row["metric"]) for row in dataset["points"]]
    baseline = pct(dataset["baseline"])

    ax.plot(
        xs,
        ys,
        color=dataset["color"],
        marker="o",
        linewidth=2.3,
        markersize=6,
        label="Single-layer override",
    )
    ax.scatter(
        [dataset["best"]["layer"]],
        [pct(dataset["best"]["metric"])],
        color=dataset["color"],
        edgecolors="black",
        linewidths=0.8,
        marker="*",
        s=180,
        zorder=5,
        label=f"Best layer = L{dataset['best']['layer']}",
    )
    ax.axhline(
        baseline,
        color=dataset["color"],
        linestyle="--",
        linewidth=1.6,
        alpha=0.75,
        label=dataset["baseline_label"],
    )

    ax.set_title(dataset["title"], fontsize=12, fontweight="bold")
    ax.set_xlabel("InternVL Extraction Layer")
    ax.set_ylabel(dataset["metric_label"])
    ax.set_xticks(xs)
    ax.grid(True, axis="y", alpha=0.25)

    y_floor = min(min(ys), baseline) - 1.3
    y_ceil = max(max(ys), baseline) + 1.3
    ax.set_ylim(y_floor, y_ceil)

    ax.text(
        xs[0],
        baseline + 0.2,
        f"{dataset['baseline_label']}: {baseline:.2f}%",
        color=dataset["color"],
        fontsize=8,
        ha="left",
        va="bottom",
    )
    ax.text(
        dataset["best"]["layer"],
        pct(dataset["best"]["metric"]) + 0.55,
        f"best L{dataset['best']['layer']}",
        color=dataset["color"],
        fontsize=8,
        ha="center",
        va="bottom",
        fontweight="bold",
    )
    annotate_points(ax, xs, ys, dataset["color"])


def make_figure(out_dir: str, output_stem: str, datasets: Sequence[Dict]) -> Dict[str, str]:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.titlesize": 12,
        }
    )

    fig, axes = plt.subplots(1, len(datasets), figsize=(12.0, 4.8))
    if len(datasets) == 1:
        axes = [axes]

    for ax, dataset in zip(axes, datasets):
        plot_panel(ax, dataset)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle(
        "Layer Match Figure: Measured Downstream Accuracy by Extraction Layer",
        y=0.98,
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.01,
        "Protocol: no-retrain extraction-layer override on the unified epoch-5 checkpoint. "
        "Dashed lines show saved downstream baselines.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=[0.02, 0.06, 0.98, 0.90])

    png_path = str(Path(out_dir) / f"{output_stem}.png")
    pdf_path = str(Path(out_dir) / f"{output_stem}.pdf")
    fig.savefig(png_path, dpi=220)
    fig.savefig(pdf_path)
    plt.close(fig)
    return {"png": png_path, "pdf": pdf_path}


def write_summary(
    out_dir: str,
    output_stem: str,
    layers: Sequence[int],
    datasets: Sequence[Dict],
    figure_paths: Dict[str, str],
) -> Dict[str, str]:
    summary = {
        "status": "ok",
        "layers": list(layers),
        "datasets": {},
        "figures": figure_paths,
    }

    lines = [
        "# Paper Layer Match Figure Summary",
        "",
        "## Protocol",
        "",
        "- Measured no-retrain extraction-layer override sweep on the unified epoch-5 checkpoint.",
        "- Layers tested: `" + ", ".join(str(x) for x in layers) + "`.",
        "- Dashed baselines come from separately saved downstream eval summaries.",
        "- `Trained fused 8-9-10` references are reported in the table below but are not mixed into the plotted curve.",
        "",
        "## Outputs",
        "",
        f"- PNG: `{figure_paths['png']}`",
        f"- PDF: `{figure_paths['pdf']}`",
        "",
        "## Table",
        "",
        "| Dataset | Baseline | Best single layer | Layer 9 | Delta best-baseline | Trained fused 8-9-10 |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for dataset in datasets:
        best = dataset["best"]
        row = {
            "title": dataset["title"],
            "baseline": dataset["baseline"],
            "best_layer": int(best["layer"]),
            "best_metric": float(best["metric"]),
            "layer9_metric": float(dataset["layer9"]["metric"]),
            "delta_best_vs_baseline": float(best["metric"] - dataset["baseline"]),
            "trained_reference": float(dataset["trained_reference"]),
            "sources": {
                "baseline": dataset["baseline_path"],
                "trained_reference": dataset["trained_reference_path"],
                "layer_points": [row["path"] for row in dataset["points"]],
            },
        }
        summary["datasets"][dataset["key"]] = row
        lines.append(
            "| {title} | {baseline} | L{best_layer}: {best_metric} | {layer9} | {delta} | {trained} |".format(
                title=dataset["title"],
                baseline=fmt_pct(dataset["baseline"]),
                best_layer=row["best_layer"],
                best_metric=fmt_pct(row["best_metric"]),
                layer9=fmt_pct(row["layer9_metric"]),
                delta=fmt_pct(row["delta_best_vs_baseline"]),
                trained=fmt_pct(row["trained_reference"]),
            )
        )

    json_path = str(Path(out_dir) / f"{output_stem}.json")
    md_path = str(Path(out_dir) / f"{output_stem}.md")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    return {"json": json_path, "md": md_path}


def main() -> None:
    args = parse_args()
    ensure_dir(args.results_dir)

    datasets = [
        collect_dataset(args.results_dir, dataset_key, args.layers)
        for dataset_key in ("flickr_val", "affordance")
    ]
    figure_paths = make_figure(args.results_dir, args.output_stem, datasets)
    summary_paths = write_summary(
        args.results_dir,
        args.output_stem,
        args.layers,
        datasets,
        figure_paths,
    )

    print(f"[DONE] Figure PNG -> {figure_paths['png']}")
    print(f"[DONE] Figure PDF -> {figure_paths['pdf']}")
    print(f"[DONE] Summary JSON -> {summary_paths['json']}")
    print(f"[DONE] Summary MD -> {summary_paths['md']}")


if __name__ == "__main__":
    main()
