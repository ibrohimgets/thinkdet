#!/usr/bin/env python3
"""Replot a layer-probe curve with minimal styling and highlighted layers."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt


ROOT = "/home/iibrohimm/project/next_step"
DEFAULT_INPUT = (
    f"{ROOT}/thinkdet/results/layer_probe_flickr_val/layer_probe_results.json"
)
DEFAULT_OUTPUT = (
    f"{ROOT}/thinkdet/results/layer_probe_flickr_val/layer_probe_focus_8_9_10.png"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--highlight_layers", type=int, nargs="+", default=[8, 9, 10])
    return parser.parse_args()


def load_rows(path: str) -> List[Dict]:
    with open(path, "r") as f:
        payload = json.load(f)
    rows = payload["layers"]
    rows.sort(key=lambda row: int(row["layer"]))
    return rows


def ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def build_lookup(rows: Sequence[Dict]) -> Dict[int, Dict]:
    return {int(row["layer"]): row for row in rows}


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input)
    ensure_dir(args.output)

    layers = [int(row["layer"]) for row in rows]
    scores = [float(row["composite_score"]) for row in rows]
    lookup = build_lookup(rows)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig, ax = plt.subplots(figsize=(9.2, 5.2), dpi=180)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.plot(
        layers,
        scores,
        color="#4f83cc",
        linewidth=2.2,
        marker="o",
        markersize=3.5,
        markerfacecolor="white",
        markeredgewidth=0.9,
        alpha=0.95,
        zorder=2,
    )

    highlight_palette = {
        8: "#f39c12",
        9: "#2ca02c",
        10: "#d62728",
    }

    for layer in args.highlight_layers:
        row = lookup.get(int(layer))
        if row is None:
            continue
        x = int(row["layer"])
        y = float(row["composite_score"])
        color = highlight_palette.get(x, "#222222")

        ax.scatter(
            [x],
            [y],
            s=78,
            color=color,
            edgecolors="white",
            linewidths=0.8,
            zorder=4,
        )

        if x == 9:
            dx, dy = 0.55, 0.020
        elif x < 9:
            dx, dy = -1.15, 0.018
        else:
            dx, dy = 0.35, -0.030

        ax.annotate(
            f"L{x}: {y:.6f}",
            xy=(x, y),
            xytext=(x + dx, y + dy),
            fontsize=10,
            color=color,
            fontweight="bold",
            arrowprops={
                "arrowstyle": "-",
                "color": color,
                "lw": 1.0,
                "alpha": 0.9,
            },
            zorder=5,
        )

    ax.set_xlim(min(layers) - 0.5, max(layers) + 0.5)
    y_min = min(scores)
    y_max = max(scores)
    y_pad = max((y_max - y_min) * 0.12, 0.03)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)

    ax.set_xticks(layers)
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Composite Score", fontsize=12)
    ax.grid(True, axis="both", color="#d9d9d9", linewidth=0.8, alpha=0.75)

    fig.tight_layout()
    fig.savefig(args.output, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {args.output}")


if __name__ == "__main__":
    main()
