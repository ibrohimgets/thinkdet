#!/usr/bin/env python
import argparse
import json
import os
import re
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize trained MLLM-layer functional-query eval outputs."
    )
    parser.add_argument("results", nargs="+", help="Result JSON files or directories")
    parser.add_argument("--baseline_layer", type=int, default=9)
    return parser.parse_args()


def iter_result_files(paths):
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            yield from sorted(path.glob("layer*.json"))
        else:
            yield path


def layer_from_payload(path, payload):
    layers = payload.get("ckpt_meta", {}).get("extract_layers") or []
    if len(layers) == 1:
        return int(layers[0])
    match = re.search(r"layer(\d+)", path.stem)
    if match:
        return int(match.group(1))
    raise ValueError(f"Could not infer layer from {path}")


def pct(value):
    return 100.0 * float(value)


def main():
    args = parse_args()
    rows = []
    seen = set()

    for path in iter_result_files(args.results):
        if not path.exists():
            continue
        with path.open("r") as f:
            payload = json.load(f)
        layer = layer_from_payload(path, payload)
        if layer in seen:
            raise ValueError(f"Duplicate result for layer {layer}")
        seen.add(layer)
        summary = payload["summary"]
        rows.append(
            {
                "layer": layer,
                "hit50_top1": float(summary["hit@0.5_top1"]),
                "hit50_top5": float(summary["hit@0.5_topk"]),
                "mean_iou_top1": float(summary["mean_best_iou_top1"]),
                "mean_iou_top5": float(summary["mean_best_iou_topk"]),
                "n_samples": int(summary["n_samples"]),
                "path": str(path),
            }
        )

    rows.sort(key=lambda r: r["layer"])
    if not rows:
        raise SystemExit("No result JSON files found.")

    ranked = sorted(rows, key=lambda r: (-r["hit50_top1"], -r["hit50_top5"], r["layer"]))
    winner = ranked[0]
    baseline = next((r for r in rows if r["layer"] == args.baseline_layer), None)

    lines = [
        "| Layer | Hit@0.5 Top-1 | Hit@0.5 Top-5 | Mean IoU Top-1 | Mean IoU Top-5 | N |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['layer']} | {pct(row['hit50_top1']):.3f}% | "
            f"{pct(row['hit50_top5']):.3f}% | {row['mean_iou_top1']:.6f} | "
            f"{row['mean_iou_top5']:.6f} | {row['n_samples']} |"
        )

    lines.append("")
    lines.append(
        f"Winner: layer {winner['layer']} "
        f"(Top-1 {pct(winner['hit50_top1']):.3f}%, Top-5 {pct(winner['hit50_top5']):.3f}%)."
    )
    if baseline is not None:
        lines.append(
            f"Delta vs layer {args.baseline_layer}: "
            f"Top-1 {pct(winner['hit50_top1'] - baseline['hit50_top1']):+.3f} pp, "
            f"Top-5 {pct(winner['hit50_top5'] - baseline['hit50_top5']):+.3f} pp."
        )

    out_dir = Path(args.results[0]) if len(args.results) == 1 else Path.cwd()
    if not out_dir.is_dir():
        out_dir = out_dir.parent
    summary = {
        "rows": rows,
        "winner": winner,
        "baseline_layer": args.baseline_layer,
        "delta_vs_baseline": None
        if baseline is None
        else {
            "hit50_top1": winner["hit50_top1"] - baseline["hit50_top1"],
            "hit50_top5": winner["hit50_top5"] - baseline["hit50_top5"],
        },
    }

    md_path = out_dir / "summary.md"
    json_path = out_dir / "summary.json"
    md_path.write_text("\n".join(lines) + "\n")
    json_path.write_text(json.dumps(summary, indent=2) + "\n")

    print("\n".join(lines))
    print(f"\nSaved: {md_path}")
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
