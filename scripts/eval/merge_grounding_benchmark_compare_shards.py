#!/usr/bin/env python3
"""Merge sharded outputs from eval_grounding_benchmark_compare.py."""

import argparse
import csv
import json
import os
from collections import Counter, defaultdict


FALLBACK_STAGE_ORDER = [
    "primary",
    "llm_feedback",
    "prompt_refine",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True, type=str)
    p.add_argument("shards", nargs="+")
    return p.parse_args()


def summarize_rows(rows, top_k):
    n = max(len(rows), 1)
    return {
        "n_samples": len(rows),
        "mean_best_iou_top1": sum(r["top1_iou"] for r in rows) / n,
        "mean_best_iou_topk": sum(r["topk_best_iou"] for r in rows) / n,
        "hit@0.5_top1": sum(r["hit50_top1"] for r in rows) / n,
        "hit@0.5_topk": sum(r["hit50_topk"] for r in rows) / n,
        "hit@0.75_top1": sum(r["hit75_top1"] for r in rows) / n,
        "hit@0.75_topk": sum(r["hit75_topk"] for r in rows) / n,
        "mean_top1_score": sum(r["top1_score"] for r in rows) / n,
        "top_k": int(top_k),
    }


def save_markdown_and_csv(output_json, payload):
    base = os.path.splitext(output_json)[0]
    md_path = base + ".md"
    csv_path = base + ".csv"

    with open(md_path, "w") as f:
        f.write("# Grounding Benchmark Comparison\n\n")
        f.write(f"- benchmark: `{payload['benchmark_path']}`\n")
        f.write(f"- benchmark_name: `{payload.get('benchmark_name', '')}`\n")
        f.write(f"- split: `{payload['split']}`\n")
        f.write(f"- n_samples: {payload['n_samples']}\n")
        f.write(f"- top_k: {payload['top_k']}\n\n")

        f.write("## Overall\n\n")
        f.write("| model | hit@0.5_top1 | hit@0.5_topk | mean_best_iou_top1 | mean_best_iou_topk |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for model_name in ["baseline", "thinkdet", "thinkdet_fallback"]:
            row = payload["results"][model_name]["overall"]
            f.write(
                f"| {model_name} | {row['hit@0.5_top1']:.4f} | {row['hit@0.5_topk']:.4f} | "
                f"{row['mean_best_iou_top1']:.4f} | {row['mean_best_iou_topk']:.4f} |\n"
            )

        f.write("\n## Fallback Stage Counts\n\n")
        for stage in FALLBACK_STAGE_ORDER:
            f.write(f"- {stage}: {payload['fallback_stage_counts'].get(stage, 0)}\n")

    with open(csv_path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow([
            "model",
            "n_samples",
            "hit@0.5_top1",
            "hit@0.5_topk",
            "mean_best_iou_top1",
            "mean_best_iou_topk",
            "mean_top1_score",
        ])
        for model_name in ["baseline", "thinkdet", "thinkdet_fallback"]:
            row = payload["results"][model_name]["overall"]
            wr.writerow([
                model_name,
                row["n_samples"],
                f"{row['hit@0.5_top1']:.6f}",
                f"{row['hit@0.5_topk']:.6f}",
                f"{row['mean_best_iou_top1']:.6f}",
                f"{row['mean_best_iou_topk']:.6f}",
                f"{row['mean_top1_score']:.6f}",
            ])

    return md_path, csv_path


def main():
    args = parse_args()
    docs = []
    for path in args.shards:
        with open(path, "r") as f:
            docs.append(json.load(f))
    if not docs:
        raise RuntimeError("No shard files provided")

    meta = docs[0]
    top_k = int(meta["top_k"])

    per_sample = []
    seen = set()
    group_name_map = {}
    fallback_stage_counts = Counter()
    per_group_rows = defaultdict(lambda: {"baseline": [], "thinkdet": [], "thinkdet_fallback": []})
    mode_rows = {"baseline": [], "thinkdet": [], "thinkdet_fallback": []}

    for doc in docs:
        group_name_map.update(doc.get("group_name_map", {}))
        fallback_stage_counts.update(doc.get("fallback_stage_counts", {}))
        for row in doc.get("per_sample", []):
            key = row.get("benchmark_id")
            if key in seen:
                continue
            seen.add(key)
            per_sample.append(row)
            group_id = row.get("group_id")
            for mode_name in ["baseline", "thinkdet", "thinkdet_fallback"]:
                metrics = row[mode_name]
                mode_rows[mode_name].append(metrics)
                if group_id is not None:
                    per_group_rows[group_id][mode_name].append(metrics)

    results = {}
    for mode_name in ["baseline", "thinkdet", "thinkdet_fallback"]:
        results[mode_name] = {
            "overall": summarize_rows(mode_rows[mode_name], top_k),
            "per_group": {},
        }
    for group_id, rows_by_mode in sorted(per_group_rows.items()):
        for mode_name in ["baseline", "thinkdet", "thinkdet_fallback"]:
            results[mode_name]["per_group"][group_id] = summarize_rows(rows_by_mode[mode_name], top_k)

    payload = {
        "status": "ok",
        "benchmark_path": meta["benchmark_path"],
        "benchmark_name": meta.get("benchmark_name"),
        "benchmark_version": meta.get("benchmark_version"),
        "split": meta["split"],
        "n_samples": len(per_sample),
        "top_k": top_k,
        "thinkdet_checkpoint": meta.get("thinkdet_checkpoint"),
        "checkpoint_meta": meta.get("checkpoint_meta"),
        "fallback_config": meta.get("fallback_config"),
        "group_name_map": group_name_map,
        "results": results,
        "fallback_stage_counts": {stage: int(fallback_stage_counts.get(stage, 0)) for stage in FALLBACK_STAGE_ORDER},
        "per_sample": per_sample,
        "source_shards": args.shards,
    }

    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)

    md_path, csv_path = save_markdown_and_csv(args.output, payload)
    print(f"[saved] {args.output}")
    print(f"[saved] {md_path}")
    print(f"[saved] {csv_path}")


if __name__ == "__main__":
    main()
