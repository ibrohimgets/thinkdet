#!/usr/bin/env python3
"""Summarize score distributions and fallback trigger rates from saved eval JSON."""

import argparse
import json
from typing import Any, Dict, List, Optional


DEFAULT_TOP1_THRESHOLDS = "0.05,0.08,0.10,0.12,0.15,0.20,0.30"
DEFAULT_MARGIN_THRESHOLDS = "0.005,0.01,0.02,0.05"


def parse_float_list(text: str) -> List[float]:
    vals = []
    for part in str(text).split(","):
        part = part.strip()
        if part:
            vals.append(float(part))
    return vals


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def percentile(sorted_vals: List[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    pos = max(0.0, min(1.0, q)) * (len(sorted_vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return float(sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac)


def summarize_values(vals: List[float]) -> Optional[Dict[str, float]]:
    if not vals:
        return None
    ordered = sorted(float(v) for v in vals)
    return {
        "count": len(ordered),
        "min": float(ordered[0]),
        "p10": percentile(ordered, 0.10),
        "p25": percentile(ordered, 0.25),
        "p50": percentile(ordered, 0.50),
        "p75": percentile(ordered, 0.75),
        "p90": percentile(ordered, 0.90),
        "max": float(ordered[-1]),
        "mean": float(sum(ordered) / len(ordered)),
    }


def mean_from_records(records: List[Dict[str, Any]], field: str) -> Optional[float]:
    vals = [float(r[field]) for r in records if r.get(field) is not None]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def normalize_records(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = payload.get("per_sample") or []
    records = []
    for row in rows:
        thinkdet = row.get("thinkdet")
        if isinstance(thinkdet, dict) and isinstance(thinkdet.get("stats"), dict):
            stats = thinkdet["stats"]
            fallback = row.get("fallback") or {}
            records.append(
                {
                    "top1": float(stats["top1"]),
                    "top2": float(stats["top2"]),
                    "top3_mean": float(stats["top3_mean"]),
                    "margin": float(stats["margin"]),
                    "gate_mean": float(stats.get("gate_mean", 0.0)),
                    "reliability": float(stats["reliability"]),
                    "top1_correct": int(thinkdet.get("top1_correct", 0)),
                    "no_det": int(thinkdet.get("no_det", 0)),
                    "selected_stage": fallback.get("selected_stage"),
                    "fallback_top1_correct": (
                        None
                        if fallback.get("top1_correct") is None
                        else int(fallback["top1_correct"])
                    ),
                }
            )
            continue

        if "top1_score" in row:
            records.append(
                {
                    "top1": float(row["top1_score"]),
                    "top2": None,
                    "top3_mean": None,
                    "margin": None,
                    "gate_mean": None,
                    "reliability": None,
                    "top1_correct": int(row.get("top1_correct_iou50", row.get("hit50_top1", 0))),
                    "no_det": None,
                    "selected_stage": row.get("fallback_selected_stage"),
                    "fallback_top1_correct": None,
                }
            )
    return records


def summarize_trigger(records: List[Dict[str, Any]], min_top1: float, min_margin: Optional[float]) -> Dict[str, Any]:
    triggered = []
    kept = []
    for rec in records:
        fire = rec["top1"] < min_top1
        if min_margin is not None and rec.get("margin") is not None:
            fire = fire or (float(rec["margin"]) < min_margin)
        if fire:
            triggered.append(rec)
        else:
            kept.append(rec)

    result = {
        "min_top1": float(min_top1),
        "trigger_rate": float(len(triggered) / max(len(records), 1)),
        "trigger_count": int(len(triggered)),
        "triggered_top1_acc": mean_from_records(triggered, "top1_correct"),
        "kept_top1_acc": mean_from_records(kept, "top1_correct"),
    }
    if min_margin is not None:
        result["min_margin"] = float(min_margin)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--top1_thresholds", default=DEFAULT_TOP1_THRESHOLDS)
    parser.add_argument("--margin_thresholds", default=DEFAULT_MARGIN_THRESHOLDS)
    args = parser.parse_args()

    payload = load_json(args.input)
    records = normalize_records(payload)
    if not records:
        raise RuntimeError(f"No usable per-sample records found in {args.input}")

    top1_vals = [float(r["top1"]) for r in records]
    margin_vals = [float(r["margin"]) for r in records if r.get("margin") is not None]
    gate_vals = [float(r["gate_mean"]) for r in records if r.get("gate_mean") is not None]

    out = {
        "input": args.input,
        "n_samples": len(records),
        "top1_score": summarize_values(top1_vals),
        "margin": summarize_values(margin_vals),
        "gate_mean": summarize_values(gate_vals),
        "overall_top1_acc": mean_from_records(records, "top1_correct"),
        "top1_trigger_sweep": [],
        "top1_margin_trigger_sweep": [],
    }

    for thr in parse_float_list(args.top1_thresholds):
        out["top1_trigger_sweep"].append(summarize_trigger(records, thr, None))

    if margin_vals:
        for top1_thr in parse_float_list(args.top1_thresholds):
            for margin_thr in parse_float_list(args.margin_thresholds):
                out["top1_margin_trigger_sweep"].append(
                    summarize_trigger(records, top1_thr, margin_thr)
                )

    stage_counts: Dict[str, int] = {}
    for rec in records:
        stage = rec.get("selected_stage")
        if stage:
            stage_counts[stage] = stage_counts.get(stage, 0) + 1
    if stage_counts:
        out["selected_stage_counts"] = stage_counts

    rendered = json.dumps(out, indent=2)
    print(rendered)
    if args.output:
        with open(args.output, "w") as f:
            f.write(rendered + "\n")


if __name__ == "__main__":
    main()
