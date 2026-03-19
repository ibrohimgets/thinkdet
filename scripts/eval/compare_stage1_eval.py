#!/usr/bin/env python3
"""
Compare clean Stage-1 evaluation outputs and produce delta vs baseline.

By default, both input runs must have zero runtime errors.
"""

import argparse
import json
import os
import time


METRIC_KEYS = [
    "mAP",
    "mAP_50",
    "mAP_75",
    "mAP_S",
    "mAP_M",
    "mAP_L",
    "mAR_1",
    "mAR_10",
    "mAR_100",
    "num_predictions",
    "num_images",
    "eval_time_min",
]


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def extract_metrics(doc):
    if "metrics" in doc:
        return doc["metrics"]
    raise KeyError(f"Missing 'metrics' in {doc}")


def extract_runtime_errors(doc):
    runtime = doc.get("runtime", {})
    return int(runtime.get("errors_sum", 0))


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, type=str)
    ap.add_argument("--trained", required=True, type=str)
    ap.add_argument("--output", required=True, type=str)
    ap.add_argument(
        "--allow_errors",
        action="store_true",
        help="Allow comparison even if runtime errors were reported.",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    base_doc = load_json(args.baseline)
    train_doc = load_json(args.trained)

    base_errors = extract_runtime_errors(base_doc)
    train_errors = extract_runtime_errors(train_doc)
    if not args.allow_errors and (base_errors > 0 or train_errors > 0):
        raise SystemExit(
            f"Refusing comparison: runtime errors detected "
            f"(baseline={base_errors}, trained={train_errors})."
        )

    base_metrics = extract_metrics(base_doc)
    train_metrics = extract_metrics(train_doc)

    delta = {}
    for key in METRIC_KEYS:
        b = base_metrics.get(key)
        t = train_metrics.get(key)
        if isinstance(b, (int, float)) and isinstance(t, (int, float)):
            delta[key] = t - b

    out = {
        "status": "ok",
        "baseline_path": args.baseline,
        "trained_path": args.trained,
        "baseline_mode": base_doc.get("mode"),
        "trained_mode": train_doc.get("mode"),
        "baseline_metrics": base_metrics,
        "trained_metrics": train_metrics,
        "delta_trained_minus_baseline": delta,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)

    print(f"[done] {args.output}")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

