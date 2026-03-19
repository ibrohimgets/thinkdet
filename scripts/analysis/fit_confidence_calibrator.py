#!/usr/bin/env python3
"""Fit a post-hoc confidence calibrator for ThinkDet per-sample outputs.

The calibrator is a logistic model over primary ThinkDet score diagnostics:
    calibrated_conf = sigmoid(w * z(features) + b)

By default it uses:
    - top1
    - margin
    - gate_mean

Inputs must be eval JSONs with `per_sample` records containing:
    row["thinkdet"]["stats"][...]
    row["thinkdet"]["top1_correct"]

Optionally writes an annotated copy of the input JSON with calibrated
confidence attached to each per-sample ThinkDet record.
"""

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


DEFAULT_FEATURES = ("top1", "margin", "gate_mean")


@dataclass
class SampleRecord:
    key: str
    features: List[float]
    label: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+", required=True, help="Eval JSON path(s)")
    parser.add_argument(
        "--features",
        default=",".join(DEFAULT_FEATURES),
        help="Comma-separated feature names from thinkdet.stats",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Where to save fitted calibrator JSON",
    )
    parser.add_argument(
        "--annotate_input",
        default="",
        help="If set, write an annotated copy of the first input JSON here",
    )
    parser.add_argument(
        "--max_iters",
        type=int,
        default=200,
        help="LBFGS iterations",
    )
    parser.add_argument(
        "--l2",
        type=float,
        default=1e-3,
        help="L2 regularization on weights",
    )
    parser.add_argument(
        "--ece_bins",
        type=int,
        default=10,
        help="Bins for Expected Calibration Error",
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def parse_features(text: str) -> List[str]:
    features = [part.strip() for part in str(text).split(",") if part.strip()]
    if not features:
        raise ValueError("At least one feature is required")
    return features


def build_record_key(row: Dict[str, Any], source_tag: str, fallback_index: int) -> str:
    for field in ("benchmark_id", "sample_id"):
        value = row.get(field)
        if value is not None:
            return f"{field}:{value}"
    return f"{source_tag}:{fallback_index}"


def extract_records(
    payload: Dict[str, Any],
    feature_names: Sequence[str],
    source_tag: str,
) -> List[SampleRecord]:
    rows = payload.get("per_sample") or []
    records: List[SampleRecord] = []
    for idx, row in enumerate(rows):
        thinkdet = row.get("thinkdet")
        if not isinstance(thinkdet, dict):
            continue
        stats = thinkdet.get("stats")
        if not isinstance(stats, dict):
            continue
        if thinkdet.get("top1_correct") is None:
            continue

        feats: List[float] = []
        missing = False
        for name in feature_names:
            value = stats.get(name)
            if value is None:
                missing = True
                break
            feats.append(float(value))
        if missing:
            continue

        records.append(
            SampleRecord(
                key=build_record_key(row, source_tag, idx),
                features=feats,
                label=int(thinkdet["top1_correct"]),
            )
        )
    return records


def dedupe_records(records: Iterable[SampleRecord]) -> Tuple[List[SampleRecord], int]:
    deduped: List[SampleRecord] = []
    seen = set()
    skipped = 0
    for record in records:
        if record.key in seen:
            skipped += 1
            continue
        seen.add(record.key)
        deduped.append(record)
    return deduped, skipped


def tensorize(records: Sequence[SampleRecord]) -> Tuple[torch.Tensor, torch.Tensor]:
    x = torch.tensor([r.features for r in records], dtype=torch.float64)
    y = torch.tensor([r.label for r in records], dtype=torch.float64)
    return x, y


def standardize_features(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = x.mean(dim=0)
    std = x.std(dim=0, unbiased=False)
    std = torch.where(std < 1e-8, torch.ones_like(std), std)
    z = (x - mean) / std
    return z, mean, std


def fit_logistic_calibrator(
    x: torch.Tensor,
    y: torch.Tensor,
    max_iters: int,
    l2: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    weights = torch.zeros(x.shape[1], dtype=torch.float64, requires_grad=True)
    bias = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [weights, bias],
        max_iter=max_iters,
        tolerance_grad=1e-12,
        tolerance_change=1e-12,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        logits = x.matmul(weights) + bias
        loss = F.binary_cross_entropy_with_logits(logits, y)
        if l2 > 0:
            loss = loss + l2 * weights.pow(2).sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    return weights.detach(), bias.detach()


def sigmoid_probs(x: torch.Tensor, weights: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x.matmul(weights) + bias)


def clamp_probs(p: torch.Tensor) -> torch.Tensor:
    return p.clamp(min=1e-6, max=1.0 - 1e-6)


def log_loss(p: torch.Tensor, y: torch.Tensor) -> float:
    q = clamp_probs(p)
    return float((-(y * torch.log(q) + (1.0 - y) * torch.log(1.0 - q))).mean().item())


def brier_score(p: torch.Tensor, y: torch.Tensor) -> float:
    return float(((p - y) ** 2).mean().item())


def ece_score(p: torch.Tensor, y: torch.Tensor, bins: int) -> float:
    q = clamp_probs(p)
    edges = torch.linspace(0.0, 1.0, steps=bins + 1, dtype=torch.float64)
    total = float(len(q))
    ece = 0.0
    for i in range(bins):
        lo = edges[i]
        hi = edges[i + 1]
        if i == bins - 1:
            mask = (q >= lo) & (q <= hi)
        else:
            mask = (q >= lo) & (q < hi)
        count = int(mask.sum().item())
        if count == 0:
            continue
        acc = float(y[mask].mean().item())
        conf = float(q[mask].mean().item())
        ece += abs(acc - conf) * (count / total)
    return float(ece)


def auc_score(p: torch.Tensor, y: torch.Tensor) -> Optional[float]:
    pos = p[y > 0.5]
    neg = p[y <= 0.5]
    n_pos = int(pos.numel())
    n_neg = int(neg.numel())
    if n_pos == 0 or n_neg == 0:
        return None
    better = 0.0
    for value in pos.tolist():
        better += sum(1.0 for other in neg.tolist() if value > other)
        better += 0.5 * sum(1.0 for other in neg.tolist() if value == other)
    return float(better / (n_pos * n_neg))


def summarize_prob_stats(p: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
    correct = p[y > 0.5]
    incorrect = p[y <= 0.5]
    out = {
        "mean": float(p.mean().item()),
        "p50": float(p.median().item()),
    }
    if correct.numel() > 0:
        out["mean_on_correct"] = float(correct.mean().item())
    if incorrect.numel() > 0:
        out["mean_on_incorrect"] = float(incorrect.mean().item())
    return out


def build_metrics(
    raw_primary: torch.Tensor,
    calibrated: torch.Tensor,
    labels: torch.Tensor,
    ece_bins: int,
) -> Dict[str, Any]:
    return {
        "n_samples": int(labels.numel()),
        "positive_rate": float(labels.mean().item()),
        "raw_top1": {
            "log_loss": log_loss(raw_primary, labels),
            "brier": brier_score(raw_primary, labels),
            "ece": ece_score(raw_primary, labels, ece_bins),
            "auc": auc_score(raw_primary, labels),
            "prob_stats": summarize_prob_stats(raw_primary, labels),
        },
        "calibrated": {
            "log_loss": log_loss(calibrated, labels),
            "brier": brier_score(calibrated, labels),
            "ece": ece_score(calibrated, labels, ece_bins),
            "auc": auc_score(calibrated, labels),
            "prob_stats": summarize_prob_stats(calibrated, labels),
        },
    }


def apply_calibrator_to_payload(
    payload: Dict[str, Any],
    feature_names: Sequence[str],
    means: Sequence[float],
    stds: Sequence[float],
    weights: Sequence[float],
    bias: float,
) -> Dict[str, Any]:
    rows = payload.get("per_sample") or []
    means_t = torch.tensor(means, dtype=torch.float64)
    stds_t = torch.tensor(stds, dtype=torch.float64)
    weights_t = torch.tensor(weights, dtype=torch.float64)
    bias_t = torch.tensor([bias], dtype=torch.float64)

    for row in rows:
        thinkdet = row.get("thinkdet")
        if not isinstance(thinkdet, dict):
            continue
        stats = thinkdet.get("stats")
        if not isinstance(stats, dict):
            continue
        values = []
        missing = False
        for name in feature_names:
            value = stats.get(name)
            if value is None:
                missing = True
                break
            values.append(float(value))
        if missing:
            continue
        x = torch.tensor(values, dtype=torch.float64)
        z = (x - means_t) / stds_t
        prob = float(torch.sigmoid(z.dot(weights_t) + bias_t[0]).item())
        thinkdet["calibrated_confidence"] = prob

    payload.setdefault("calibration", {})
    payload["calibration"]["feature_names"] = list(feature_names)
    payload["calibration"]["source"] = "fit_confidence_calibrator.py"
    payload["calibration"]["field"] = "thinkdet.calibrated_confidence"
    return payload


def main() -> None:
    args = parse_args()
    feature_names = parse_features(args.features)
    input_paths = [Path(path) for path in args.input]

    all_records: List[SampleRecord] = []
    for path in input_paths:
        payload = load_json(path)
        all_records.extend(extract_records(payload, feature_names, str(path)))

    records, duplicates_skipped = dedupe_records(all_records)
    if not records:
        raise RuntimeError("No usable ThinkDet per-sample records found")

    x_raw, y = tensorize(records)
    z, means, stds = standardize_features(x_raw)
    weights, bias = fit_logistic_calibrator(z, y, max_iters=args.max_iters, l2=args.l2)
    calibrated = sigmoid_probs(z, weights, bias)
    raw_primary = clamp_probs(x_raw[:, 0])

    payload = {
        "status": "ok",
        "inputs": [str(path) for path in input_paths],
        "duplicates_skipped": int(duplicates_skipped),
        "feature_names": list(feature_names),
        "feature_means": [float(v) for v in means.tolist()],
        "feature_stds": [float(v) for v in stds.tolist()],
        "weights": [float(v) for v in weights.tolist()],
        "bias": float(bias.item()),
        "metrics": build_metrics(raw_primary, calibrated, y, args.ece_bins),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")

    print(json.dumps(payload, indent=2))
    print(f"[saved] {output_path}")

    if args.annotate_input:
        annotated_payload = load_json(input_paths[0])
        annotated_payload = apply_calibrator_to_payload(
            annotated_payload,
            feature_names=feature_names,
            means=payload["feature_means"],
            stds=payload["feature_stds"],
            weights=payload["weights"],
            bias=payload["bias"],
        )
        annotated_path = Path(args.annotate_input)
        annotated_path.parent.mkdir(parents=True, exist_ok=True)
        with annotated_path.open("w") as f:
            json.dump(annotated_payload, f, indent=2)
            f.write("\n")
        print(f"[saved] {annotated_path}")


if __name__ == "__main__":
    main()
