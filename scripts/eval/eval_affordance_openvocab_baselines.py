"""
Evaluate external open-vocabulary baselines (e.g., OWL-ViT) on the held-out
ThinkDet affordance benchmark and report calibration metrics.

This script is intentionally independent from ThinkDet/GroundingDINO codepaths
so we can benchmark external models with a common metric protocol.
"""

import argparse
import datetime
import json
import math
import os
import sys
from collections import defaultdict

import torch
from PIL import Image

ROOT = "/home/iibrohimm/project/next_step"
sys.path.insert(0, ROOT)


DEFAULT_BENCH = f"{ROOT}/thinkdet/data/benchmarks/affordance_coco_val_heldout_v1.json"
DEFAULT_OUT = (
    f"{ROOT}/thinkdet/results/eval/"
    f"affordance_openvocab_baselines_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", type=str, default=DEFAULT_BENCH)
    p.add_argument("--split", type=str, default="test", choices=["dev", "test", "all"])
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--max_samples", type=int, default=0, help="If >0, evaluate only first N samples after split filter.")
    p.add_argument("--output", type=str, default=DEFAULT_OUT)
    p.add_argument(
        "--models",
        nargs="+",
        default=["owlvit", "glip_t"],
        choices=["owlvit", "glip_t"],
        help="Baseline models to evaluate. glip_t requires local GLIP assets/code.",
    )
    p.add_argument(
        "--owlvit_model",
        type=str,
        default="google/owlvit-base-patch32",
        help="HF model id for OWL-ViT. Must be locally cached when offline.",
    )
    p.add_argument(
        "--hf_local_only",
        action="store_true",
        default=True,
        help="Load HF models from local cache only (default on).",
    )
    p.add_argument(
        "--no_hf_local_only",
        dest="hf_local_only",
        action="store_false",
        help="Allow HF network fetches if available.",
    )
    return p.parse_args()


def iou_xyxy(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (aa + bb - inter + 1e-9)


def xywh_to_xyxy(box):
    x, y, w, h = box
    return [x, y, x + w, y + h]


class MetricAccumulator:
    def __init__(self, top_k):
        self.top_k = top_k
        self.n = 0
        self.sum_best_iou_top1 = 0.0
        self.sum_best_iou_topk = 0.0
        self.hit50_top1 = 0
        self.hit50_topk = 0
        self.hit75_top1 = 0
        self.hit75_topk = 0
        self.prec50_top1_sum = 0.0
        self.prec50_topk_sum = 0.0
        self.rec50_top1_sum = 0.0
        self.rec50_topk_sum = 0.0

    def update(self, pred_boxes, gt_boxes):
        self.n += 1
        k = min(self.top_k, len(pred_boxes))
        if k == 0:
            return

        best_iou_per_pred = []
        for p in pred_boxes[:k]:
            ious = [iou_xyxy(p["box_abs_xyxy"], g) for g in gt_boxes]
            best_iou_per_pred.append(max(ious) if ious else 0.0)

        best1 = best_iou_per_pred[0]
        bestk = max(best_iou_per_pred)
        self.sum_best_iou_top1 += best1
        self.sum_best_iou_topk += bestk

        m50 = [x >= 0.5 for x in best_iou_per_pred]
        m75 = [x >= 0.75 for x in best_iou_per_pred]

        self.hit50_top1 += 1 if m50[0] else 0
        self.hit50_topk += 1 if any(m50) else 0
        self.hit75_top1 += 1 if m75[0] else 0
        self.hit75_topk += 1 if any(m75) else 0
        self.prec50_top1_sum += 1.0 if m50[0] else 0.0
        self.prec50_topk_sum += float(sum(m50)) / float(k)
        self.rec50_top1_sum += 1.0 if m50[0] else 0.0
        self.rec50_topk_sum += 1.0 if any(m50) else 0.0

    def to_dict(self):
        n = max(self.n, 1)
        return {
            "n_samples": self.n,
            "mean_best_iou_top1": self.sum_best_iou_top1 / n,
            "mean_best_iou_topk": self.sum_best_iou_topk / n,
            "hit@0.5_top1": self.hit50_top1 / n,
            "hit@0.5_topk": self.hit50_topk / n,
            "hit@0.75_top1": self.hit75_top1 / n,
            "hit@0.75_topk": self.hit75_topk / n,
            "precision@0.5_top1": self.prec50_top1_sum / n,
            "precision@0.5_topk": self.prec50_topk_sum / n,
            "recall@0.5_top1": self.rec50_top1_sum / n,
            "recall@0.5_topk": self.rec50_topk_sum / n,
        }


def ece_10(scores, labels):
    n = len(scores)
    if n == 0:
        return float("nan")
    edges = [i / 10.0 for i in range(11)]
    ece = 0.0
    bins = []
    for bi in range(10):
        lo, hi = edges[bi], edges[bi + 1]
        if bi < 9:
            idx = [i for i, s in enumerate(scores) if lo <= s < hi]
        else:
            idx = [i for i, s in enumerate(scores) if lo <= s <= hi]
        if not idx:
            bins.append({"bin": bi, "count": 0})
            continue
        conf = sum(scores[i] for i in idx) / len(idx)
        acc = sum(1.0 if labels[i] else 0.0 for i in idx) / len(idx)
        gap = abs(acc - conf)
        ece += (len(idx) / n) * gap
        bins.append({
            "bin": bi,
            "count": len(idx),
            "mean_conf": conf,
            "accuracy": acc,
            "abs_gap": gap,
        })
    return ece, bins


def brier_score(scores, labels):
    n = len(scores)
    if n == 0:
        return float("nan")
    return sum((float(scores[i]) - (1.0 if labels[i] else 0.0)) ** 2 for i in range(n)) / n


def nll_score(scores, labels, eps=1e-12):
    n = len(scores)
    if n == 0:
        return float("nan")
    total = 0.0
    for s, y in zip(scores, labels):
        p = min(max(float(s), eps), 1.0 - eps)
        total += -math.log(p) if y else -math.log(1.0 - p)
    return total / n


def auroc_pairwise(scores, labels):
    pos = [float(s) for s, y in zip(scores, labels) if y]
    neg = [float(s) for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    total = len(pos) * len(neg)
    for sp in pos:
        for sn in neg:
            if sp > sn:
                wins += 1.0
            elif sp == sn:
                wins += 0.5
    return wins / total


def calibration_from_per_sample(per_sample, iou_thresh=0.5):
    scores = [min(max(float(s.get("top1_score", 0.0)), 0.0), 1.0) for s in per_sample]
    labels = [float(s.get("top1_iou", 0.0)) >= iou_thresh for s in per_sample]
    ece, bins = ece_10(scores, labels)
    return {
        "n": len(per_sample),
        "iou_threshold_correct": iou_thresh,
        "accuracy": (sum(1 for y in labels if y) / len(labels)) if labels else float("nan"),
        "mean_score": (sum(scores) / len(scores)) if scores else float("nan"),
        "ece@10": ece,
        "brier": brier_score(scores, labels),
        "nll": nll_score(scores, labels),
        "auroc_score_to_correctness": auroc_pairwise(scores, labels),
        "ece_bins": bins,
    }


def evaluate_model(model_label, infer_topk_fn, samples, args):
    overall = MetricAccumulator(args.top_k)
    per_aff = defaultdict(lambda: MetricAccumulator(args.top_k))
    per_sample = []

    for i, s in enumerate(samples):
        image_pil = Image.open(s["image_path"]).convert("RGB")
        preds = infer_topk_fn(image_pil, s["prompt"], args.top_k)
        gt_boxes = [xywh_to_xyxy(t["bbox_xywh"]) for t in s["positive_targets"]]

        overall.update(preds, gt_boxes)
        per_aff[s["affordance_id"]].update(preds, gt_boxes)

        best_top1 = 0.0
        if preds:
            best_top1 = max(iou_xyxy(preds[0]["box_abs_xyxy"], g) for g in gt_boxes)
        per_sample.append({
            "benchmark_id": s["benchmark_id"],
            "image_id": s["image_id"],
            "affordance_id": s["affordance_id"],
            "prompt": s["prompt"],
            "top1_iou": best_top1,
            "top1_score": preds[0]["score"] if preds else 0.0,
        })

        if (i + 1) % args.log_every == 0:
            cur = overall.to_dict()
            print(
                f"  [{model_label}] {i+1}/{len(samples)} "
                f"hit@0.5_top1={cur['hit@0.5_top1']:.4f} "
                f"miou_top1={cur['mean_best_iou_top1']:.4f}"
            )

    return {
        "overall": overall.to_dict(),
        "per_affordance": {k: v.to_dict() for k, v in sorted(per_aff.items())},
        "per_sample": per_sample,
        "calibration": calibration_from_per_sample(per_sample),
    }


def build_owlvit_runner(model_name, device, hf_local_only):
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
    from transformers import OwlViTForObjectDetection, OwlViTProcessor

    print(f"  Loading OWL-ViT from {model_name} (local_only={hf_local_only})")
    processor = OwlViTProcessor.from_pretrained(model_name, local_files_only=hf_local_only)
    model = OwlViTForObjectDetection.from_pretrained(model_name, local_files_only=hf_local_only)
    model = model.to(device).eval()

    @torch.no_grad()
    def infer_topk(image_pil, prompt, top_k):
        inputs = processor(text=[[prompt]], images=image_pil, return_tensors="pt")
        inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
        outputs = model(**inputs)
        target_sizes = torch.tensor([[image_pil.size[1], image_pil.size[0]]], device=device)

        result = None
        if hasattr(processor, "post_process_grounded_object_detection"):
            try:
                processed = processor.post_process_grounded_object_detection(
                    outputs=outputs,
                    target_sizes=target_sizes,
                    threshold=0.0,
                    text_labels=[[prompt]],
                )
                result = processed[0]
            except TypeError:
                processed = processor.post_process_grounded_object_detection(
                    outputs=outputs,
                    target_sizes=target_sizes,
                    threshold=0.0,
                )
                result = processed[0]
        if result is None:
            processed = processor.post_process_object_detection(
                outputs=outputs,
                target_sizes=target_sizes,
                threshold=0.0,
            )
            result = processed[0]

        boxes = result["boxes"].detach().cpu()
        scores = result["scores"].detach().cpu()

        if boxes.numel() == 0:
            return []

        scores_sorted, idx = torch.sort(scores, descending=True)
        k = min(int(top_k), int(scores_sorted.shape[0]))
        preds = []
        for j in range(k):
            bi = int(idx[j].item())
            x1, y1, x2, y2 = boxes[bi].tolist()
            preds.append({
                "score": float(scores[bi].item()),
                "box_abs_xyxy": [float(x1), float(y1), float(x2), float(y2)],
            })
        return preds

    return infer_topk


def build_glip_t_runner(device):
    raise RuntimeError(
        "GLIP-T runner is not available in this workspace. "
        "No local GLIP-T code+weights were found (offline environment)."
    )


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    with open(args.benchmark, "r") as f:
        bench = json.load(f)

    samples = bench["samples"]
    if args.split != "all":
        samples = [s for s in samples if s.get("split") == args.split]
    if args.max_samples and args.max_samples > 0:
        samples = samples[: args.max_samples]
    if not samples:
        raise RuntimeError(f"No samples for split={args.split}")

    print("=" * 72)
    print("Affordance benchmark: external open-vocab baselines")
    print(f"benchmark: {args.benchmark}")
    print(f"split: {args.split}  n_samples: {len(samples)}  top_k: {args.top_k}")
    if args.max_samples:
        print(f"max_samples: {args.max_samples}")
    print(f"device: {device}")
    print(f"models: {args.models}")
    print("=" * 72)

    payload = {
        "status": "ok",
        "timestamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
        "benchmark_path": args.benchmark,
        "split": args.split,
        "n_samples": len(samples),
        "top_k": args.top_k,
        "device": str(device),
        "requested_models": list(args.models),
        "results": {},
        "errors": {},
        "config": {
            "owlvit_model": args.owlvit_model,
            "hf_local_only": bool(args.hf_local_only),
        },
    }

    for model_key in args.models:
        print(f"\n[{model_key}] Preparing...")
        try:
            if model_key == "owlvit":
                runner = build_owlvit_runner(args.owlvit_model, device, args.hf_local_only)
            elif model_key == "glip_t":
                runner = build_glip_t_runner(device)
            else:
                raise ValueError(f"Unsupported model_key={model_key}")

            print(f"[{model_key}] Evaluating...")
            payload["results"][model_key] = evaluate_model(model_key, runner, samples, args)

            cal = payload["results"][model_key]["calibration"]
            ov = payload["results"][model_key]["overall"]
            print(
                f"[{model_key}] done | hit@0.5_top1={ov['hit@0.5_top1']:.4f} "
                f"ece@10={cal['ece@10']:.4f} brier={cal['brier']:.4f} "
                f"auroc={cal['auroc_score_to_correctness']:.4f}"
            )
        except Exception as e:
            payload["errors"][model_key] = repr(e)
            print(f"[{model_key}] ERROR: {e}")
        finally:
            torch.cuda.empty_cache()

    if payload["errors"] and not payload["results"]:
        payload["status"] = "error"
    elif payload["errors"]:
        payload["status"] = "partial"

    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)

    print("\n" + "=" * 72)
    print(f"Saved: {args.output}")
    print(f"Status: {payload['status']}")
    for k, v in payload["results"].items():
        ov = v["overall"]
        cal = v["calibration"]
        print(
            f"{k:>10} | hit@0.5={ov['hit@0.5_top1']:.4f} "
            f"ECE@10={cal['ece@10']:.4f} Brier={cal['brier']:.4f} "
            f"AUROC={cal['auroc_score_to_correctness']:.4f}"
        )
    if payload["errors"]:
        print("Errors:")
        for k, e in payload["errors"].items():
            print(f"  - {k}: {e}")
    print("=" * 72)


if __name__ == "__main__":
    main()
