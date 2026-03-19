#!/usr/bin/env python3
"""
Run GroundingDINO baseline-only on a commonsense object grounding benchmark.
"""

import argparse
import csv
import datetime
import json
import os
import sys
from collections import defaultdict

import torch
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF

ROOT = "/home/iibrohimm/project/next_step"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "GroundingDINO", "GroundingDINO"))

from groundingdino.util.inference import load_model as load_gd_model
from groundingdino.util.misc import NestedTensor


DEFAULT_BENCH = (
    f"{ROOT}/thinkdet/data/benchmarks/commonsense_refcoco_objects_coco_val_v1.json"
)
DEFAULT_OUT = (
    f"{ROOT}/thinkdet/results/eval/"
    f"commonsense_baseline_eval_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
)
GD_CONFIG = (
    f"{ROOT}/GroundingDINO/GroundingDINO/groundingdino/config/"
    "GroundingDINO_SwinT_OGC.py"
)
GD_WEIGHTS = f"{ROOT}/GroundingDINO/GroundingDINO/weights/groundingdino_swint_ogc.pth"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", type=str, default=DEFAULT_BENCH)
    p.add_argument("--split", type=str, default="test", choices=["test", "all"])
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument("--output", type=str, default=DEFAULT_OUT)
    p.add_argument("--gd_config", type=str, default=GD_CONFIG)
    p.add_argument("--gd_weights", type=str, default=GD_WEIGHTS)
    return p.parse_args()


def build_dino_transform():
    normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def transform(image_pil):
        w, h = image_pil.size
        scale = 800 / min(w, h)
        if scale * max(w, h) > 1333:
            scale = 1333 / max(w, h)
        nw, nh = int(w * scale), int(h * scale)
        img = image_pil.resize((nw, nh), Image.BILINEAR)
        return normalize(TF.to_tensor(img))

    return transform


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


def get_special_tokens(model):
    tokens = getattr(model, "specical_tokens", None)
    if tokens is not None:
        return tokens
    tokens = getattr(model, "special_tokens", None)
    if tokens is not None:
        return tokens
    return model.tokenizer.all_special_ids


def build_positive_map_for_query(tokenizer, special_tokens, query_text, max_text_len=512):
    tokenized = tokenizer(query_text, return_tensors="pt")
    input_ids = tokenized["input_ids"][0]

    positive_map = torch.zeros(1, max_text_len, dtype=torch.float32)
    special_set = set(int(x) for x in special_tokens)
    for pos, tok_id in enumerate(input_ids.tolist()):
        if pos >= max_text_len:
            break
        if tok_id not in special_set:
            positive_map[0, pos] = 1.0

    row_sum = positive_map.sum(dim=-1, keepdim=True).clamp(min=1e-6)
    return positive_map / row_sum


def score_outputs_for_query(outputs, positive_map_norm):
    logits = outputs["pred_logits"][0].clamp(-50, 50)
    boxes = outputs["pred_boxes"][0]
    text_len = logits.shape[-1]
    pmap = positive_map_norm[:, :text_len].to(logits.device)
    probs = logits.sigmoid()
    scores = (probs * pmap).sum(dim=-1)
    return scores, boxes


@torch.no_grad()
def run_topk(model, image_pil, prompt, device, dino_tf, top_k):
    img_w, img_h = image_pil.size
    dino_t = dino_tf(image_pil).unsqueeze(0).to(device)
    mask = torch.zeros(1, dino_t.shape[2], dino_t.shape[3], dtype=torch.bool, device=device)
    nested = NestedTensor(dino_t, mask)
    query_text = prompt if prompt.strip().endswith(".") else (prompt.strip() + " .")
    outputs = model(samples=nested, captions=[query_text])

    positive_map_norm = build_positive_map_for_query(
        model.tokenizer, get_special_tokens(model), query_text, max_text_len=512
    )
    scores, boxes = score_outputs_for_query(outputs, positive_map_norm)
    vals, idx = scores.topk(min(top_k, scores.shape[0]))

    results = []
    for j in range(len(vals)):
        cx, cy, bw, bh = boxes[idx[j]].tolist()
        x1 = (cx - bw / 2) * img_w
        y1 = (cy - bh / 2) * img_h
        x2 = (cx + bw / 2) * img_w
        y2 = (cy + bh / 2) * img_h
        results.append({
            "score": float(vals[j].item()),
            "box_abs_xyxy": [x1, y1, x2, y2],
        })
    return results


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

        self.hit50_top1 += 1 if best1 >= 0.5 else 0
        self.hit50_topk += 1 if bestk >= 0.5 else 0
        self.hit75_top1 += 1 if best1 >= 0.75 else 0
        self.hit75_topk += 1 if bestk >= 0.75 else 0

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
        }


def evaluate(model, samples, args, dino_tf, device):
    overall = MetricAccumulator(args.top_k)
    per_object = defaultdict(lambda: MetricAccumulator(args.top_k))
    per_sample = []

    for i, s in enumerate(samples):
        image_pil = Image.open(s["image_path"]).convert("RGB")
        preds = run_topk(model, image_pil, s["prompt"], device, dino_tf, args.top_k)
        gt_boxes = [xywh_to_xyxy(t["bbox_xywh"]) for t in s["positive_targets"]]

        overall.update(preds, gt_boxes)
        per_object[s["object_id"]].update(preds, gt_boxes)

        best_top1 = 0.0
        if preds:
            best_top1 = max(iou_xyxy(preds[0]["box_abs_xyxy"], g) for g in gt_boxes)
        per_sample.append({
            "benchmark_id": s["benchmark_id"],
            "image_id": s["image_id"],
            "object_id": s["object_id"],
            "object_name": s["object_name"],
            "prompt_index": s["prompt_index"],
            "prompt": s["prompt"],
            "top1_iou": best_top1,
            "top1_score": preds[0]["score"] if preds else 0.0,
        })

        if (i + 1) % args.log_every == 0:
            cur = overall.to_dict()
            print(
                f"  [baseline] {i+1}/{len(samples)} "
                f"hit@0.5_top1={cur['hit@0.5_top1']:.4f} "
                f"miou_top1={cur['mean_best_iou_top1']:.4f}"
            )

    return {
        "overall": overall.to_dict(),
        "per_object": {k: v.to_dict() for k, v in sorted(per_object.items())},
        "per_sample": per_sample,
    }


def save_markdown_and_csv(output_json, payload):
    base = os.path.splitext(output_json)[0]
    md_path = base + ".md"
    csv_path = base + ".csv"

    rows = []
    name_map = payload.get("object_name_map", {})
    for object_id, metrics in payload["results"]["baseline"]["per_object"].items():
        rows.append((object_id, name_map.get(object_id, object_id), metrics))
    rows.sort(key=lambda x: x[2]["hit@0.5_top1"], reverse=True)

    with open(md_path, "w") as f:
        f.write("# Commonsense Baseline Results\n\n")
        f.write(f"- benchmark: `{payload['benchmark_path']}`\n")
        f.write(f"- split: `{payload['split']}`\n")
        f.write(f"- n_samples: {payload['n_samples']}\n")
        f.write(f"- top_k: {payload['top_k']}\n\n")

        overall = payload["results"]["baseline"]["overall"]
        f.write("## Overall\n\n")
        for key in [
            "mean_best_iou_top1",
            "mean_best_iou_topk",
            "hit@0.5_top1",
            "hit@0.5_topk",
            "hit@0.75_top1",
            "hit@0.75_topk",
        ]:
            f.write(f"- {key}: {overall[key]:.4f}\n")

        f.write("\n## Top Objects By Hit@0.5 Top1\n\n")
        f.write("| object | hit@0.5_top1 | mean_best_iou_top1 | n |\n")
        f.write("|---|---:|---:|---:|\n")
        for _, name, metrics in rows[:20]:
            f.write(
                f"| {name} | {metrics['hit@0.5_top1']:.4f} | "
                f"{metrics['mean_best_iou_top1']:.4f} | {metrics['n_samples']} |\n"
            )

        f.write("\n## Bottom Objects By Hit@0.5 Top1\n\n")
        f.write("| object | hit@0.5_top1 | mean_best_iou_top1 | n |\n")
        f.write("|---|---:|---:|---:|\n")
        for _, name, metrics in rows[-20:]:
            f.write(
                f"| {name} | {metrics['hit@0.5_top1']:.4f} | "
                f"{metrics['mean_best_iou_top1']:.4f} | {metrics['n_samples']} |\n"
            )

    with open(csv_path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["object_id", "object_name", "n_samples", "hit@0.5_top1", "mean_best_iou_top1"])
        for object_id, name, metrics in rows:
            wr.writerow([
                object_id,
                name,
                metrics["n_samples"],
                f"{metrics['hit@0.5_top1']:.6f}",
                f"{metrics['mean_best_iou_top1']:.6f}",
            ])

    return md_path, csv_path


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    with open(args.benchmark, "r") as f:
        bench = json.load(f)

    samples = bench["samples"]
    if args.split != "all":
        samples = [s for s in samples if s.get("split", "test") == args.split]
    if args.max_samples > 0:
        samples = samples[: args.max_samples]
    if not samples:
        raise RuntimeError("No samples selected for evaluation")

    print("=" * 70)
    print("Commonsense benchmark baseline evaluation")
    print(f"benchmark: {args.benchmark}")
    print(f"split: {args.split}  n_samples: {len(samples)}  top_k: {args.top_k}")
    print(f"device: {device}")
    print("=" * 70)

    dino_tf = build_dino_transform()

    print("\n[1/1] Loading baseline...")
    model = load_gd_model(args.gd_config, args.gd_weights, device="cpu").to(device).eval()

    print("[1/1] Evaluating baseline...")
    result = evaluate(model, samples, args, dino_tf, device)

    object_name_map = {}
    for obj in bench.get("objects", []):
        object_name_map[obj["object_id"]] = obj["object_name"]

    payload = {
        "status": "ok",
        "benchmark_path": args.benchmark,
        "split": args.split,
        "n_samples": len(samples),
        "top_k": args.top_k,
        "device": str(device),
        "object_name_map": object_name_map,
        "results": {"baseline": result},
    }

    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)

    md_path, csv_path = save_markdown_and_csv(args.output, payload)
    print(f"[saved] {args.output}")
    print(f"[saved] {md_path}")
    print(f"[saved] {csv_path}")


if __name__ == "__main__":
    main()
