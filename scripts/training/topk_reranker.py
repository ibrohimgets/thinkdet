#!/usr/bin/env python3
"""Build and train a lightweight top-k reranker for ThinkDet/GroundingDINO.

This is the bridge experiment from the GroundingDINO selector work:
export detector top-k candidates, attach IoU-derived labels, and train a small
proposal scorer. It is intentionally detector-agnostic and keeps a 4-way
reasoning_type field so we can evaluate functional, attribute, relational, and
interaction slices instead of only affordance prompts.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image


ROOT = "/home/iibrohimm/project/next_step"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "GroundingDINO", "GroundingDINO"))
sys.path.insert(0, os.path.join(ROOT, "GroundingDINO"))

from groundingdino.util.inference import load_model as load_gd_model
from groundingdino.util.misc import NestedTensor, nested_tensor_from_tensor_list
from thinkdet.data.refcoco_grounding import build_refcoco_eval
from thinkdet.models.arch import DEFAULT_INJECTION_LAYERS, ThinkDetModel
from thinkdet.models.projector import build_internvl_transform


GD_CONFIG = f"{ROOT}/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GD_WEIGHTS = f"{ROOT}/GroundingDINO/weights/groundingdino_swint_ogc.pth"
INTERNVL_PATH = f"{ROOT}/InternVL3_5-1B"
DEFAULT_AFFORDANCE = f"{ROOT}/thinkdet/data/benchmarks/affordance_coco_val_heldout_v1.json"
DEFAULT_REFCOPLUS_DATA = f"{ROOT}/dataSets/refer/data"
DEFAULT_REFCOPLUS_IMAGES = f"{ROOT}/dataSets/coco/train2017"
DEFAULT_THINKDET_CKPT = (
    f"{ROOT}/thinkdet/checkpoints/unified/"
    "layer9_kd0p05_l1_1e-4_20260224_135540/"
    "thinkdet_unified_epoch5.pth"
)

TYPE_ORDER = ["functional", "attribute", "relational", "interaction"]
RELATIONAL_KEYWORDS = [
    "left",
    "right",
    "next to",
    "near",
    "behind",
    "front",
    "between",
    "above",
    "below",
    "closest",
    "farthest",
    "middle",
    "top",
    "bottom",
    "under",
    "over",
]
INTERACTION_KEYWORDS = [
    "holding",
    "hold",
    "carrying",
    "carry",
    "feeding",
    "riding",
    "ride",
    "using",
    "playing",
    "talking on",
    "drinking from",
    "eating with",
]
FUNCTIONAL_KEYWORDS = [
    "something to",
    "used to",
    "used for",
    "you sit",
    "you drink",
    "you eat",
    "people ride",
    "drink from",
    "eat with",
    "cut with",
]


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def infer_reasoning_type(query: str, source: str | None = None) -> str:
    lowered = " ".join(query.lower().strip().split())
    if source == "affordance":
        return "functional"
    if any(keyword in lowered for keyword in FUNCTIONAL_KEYWORDS):
        return "functional"
    if any(keyword in lowered for keyword in RELATIONAL_KEYWORDS):
        return "relational"
    if any(keyword in lowered for keyword in INTERACTION_KEYWORDS):
        return "interaction"
    return "attribute"


def cxcywh_to_xyxy(box: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = box.unbind(-1)
    return torch.stack([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], dim=-1)


def xywh_abs_to_xyxy_norm(box: list[float], width: int, height: int) -> list[float]:
    x, y, w, h = [float(v) for v in box]
    return [x / width, y / height, (x + w) / width, (y + h) / height]


def pairwise_iou_xyxy(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter
    return inter / (union + 1e-6)


def geometry_features_xyxy_norm(box: list[float]) -> list[float]:
    x1, y1, x2, y2 = [float(v) for v in box]
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    area = w * h
    aspect = w / max(h, 1e-6)
    return [x1, y1, x2, y2, w, h, area, aspect]


def build_dino_transform():
    normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def transform(image_pil):
        width, height = image_pil.size
        scale = 800 / min(width, height)
        if scale * max(width, height) > 1333:
            scale = 1333 / max(width, height)
        new_width, new_height = int(width * scale), int(height * scale)
        image = image_pil.resize((new_width, new_height), Image.BILINEAR)
        return normalize(TF.to_tensor(image))

    return transform


def get_special_tokens(model):
    tokens = getattr(model, "specical_tokens", None)
    if tokens is not None:
        return tokens
    tokens = getattr(model, "special_tokens", None)
    if tokens is not None:
        return tokens
    return model.tokenizer.all_special_ids


def build_positive_map(tokenizer, special_tokens, query_text: str, max_text_len: int = 512) -> torch.Tensor:
    tokenized = tokenizer(query_text, return_tensors="pt")
    input_ids = tokenized["input_ids"][0]
    positive_map = torch.zeros(1, max_text_len, dtype=torch.float32)
    special_set = set(int(token) for token in special_tokens)
    for pos, token_id in enumerate(input_ids.tolist()):
        if pos >= max_text_len:
            break
        if token_id not in special_set:
            positive_map[0, pos] = 1.0
    return positive_map / positive_map.sum(dim=-1, keepdim=True).clamp(min=1e-6)


def score_outputs(outputs: dict, positive_map_norm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    logits = outputs["pred_logits"][0].clamp(-50, 50)
    boxes = outputs["pred_boxes"][0]
    text_len = logits.shape[-1]
    pmap = positive_map_norm[:, :text_len].to(logits.device)
    scores = (logits.sigmoid() * pmap).sum(dim=-1)
    return scores, boxes


def outputs_to_topk(outputs: dict, positive_map_norm: torch.Tensor, top_k: int) -> list[dict]:
    scores, boxes = score_outputs(outputs, positive_map_norm)
    vals, idx = scores.topk(min(int(top_k), scores.shape[0]))
    rows = []
    for rank in range(len(vals)):
        box_cxcywh = boxes[idx[rank]].detach().cpu()
        box_xyxy = cxcywh_to_xyxy(box_cxcywh.unsqueeze(0))[0].tolist()
        rows.append(
            {
                "rank": int(rank),
                "score": float(vals[rank].detach().cpu().item()),
                "box_xyxy_norm": [float(v) for v in box_xyxy],
                "features": [
                    float(vals[rank].detach().cpu().item()),
                    float(rank) / max(float(top_k - 1), 1.0),
                    *geometry_features_xyxy_norm(box_xyxy),
                ],
            }
        )
    return rows


def attach_labels(candidates: list[dict], gt_boxes_xyxy_norm: list[list[float]], iou_threshold: float) -> tuple[list[dict], int, float]:
    if not candidates or not gt_boxes_xyxy_norm:
        return candidates, -1, 0.0
    candidate_boxes = torch.tensor([row["box_xyxy_norm"] for row in candidates], dtype=torch.float32)
    gt_boxes = torch.tensor(gt_boxes_xyxy_norm, dtype=torch.float32)
    best_ious = pairwise_iou_xyxy(candidate_boxes, gt_boxes).max(dim=1).values
    best_index = int(best_ious.argmax().item())
    best_iou = float(best_ious[best_index].item())
    for idx, row in enumerate(candidates):
        row["iou_to_gt"] = float(best_ious[idx].item())
        row["is_positive"] = bool(best_ious[idx].item() >= iou_threshold)
    target_index = best_index if best_iou >= iou_threshold else -1
    return candidates, target_index, best_iou


def load_detector(args: argparse.Namespace, device: torch.device):
    if args.detector == "baseline":
        model = load_gd_model(args.gd_config, args.gd_weights, device="cpu").to(device).eval()
        return model, model.tokenizer, get_special_tokens(model)

    checkpoint = torch.load(args.thinkdet_checkpoint, map_location="cpu")
    extract_layers = checkpoint.get("extract_layers") or [checkpoint.get("extract_layer", 9)]
    model = ThinkDetModel(
        grounding_dino=load_gd_model(args.gd_config, args.gd_weights, device="cpu"),
        internvl_path=args.internvl_path,
        extract_layer=max(extract_layers),
        extract_layers=extract_layers,
        layer_fusion=checkpoint.get("layer_fusion", "mean"),
        injection_layers=checkpoint.get("injection_layers", DEFAULT_INJECTION_LAYERS),
        tma_m=checkpoint.get("tma_m", 8),
        tma_n_heads=checkpoint.get("tma_n_heads", 8),
        fusion_mode=checkpoint.get("fusion_mode", "concat"),
    )
    model.load_state_dict(checkpoint["trainable_state_dict"], strict=False)
    model = model.to(device).eval()
    return model, model.grounding_dino.tokenizer, get_special_tokens(model.grounding_dino)


@torch.no_grad()
def run_detector(model, args: argparse.Namespace, sample: dict, tokenizer, special_tokens, device: torch.device) -> list[dict]:
    query_text = sample["query"].strip().lower()
    if not query_text.endswith("."):
        query_text += " ."
    pmap = build_positive_map(tokenizer, special_tokens, query_text)

    if args.detector == "baseline":
        outputs = model(samples=sample["dino_nested"], captions=[query_text])
    else:
        outputs, _ = model(
            sample["internvl_image"].unsqueeze(0).to(device),
            [query_text],
            {"samples": sample["dino_nested"], "captions": [query_text]},
        )
    return outputs_to_topk(outputs, pmap, args.top_k)


def iter_affordance_samples(args: argparse.Namespace) -> Iterable[dict]:
    dino_tf = build_dino_transform()
    internvl_tf = build_internvl_transform(448)
    payload = json.load(open(args.affordance_benchmark))
    samples = payload["samples"]
    if args.affordance_split != "all":
        samples = [row for row in samples if row.get("split") == args.affordance_split]
    for row in samples:
        image = Image.open(row["image_path"]).convert("RGB")
        dino_image = dino_tf(image).to(args.device)
        mask = torch.zeros(1, dino_image.shape[1], dino_image.shape[2], dtype=torch.bool, device=args.device)
        gt_boxes = [
            xywh_abs_to_xyxy_norm(target["bbox_xywh"], row["width"], row["height"])
            for target in row["positive_targets"]
        ]
        yield {
            "source": "affordance",
            "split": row["split"],
            "sample_id": row["benchmark_id"],
            "image_id": int(row["image_id"]),
            "image_path": row["image_path"],
            "query": row["prompt"],
            "reasoning_type": "functional",
            "task_label": row["affordance_id"],
            "gt_boxes_xyxy_norm": gt_boxes,
            "dino_nested": NestedTensor(dino_image.unsqueeze(0), mask),
            "internvl_image": internvl_tf(image).to(args.device),
        }


def iter_refcoco_samples(args: argparse.Namespace) -> Iterable[dict]:
    split_by = args.refcoco_split_by or ("umd" if args.refcoco_dataset == "refcocog" else "unc")
    dataset = build_refcoco_eval(
        data_root=args.refcoco_data_root,
        image_dir=args.refcoco_image_dir,
        dataset_name=args.refcoco_dataset,
        split_by=split_by,
        split=args.refcoco_split,
    )
    for idx in range(len(dataset)):
        row = dataset[idx]
        image_path = os.path.join(args.refcoco_image_dir, f"{int(row['image_id']):012d}.jpg")
        gt_xyxy = cxcywh_to_xyxy(row["box"])[0].tolist()
        yield {
            "source": args.refcoco_dataset,
            "split": args.refcoco_split,
            "sample_id": f"{args.refcoco_dataset}_{args.refcoco_split}_{idx}",
            "image_id": int(row["image_id"]),
            "image_path": image_path,
            "query": row["expression"],
            "reasoning_type": infer_reasoning_type(row["expression"], source=args.refcoco_dataset),
            "task_label": args.refcoco_dataset,
            "gt_boxes_xyxy_norm": [[float(v) for v in gt_xyxy]],
            "dino_nested": nested_tensor_from_tensor_list([row["dino_image"].to(args.device)]),
            "internvl_image": row["internvl_image"].to(args.device),
        }


def serialize_sample(sample: dict, candidates: list[dict], target_index: int, best_iou: float) -> dict:
    return {
        "source": sample["source"],
        "split": sample["split"],
        "sample_id": sample["sample_id"],
        "image_id": sample["image_id"],
        "image_path": sample["image_path"],
        "query": sample["query"],
        "reasoning_type": sample["reasoning_type"],
        "task_label": sample["task_label"],
        "gt_boxes_xyxy_norm": sample["gt_boxes_xyxy_norm"],
        "target_index": int(target_index),
        "best_candidate_iou": float(best_iou),
        "top1_iou": float(candidates[0].get("iou_to_gt", 0.0)) if candidates else 0.0,
        "recall_at_k": bool(target_index >= 0),
        "candidates": candidates,
    }


def build_cache(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    model, tokenizer, special_tokens = load_detector(args, device)
    source_iters = []
    if "affordance" in args.sources:
        source_iters.append(("affordance", iter_affordance_samples(args)))
    if "refcoco" in args.sources:
        source_iters.append(("refcoco", iter_refcoco_samples(args)))

    rows = []
    type_counts = Counter()
    for _, source_iter in source_iters:
        source_count = 0
        for sample in source_iter:
            if args.max_samples > 0 and len(rows) >= args.max_samples:
                break
            if args.max_samples_per_source > 0 and source_count >= args.max_samples_per_source:
                break
            if args.max_samples_per_type > 0 and type_counts[sample["reasoning_type"]] >= args.max_samples_per_type:
                continue
            candidates = run_detector(model, args, sample, tokenizer, special_tokens, device)
            candidates, target_index, best_iou = attach_labels(candidates, sample["gt_boxes_xyxy_norm"], args.iou_threshold)
            rows.append(serialize_sample(sample, candidates, target_index, best_iou))
            source_count += 1
            type_counts[sample["reasoning_type"]] += 1
            if len(rows) % max(1, args.log_every) == 0:
                print(f"cached={len(rows)}")
            if args.max_samples_per_type > 0 and all(type_counts[label] >= args.max_samples_per_type for label in TYPE_ORDER):
                break
        if args.max_samples > 0 and len(rows) >= args.max_samples:
            break
        if args.max_samples_per_type > 0 and all(type_counts[label] >= args.max_samples_per_type for label in TYPE_ORDER):
            break

    summary = summarize_rows(rows)
    payload = {
        "status": "ok",
        "detector": args.detector,
        "top_k": int(args.top_k),
        "iou_threshold": float(args.iou_threshold),
        "sources": args.sources,
        "summary": summary,
        "rows": rows,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": summary, "output": str(output_path)}, indent=2))
    return payload


def summarize_rows(rows: list[dict]) -> dict:
    total = len(rows)
    by_type = {}
    for label in TYPE_ORDER:
        subset = [row for row in rows if row["reasoning_type"] == label]
        denom = max(len(subset), 1)
        by_type[label] = {
            "total": len(subset),
            "top1_acc": sum(row["top1_iou"] >= 0.5 for row in subset) / denom,
            "recall_at_k": sum(row["recall_at_k"] for row in subset) / denom,
        }
    return {
        "total": total,
        "source_distribution": dict(Counter(row["source"] for row in rows)),
        "split_distribution": dict(Counter(row["split"] for row in rows)),
        "reasoning_type_distribution": dict(Counter(row["reasoning_type"] for row in rows)),
        "top1_acc": sum(row["top1_iou"] >= 0.5 for row in rows) / max(total, 1),
        "recall_at_k": sum(row["recall_at_k"] for row in rows) / max(total, 1),
        "by_reasoning_type": by_type,
    }


class TopKReranker(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.scorer(features).squeeze(-1)


def tensorize(rows: list[dict], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, list[dict]]:
    usable = [row for row in rows if row["target_index"] >= 0 and row["candidates"]]
    if not usable:
        raise RuntimeError("No rows with a positive top-k target.")
    features = torch.tensor(
        [[candidate["features"] for candidate in row["candidates"]] for row in usable],
        dtype=torch.float32,
        device=device,
    )
    targets = torch.tensor([row["target_index"] for row in usable], dtype=torch.long, device=device)
    return features, targets, usable


def evaluate_reranker(model: TopKReranker, rows: list[dict], device: torch.device) -> dict:
    if not rows:
        return {"total": 0, "accuracy": None, "by_reasoning_type": {}}
    correct = 0
    base_correct = 0
    by_type = defaultdict(lambda: {"total": 0, "correct": 0, "base_correct": 0})
    for row in rows:
        candidates = row["candidates"]
        features = torch.tensor([[candidate["features"] for candidate in candidates]], dtype=torch.float32, device=device)
        pred = int(model(features).argmax(dim=1).item())
        hit = candidates[pred].get("iou_to_gt", 0.0) >= 0.5
        base_hit = candidates[0].get("iou_to_gt", 0.0) >= 0.5
        correct += int(hit)
        base_correct += int(base_hit)
        bucket = by_type[row["reasoning_type"]]
        bucket["total"] += 1
        bucket["correct"] += int(hit)
        bucket["base_correct"] += int(base_hit)
    by_type_out = {
        key: {
            "total": value["total"],
            "base_acc": value["base_correct"] / max(value["total"], 1),
            "reranker_acc": value["correct"] / max(value["total"], 1),
        }
        for key, value in sorted(by_type.items())
    }
    return {
        "total": len(rows),
        "base_acc": base_correct / len(rows),
        "reranker_acc": correct / len(rows),
        "by_reasoning_type": by_type_out,
    }


def train_reranker(args: argparse.Namespace) -> dict:
    payload = json.loads(Path(args.cache).read_text())
    rows = payload["rows"]
    random.Random(args.seed).shuffle(rows)
    if args.train_splits:
        train_splits = set(parse_csv(args.train_splits))
        eval_splits = set(parse_csv(args.eval_splits))
        train_rows = [row for row in rows if row["split"] in train_splits]
        eval_rows = [row for row in rows if row["split"] in eval_splits]
    else:
        split_at = int(len(rows) * args.train_ratio)
        train_rows = rows[:split_at]
        eval_rows = rows[split_at:]

    device = torch.device(args.device)
    train_features, train_targets, train_usable = tensorize(train_rows, device)
    input_dim = int(train_features.shape[-1])
    model = TopKReranker(input_dim=input_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best = {"epoch": 0, "eval_acc": -1.0, "state_dict": None}
    for epoch in range(1, args.epochs + 1):
        model.train()
        logits = model(train_features)
        loss = F.cross_entropy(logits, train_targets)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if epoch == 1 or epoch % args.eval_every == 0 or epoch == args.epochs:
            model.eval()
            with torch.no_grad():
                eval_result = evaluate_reranker(model, eval_rows, device)
            eval_acc = float(eval_result.get("reranker_acc") or 0.0)
            if eval_acc > best["eval_acc"]:
                best = {
                    "epoch": epoch,
                    "eval_acc": eval_acc,
                    "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                }
            print(f"epoch={epoch} loss={loss.item():.4f} eval_acc={eval_acc:.4f}")

    if best["state_dict"] is not None:
        model.load_state_dict(best["state_dict"])
    final_eval = evaluate_reranker(model, eval_rows, device)
    result = {
        "status": "ok",
        "cache": str(Path(args.cache).resolve()),
        "train_rows": len(train_rows),
        "train_usable_rows": len(train_usable),
        "eval_rows": len(eval_rows),
        "best_epoch": best["epoch"],
        "input_dim": input_dim,
        "final_eval": final_eval,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": input_dim,
            "result": result,
        },
        output_path,
    )
    result_path = output_path.with_name(output_path.stem + "_metrics.json")
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"checkpoint={output_path}")
    print(f"results={result_path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Build/train a 4-way-aware top-k reranker cache for ThinkDet.")
    parser.add_argument("--mode", choices=["build-cache", "train"], default="build-cache")
    parser.add_argument("--sources", nargs="+", choices=["affordance", "refcoco"], default=["affordance", "refcoco"])
    parser.add_argument("--detector", choices=["baseline", "thinkdet"], default="baseline")
    parser.add_argument("--output", default=f"{ROOT}/thinkdet/results/analysis/topk_reranker_cache.json")
    parser.add_argument("--cache", default=f"{ROOT}/thinkdet/results/analysis/topk_reranker_cache.json")
    parser.add_argument("--gd_config", default=GD_CONFIG)
    parser.add_argument("--gd_weights", default=GD_WEIGHTS)
    parser.add_argument("--internvl_path", default=INTERNVL_PATH)
    parser.add_argument("--thinkdet_checkpoint", default=DEFAULT_THINKDET_CKPT)
    parser.add_argument("--affordance_benchmark", default=DEFAULT_AFFORDANCE)
    parser.add_argument("--affordance_split", choices=["dev", "test", "all"], default="all")
    parser.add_argument("--refcoco_data_root", default=DEFAULT_REFCOPLUS_DATA)
    parser.add_argument("--refcoco_image_dir", default=DEFAULT_REFCOPLUS_IMAGES)
    parser.add_argument("--refcoco_dataset", choices=["refcoco", "refcoco+", "refcocog"], default="refcoco+")
    parser.add_argument("--refcoco_split_by", default=None)
    parser.add_argument("--refcoco_split", default="val")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--iou_threshold", type=float, default=0.5)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--max_samples_per_source", type=int, default=0)
    parser.add_argument("--max_samples_per_type", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--train_splits", default="", help="Comma-separated splits for training; empty uses random train_ratio.")
    parser.add_argument("--eval_splits", default="test,val", help="Comma-separated splits for eval when train_splits is set.")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--eval_every", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.mode == "build-cache":
        build_cache(args)
    else:
        train_reranker(args)


if __name__ == "__main__":
    main()
