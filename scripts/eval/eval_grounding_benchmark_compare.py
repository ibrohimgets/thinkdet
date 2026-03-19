#!/usr/bin/env python3
"""
Compare GroundingDINO baseline vs ThinkDet vs ThinkDet with optional LLM fallback
on a benchmark JSON artifact.

Supported benchmark sample schemas:
- Commonsense COCO benchmark rows with `positive_targets` (bbox_xywh)
- Phrase-grounding benchmark rows with `target_boxes_abs_xyxy`
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
from thinkdet.inference.fallback import (
    FallbackPolicyConfig,
    InternVLPromptRefiner,
    InternVLYesNoReranker,
    ScoredCandidateSet,
    apply_confidence_calibrator_to_stats,
    apply_fallback_policy,
    is_unreliable as fallback_is_unreliable,
    load_confidence_calibrator,
    summarize_scores,
)
from thinkdet.models.arch import ThinkDetModel, DEFAULT_INJECTION_LAYERS


DEFAULT_BENCH = (
    f"{ROOT}/thinkdet/data/benchmarks/commonsense_refcoco_objects_coco_val_10k_v1.json"
)
DEFAULT_CKPT = (
    f"{ROOT}/thinkdet/checkpoints/unified/"
    "layer9_kd0p05_l1_1e-4_20260224_135540/"
    "thinkdet_unified_epoch5.pth"
)
DEFAULT_OUT = (
    f"{ROOT}/thinkdet/results/eval/"
    f"grounding_benchmark_compare_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
)
GD_CONFIG = (
    f"{ROOT}/GroundingDINO/GroundingDINO/groundingdino/config/"
    "GroundingDINO_SwinT_OGC.py"
)
GD_WEIGHTS = f"{ROOT}/GroundingDINO/GroundingDINO/weights/groundingdino_swint_ogc.pth"
INTERNVL_PATH = f"{ROOT}/InternVL3_5-1B"

FALLBACK_STAGE_ORDER = [
    "primary",
    "llm_feedback",
    "prompt_refine",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", type=str, default=DEFAULT_BENCH)
    p.add_argument("--split", type=str, default="all")
    p.add_argument("--shard_id", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--log_every", type=int, default=25)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--output", type=str, default=DEFAULT_OUT)

    p.add_argument("--gd_config", type=str, default=GD_CONFIG)
    p.add_argument("--gd_weights", type=str, default=GD_WEIGHTS)
    p.add_argument("--internvl_path", type=str, default=INTERNVL_PATH)
    p.add_argument("--thinkdet_checkpoint", type=str, default=DEFAULT_CKPT)
    p.add_argument("--extract_layer", type=int, default=None)
    p.add_argument("--extract_layers", type=int, nargs="+", default=None)
    p.add_argument("--layer_fusion", type=str, default=None, choices=["mean", "last"])

    p.add_argument("--enable_fallback", action="store_true")
    p.add_argument(
        "--disable_feedback",
        action="store_true",
        help="Disable LLM yes/no reranking and keep only prompt-refinement fallback.",
    )
    p.add_argument("--fallback_min_top1", type=float, default=0.20)
    p.add_argument(
        "--fallback_min_confidence",
        type=float,
        default=None,
        help="If set with --confidence_calibrator, route fallback on calibrated_confidence instead of raw top1.",
    )
    p.add_argument("--fallback_min_margin", type=float, default=0.02)
    p.add_argument("--fallback_min_gate", type=float, default=None)
    p.add_argument(
        "--confidence_calibrator",
        type=str,
        default="",
        help="Path to a JSON confidence calibrator from fit_confidence_calibrator.py",
    )
    p.add_argument("--llm_feedback_top_k", type=int, default=20)
    p.add_argument("--llm_feedback_weight", type=float, default=0.20)
    p.add_argument("--llm_feedback_max_new_tokens", type=int, default=6)
    p.add_argument("--llm_feedback_temperature", type=float, default=0.0)
    p.add_argument("--llm_feedback_improve_margin", type=float, default=0.01)
    p.add_argument("--prompt_refine", action="store_true")
    p.add_argument("--prompt_refine_max_variants", type=int, default=2)
    p.add_argument("--prompt_refine_max_new_tokens", type=int, default=96)
    p.add_argument("--prompt_refine_temperature", type=float, default=0.0)
    p.add_argument("--prompt_refine_improve_margin", type=float, default=0.01)
    p.add_argument(
        "--prompt_refine_require_semantic_preservation",
        dest="prompt_refine_require_semantic_preservation",
        action="store_true",
    )
    p.add_argument(
        "--no_prompt_refine_require_semantic_preservation",
        dest="prompt_refine_require_semantic_preservation",
        action="store_false",
    )
    p.set_defaults(prompt_refine_require_semantic_preservation=True)
    p.add_argument("--prompt_refine_min_shared_terms", type=int, default=1)
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


def build_internvl_transform():
    return T.Compose([
        T.Resize((448, 448), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


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


def outputs_to_predictions(outputs, positive_map_norm, image_size, top_k):
    img_w, img_h = image_size
    scores, boxes = score_outputs_for_query(outputs, positive_map_norm)
    vals, idx = scores.topk(min(int(top_k), scores.shape[0]))
    preds = []
    for j in range(len(vals)):
        box = boxes[idx[j]].detach().cpu()
        cx, cy, bw, bh = box.tolist()
        preds.append({
            "score": float(vals[j].item()),
            "box_abs_xyxy": [
                (cx - bw / 2.0) * img_w,
                (cy - bh / 2.0) * img_h,
                (cx + bw / 2.0) * img_w,
                (cy + bh / 2.0) * img_h,
            ],
        })
    return preds


def build_feedback_predictions(outputs, positive_map_norm, image_size, top_k):
    img_w, img_h = image_size
    scores, boxes = score_outputs_for_query(outputs, positive_map_norm)
    vals, idx = scores.topk(min(int(top_k), scores.shape[0]))
    preds = []
    for j in range(len(vals)):
        box = boxes[idx[j]].detach().cpu()
        cx, cy, bw, bh = box.tolist()
        preds.append({
            "score": float(vals[j].item()),
            "combined_score": float(vals[j].item()),
            "box_cxcywh": [cx, cy, bw, bh],
            "box_abs_xyxy": [
                (cx - bw / 2.0) * img_w,
                (cy - bh / 2.0) * img_h,
                (cx + bw / 2.0) * img_w,
                (cy + bh / 2.0) * img_h,
            ],
        })
    return preds


def build_scored_bundle(name, outputs, positive_map_norm, query_text, aux=None):
    scores, _ = score_outputs_for_query(outputs, positive_map_norm)
    return ScoredCandidateSet(
        name=name,
        scores=scores.detach().cpu().tolist(),
        payload={
            "type": "outputs",
            "outputs": outputs,
            "positive_map_norm": positive_map_norm,
        },
        aux=aux,
        query_text=query_text,
    )


def bundle_to_predictions(bundle, image_size, top_k):
    payload = bundle.payload
    if payload["type"] == "outputs":
        return outputs_to_predictions(
            payload["outputs"],
            payload["positive_map_norm"],
            image_size,
            top_k,
        )
    if payload["type"] == "reranked_preds":
        return list(payload["preds"][: int(top_k)])
    raise ValueError(f"Unsupported payload type: {payload['type']}")


def evaluate_predictions(preds, gt_boxes, top_k):
    k = min(int(top_k), len(preds))
    if k == 0:
        return {
            "top1_iou": 0.0,
            "topk_best_iou": 0.0,
            "hit50_top1": 0.0,
            "hit50_topk": 0.0,
            "hit75_top1": 0.0,
            "hit75_topk": 0.0,
            "top1_score": 0.0,
        }

    best_iou_per_pred = []
    for pred in preds[:k]:
        ious = [iou_xyxy(pred["box_abs_xyxy"], g) for g in gt_boxes]
        best_iou_per_pred.append(max(ious) if ious else 0.0)

    top1 = best_iou_per_pred[0]
    topk = max(best_iou_per_pred)
    return {
        "top1_iou": top1,
        "topk_best_iou": topk,
        "hit50_top1": 1.0 if top1 >= 0.5 else 0.0,
        "hit50_topk": 1.0 if topk >= 0.5 else 0.0,
        "hit75_top1": 1.0 if top1 >= 0.75 else 0.0,
        "hit75_topk": 1.0 if topk >= 0.75 else 0.0,
        "top1_score": float(preds[0]["score"]),
    }


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


def is_unreliable(stats, cfg):
    return fallback_is_unreliable(stats, cfg)


def resolve_confidence_calibration(args):
    calibrator_path = str(args.confidence_calibrator or "").strip()
    min_confidence = args.fallback_min_confidence

    if min_confidence is not None and not calibrator_path:
        raise RuntimeError("--fallback_min_confidence requires --confidence_calibrator")

    if calibrator_path and min_confidence is None:
        min_confidence = 0.50

    calibrator = None
    if calibrator_path:
        calibrator = load_confidence_calibrator(calibrator_path)

    return calibrator_path, min_confidence, calibrator


def load_baseline_model(args, device):
    return load_gd_model(args.gd_config, args.gd_weights, device="cpu").to(device).eval()


def load_unified_model(args, device):
    ckpt = torch.load(args.thinkdet_checkpoint, map_location="cpu")
    inject_layers = ckpt.get("injection_layers", DEFAULT_INJECTION_LAYERS)
    tma_m = ckpt.get("tma_m", 8)
    tma_nheads = ckpt.get("tma_n_heads", 8) or 8
    checkpoint_extract_layers = ckpt.get("extract_layers") or [ckpt.get("extract_layer", 9)]
    checkpoint_layer_fusion = ckpt.get("layer_fusion", "mean") or "mean"
    if args.extract_layers:
        ext_layers = sorted(set(int(x) for x in args.extract_layers))
    elif args.extract_layer is not None:
        ext_layers = [int(args.extract_layer)]
    else:
        ext_layers = [int(x) for x in checkpoint_extract_layers]
    layer_fusion = args.layer_fusion or checkpoint_layer_fusion
    fusion_mode = ckpt.get("fusion_mode", "concat") or "concat"

    gd = load_gd_model(args.gd_config, args.gd_weights, device="cpu")
    model = ThinkDetModel(
        grounding_dino=gd,
        internvl_path=args.internvl_path,
        extract_layer=max(ext_layers),
        extract_layers=ext_layers,
        layer_fusion=layer_fusion,
        injection_layers=inject_layers,
        tma_m=tma_m,
        tma_n_heads=tma_nheads,
        fusion_mode=fusion_mode,
    )
    model.load_state_dict(ckpt["trainable_state_dict"], strict=False)
    model = model.to(device).eval()
    return model, {
        "checkpoint_extract_layers": [int(x) for x in checkpoint_extract_layers],
        "checkpoint_layer_fusion": checkpoint_layer_fusion,
        "extract_layers": ext_layers,
        "layer_fusion": layer_fusion,
        "injection_layers": inject_layers,
        "tma_m": tma_m,
        "tma_n_heads": tma_nheads,
        "fusion_mode": fusion_mode,
    }


@torch.no_grad()
def run_baseline_outputs(model, image_pil, prompt, device, dino_tf):
    dino_t = dino_tf(image_pil).unsqueeze(0).to(device)
    mask = torch.zeros(1, dino_t.shape[2], dino_t.shape[3], dtype=torch.bool, device=device)
    nested = NestedTensor(dino_t, mask)
    query_text = prompt if prompt.strip().endswith(".") else (prompt.strip() + " .")
    outputs = model(samples=nested, captions=[query_text])
    return query_text, outputs


@torch.no_grad()
def run_thinkdet_outputs(model, image_pil, prompt, device, dino_tf, ivl_tf):
    dino_t = dino_tf(image_pil).unsqueeze(0).to(device)
    mask = torch.zeros(1, dino_t.shape[2], dino_t.shape[3], dtype=torch.bool, device=device)
    nested = NestedTensor(dino_t, mask)
    ivl_t = ivl_tf(image_pil).unsqueeze(0).to(device)
    query_text = prompt if prompt.strip().endswith(".") else (prompt.strip() + " .")
    outputs, aux = model(
        ivl_t,
        [prompt],
        {"samples": nested, "captions": [query_text]},
    )
    return query_text, outputs, aux


def extract_gt_boxes(sample):
    if "positive_targets" in sample:
        return [xywh_to_xyxy(t["bbox_xywh"]) for t in sample["positive_targets"]]
    if "target_boxes_abs_xyxy" in sample:
        return [[float(v) for v in box] for box in sample["target_boxes_abs_xyxy"]]
    if "gt_boxes" in sample:
        return [[float(v) for v in box] for box in sample["gt_boxes"]]
    raise KeyError("Unsupported benchmark sample schema: missing GT boxes")


def extract_group_info(sample):
    if "object_id" in sample:
        return sample.get("object_id"), sample.get("object_name", sample.get("object_id"))
    return None, None


def metric_row_to_jsonable(row):
    return {
        "top1_iou": float(row["top1_iou"]),
        "topk_best_iou": float(row["topk_best_iou"]),
        "hit50_top1": float(row["hit50_top1"]),
        "hit50_topk": float(row["hit50_topk"]),
        "hit75_top1": float(row["hit75_top1"]),
        "hit75_topk": float(row["hit75_topk"]),
        "top1_score": float(row["top1_score"]),
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

        if payload.get("group_name_map") and payload["results"]["thinkdet_fallback"]["per_group"]:
            sorted_rows = sorted(
                payload["results"]["thinkdet_fallback"]["per_group"].items(),
                key=lambda kv: kv[1]["hit@0.5_top1"],
                reverse=True,
            )
            f.write("\n## Top Groups By Fallback Hit@0.5 Top1\n\n")
            f.write("| group | hit@0.5_top1 | mean_best_iou_top1 | n |\n")
            f.write("|---|---:|---:|---:|\n")
            for group_id, metrics in sorted_rows[:20]:
                f.write(
                    f"| {payload['group_name_map'].get(group_id, group_id)} | "
                    f"{metrics['hit@0.5_top1']:.4f} | {metrics['mean_best_iou_top1']:.4f} | "
                    f"{metrics['n_samples']} |\n"
                )

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
    calibrator_path, fallback_min_confidence, confidence_calibrator = resolve_confidence_calibration(args)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    with open(args.benchmark, "r") as f:
        bench = json.load(f)

    samples = bench["samples"]
    if args.split != "all":
        samples = [s for s in samples if s.get("split", "all") == args.split]
    if args.max_samples > 0:
        samples = samples[: args.max_samples]
    if args.num_shards > 1:
        samples = samples[args.shard_id::args.num_shards]
    if not samples:
        raise RuntimeError("No samples selected for evaluation")

    print("=" * 70)
    print("Grounding benchmark compare")
    print(f"benchmark: {args.benchmark}")
    print(f"split: {args.split}  n_samples: {len(samples)}  top_k: {args.top_k}")
    print(f"device: {device}")
    print(f"fallback: enabled={bool(args.enable_fallback)} prompt_refine={bool(args.prompt_refine)}")
    print("=" * 70)

    dino_tf = build_dino_transform()
    ivl_tf = build_internvl_transform()

    print("\n[1/2] Loading baseline GroundingDINO...")
    baseline = load_baseline_model(args, device)
    print("[2/2] Loading unified ThinkDet...")
    thinkdet, ckpt_meta = load_unified_model(args, device)

    baseline_tokenizer = baseline.tokenizer
    baseline_special_tokens = get_special_tokens(baseline)
    thinkdet_tokenizer = thinkdet.grounding_dino.tokenizer
    thinkdet_special_tokens = get_special_tokens(thinkdet.grounding_dino)

    llm_reranker = None
    llm_prompt_refiner = None
    if args.enable_fallback and not args.disable_feedback:
        llm_reranker = InternVLYesNoReranker(
            internvl_model=thinkdet.feature_extractor.internvl,
            tokenizer=thinkdet.feature_extractor.tokenizer,
            device=device,
            image_transform=ivl_tf,
            weight=args.llm_feedback_weight,
            max_new_tokens=args.llm_feedback_max_new_tokens,
            temperature=args.llm_feedback_temperature,
        )
        if args.prompt_refine:
            llm_prompt_refiner = InternVLPromptRefiner(
                internvl_model=thinkdet.feature_extractor.internvl,
                tokenizer=thinkdet.feature_extractor.tokenizer,
                device=device,
                image_transform=ivl_tf,
                max_new_tokens=args.prompt_refine_max_new_tokens,
                temperature=args.prompt_refine_temperature,
                num_candidates=args.prompt_refine_max_variants,
            )

    fallback_cfg = FallbackPolicyConfig(
        min_top1=args.fallback_min_top1,
        min_confidence=fallback_min_confidence,
        min_margin=args.fallback_min_margin,
        min_gate=args.fallback_min_gate,
        confidence_calibrator=confidence_calibrator,
        confidence_calibrator_path=calibrator_path or None,
        feedback_improve_margin=args.llm_feedback_improve_margin,
        refine_improve_margin=args.prompt_refine_improve_margin,
        refine_require_semantic_preservation=args.prompt_refine_require_semantic_preservation,
        refine_min_shared_terms=args.prompt_refine_min_shared_terms,
    )

    mode_rows = {
        "baseline": [],
        "thinkdet": [],
        "thinkdet_fallback": [],
    }
    per_group_rows = defaultdict(lambda: {"baseline": [], "thinkdet": [], "thinkdet_fallback": []})
    group_name_map = {}
    fallback_stage_counts = {stage: 0 for stage in FALLBACK_STAGE_ORDER}
    per_sample = []

    print("[eval] Running comparison...")
    with torch.no_grad():
        for idx, sample in enumerate(samples, start=1):
            image_pil = Image.open(sample["image_path"]).convert("RGB")
            gt_boxes = extract_gt_boxes(sample)
            group_id, group_name = extract_group_info(sample)
            if group_id is not None and group_name:
                group_name_map[group_id] = group_name

            base_query, base_out = run_baseline_outputs(
                baseline, image_pil, sample["prompt"], device, dino_tf
            )
            base_pmap = build_positive_map_for_query(
                baseline_tokenizer,
                baseline_special_tokens,
                base_query,
                max_text_len=512,
            )
            base_preds = outputs_to_predictions(base_out, base_pmap, image_pil.size, args.top_k)
            base_metrics = evaluate_predictions(base_preds, gt_boxes, args.top_k)
            mode_rows["baseline"].append(base_metrics)
            if group_id is not None:
                per_group_rows[group_id]["baseline"].append(base_metrics)

            td_query, td_out, td_aux = run_thinkdet_outputs(
                thinkdet, image_pil, sample["prompt"], device, dino_tf, ivl_tf
            )
            td_pmap = build_positive_map_for_query(
                thinkdet_tokenizer,
                thinkdet_special_tokens,
                td_query,
                max_text_len=512,
            )
            primary_bundle = build_scored_bundle(
                "thinkdet",
                td_out,
                td_pmap,
                td_query,
                aux=td_aux,
            )
            primary_preds = bundle_to_predictions(primary_bundle, image_pil.size, args.top_k)
            td_metrics = evaluate_predictions(primary_preds, gt_boxes, args.top_k)
            mode_rows["thinkdet"].append(td_metrics)
            if group_id is not None:
                per_group_rows[group_id]["thinkdet"].append(td_metrics)

            selected_bundle = primary_bundle
            selected_info = {
                "selected_stage": "primary",
                "selected_name": "thinkdet",
            }

            if args.enable_fallback:
                rerank_bundle = None
                rerank_fn = None
                primary_stats = summarize_scores(primary_bundle.scores, primary_bundle.aux)
                apply_confidence_calibrator_to_stats(primary_stats, confidence_calibrator)
                candidate_stats = primary_stats

                if is_unreliable(primary_stats, fallback_cfg) and llm_reranker is not None:
                    feedback_preds = build_feedback_predictions(
                        td_out,
                        td_pmap,
                        image_pil.size,
                        top_k=args.llm_feedback_top_k,
                    )
                    if feedback_preds:
                        reranked_preds = llm_reranker.rerank_predictions(
                            image_pil,
                            td_query,
                            feedback_preds,
                        )
                        rerank_bundle = ScoredCandidateSet(
                            name="thinkdet_llm_feedback",
                            scores=[float(p.get("combined_score", p["score"])) for p in reranked_preds],
                            payload={
                                "type": "reranked_preds",
                                "preds": reranked_preds,
                                "outputs": td_out,
                                "positive_map_norm": td_pmap,
                            },
                            aux=td_aux,
                            query_text=td_query,
                        )
                        rerank_stats = summarize_scores(rerank_bundle.scores, rerank_bundle.aux)
                        if (
                            rerank_stats["reliability"]
                            >= primary_stats["reliability"] + fallback_cfg.feedback_improve_margin
                        ):
                            candidate_stats = rerank_stats
                        rerank_fn = lambda _candidate_set, bundle=rerank_bundle: bundle

                refined_candidates = []
                if llm_prompt_refiner is not None and is_unreliable(candidate_stats, fallback_cfg):
                    refinements = llm_prompt_refiner.refine_prompts(td_query, image_pil=image_pil)
                    for ref_idx, ref_query in enumerate(refinements[: args.prompt_refine_max_variants]):
                        ref_q, ref_out, ref_aux = run_thinkdet_outputs(
                            thinkdet, image_pil, ref_query, device, dino_tf, ivl_tf
                        )
                        ref_pmap = build_positive_map_for_query(
                            thinkdet_tokenizer,
                            thinkdet_special_tokens,
                            ref_q,
                            max_text_len=512,
                        )
                        refined_candidates.append(
                            build_scored_bundle(
                                f"thinkdet_refine_{ref_idx+1}",
                                ref_out,
                                ref_pmap,
                                ref_q,
                                aux=ref_aux,
                            )
                        )

                if rerank_fn is not None or refined_candidates:
                    selected_bundle, selected_info = apply_fallback_policy(
                        primary=primary_bundle,
                        rerank_fn=rerank_fn,
                        refined_candidates=refined_candidates,
                        config=fallback_cfg,
                    )

            fb_preds = bundle_to_predictions(selected_bundle, image_pil.size, args.top_k)
            fb_metrics = evaluate_predictions(fb_preds, gt_boxes, args.top_k)
            mode_rows["thinkdet_fallback"].append(fb_metrics)
            if group_id is not None:
                per_group_rows[group_id]["thinkdet_fallback"].append(fb_metrics)
            fallback_stage_counts[selected_info["selected_stage"]] += 1

            per_sample.append({
                "benchmark_id": sample.get("benchmark_id", str(idx - 1)),
                "image_id": sample.get("image_id"),
                "prompt": sample.get("prompt"),
                "group_id": group_id,
                "group_name": group_name,
                "fallback_selected_stage": selected_info["selected_stage"],
                "baseline": metric_row_to_jsonable(base_metrics),
                "thinkdet": metric_row_to_jsonable(td_metrics),
                "thinkdet_fallback": metric_row_to_jsonable(fb_metrics),
            })

            if idx % args.log_every == 0 or idx == len(samples):
                base_summary = summarize_rows(mode_rows["baseline"], args.top_k)
                td_summary = summarize_rows(mode_rows["thinkdet"], args.top_k)
                fb_summary = summarize_rows(mode_rows["thinkdet_fallback"], args.top_k)
                print(
                    f"  [{idx}/{len(samples)}] "
                    f"base_hit50={base_summary['hit@0.5_top1']:.4f} "
                    f"td_hit50={td_summary['hit@0.5_top1']:.4f} "
                    f"fb_hit50={fb_summary['hit@0.5_top1']:.4f}"
                )

    results = {}
    for mode_name in ["baseline", "thinkdet", "thinkdet_fallback"]:
        results[mode_name] = {
            "overall": summarize_rows(mode_rows[mode_name], args.top_k),
            "per_group": {},
        }

    if group_name_map:
        for group_id, rows_by_mode in sorted(per_group_rows.items()):
            for mode_name in ["baseline", "thinkdet", "thinkdet_fallback"]:
                results[mode_name]["per_group"][group_id] = summarize_rows(
                    rows_by_mode[mode_name], args.top_k
                )

    payload = {
        "status": "ok",
        "benchmark_path": args.benchmark,
        "benchmark_name": bench.get("benchmark_name", os.path.splitext(os.path.basename(args.benchmark))[0]),
        "benchmark_version": bench.get("version"),
        "split": args.split,
        "n_samples": len(samples),
        "top_k": args.top_k,
        "device": str(device),
        "thinkdet_checkpoint": args.thinkdet_checkpoint,
        "checkpoint_meta": ckpt_meta,
        "fallback_config": {
            "enabled": bool(args.enable_fallback),
            "feedback_enabled": bool(args.enable_fallback and not args.disable_feedback),
            "min_top1": float(args.fallback_min_top1),
            "min_confidence": None if fallback_min_confidence is None else float(fallback_min_confidence),
            "min_margin": float(args.fallback_min_margin),
            "min_gate": None if args.fallback_min_gate is None else float(args.fallback_min_gate),
            "confidence_calibrator": calibrator_path or None,
            "llm_feedback_top_k": int(args.llm_feedback_top_k),
            "llm_feedback_weight": float(args.llm_feedback_weight),
            "llm_feedback_max_new_tokens": int(args.llm_feedback_max_new_tokens),
            "llm_feedback_temperature": float(args.llm_feedback_temperature),
            "llm_feedback_improve_margin": float(args.llm_feedback_improve_margin),
            "prompt_refine": bool(args.prompt_refine),
            "prompt_refine_max_variants": int(args.prompt_refine_max_variants),
            "prompt_refine_max_new_tokens": int(args.prompt_refine_max_new_tokens),
            "prompt_refine_temperature": float(args.prompt_refine_temperature),
            "prompt_refine_improve_margin": float(args.prompt_refine_improve_margin),
            "prompt_refine_require_semantic_preservation": bool(
                args.prompt_refine_require_semantic_preservation
            ),
            "prompt_refine_min_shared_terms": int(args.prompt_refine_min_shared_terms),
        },
        "group_name_map": group_name_map,
        "results": results,
        "fallback_stage_counts": fallback_stage_counts,
        "per_sample": per_sample,
    }

    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)

    md_path, csv_path = save_markdown_and_csv(args.output, payload)
    print(f"[saved] {args.output}")
    print(f"[saved] {md_path}")
    print(f"[saved] {csv_path}")


if __name__ == "__main__":
    main()
