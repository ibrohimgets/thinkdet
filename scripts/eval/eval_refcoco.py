"""
RefCOCO benchmark eval (REC-style):
  - top1 accuracy at IoU >= threshold
  - detection-level FP stats (optional, via conf_thresh)

Evaluates baseline and ThinkDet checkpoint on a single clean prompt
per sample (no prompt variants).
"""

import argparse
import datetime
import json
import os
import sys
import time

ROOT = "/home/iibrohimm/project/next_step"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "GroundingDINO", "GroundingDINO"))

import torch
import torch.distributed as dist
from PIL import Image
from groundingdino.util.misc import nested_tensor_from_tensor_list
from thinkdet.data.refcoco_grounding import build_refcoco_eval
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
from thinkdet.models.projector import build_internvl_transform
from groundingdino.util.inference import load_model as load_gd_model


GD_CONFIG = f"{ROOT}/GroundingDINO/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GD_WEIGHTS = f"{ROOT}/GroundingDINO/GroundingDINO/weights/groundingdino_swint_ogc.pth"
INTERNVL_PATH = f"{ROOT}/InternVL3_5-1B"

DEFAULT_STAGE2 = (
    f"{ROOT}/thinkdet/checkpoints/stage2_tma/"
    "layer9_tma_m8_8gpu_20260219_010134/"
    "thinkdet_tma_stage2_epoch2.pth"
)

FALLBACK_STAGE_ORDER = [
    "primary",
    "llm_feedback",
    "prompt_refine",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="refcoco",
                        choices=["refcoco", "refcoco+", "refcocog"])
    parser.add_argument("--split_by", type=str, default=None,
                        help="Defaults: refcoco/refcoco+=unc, refcocog=umd")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--max_samples", type=int, default=0,
                        help="0 means full split")
    parser.add_argument("--conf_thresh", type=float, default=0.3)
    parser.add_argument("--iou_thresh", type=float, default=0.5)
    parser.add_argument("--log_every", type=int, default=200)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--distributed", action="store_true",
                        help="Enable torch.distributed evaluation (use torchrun).")
    parser.add_argument("--shard_rank", type=int, default=-1,
                        help="Manual sharding rank (no torch.distributed).")
    parser.add_argument("--shard_world_size", type=int, default=1,
                        help="Manual sharding world size.")
    parser.add_argument("--shard_dir", type=str, default="",
                        help="If set, write shard result JSON here.")
    parser.add_argument("--aggregate_shards", action="store_true",
                        help="Aggregate shard_dir results into --output and exit.")

    parser.add_argument("--data_root", type=str,
                        default=f"{ROOT}/dataSets/refer/data")
    parser.add_argument("--image_dir", type=str,
                        default=f"{ROOT}/dataSets/coco/train2017")
    parser.add_argument("--gd_config", type=str, default=GD_CONFIG)
    parser.add_argument("--gd_weights", type=str, default=GD_WEIGHTS)
    parser.add_argument("--internvl_path", type=str, default=INTERNVL_PATH)

    parser.add_argument("--thinkdet_checkpoint", type=str, default=DEFAULT_STAGE2)
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--skip_baseline", action="store_true")
    parser.add_argument(
        "--save_per_sample",
        action="store_true",
        help="Save per-sample score and fallback diagnostics in the output JSON.",
    )
    parser.add_argument("--enable_fallback", action="store_true")
    parser.add_argument("--fallback_min_top1", type=float, default=0.20)
    parser.add_argument(
        "--fallback_min_confidence",
        type=float,
        default=None,
        help="If set with --confidence_calibrator, route fallback on calibrated_confidence instead of raw top1.",
    )
    parser.add_argument("--fallback_min_margin", type=float, default=0.02)
    parser.add_argument("--fallback_min_gate", type=float, default=None)
    parser.add_argument(
        "--confidence_calibrator",
        type=str,
        default="",
        help="Path to a JSON confidence calibrator from fit_confidence_calibrator.py",
    )
    parser.add_argument("--llm_feedback_top_k", type=int, default=20)
    parser.add_argument("--llm_feedback_weight", type=float, default=0.20)
    parser.add_argument("--llm_feedback_max_new_tokens", type=int, default=48)
    parser.add_argument("--llm_feedback_temperature", type=float, default=0.0)
    parser.add_argument("--llm_feedback_improve_margin", type=float, default=0.01)
    parser.add_argument("--prompt_refine", action="store_true")
    parser.add_argument("--prompt_refine_max_variants", type=int, default=2)
    parser.add_argument("--prompt_refine_max_new_tokens", type=int, default=96)
    parser.add_argument("--prompt_refine_temperature", type=float, default=0.0)
    parser.add_argument("--prompt_refine_improve_margin", type=float, default=0.01)
    parser.add_argument(
        "--prompt_refine_require_semantic_preservation",
        dest="prompt_refine_require_semantic_preservation",
        action="store_true",
    )
    parser.add_argument(
        "--no_prompt_refine_require_semantic_preservation",
        dest="prompt_refine_require_semantic_preservation",
        action="store_false",
    )
    parser.set_defaults(prompt_refine_require_semantic_preservation=True)
    parser.add_argument("--prompt_refine_min_shared_terms", type=int, default=1)
    return parser.parse_args()


DEFAULT_SPLIT_BY = {
    "refcoco": "unc",
    "refcoco+": "unc",
    "refcocog": "umd",
}


def resolve_requested_device(requested_device):
    requested = str(requested_device)
    if requested.startswith("cuda"):
        # In this environment torch.cuda.is_available() can return false when NVML
        # probes fail, even though explicit CUDA device use still works.
        return torch.device(requested)
    return torch.device("cpu")


def setup_distributed(requested_device, use_distributed):
    if not use_distributed:
        return False, 0, 1, resolve_requested_device(requested_device)

    if not dist.is_available():
        raise RuntimeError("torch.distributed not available")
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    return True, rank, world_size, device


def get_special_tokens(model):
    tokens = getattr(model, "special_tokens", None)
    if tokens is not None:
        return tokens
    tokenizer = model.tokenizer
    return tokenizer.all_special_ids


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
    positive_map_norm = positive_map / row_sum
    return positive_map_norm


def box_cxcywh_to_xyxy(box):
    cx, cy, w, h = box.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)


def pairwise_iou_xyxy(boxes1, boxes2):
    # boxes1: [N,4], boxes2: [M,4]
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter
    return inter / (union + 1e-6)


def score_outputs(outputs, positive_map_norm):
    logits = outputs["pred_logits"][0].clamp(-50, 50)
    boxes = outputs["pred_boxes"][0]
    text_len = logits.shape[-1]
    pmap = positive_map_norm[:, :text_len].to(logits.device)
    probs = logits.sigmoid()
    scores = (probs * pmap).sum(dim=-1)
    return scores, boxes


def build_scored_bundle(name, outputs, positive_map_norm, query_text, aux=None):
    scores, _ = score_outputs(outputs, positive_map_norm)
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


def build_feedback_predictions(outputs, positive_map_norm, image_size, top_k):
    img_w, img_h = image_size
    scores, boxes = score_outputs(outputs, positive_map_norm)
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


def evaluate_reranked_predictions(
    preds,
    gt_box,
    base_outputs,
    positive_map_norm,
    conf_thresh=0.3,
    iou_thresh=0.5,
):
    metrics = evaluate_single_prediction(
        base_outputs,
        gt_box,
        positive_map_norm,
        conf_thresh=conf_thresh,
        iou_thresh=iou_thresh,
    )

    if not preds:
        metrics["top1_correct"] = 0
        metrics["top1_fp"] = 1
        return metrics

    pred_top = torch.tensor(
        [preds[0]["box_cxcywh"]],
        dtype=gt_box.dtype,
        device=gt_box.device,
    )
    top_iou = pairwise_iou_xyxy(
        box_cxcywh_to_xyxy(pred_top),
        box_cxcywh_to_xyxy(gt_box),
    )[0, 0].item()
    metrics["top1_correct"] = 1 if top_iou >= iou_thresh else 0
    metrics["top1_fp"] = 1 - metrics["top1_correct"]
    return metrics


def evaluate_selected_bundle(selected_bundle, gt_box, conf_thresh=0.3, iou_thresh=0.5):
    payload = selected_bundle.payload
    if payload["type"] == "outputs":
        return evaluate_single_prediction(
            payload["outputs"],
            gt_box,
            payload["positive_map_norm"],
            conf_thresh=conf_thresh,
            iou_thresh=iou_thresh,
        )
    if payload["type"] == "reranked_preds":
        return evaluate_reranked_predictions(
            payload["preds"],
            gt_box,
            payload["outputs"],
            payload["positive_map_norm"],
            conf_thresh=conf_thresh,
            iou_thresh=iou_thresh,
        )
    raise ValueError(f"Unsupported payload type: {payload['type']}")


def evaluate_single_prediction(outputs, gt_box, positive_map_norm, conf_thresh=0.3, iou_thresh=0.5):
    scores, boxes = score_outputs(outputs, positive_map_norm)
    best_idx = int(scores.argmax().item())
    pred_top = boxes[best_idx].unsqueeze(0)
    top_iou = pairwise_iou_xyxy(
        box_cxcywh_to_xyxy(pred_top),
        box_cxcywh_to_xyxy(gt_box),
    )[0, 0].item()
    top1_correct = 1 if top_iou >= iou_thresh else 0
    top1_fp = 1 - top1_correct

    keep = scores > conf_thresh
    det_total = int(keep.sum().item())
    det_fp = 0
    no_det = 0
    img_has_fp = 0
    if det_total > 0:
        kept_boxes = boxes[keep]
        ious = pairwise_iou_xyxy(
            box_cxcywh_to_xyxy(kept_boxes),
            box_cxcywh_to_xyxy(gt_box),
        )[:, 0]
        det_fp = int((ious < iou_thresh).sum().item())
        img_has_fp = 1 if det_fp > 0 else 0
    else:
        no_det = 1

    return {
        "samples": 1,
        "top1_correct": top1_correct,
        "top1_fp": top1_fp,
        "det_total": det_total,
        "det_fp": det_fp,
        "no_det": no_det,
        "img_has_fp": img_has_fp,
    }


def init_counters():
    return {
        "samples": 0,
        "top1_correct": 0,
        "top1_fp": 0,
        "det_total": 0,
        "det_fp": 0,
        "no_det": 0,
        "img_has_fp": 0,
    }


def merge_counters(dst, src):
    for k in dst.keys():
        dst[k] += int(src[k])

def counters_to_tensor(c):
    return torch.tensor([
        float(c["samples"]),
        float(c["top1_correct"]),
        float(c["top1_fp"]),
        float(c["det_total"]),
        float(c["det_fp"]),
        float(c["no_det"]),
        float(c["img_has_fp"]),
    ], dtype=torch.float64)


def tensor_to_counters(t):
    return {
        "samples": int(round(t[0].item())),
        "top1_correct": int(round(t[1].item())),
        "top1_fp": int(round(t[2].item())),
        "det_total": int(round(t[3].item())),
        "det_fp": int(round(t[4].item())),
        "no_det": int(round(t[5].item())),
        "img_has_fp": int(round(t[6].item())),
    }


def summarize_counters(c):
    n = max(c["samples"], 1)
    det_total = max(c["det_total"], 1)
    return {
        "samples": c["samples"],
        "top1_acc": c["top1_correct"] / n,
        "top1_fp_rate": c["top1_fp"] / n,
        "det_fp_rate": c["det_fp"] / det_total,
        "no_det_rate": c["no_det"] / n,
        "img_has_fp_rate": c["img_has_fp"] / n,
    }


def is_weak_by_config(stats, cfg):
    return fallback_is_unreliable(stats, cfg)


def serialize_stats(stats):
    return {k: float(v) for k, v in stats.items()}


def load_thinkdet(args, device):
    ckpt = torch.load(args.thinkdet_checkpoint, map_location="cpu")
    extract_layers = ckpt.get("extract_layers") or [ckpt.get("extract_layer", 9)]
    inject_layers = ckpt.get("injection_layers", DEFAULT_INJECTION_LAYERS)
    tma_m = ckpt.get("tma_m", 8)
    tma_n_heads = ckpt.get("tma_n_heads", 8)
    fusion_mode = ckpt.get("fusion_mode", "concat")
    layer_fusion = ckpt.get("layer_fusion", "mean")
    gd = load_gd_model(args.gd_config, args.gd_weights, device="cpu")
    model = ThinkDetModel(
        grounding_dino=gd,
        internvl_path=args.internvl_path,
        extract_layer=max(extract_layers),
        extract_layers=extract_layers,
        layer_fusion=layer_fusion,
        injection_layers=inject_layers,
        tma_m=tma_m,
        tma_n_heads=tma_n_heads,
        fusion_mode=fusion_mode,
    )
    model.load_state_dict(ckpt["trainable_state_dict"], strict=False)
    model = model.to(device).eval()
    return model


def resolve_confidence_calibration(args):
    calibrator_path = str(args.confidence_calibrator or "").strip()
    min_confidence = args.fallback_min_confidence

    if min_confidence is not None and not calibrator_path:
        raise RuntimeError("--fallback_min_confidence requires --confidence_calibrator")

    if calibrator_path and min_confidence is None:
        min_confidence = 0.50

    calibrator = None
    if calibrator_path and not args.aggregate_shards:
        calibrator = load_confidence_calibrator(calibrator_path)

    return calibrator_path, min_confidence, calibrator


def main():
    args = parse_args()
    calibrator_path, fallback_min_confidence, confidence_calibrator = resolve_confidence_calibration(args)
    split_by = args.split_by or DEFAULT_SPLIT_BY[args.dataset_name]
    if args.aggregate_shards:
        if not args.shard_dir:
            raise RuntimeError("--aggregate_shards requires --shard_dir")
        if not args.output:
            raise RuntimeError("--aggregate_shards requires --output")
        # Aggregate shard summaries.
        totals = {"thinkdet": init_counters()}
        if not args.skip_baseline:
            totals["baseline"] = init_counters()
        if args.enable_fallback:
            totals["thinkdet_fallback"] = init_counters()
        fallback_stage_counts = {stage: 0 for stage in FALLBACK_STAGE_ORDER}
        per_sample = []
        shard_files = sorted(
            f for f in os.listdir(args.shard_dir) if f.endswith(".json")
        )
        if not shard_files:
            raise RuntimeError(f"No shard jsons in {args.shard_dir}")
        for fn in shard_files:
            payload = json.load(open(os.path.join(args.shard_dir, fn)))
            for k in payload["counters"].keys():
                if k not in totals:
                    totals[k] = init_counters()
                merge_counters(totals[k], payload["counters"][k])
            for stage in FALLBACK_STAGE_ORDER:
                fallback_stage_counts[stage] += int(
                    payload.get("fallback_stage_counts", {}).get(stage, 0)
                )
            if args.save_per_sample:
                per_sample.extend(payload.get("per_sample", []))
        summary = {
            "thinkdet": summarize_counters(totals["thinkdet"]),
        }
        if "baseline" in totals:
            summary["baseline"] = summarize_counters(totals["baseline"])
        if "thinkdet_fallback" in totals:
            summary["thinkdet_fallback"] = summarize_counters(totals["thinkdet_fallback"])
        out = {
            "status": "ok",
            "timestamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
            "dataset": {
                "name": args.dataset_name,
                "split_by": split_by,
                "split": args.split,
            },
            "config": {
                "conf_thresh": args.conf_thresh,
                "iou_thresh": args.iou_thresh,
                "thinkdet_checkpoint": args.thinkdet_checkpoint,
                "skip_baseline": bool(args.skip_baseline),
                "shard_world_size": args.shard_world_size,
                "shard_dir": args.shard_dir,
                "fallback": {
                    "enabled": bool(args.enable_fallback),
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
                },
            },
            "summary": summary,
            "fallback_stage_counts": fallback_stage_counts,
        }
        if args.save_per_sample:
            out["per_sample"] = per_sample
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        json.dump(out, open(args.output, "w"), indent=2)
        print("Saved aggregated:", args.output)
        print("ThinkDet top1_acc:", summary["thinkdet"]["top1_acc"])
        if "baseline" in summary:
            print("Baseline top1_acc:", summary["baseline"]["top1_acc"])
        if "thinkdet_fallback" in summary:
            print("ThinkDet fallback top1_acc:", summary["thinkdet_fallback"]["top1_acc"])
        return

    if args.distributed:
        is_dist, rank, world_size, device = setup_distributed(args.device, True)
    else:
        is_dist = False
        rank = 0 if args.shard_rank < 0 else args.shard_rank
        world_size = args.shard_world_size if args.shard_world_size > 0 else 1
        device = resolve_requested_device(args.device)
    is_main = rank == 0

    if args.distributed and args.save_per_sample:
        raise RuntimeError(
            "--save_per_sample is not supported with --distributed; use manual "
            "sharding plus --aggregate_shards instead."
        )

    if not args.output:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"{ROOT}/thinkdet/results/eval/refcoco_eval_{args.dataset_name}_{args.split}_{ts}.json"

    if is_main:
        print("=" * 80)
        print("RefCOCO Benchmark Eval")
        print("=" * 80)
        print(f"dataset:  {args.dataset_name} / {split_by} / {args.split}")
        print(f"device:   {device}")
        print(f"distributed: {is_dist}  world_size: {world_size}")
        print(f"max_samples: {args.max_samples}")
        print(f"conf_thresh: {args.conf_thresh}  iou_thresh: {args.iou_thresh}")
        print(f"thinkdet_checkpoint: {args.thinkdet_checkpoint}")
        print(f"skip_baseline: {args.skip_baseline}")
        print(
            f"fallback: {args.enable_fallback}  "
            f"prompt_refine: {args.prompt_refine}"
        )
        print()

    baseline = None
    if not args.skip_baseline:
        baseline = load_gd_model(args.gd_config, args.gd_weights, device="cpu").to(device).eval()
    thinkdet = load_thinkdet(args, device)
    token_model = baseline if baseline is not None else thinkdet.grounding_dino
    tokenizer = token_model.tokenizer
    special_tokens = get_special_tokens(token_model)
    ivl_tf = build_internvl_transform(448)

    llm_reranker = None
    llm_prompt_refiner = None
    if args.enable_fallback:
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

    dataset = build_refcoco_eval(
        data_root=args.data_root,
        image_dir=args.image_dir,
        dataset_name=args.dataset_name,
        split_by=split_by,
        split=args.split,
    )
    total_dataset = len(dataset)
    total_eval = min(total_dataset, args.max_samples) if args.max_samples > 0 else total_dataset
    indices = list(range(rank, total_eval, world_size))

    counters = {
        "thinkdet": init_counters(),
    }
    if baseline is not None:
        counters["baseline"] = init_counters()
    fallback_stage_counts = {stage: 0 for stage in FALLBACK_STAGE_ORDER}
    if args.enable_fallback:
        counters["thinkdet_fallback"] = init_counters()
    per_sample = [] if args.save_per_sample else None

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

    t0 = time.time()
    with torch.no_grad():
        for local_i, idx in enumerate(indices):
            sample = dataset[idx]
            expression = sample["expression"]
            gt_box = sample["box"].to(device)
            internvl_img = sample["internvl_image"].unsqueeze(0).to(device)
            dino_img = sample["dino_image"].to(device)
            dino_nested = nested_tensor_from_tensor_list([dino_img])
            image_path = os.path.join(args.image_dir, f"{int(sample['image_id']):012d}.jpg")
            image_pil = Image.open(image_path).convert("RGB")

            query_text = expression.strip().lower() + " ."
            pmap_norm = build_positive_map_for_query(
                tokenizer, special_tokens, query_text, max_text_len=512
            )

            if baseline is not None:
                base_out = baseline(samples=dino_nested, captions=[query_text])
                base_metrics = evaluate_single_prediction(
                    base_out,
                    gt_box,
                    pmap_norm,
                    conf_thresh=args.conf_thresh,
                    iou_thresh=args.iou_thresh,
                )
                merge_counters(counters["baseline"], base_metrics)
            else:
                base_out = None
                base_metrics = None

            dino_inputs = {"samples": dino_nested, "captions": [query_text]}
            td_out, td_aux = thinkdet(internvl_img, [query_text], dino_inputs)
            td_metrics = evaluate_single_prediction(
                td_out,
                gt_box,
                pmap_norm,
                conf_thresh=args.conf_thresh,
                iou_thresh=args.iou_thresh,
            )
            merge_counters(counters["thinkdet"], td_metrics)

            primary_bundle = None
            primary_stats = None
            primary_is_weak = None
            if args.enable_fallback or args.save_per_sample:
                primary_bundle = build_scored_bundle(
                    "thinkdet",
                    td_out,
                    pmap_norm,
                    query_text,
                    aux=td_aux,
                )
                primary_stats = summarize_scores(primary_bundle.scores, primary_bundle.aux)
                apply_confidence_calibrator_to_stats(primary_stats, confidence_calibrator)
                primary_is_weak = is_weak_by_config(primary_stats, fallback_cfg)

            base_stats = None
            if args.save_per_sample and base_out is not None:
                base_scores, _ = score_outputs(base_out, pmap_norm)
                base_stats = summarize_scores(base_scores.detach().cpu().tolist())

            selected_bundle = None
            selected_info = None
            fallback_metrics = None

            if args.enable_fallback:
                refined_candidates = []
                if llm_prompt_refiner is not None:
                    refinements = llm_prompt_refiner.refine_prompts(query_text, image_pil=image_pil)
                    for ref_idx, ref_query in enumerate(refinements[:args.prompt_refine_max_variants]):
                        ref_pmap = build_positive_map_for_query(
                            tokenizer,
                            special_tokens,
                            ref_query,
                            max_text_len=512,
                        )
                        ref_inputs = {"samples": dino_nested, "captions": [ref_query]}
                        ref_out, ref_aux = thinkdet(internvl_img, [ref_query], ref_inputs)
                        refined_candidates.append(
                            build_scored_bundle(
                                f"thinkdet_refine_{ref_idx+1}",
                                ref_out,
                                ref_pmap,
                                ref_query,
                                aux=ref_aux,
                            )
                        )

                def rerank_fn(candidate_set):
                    if llm_reranker is None:
                        return None
                    payload = candidate_set.payload
                    preds = build_feedback_predictions(
                        payload["outputs"],
                        payload["positive_map_norm"],
                        image_pil.size,
                        top_k=args.llm_feedback_top_k,
                    )
                    if not preds:
                        return None
                    reranked_preds = llm_reranker.rerank_predictions(
                        image_pil,
                        candidate_set.query_text or query_text,
                        preds,
                    )
                    return ScoredCandidateSet(
                        name=f"{candidate_set.name}_llm_feedback",
                        scores=[float(p.get("combined_score", p["score"])) for p in reranked_preds],
                        payload={
                            "type": "reranked_preds",
                            "preds": reranked_preds,
                            "outputs": payload["outputs"],
                            "positive_map_norm": payload["positive_map_norm"],
                        },
                        aux=candidate_set.aux,
                        query_text=candidate_set.query_text,
                    )

                selected_bundle, selected_info = apply_fallback_policy(
                    primary=primary_bundle,
                    rerank_fn=rerank_fn,
                    refined_candidates=refined_candidates,
                    config=fallback_cfg,
                )
                fallback_metrics = evaluate_selected_bundle(
                    selected_bundle,
                    gt_box,
                    conf_thresh=args.conf_thresh,
                    iou_thresh=args.iou_thresh,
                )
                merge_counters(counters["thinkdet_fallback"], fallback_metrics)
                fallback_stage_counts[selected_info["selected_stage"]] += 1

            if args.save_per_sample:
                record = {
                    "image_id": int(sample["image_id"]),
                    "expression": expression,
                    "query_text": query_text,
                    "thinkdet": {
                        "stats": serialize_stats(primary_stats),
                        "weak_under_config": bool(primary_is_weak),
                        "top1_correct": int(td_metrics["top1_correct"]),
                        "no_det": int(td_metrics["no_det"]),
                    },
                }
                if base_stats is not None and base_metrics is not None:
                    record["baseline"] = {
                        "stats": serialize_stats(base_stats),
                        "top1_correct": int(base_metrics["top1_correct"]),
                        "no_det": int(base_metrics["no_det"]),
                    }
                if selected_info is not None and fallback_metrics is not None:
                    record["fallback"] = {
                        "selected_stage": selected_info["selected_stage"],
                        "selected_name": selected_info["selected_name"],
                        "selected_query_text": selected_info["selected_query_text"],
                        "selected_stats": serialize_stats(selected_info["selected_stats"]),
                        "top1_correct": int(fallback_metrics["top1_correct"]),
                        "no_det": int(fallback_metrics["no_det"]),
                    }
                per_sample.append(record)

            if is_main and (local_i + 1) % max(1, args.log_every) == 0:
                elapsed = time.time() - t0
                rate = (local_i + 1) / max(elapsed, 1e-6)
                print(f"[rank{rank}] {local_i+1}/{len(indices)} samples ({rate:.2f} samples/s)")

    if is_dist:
        for name in list(counters.keys()):
            t = counters_to_tensor(counters[name]).to(device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            counters[name] = tensor_to_counters(t.cpu())
        if args.enable_fallback:
            t = torch.tensor(
                [float(fallback_stage_counts[stage]) for stage in FALLBACK_STAGE_ORDER],
                dtype=torch.float64,
                device=device,
            )
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            fallback_stage_counts = {
                stage: int(round(t[idx].item()))
                for idx, stage in enumerate(FALLBACK_STAGE_ORDER)
            }
    elif args.shard_dir:
        os.makedirs(args.shard_dir, exist_ok=True)
        shard_payload = {
            "rank": rank,
            "world_size": world_size,
            "counters": counters,
            "fallback_stage_counts": fallback_stage_counts,
        }
        if args.save_per_sample:
            shard_payload["per_sample"] = per_sample
        shard_path = os.path.join(args.shard_dir, f"shard_rank{rank}.json")
        with open(shard_path, "w") as f:
            json.dump(shard_payload, f, indent=2)
        if not is_main:
            return

    if not is_main:
        return

    summary = {
        "thinkdet": summarize_counters(counters["thinkdet"]),
    }
    if "baseline" in counters:
        summary["baseline"] = summarize_counters(counters["baseline"])
    if "thinkdet_fallback" in counters:
        summary["thinkdet_fallback"] = summarize_counters(counters["thinkdet_fallback"])

    payload = {
        "status": "ok",
        "timestamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
        "dataset": {
            "name": args.dataset_name,
            "split_by": split_by,
            "split": args.split,
            "total_samples": total_dataset,
            "evaluated_samples": total_eval,
        },
            "config": {
                "conf_thresh": args.conf_thresh,
                "iou_thresh": args.iou_thresh,
                "thinkdet_checkpoint": args.thinkdet_checkpoint,
                "skip_baseline": bool(args.skip_baseline),
                "fallback": {
                    "enabled": bool(args.enable_fallback),
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
        },
        "summary": summary,
        "fallback_stage_counts": fallback_stage_counts,
    }
    if args.save_per_sample:
        payload["per_sample"] = per_sample

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)

    print("\nSaved:", args.output)
    print("ThinkDet top1_acc:", summary["thinkdet"]["top1_acc"])
    if "baseline" in summary:
        print("Baseline top1_acc:", summary["baseline"]["top1_acc"])
    if "thinkdet_fallback" in summary:
        print("ThinkDet fallback top1_acc:", summary["thinkdet_fallback"]["top1_acc"])


if __name__ == "__main__":
    main()
