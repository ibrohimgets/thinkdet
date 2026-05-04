#!/usr/bin/env python3
"""
ThinkDet Prompt Robustness Benchmark

Compares baseline GroundingDINO vs ThinkDet on the same RefCOCO samples using
three prompt settings:
  1) clean
  2) ambiguous
  3) adversarial

Metrics (per setting, per model):
  - Acc@0.5 (top-1 REC style)
  - Top-1 hallucination rate (1 - top-1 accuracy)
  - Detection FPR over all predictions above score threshold
  - Image-level FP rate, no-detection rate, mean detections/image

Usage example (single GPU):
  python scripts/eval/eval_prompt_robustness.py \
    --thinkdet_checkpoint /path/to/thinkdet_refcoco_epoch10.pth \
    --dataset_name refcoco --split val \
    --max_samples 2000 \
    --output /tmp/prompt_robustness.json

Usage example (8 GPUs):
  torchrun --nproc_per_node=8 scripts/eval/eval_prompt_robustness.py \
    --thinkdet_checkpoint /path/to/thinkdet_refcoco_epoch10.pth \
    --dataset_name refcoco --split val \
    --max_samples 8000 \
    --output /tmp/prompt_robustness_8gpu.json
"""

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict

import torch
import torch.distributed as dist
from PIL import Image

ROOT = "/home/iibrohimm/project/next_step"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "GroundingDINO", "GroundingDINO"))

from groundingdino.util.inference import load_model as load_gd_model
from groundingdino.util.misc import nested_tensor_from_tensor_list

from thinkdet.data.refcoco_grounding import build_refcoco_eval
from thinkdet.inference.fallback import (
    FallbackPolicyConfig,
    InternVLPromptRefiner,
    InternVLYesNoReranker,
    ScoredCandidateSet,
    apply_fallback_policy,
)
from thinkdet.models.arch import ThinkDetModel, DEFAULT_INJECTION_LAYERS
from thinkdet.models.projector import build_internvl_transform


ALLOWED_LAYER_SETUPS = {
    "layer9": [9],
    "layer10": [10],
    "fusion_9_10": [9, 10],
    "fusion_9_10_11": [9, 10, 11],
}

DEFAULT_SPLIT_BY = {
    "refcoco": "unc",
    "refcoco+": "unc",
    "refcocog": "umd",
}

STOPWORDS = {
    "a", "an", "the", "this", "that", "these", "those", "is", "are", "was", "were",
    "be", "being", "been", "to", "of", "in", "on", "at", "for", "from", "with",
    "without", "and", "or", "but", "as", "by", "it", "its", "there", "here", "who",
    "whom", "which", "what", "where", "when", "why", "how", "near", "next", "beside",
    "behind", "front", "left", "right", "upper", "lower", "top", "bottom", "middle",
    "center", "centre", "object", "thing", "item", "entity", "person", "man", "woman",
}

DIRECTION_SWAP = {
    "left": "right",
    "right": "left",
    "above": "below",
    "below": "above",
    "top": "bottom",
    "bottom": "top",
    "front": "back",
    "back": "front",
    "inside": "outside",
    "outside": "inside",
}

COLOR_SWAP = {
    "red": "blue",
    "blue": "red",
    "green": "purple",
    "purple": "green",
    "black": "white",
    "white": "black",
    "yellow": "brown",
    "brown": "yellow",
    "orange": "gray",
    "grey": "orange",
    "gray": "orange",
    "pink": "black",
}

FALLBACK_STAGE_ORDER = [
    "primary",
    "llm_feedback",
    "prompt_refine",
]


def normalize_query(text):
    text = " ".join(str(text).strip().lower().split())
    text = re.sub(r"\s*\.\s*$", "", text)
    if not text:
        text = "object"
    return f"{text} ."


def tokenize_words(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def extract_anchor_words(text, max_words=3):
    anchors = []
    seen = set()
    for tok in tokenize_words(text):
        if len(tok) <= 2 or tok in STOPWORDS:
            continue
        if tok in seen:
            continue
        seen.add(tok)
        anchors.append(tok)
        if len(anchors) >= max_words:
            break
    if not anchors:
        toks = tokenize_words(text)
        if toks:
            anchors = [toks[0]]
        else:
            anchors = ["object"]
    return anchors


def make_ambiguous_query(expression):
    anchors = extract_anchor_words(expression, max_words=1)
    return normalize_query(anchors[0])


def make_adversarial_query(expression):
    words = tokenize_words(expression)
    swapped = []
    changed = False
    for w in words:
        if w in DIRECTION_SWAP:
            swapped.append(DIRECTION_SWAP[w])
            changed = True
        elif w in COLOR_SWAP:
            swapped.append(COLOR_SWAP[w])
            changed = True
        else:
            swapped.append(w)

    if not swapped:
        return normalize_query("different object on the opposite side")

    if not changed:
        anchor = extract_anchor_words(expression, max_words=1)[0]
        return normalize_query(f"{anchor} on the opposite side")

    return normalize_query(" ".join(swapped))


def build_prompt_variants(expression):
    clean = normalize_query(expression)
    ambiguous = make_ambiguous_query(expression)
    adversarial = make_adversarial_query(expression)
    return {
        "clean": clean,
        "ambiguous": ambiguous,
        "adversarial": adversarial,
    }


def setup_distributed(device_arg):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return False, 0, 1, torch.device(device_arg)

    if not torch.cuda.is_available():
        raise RuntimeError("Distributed evaluation requires CUDA")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    device = torch.device(f"cuda:{local_rank}")
    return True, rank, world_size, device


def cleanup_distributed(is_dist):
    if is_dist and dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def box_cxcywh_to_xyxy(x):
    cx, cy, w, h = x.unbind(-1)
    return torch.stack([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], dim=-1)


def pairwise_iou_xyxy(boxes1, boxes2):
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    a1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    a2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = a1[:, None] + a2[None, :] - inter
    return inter / (union + 1e-6)


def get_special_tokens(model):
    tokens = getattr(model, "specical_tokens", None)
    if tokens is not None:
        return tokens

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


def score_outputs(outputs, positive_map_norm):
    logits = outputs["pred_logits"][0].clamp(-50, 50)  # [nq, text_len]
    boxes = outputs["pred_boxes"][0]                   # [nq, 4]
    text_len = logits.shape[-1]
    pmap = positive_map_norm[:, :text_len].to(logits.device)
    probs = logits.sigmoid()
    scores = (probs * pmap).sum(dim=-1)                # [nq]
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


def evaluate_selected_bundle(
    selected_bundle,
    gt_box,
    conf_thresh=0.3,
    iou_thresh=0.5,
):
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

    # Top-1 metric (REC style)
    best_idx = int(scores.argmax().item())
    pred_top = boxes[best_idx].unsqueeze(0)
    top_iou = pairwise_iou_xyxy(
        box_cxcywh_to_xyxy(pred_top),
        box_cxcywh_to_xyxy(gt_box),
    )[0, 0].item()
    top1_correct = 1 if top_iou >= iou_thresh else 0
    top1_fp = 1 - top1_correct

    # Detection-level FPR (all predictions above conf threshold)
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
        "num_samples": c["samples"],
        "acc_at_05": c["top1_correct"] / n,
        "top1_hallucination_rate": c["top1_fp"] / n,
        "detection_fpr": c["det_fp"] / det_total,
        "image_fp_rate": c["img_has_fp"] / n,
        "no_detection_rate": c["no_det"] / n,
        "mean_detections_per_image": c["det_total"] / n,
        "raw": c,
    }


def choose_layers(args, checkpoint_dict):
    if args.layer_setup:
        return list(ALLOWED_LAYER_SETUPS[args.layer_setup]), "mean"

    ck_layers = checkpoint_dict.get("extract_layers")
    if ck_layers:
        return [int(x) for x in ck_layers], checkpoint_dict.get("layer_fusion", "mean")

    ck_layer = checkpoint_dict.get("extract_layer")
    if ck_layer is not None:
        return [int(ck_layer)], checkpoint_dict.get("layer_fusion", "mean")

    if args.extract_layers:
        return [int(x) for x in args.extract_layers], args.layer_fusion

    return [10], "mean"


def load_models(args, device, is_main):
    gd_config = args.gd_config
    gd_weights = args.gd_weights

    if is_main:
        print("[model] Loading baseline GroundingDINO ...")
    baseline = load_gd_model(gd_config, gd_weights, device="cpu").to(device).eval()

    thinkdet = None
    thinkdet_meta = {}
    if args.thinkdet_checkpoint:
        if is_main:
            print(f"[model] Loading ThinkDet checkpoint: {args.thinkdet_checkpoint}")
        if not os.path.exists(args.thinkdet_checkpoint):
            raise FileNotFoundError(f"Checkpoint not found: {args.thinkdet_checkpoint}")

        ckpt = torch.load(args.thinkdet_checkpoint, map_location="cpu")
        extract_layers, layer_fusion = choose_layers(args, ckpt)
        uncertainty_enabled = ckpt.get("uncertainty_enabled", not args.disable_uncertainty)
        uncertainty_num_variants = int(
            ckpt.get("uncertainty_num_variants", args.uncertainty_variants)
        )
        uncertainty_beta = float(ckpt.get("uncertainty_beta", args.uncertainty_beta))
        uncertainty_min_reliability = float(
            ckpt.get("uncertainty_min_reliability", args.uncertainty_min_reliability)
        )
        delta_gain = float(ckpt.get("delta_gain", 1.0))
        inject_layers = ckpt.get("injection_layers", DEFAULT_INJECTION_LAYERS)
        tma_m = int(ckpt.get("tma_m", 8))
        tma_n_heads = int(ckpt.get("tma_n_heads", 8))
        fusion_mode = ckpt.get("fusion_mode", "concat")

        gd_thinkdet = load_gd_model(gd_config, gd_weights, device="cpu")
        thinkdet = ThinkDetModel(
            grounding_dino=gd_thinkdet,
            internvl_path=args.internvl_path,
            extract_layer=max(extract_layers),
            extract_layers=extract_layers,
            layer_fusion=layer_fusion,
            injection_layers=inject_layers,
            d_model=256,
            tma_m=tma_m,
            tma_n_heads=tma_n_heads,
            fusion_mode=fusion_mode,
        )

        state = ckpt.get("trainable_state_dict", ckpt.get("state_dict", ckpt))
        missing, unexpected = thinkdet.load_state_dict(state, strict=False)
        thinkdet = thinkdet.to(device).eval()

        thinkdet_meta = {
            "extract_layers": extract_layers,
            "layer_fusion": layer_fusion,
            "uncertainty_enabled": uncertainty_enabled,
            "uncertainty_num_variants": uncertainty_num_variants,
            "uncertainty_beta": uncertainty_beta,
            "uncertainty_min_reliability": uncertainty_min_reliability,
            "delta_gain": delta_gain,
            "injection_layers": inject_layers,
            "tma_m": tma_m,
            "tma_n_heads": tma_n_heads,
            "fusion_mode": fusion_mode,
            "missing_keys": len(missing),
            "unexpected_keys": len(unexpected),
        }

        if is_main:
            print(
                "[model] ThinkDet config: "
                f"layers={extract_layers} inj={inject_layers} "
                f"fusion={layer_fusion}/{fusion_mode} "
                f"uncertainty={uncertainty_enabled} variants={uncertainty_num_variants} "
                f"delta_gain={delta_gain}"
            )
            print(
                f"[model] Checkpoint load: missing={len(missing)} unexpected={len(unexpected)}"
            )
    elif is_main:
        print("[model] ThinkDet checkpoint not provided. Running baseline-only.")

    return baseline, thinkdet, thinkdet_meta


def evaluate(args):
    is_dist, rank, world_size, device = setup_distributed(args.device)
    is_main = rank == 0

    try:
        split_by = args.split_by or DEFAULT_SPLIT_BY[args.dataset_name]

        if is_main:
            print("=" * 80)
            print("ThinkDet Prompt Robustness Benchmark")
            print("=" * 80)
            print(f"dataset:      {args.dataset_name} / {split_by} / {args.split}")
            print(f"device:       {device}")
            print(f"world_size:   {world_size}")
            print(f"max_samples:  {args.max_samples}")
            print(f"conf_thresh:  {args.conf_thresh}")
            print(f"iou_thresh:   {args.iou_thresh}")
            print()

        baseline, thinkdet, thinkdet_meta = load_models(args, device, is_main)
        tokenizer = baseline.tokenizer
        special_tokens = get_special_tokens(baseline)
        ivl_tf = build_internvl_transform(448)

        llm_reranker = None
        llm_prompt_refiner = None
        if thinkdet is not None and args.enable_fallback:
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

        if is_main:
            print(f"[data] Total dataset samples: {total_dataset}")
            print(f"[data] Evaluating samples:    {total_eval}")
            if is_dist:
                print(f"[data] Rank-0 shard size:     {len(indices)}")
            print()

        counters = {
            "baseline": {
                "clean": init_counters(),
                "ambiguous": init_counters(),
                "adversarial": init_counters(),
            },
        }
        fallback_stage_counts = {}
        if thinkdet is not None:
            counters["thinkdet"] = {
                "clean": init_counters(),
                "ambiguous": init_counters(),
                "adversarial": init_counters(),
            }
            if args.enable_fallback:
                counters["thinkdet_fallback"] = {
                    "clean": init_counters(),
                    "ambiguous": init_counters(),
                    "adversarial": init_counters(),
                }
                fallback_stage_counts = {
                    mode_name: {stage: 0 for stage in FALLBACK_STAGE_ORDER}
                    for mode_name in ("clean", "ambiguous", "adversarial")
                }

        fallback_cfg = FallbackPolicyConfig(
            min_top1=args.fallback_min_top1,
            min_margin=args.fallback_min_margin,
            min_gate=args.fallback_min_gate,
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
                variants = build_prompt_variants(expression)

                for mode_name, query_text in variants.items():
                    pmap_norm = build_positive_map_for_query(
                        tokenizer, special_tokens, query_text, max_text_len=512
                    )

                    base_out = baseline(samples=dino_nested, captions=[query_text])
                    base_metrics = evaluate_single_prediction(
                        base_out,
                        gt_box,
                        pmap_norm,
                        conf_thresh=args.conf_thresh,
                        iou_thresh=args.iou_thresh,
                    )
                    merge_counters(counters["baseline"][mode_name], base_metrics)

                    if thinkdet is not None:
                        dino_inputs = {"samples": dino_nested, "captions": [query_text]}
                        td_out, td_aux = thinkdet(internvl_img, [query_text], dino_inputs)
                        td_metrics = evaluate_single_prediction(
                            td_out,
                            gt_box,
                            pmap_norm,
                            conf_thresh=args.conf_thresh,
                            iou_thresh=args.iou_thresh,
                        )
                        merge_counters(counters["thinkdet"][mode_name], td_metrics)

                        if args.enable_fallback:
                            primary_bundle = build_scored_bundle(
                                "thinkdet",
                                td_out,
                                pmap_norm,
                                query_text,
                                aux=td_aux,
                            )

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
                                    scores=[
                                        float(p.get("combined_score", p["score"]))
                                        for p in reranked_preds
                                    ],
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
                            merge_counters(counters["thinkdet_fallback"][mode_name], fallback_metrics)
                            stage_name = selected_info["selected_stage"]
                            if stage_name not in fallback_stage_counts[mode_name]:
                                fallback_stage_counts[mode_name][stage_name] = 0
                            fallback_stage_counts[mode_name][stage_name] += 1

                if is_main and (local_i + 1) % max(1, args.log_every) == 0:
                    elapsed = time.time() - t0
                    rate = (local_i + 1) / max(elapsed, 1e-6)
                    print(
                        f"[rank{rank}] {local_i + 1}/{len(indices)} samples "
                        f"({rate:.2f} samples/s)"
                    )

        # Reduce counters across ranks.
        if is_dist:
            for model_name in counters.keys():
                for mode_name in counters[model_name].keys():
                    t = counters_to_tensor(counters[model_name][mode_name]).to(device)
                    dist.all_reduce(t, op=dist.ReduceOp.SUM)
                    counters[model_name][mode_name] = tensor_to_counters(t.cpu())
            if thinkdet is not None and args.enable_fallback:
                for mode_name in fallback_stage_counts.keys():
                    t = torch.tensor(
                        [float(fallback_stage_counts[mode_name].get(stage, 0)) for stage in FALLBACK_STAGE_ORDER],
                        dtype=torch.float64,
                        device=device,
                    )
                    dist.all_reduce(t, op=dist.ReduceOp.SUM)
                    fallback_stage_counts[mode_name] = {
                        stage: int(round(t[idx].item()))
                        for idx, stage in enumerate(FALLBACK_STAGE_ORDER)
                    }

        if not is_main:
            return None

        elapsed = time.time() - t0
        summary = {}
        for model_name in counters.keys():
            summary[model_name] = {}
            for mode_name in counters[model_name].keys():
                summary[model_name][mode_name] = summarize_counters(
                    counters[model_name][mode_name]
                )

        deltas = {}
        if "thinkdet" in summary:
            for mode_name in ("clean", "ambiguous", "adversarial"):
                b = summary["baseline"][mode_name]
                t = summary["thinkdet"][mode_name]
                deltas[mode_name] = {
                    "delta_acc_at_05": t["acc_at_05"] - b["acc_at_05"],
                    "delta_top1_hallucination_rate": (
                        t["top1_hallucination_rate"] - b["top1_hallucination_rate"]
                    ),
                    "delta_detection_fpr": t["detection_fpr"] - b["detection_fpr"],
                    "delta_image_fp_rate": t["image_fp_rate"] - b["image_fp_rate"],
                }
        if "thinkdet_fallback" in summary:
            for mode_name in ("clean", "ambiguous", "adversarial"):
                b = summary["baseline"][mode_name]
                t = summary["thinkdet_fallback"][mode_name]
                deltas[f"{mode_name}_fallback"] = {
                    "delta_acc_at_05": t["acc_at_05"] - b["acc_at_05"],
                    "delta_top1_hallucination_rate": (
                        t["top1_hallucination_rate"] - b["top1_hallucination_rate"]
                    ),
                    "delta_detection_fpr": t["detection_fpr"] - b["detection_fpr"],
                    "delta_image_fp_rate": t["image_fp_rate"] - b["image_fp_rate"],
                }

        print("\n" + "=" * 80)
        print("Results")
        print("=" * 80)
        for model_name in summary.keys():
            print(f"\n[{model_name}]")
            for mode_name in ("clean", "ambiguous", "adversarial"):
                m = summary[model_name][mode_name]
                print(
                    f"  {mode_name:11s} "
                    f"acc@0.5={m['acc_at_05']:.4f}  "
                    f"top1_hall={m['top1_hallucination_rate']:.4f}  "
                    f"det_fpr={m['detection_fpr']:.4f}  "
                    f"img_fp={m['image_fp_rate']:.4f}  "
                    f"mean_det={m['mean_detections_per_image']:.2f}"
                )

        if deltas:
            print("\n[thinkdet - baseline deltas]")
            for mode_name in ("clean", "ambiguous", "adversarial"):
                d = deltas[mode_name]
                print(
                    f"  {mode_name:11s} "
                    f"dAcc={d['delta_acc_at_05']:+.4f}  "
                    f"dTop1Hall={d['delta_top1_hallucination_rate']:+.4f}  "
                    f"dDetFPR={d['delta_detection_fpr']:+.4f}  "
                    f"dImgFP={d['delta_image_fp_rate']:+.4f}"
                )
            if "clean_fallback" in deltas:
                print("\n[thinkdet_fallback - baseline deltas]")
                for mode_name in ("clean", "ambiguous", "adversarial"):
                    d = deltas[f"{mode_name}_fallback"]
                    print(
                        f"  {mode_name:11s} "
                        f"dAcc={d['delta_acc_at_05']:+.4f}  "
                        f"dTop1Hall={d['delta_top1_hallucination_rate']:+.4f}  "
                        f"dDetFPR={d['delta_detection_fpr']:+.4f}  "
                        f"dImgFP={d['delta_image_fp_rate']:+.4f}"
                    )

        payload = {
            "config": {
                "dataset_name": args.dataset_name,
                "split_by": split_by,
                "split": args.split,
                "max_samples": total_eval,
                "conf_thresh": args.conf_thresh,
                "iou_thresh": args.iou_thresh,
                "world_size": world_size,
                "elapsed_min": round(elapsed / 60.0, 3),
                "thinkdet_checkpoint": args.thinkdet_checkpoint,
                "thinkdet_meta": thinkdet_meta,
                "fallback": {
                    "enabled": bool(args.enable_fallback),
                    "min_top1": float(args.fallback_min_top1),
                    "min_margin": float(args.fallback_min_margin),
                    "min_gate": None if args.fallback_min_gate is None else float(args.fallback_min_gate),
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
            "deltas": deltas,
            "fallback_stage_counts": fallback_stage_counts,
        }

        if args.output:
            out_dir = os.path.dirname(args.output)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            with open(args.output, "w") as f:
                json.dump(payload, f, indent=2)
            print(f"\nSaved: {args.output}")

        return payload
    finally:
        cleanup_distributed(is_dist)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="refcoco",
                        choices=["refcoco", "refcoco+", "refcocog"])
    parser.add_argument("--split_by", type=str, default=None,
                        help="Defaults: refcoco/refcoco+=unc, refcocog=umd")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--max_samples", type=int, default=0,
                        help="0 means full split")
    parser.add_argument("--conf_thresh", type=float, default=0.30)
    parser.add_argument("--iou_thresh", type=float, default=0.50)
    parser.add_argument("--log_every", type=int, default=200)

    parser.add_argument("--data_root", type=str,
                        default=f"{ROOT}/dataSets/refer/data")
    parser.add_argument("--image_dir", type=str,
                        default=f"{ROOT}/dataSets/coco/train2017")
    parser.add_argument("--gd_config", type=str,
                        default=f"{ROOT}/GroundingDINO/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    parser.add_argument("--gd_weights", type=str,
                        default=f"{ROOT}/GroundingDINO/GroundingDINO/weights/groundingdino_swint_ogc.pth")
    parser.add_argument("--internvl_path", type=str,
                        default=f"{ROOT}/InternVL3_5-1B")

    parser.add_argument("--thinkdet_checkpoint", type=str, default=None)
    parser.add_argument("--layer_setup", type=str, default=None,
                        choices=sorted(ALLOWED_LAYER_SETUPS.keys()),
                        help="Optional override when checkpoint metadata is missing")
    parser.add_argument("--extract_layers", type=int, nargs="+", default=None)
    parser.add_argument("--layer_fusion", type=str, default="mean",
                        choices=["mean", "last"])
    parser.add_argument("--disable_uncertainty", action="store_true")
    parser.add_argument("--uncertainty_variants", type=int, default=2)
    parser.add_argument("--uncertainty_beta", type=float, default=8.0)
    parser.add_argument("--uncertainty_min_reliability", type=float, default=0.05)

    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--enable_fallback", action="store_true")
    parser.add_argument("--fallback_min_top1", type=float, default=0.20)
    parser.add_argument("--fallback_min_margin", type=float, default=0.02)
    parser.add_argument("--fallback_min_gate", type=float, default=None)
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

    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
