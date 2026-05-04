"""
Evaluate a single unified ThinkDet checkpoint on the held-out affordance benchmark.

Outputs:
    - overall summary
    - per-affordance summary
    - per-sample records for paired bootstrap / bucket analysis
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
import torchvision.transforms as T
import torchvision.transforms.functional as TF

ROOT = "/home/iibrohimm/project/next_step"
sys.path.insert(0, ROOT)
GROUNDING_DINO_ROOT = os.path.join(ROOT, "GroundingDINO")
sys.path.insert(0, GROUNDING_DINO_ROOT)

from thinkdet.models.arch import ThinkDetModel, DEFAULT_INJECTION_LAYERS
from groundingdino.util.inference import load_model as load_gd_model
from groundingdino.util.misc import NestedTensor
from thinkdet.inference.fallback import InternVLYesNoReranker


DEFAULT_BENCH = f"{ROOT}/thinkdet/data/benchmarks/affordance_coco_val_heldout_v1.json"
DEFAULT_CKPT = (
    f"{ROOT}/thinkdet/checkpoints/unified/"
    "layer9_kd0p05_l1_1e-4_20260224_135540/"
    "thinkdet_unified_epoch5.pth"
)
DEFAULT_OUT = (
    f"{ROOT}/thinkdet/results/layer_ablation/"
    f"affordance_layer_eval_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
)
GD_CONFIG = f"{GROUNDING_DINO_ROOT}/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GD_WEIGHTS = f"{GROUNDING_DINO_ROOT}/weights/groundingdino_swint_ogc.pth"
INTERNVL_PATH = f"{ROOT}/InternVL3_5-1B"


def patch_groundingdino_ms_deform_attn():
    """Use the PyTorch attention fallback when GroundingDINO custom ops are absent."""
    try:
        from groundingdino.models.GroundingDINO import ms_deform_attn
    except Exception:
        return
    if hasattr(ms_deform_attn, "_C"):
        return

    def fallback_apply(
        value,
        value_spatial_shapes,
        value_level_start_index,
        sampling_locations,
        attention_weights,
        im2col_step,
    ):
        return ms_deform_attn.multi_scale_deformable_attn_pytorch(
            value,
            value_spatial_shapes,
            sampling_locations,
            attention_weights,
        )

    ms_deform_attn.MultiScaleDeformableAttnFunction.apply = staticmethod(fallback_apply)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=str, default=DEFAULT_BENCH)
    parser.add_argument("--split", type=str, default="test", choices=["dev", "test", "all"])
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output", type=str, default=DEFAULT_OUT)

    parser.add_argument("--gd_config", type=str, default=GD_CONFIG)
    parser.add_argument("--gd_weights", type=str, default=GD_WEIGHTS)
    parser.add_argument("--internvl_path", type=str, default=INTERNVL_PATH)
    parser.add_argument("--thinkdet_checkpoint", type=str, default=DEFAULT_CKPT)
    parser.add_argument("--extract_layer", type=int, default=None)
    parser.add_argument("--extract_layers", type=int, nargs="+", default=None)
    parser.add_argument("--layer_fusion", type=str, default=None, choices=["mean", "last"])
    parser.add_argument(
        "--force_gate0",
        action="store_true",
        help="Disable the adapter path and run the checkpoint as a gate-off control.",
    )
    parser.add_argument(
        "--force_gate_value",
        type=float,
        default=None,
        help="Override the post-tanh adapter gate value at eval time (range: -0.99 to 0.99).",
    )
    parser.add_argument(
        "--rerank_vlm_delta",
        action="store_true",
        help="Rerank learned candidates by how much their score rises relative to a gate-off pass.",
    )
    parser.add_argument(
        "--rerank_top_k",
        type=int,
        default=20,
        help="Candidate pool size used before VLM-delta reranking.",
    )
    parser.add_argument(
        "--rerank_delta_lambda",
        type=float,
        default=2.0,
        help="Weight for the normalized VLM-delta term in reranking.",
    )
    parser.add_argument(
        "--rerank_match_iou",
        type=float,
        default=0.3,
        help="IoU threshold for matching learned boxes to gate-off boxes during reranking.",
    )
    parser.add_argument(
        "--rerank_affordances",
        type=str,
        nargs="+",
        default=None,
        help="Optional affordance allow-list for VLM-delta reranking (for example: carry_in talk_on).",
    )
    parser.add_argument(
        "--llm_judge",
        action="store_true",
        help="Use InternVL evidence-check feedback to rerank a small candidate set.",
    )
    parser.add_argument(
        "--llm_judge_pool_k",
        type=int,
        default=5,
        help="Number of candidate boxes passed to the LLM judge.",
    )
    parser.add_argument(
        "--llm_judge_weight",
        type=float,
        default=0.2,
        help="Combined score = base_candidate_score + weight * llm_yes_no_score.",
    )
    parser.add_argument(
        "--llm_judge_max_new_tokens",
        type=int,
        default=48,
        help="Max generation tokens for the InternVL evidence-check judge.",
    )
    parser.add_argument(
        "--llm_judge_temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for the InternVL yes/no judge.",
    )
    parser.add_argument(
        "--llm_judge_affordances",
        type=str,
        nargs="+",
        default=None,
        help="Optional affordance allow-list for the LLM judge.",
    )
    return parser.parse_args()


def build_dino_transform():
    normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def transform(image_pil):
        w, h = image_pil.size
        scale = 800 / min(w, h)
        if scale * max(w, h) > 1333:
            scale = 1333 / max(w, h)
        nw, nh = int(w * scale), int(h * scale)
        img = image_pil.resize((nw, nh), Image.BILINEAR)
        t = TF.to_tensor(img)
        return normalize(t)

    return transform


def build_internvl_transform():
    return T.Compose([
        T.Resize((448, 448), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def xywh_to_xyxy(box):
    x, y, w, h = box
    return [x, y, x + w, y + h]


def iou_xyxy(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (aa + bb - inter + 1e-9)


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


def resolve_query_scoring_assets(model_fn):
    if hasattr(model_fn, "grounding_dino"):
        dino_model = model_fn.grounding_dino
    else:
        dino_model = model_fn
    if not hasattr(dino_model, "tokenizer"):
        raise AttributeError("Could not resolve GroundingDINO tokenizer for query scoring")
    return dino_model.tokenizer, get_special_tokens(dino_model)


def target_to_raw_gate(value):
    if value >= 0.99:
        return 5.0
    if value <= -0.99:
        return -5.0
    return math.atanh(value)


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
    gate_override_applied = None
    if args.force_gate_value is not None:
        gate_value = float(args.force_gate_value)
        if not -0.99 <= gate_value <= 0.99:
            raise ValueError("--force_gate_value must be within [-0.99, 0.99]")
        raw_gate = target_to_raw_gate(gate_value)
        for layer in model.adapted_layers:
            layer.augmenter.alpha.data.fill_(raw_gate)
        gate_override_applied = gate_value
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
        "force_gate_value": gate_override_applied,
    }


@torch.no_grad()
def run_topk(
    model_fn,
    image_pil,
    prompt,
    device,
    dino_tf,
    ivl_tf,
    top_k,
    tokenizer,
    special_tokens,
    force_gate0=False,
):
    img_w, img_h = image_pil.size
    dino_t = dino_tf(image_pil).unsqueeze(0).to(device)
    mask = torch.zeros(1, dino_t.shape[2], dino_t.shape[3], dtype=torch.bool, device=device)
    nested = NestedTensor(dino_t, mask)
    ivl_t = ivl_tf(image_pil).unsqueeze(0).to(device)

    query_text = prompt if prompt.strip().endswith(".") else (prompt.strip() + " .")
    outputs, _ = model_fn(
        ivl_t,
        [prompt],
        {"samples": nested, "captions": [query_text]},
        force_gate0=force_gate0,
    )

    positive_map_norm = build_positive_map_for_query(
        tokenizer, special_tokens, query_text, max_text_len=512
    )
    scores, boxes = score_outputs_for_query(outputs, positive_map_norm)
    vals, idx = scores.topk(min(int(top_k), scores.shape[0]))

    results = []
    for j in range(len(vals)):
        cx, cy, bw, bh = boxes[idx[j]].tolist()
        x1 = (cx - bw / 2.0) * img_w
        y1 = (cy - bh / 2.0) * img_h
        x2 = (cx + bw / 2.0) * img_w
        y2 = (cy + bh / 2.0) * img_h
        results.append(
            {
                "score": float(vals[j].item()),
                "box_abs_xyxy": [x1, y1, x2, y2],
            }
        )
    return results


def rerank_with_vlm_delta(learned_preds, gate0_preds, delta_lambda, match_iou):
    if not learned_preds:
        return []

    det_scores = torch.tensor([p["score"] for p in learned_preds], dtype=torch.float32)
    delta_scores = []
    for pred in learned_preds:
        matched_gate0 = 0.0
        for ref in gate0_preds:
            if iou_xyxy(pred["box_abs_xyxy"], ref["box_abs_xyxy"]) >= match_iou:
                matched_gate0 = max(matched_gate0, float(ref["score"]))
        delta_scores.append(float(pred["score"]) - matched_gate0)
    delta_scores = torch.tensor(delta_scores, dtype=torch.float32)

    if len(learned_preds) > 1:
        det_scores = (det_scores - det_scores.mean()) / det_scores.std().clamp_min(1e-6)
        delta_scores = (delta_scores - delta_scores.mean()) / delta_scores.std().clamp_min(1e-6)

    combined = det_scores + float(delta_lambda) * delta_scores
    order = combined.argsort(descending=True).tolist()

    reranked = []
    for idx in order:
        pred = dict(learned_preds[idx])
        pred["detector_score"] = float(pred["score"])
        pred["rerank_score"] = float(combined[idx].item())
        pred["vlm_delta_score"] = float(delta_scores[idx].item())
        reranked.append(pred)
    return reranked


def summarize_rows(rows, top_k):
    n = max(len(rows), 1)
    return {
        "n_samples": len(rows),
        "mean_best_iou_top1": sum(r["top1_iou"] for r in rows) / n,
        "mean_best_iou_topk": sum(r["topk_best_iou"] for r in rows) / n,
        "hit@0.5_top1": sum(1 for r in rows if r["top1_iou"] >= 0.5) / n,
        "hit@0.5_topk": sum(1 for r in rows if r["topk_best_iou"] >= 0.5) / n,
        "hit@0.75_top1": sum(1 for r in rows if r["top1_iou"] >= 0.75) / n,
        "hit@0.75_topk": sum(1 for r in rows if r["topk_best_iou"] >= 0.75) / n,
        "top_k": int(top_k),
    }


def main():
    args = parse_args()
    patch_groundingdino_ms_deform_attn()
    if args.rerank_vlm_delta and (args.force_gate0 or args.force_gate_value is not None):
        raise ValueError("Use either VLM-delta reranking or force_gate overrides, not both.")
    if args.llm_judge and (args.force_gate0 or args.force_gate_value is not None):
        raise ValueError("Use either LLM judging or force_gate overrides, not both.")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    with open(args.benchmark, "r") as f:
        bench = json.load(f)
    samples = bench["samples"]
    if args.split != "all":
        samples = [s for s in samples if s.get("split") == args.split]
    if args.max_samples > 0:
        samples = samples[:args.max_samples]

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dino_tf = build_dino_transform()
    ivl_tf = build_internvl_transform()

    model, ckpt_meta = load_unified_model(args, device)
    tokenizer, special_tokens = resolve_query_scoring_assets(model)
    rerank_affordance_set = set(args.rerank_affordances or [])
    llm_judge_affordance_set = set(args.llm_judge_affordances or [])
    llm_judge = None
    if args.llm_judge:
        llm_judge = InternVLYesNoReranker(
            internvl_model=model.feature_extractor.internvl,
            tokenizer=model.feature_extractor.tokenizer,
            device=device,
            image_transform=ivl_tf,
            weight=args.llm_judge_weight,
            max_new_tokens=args.llm_judge_max_new_tokens,
            temperature=args.llm_judge_temperature,
        )

    rows = []
    per_affordance = defaultdict(list)

    for i, sample in enumerate(samples, 1):
        image_pil = Image.open(sample["image_path"]).convert("RGB")
        use_rerank = bool(args.rerank_vlm_delta)
        if use_rerank and rerank_affordance_set:
            use_rerank = sample["affordance_id"] in rerank_affordance_set
        use_llm_judge = bool(args.llm_judge)
        if use_llm_judge and llm_judge_affordance_set:
            use_llm_judge = sample["affordance_id"] in llm_judge_affordance_set

        if use_rerank or use_llm_judge:
            candidate_k = int(args.top_k)
            if use_rerank:
                candidate_k = max(candidate_k, int(args.rerank_top_k))
            if use_llm_judge:
                candidate_k = max(candidate_k, int(args.llm_judge_pool_k))
            learned_preds = run_topk(
                model,
                image_pil,
                sample["prompt"],
                device,
                dino_tf,
                ivl_tf,
                top_k=candidate_k,
                tokenizer=tokenizer,
                special_tokens=special_tokens,
                force_gate0=False,
            )
            preds = learned_preds
            if use_rerank:
                gate0_preds = run_topk(
                    model,
                    image_pil,
                    sample["prompt"],
                    device,
                    dino_tf,
                    ivl_tf,
                    top_k=candidate_k,
                    tokenizer=tokenizer,
                    special_tokens=special_tokens,
                    force_gate0=True,
                )
                preds = rerank_with_vlm_delta(
                    learned_preds,
                    gate0_preds,
                    delta_lambda=args.rerank_delta_lambda,
                    match_iou=args.rerank_match_iou,
                )
            if use_llm_judge and llm_judge is not None:
                head = preds[: int(args.llm_judge_pool_k)]
                tail = preds[int(args.llm_judge_pool_k):]
                judged = llm_judge.rerank_predictions(image_pil, sample["prompt"], head)
                preds = judged + tail
            preds = preds[: int(args.top_k)]
        else:
            preds = run_topk(
                model,
                image_pil,
                sample["prompt"],
                device,
                dino_tf,
                ivl_tf,
                top_k=args.top_k,
                tokenizer=tokenizer,
                special_tokens=special_tokens,
                force_gate0=bool(args.force_gate0),
            )

        gt_boxes = [xywh_to_xyxy(t["bbox_xywh"]) for t in sample["positive_targets"]]
        top1_box = preds[0]["box_abs_xyxy"] if preds else None
        top1_score = float(preds[0]["score"]) if preds else 0.0
        top1_rerank_score = float(preds[0].get("rerank_score", top1_score)) if preds else 0.0
        top1_vlm_delta = float(preds[0].get("vlm_delta_score", 0.0)) if preds else 0.0
        top1_llm_score = float(preds[0].get("llm_score", 0.0)) if preds else 0.0
        top1_combined_score = float(preds[0].get("combined_score", top1_score)) if preds else 0.0
        top1_iou = max((iou_xyxy(top1_box, g) for g in gt_boxes), default=0.0) if top1_box else 0.0
        topk_best_iou = max(
            (iou_xyxy(pred["box_abs_xyxy"], g) for pred in preds for g in gt_boxes),
            default=0.0,
        )

        row = {
            "benchmark_id": sample["benchmark_id"],
            "image_id": int(sample["image_id"]),
            "affordance_id": sample["affordance_id"],
            "prompt": sample["prompt"],
            "top1_iou": float(top1_iou),
            "topk_best_iou": float(topk_best_iou),
            "top1_score": float(top1_score),
            "top1_rerank_score": float(top1_rerank_score),
            "top1_vlm_delta_score": float(top1_vlm_delta),
            "top1_llm_score": float(top1_llm_score),
            "top1_combined_score": float(top1_combined_score),
            "top1_correct_iou50": int(top1_iou >= 0.5),
            "top1_correct_iou75": int(top1_iou >= 0.75),
        }
        rows.append(row)
        per_affordance[sample["affordance_id"]].append(row)

        if i % max(1, args.log_every) == 0 or i == len(samples):
            print(f"[eval] {i}/{len(samples)}")

    payload = {
        "status": "ok",
        "timestamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
        "benchmark": args.benchmark,
        "split": args.split,
        "n_samples": len(rows),
        "checkpoint": args.thinkdet_checkpoint,
        "ckpt_meta": ckpt_meta,
        "force_gate0": bool(args.force_gate0),
        "force_gate_value": args.force_gate_value,
        "rerank_vlm_delta": bool(args.rerank_vlm_delta),
        "rerank_top_k": int(args.rerank_top_k),
        "rerank_delta_lambda": float(args.rerank_delta_lambda),
        "rerank_match_iou": float(args.rerank_match_iou),
        "rerank_affordances": list(args.rerank_affordances) if args.rerank_affordances else None,
        "llm_judge": bool(args.llm_judge),
        "llm_judge_pool_k": int(args.llm_judge_pool_k),
        "llm_judge_weight": float(args.llm_judge_weight),
        "llm_judge_max_new_tokens": int(args.llm_judge_max_new_tokens),
        "llm_judge_temperature": float(args.llm_judge_temperature),
        "llm_judge_affordances": list(args.llm_judge_affordances) if args.llm_judge_affordances else None,
        "summary": summarize_rows(rows, args.top_k),
        "per_affordance": {
            aff: summarize_rows(aff_rows, args.top_k)
            for aff, aff_rows in sorted(per_affordance.items())
        },
        "per_sample": rows,
    }
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)

    print("Saved:", args.output)
    print("hit@0.5_top1:", payload["summary"]["hit@0.5_top1"])


if __name__ == "__main__":
    main()
