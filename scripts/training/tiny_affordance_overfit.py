"""
Tiny affordance overfit sanity test for ThinkDet injection.

Trains only adapter/projection/TMA parameters while keeping GroundingDINO and
InternVL frozen. The goal is not a useful model; it checks whether the injection
path can learn on 10 COCO-affordance samples and whether predictions change.
"""

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.optimize import linear_sum_assignment

ROOT = "/home/iibrohimm/project/next_step"
THINKDET_ROOT = os.path.join(ROOT, "thinkdet")
GROUNDING_DINO_ROOT = os.path.join(ROOT, "GroundingDINO")
sys.path.insert(0, ROOT)
sys.path.insert(0, GROUNDING_DINO_ROOT)
sys.path.insert(0, os.path.join(THINKDET_ROOT, "scripts", "eval"))

from groundingdino.util.inference import load_model as load_gd_model  # noqa: E402
from groundingdino.util.misc import NestedTensor  # noqa: E402
from thinkdet.models.arch import ThinkDetModel  # noqa: E402
from eval_affordance_unified import (  # noqa: E402
    GD_CONFIG,
    GD_WEIGHTS,
    INTERNVL_PATH,
    build_dino_transform,
    build_internvl_transform,
    build_positive_map_for_query,
    iou_xyxy,
    patch_groundingdino_ms_deform_attn,
    resolve_query_scoring_assets,
    score_outputs_for_query,
    xywh_to_xyxy,
)


DEFAULT_BENCH = os.path.join(
    THINKDET_ROOT, "data/benchmarks/affordance_coco_val_heldout_v2.json"
)
DEFAULT_OUT_DIR = os.path.join(THINKDET_ROOT, "results/overfit_sanity")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", default=DEFAULT_BENCH)
    p.add_argument("--layers", type=int, nargs="+", required=True)
    p.add_argument("--layer_fusion", default="mean", choices=["mean", "last"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--train_n", type=int, default=10)
    p.add_argument("--unseen_n", type=int, default=50)
    p.add_argument("--train_split", default="dev")
    p.add_argument("--unseen_split", default="test")
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=20260501)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--alpha_init", type=float, default=0.0)
    p.add_argument("--tma_m", type=int, default=8)
    p.add_argument("--gate_after_delta", action="store_true")
    p.add_argument("--lambda_delta", type=float, default=0.0)
    p.add_argument("--lambda_rank", type=float, default=0.0)
    p.add_argument("--lambda_protect", type=float, default=0.0)
    p.add_argument("--rank_margin", type=float, default=0.5)
    p.add_argument("--eval_every", type=int, default=0)
    p.add_argument(
        "--selection_metric",
        default="top1_top5",
        choices=["top1_top5", "combined_top1", "combined_regression"],
    )
    p.add_argument("--official_internvl_extraction", action="store_true")
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def layer_tag(layers):
    layers = sorted(set(int(x) for x in layers))
    if len(layers) == 1:
        return f"layer{layers[0]}"
    return "layers_" + "_".join(str(x) for x in layers)


def choose_unique_samples(samples, split, n, seed, exclude_ids=None):
    rng = random.Random(seed)
    exclude_ids = set(exclude_ids or [])
    pool = [s for s in samples if s.get("split") == split and int(s["image_id"]) not in exclude_ids]
    rng.shuffle(pool)
    out = []
    seen = set()
    for s in pool:
        image_id = int(s["image_id"])
        if image_id in seen:
            continue
        seen.add(image_id)
        out.append(s)
        if len(out) >= n:
            break
    if len(out) < n:
        raise RuntimeError(f"Could only select {len(out)} unique {split} samples, requested {n}")
    return out


def box_cxcywh_to_xyxy(x):
    cx, cy, w, h = x.unbind(-1)
    return torch.stack([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], dim=-1)


def safe_giou(b1, b2):
    b1 = torch.stack(
        [
            b1[..., 0],
            b1[..., 1],
            torch.max(b1[..., 2], b1[..., 0] + 1e-4),
            torch.max(b1[..., 3], b1[..., 1] + 1e-4),
        ],
        dim=-1,
    )
    b2 = torch.stack(
        [
            b2[..., 0],
            b2[..., 1],
            torch.max(b2[..., 2], b2[..., 0] + 1e-4),
            torch.max(b2[..., 3], b2[..., 1] + 1e-4),
        ],
        dim=-1,
    )
    a1 = (b1[:, 2] - b1[:, 0]) * (b1[:, 3] - b1[:, 1])
    a2 = (b2[:, 2] - b2[:, 0]) * (b2[:, 3] - b2[:, 1])
    lt = torch.max(b1[:, None, :2], b2[None, :, :2])
    rb = torch.min(b1[:, None, 2:], b2[None, :, 2:])
    inter = (rb - lt).clamp(min=0).prod(dim=-1)
    union = a1[:, None] + a2[None, :] - inter
    iou = inter / (union + 1e-6)
    lt2 = torch.min(b1[:, None, :2], b2[None, :, :2])
    rb2 = torch.max(b1[:, None, 2:], b2[None, :, 2:])
    area_enclose = (rb2 - lt2).clamp(min=0).prod(dim=-1)
    return iou - (area_enclose - union) / (area_enclose + 1e-6)


def pairwise_iou_cxcywh(b1, b2):
    b1 = box_cxcywh_to_xyxy(b1)
    b2 = box_cxcywh_to_xyxy(b2)
    b1 = torch.stack(
        [
            b1[..., 0],
            b1[..., 1],
            torch.max(b1[..., 2], b1[..., 0] + 1e-4),
            torch.max(b1[..., 3], b1[..., 1] + 1e-4),
        ],
        dim=-1,
    )
    b2 = torch.stack(
        [
            b2[..., 0],
            b2[..., 1],
            torch.max(b2[..., 2], b2[..., 0] + 1e-4),
            torch.max(b2[..., 3], b2[..., 1] + 1e-4),
        ],
        dim=-1,
    )
    area1 = (b1[:, 2] - b1[:, 0]) * (b1[:, 3] - b1[:, 1])
    area2 = (b2[:, 2] - b2[:, 0]) * (b2[:, 3] - b2[:, 1])
    lt = torch.max(b1[:, None, :2], b2[None, :, :2])
    rb = torch.min(b1[:, None, 2:], b2[None, :, 2:])
    inter = (rb - lt).clamp(min=0).prod(dim=-1)
    union = area1[:, None] + area2[None, :] - inter
    return inter / (union + 1e-6)


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha=0.25, gamma=2.0):
    inputs = inputs.clamp(-50, 50)
    prob = inputs.sigmoid()
    ce = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1.0 - prob) * (1.0 - targets)
    loss = ce * ((1.0 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
        loss = alpha_t * loss
    return loss.mean(1).sum() / max(num_boxes, 1)


def affordance_loss(outputs, gt_boxes, positive_map_norm):
    logits = outputs["pred_logits"][0].clamp(-50, 50)
    boxes = outputs["pred_boxes"][0]
    gt = gt_boxes.to(boxes.device)
    pmap = positive_map_norm[:, : logits.shape[-1]].to(logits.device)
    num_gt = int(gt.shape[0])
    if num_gt == 0:
        return boxes.sum() * 0.0

    with torch.no_grad():
        probs = logits.sigmoid()
        scores = (probs * pmap).sum(dim=-1)
        cls_cost = -scores[:, None].expand(-1, num_gt)
        l1_cost = torch.cdist(boxes, gt, p=1)
        giou_cost = 1.0 - safe_giou(box_cxcywh_to_xyxy(boxes), box_cxcywh_to_xyxy(gt))
        cost = (cls_cost + 5.0 * l1_cost + 2.0 * giou_cost).nan_to_num(100.0, 100.0, -100.0)
        row_ind, col_ind = linear_sum_assignment(cost.detach().cpu().numpy())

    row_ind = torch.as_tensor(row_ind, dtype=torch.long, device=boxes.device)
    col_ind = torch.as_tensor(col_ind, dtype=torch.long, device=boxes.device)
    matched_boxes = boxes[row_ind]
    matched_gt = gt[col_ind]

    l1 = F.l1_loss(matched_boxes, matched_gt, reduction="sum") / max(num_gt, 1)
    giou = safe_giou(box_cxcywh_to_xyxy(matched_boxes), box_cxcywh_to_xyxy(matched_gt))
    giou_loss = (1.0 - giou.diag()).sum() / max(num_gt, 1)

    cls_target = torch.zeros_like(logits)
    cls_target[row_ind] = pmap[0]
    focal = sigmoid_focal_loss(logits, cls_target, max(num_gt, 1))
    return 5.0 * l1 + 2.0 * giou_loss + focal


def proposal_ranking_loss(outputs, gt_boxes, positive_map_norm, margin=0.5):
    logits = outputs["pred_logits"][0].clamp(-50, 50)
    boxes = outputs["pred_boxes"][0]
    gt = gt_boxes.to(boxes.device)
    pmap = positive_map_norm[:, : logits.shape[-1]].to(logits.device)
    if int(gt.shape[0]) == 0 or int(boxes.shape[0]) < 2:
        return boxes.sum() * 0.0

    scores = (logits.sigmoid() * pmap).sum(dim=-1)
    with torch.no_grad():
        ious = pairwise_iou_cxcywh(boxes, gt)
        best_iou_per_proposal = ious.max(dim=1).values
        correct_idx = int(best_iou_per_proposal.argmax().item())
        wrong_mask = torch.ones_like(scores, dtype=torch.bool)
        wrong_mask[correct_idx] = False
        wrong_low_iou_mask = wrong_mask & (best_iou_per_proposal < 0.5)
        if bool(wrong_low_iou_mask.any()):
            wrong_mask = wrong_low_iou_mask
        if not bool(wrong_mask.any()):
            return boxes.sum() * 0.0

    correct_score = scores[correct_idx]
    best_wrong_score = scores[wrong_mask].max()
    return F.relu(best_wrong_score - correct_score + float(margin))


def make_inputs(sample, dino_tf, ivl_tf, device):
    image_pil = Image.open(sample["image_path"]).convert("RGB")
    dino_t = dino_tf(image_pil).unsqueeze(0).to(device)
    mask = torch.zeros(1, dino_t.shape[2], dino_t.shape[3], dtype=torch.bool, device=device)
    nested = NestedTensor(dino_t, mask)
    ivl_t = ivl_tf(image_pil).unsqueeze(0).to(device)
    query_text = sample["prompt"] if sample["prompt"].strip().endswith(".") else sample["prompt"].strip() + " ."
    gt = torch.tensor(
        [t["bbox_norm_cxcywh"] for t in sample["positive_targets"]],
        dtype=torch.float32,
        device=device,
    )
    return image_pil, ivl_t, {"samples": nested, "captions": [query_text]}, query_text, gt


def add_delta_hooks(model, bucket):
    handles = []
    for inj_idx, layer in zip(model.injection_layers, model.adapted_layers):
        if getattr(layer, "residual_fuser", None) is None:
            continue

        def hook(_module, inputs, output, inj_idx=inj_idx, layer=layer):
            memory_text = inputs[0].detach().float()
            out = output.detach().float()
            delta = out - memory_text
            mem_norm = memory_text.flatten(1).norm(dim=1).mean().item()
            delta_norm = delta.flatten(1).norm(dim=1).mean().item()
            bucket[int(inj_idx)]["gate"].append(float(torch.tanh(layer.augmenter.alpha).item()))
            bucket[int(inj_idx)]["memory_norm"].append(mem_norm)
            bucket[int(inj_idx)]["delta_norm"].append(delta_norm)
            bucket[int(inj_idx)]["delta_over_memory"].append(delta_norm / max(mem_norm, 1e-12))

        handles.append(layer.residual_fuser.register_forward_hook(hook))
    return handles


def summarize_delta(bucket):
    out = {}
    for inj_idx, metrics in sorted(bucket.items()):
        out[str(inj_idx)] = {
            key: float(sum(vals) / max(len(vals), 1))
            for key, vals in sorted(metrics.items())
        }
    return out


def summarize_rows(rows):
    n = max(len(rows), 1)
    return {
        "n": len(rows),
        "top1_hit@0.5": sum(r["top1_iou"] >= 0.5 for r in rows) / n,
        "top5_hit@0.5": sum(r["top5_iou"] >= 0.5 for r in rows) / n,
        "mean_top1_iou": sum(r["top1_iou"] for r in rows) / n,
        "mean_top5_iou": sum(r["top5_iou"] for r in rows) / n,
    }


@torch.no_grad()
def eval_samples(model, samples, dino_tf, ivl_tf, tokenizer, special_tokens, device, force_gate0=False):
    model.eval()
    rows = []
    delta_bucket = defaultdict(lambda: defaultdict(list))
    for sample in samples:
        image_pil, ivl_t, dino_inputs, query_text, _gt_norm = make_inputs(sample, dino_tf, ivl_tf, device)
        hooks = [] if force_gate0 else add_delta_hooks(model, delta_bucket)
        outputs, aux = model(ivl_t, [sample["prompt"]], dino_inputs, force_gate0=force_gate0)
        for h in hooks:
            h.remove()
        pmap = build_positive_map_for_query(tokenizer, special_tokens, query_text)
        scores, boxes = score_outputs_for_query(outputs, pmap)
        k = min(5, scores.shape[0])
        vals, idx = scores.topk(k)
        w, h = image_pil.size
        preds = []
        for j in range(k):
            cx, cy, bw, bh = boxes[idx[j]].detach().cpu().tolist()
            box = [
                (cx - bw / 2.0) * w,
                (cy - bh / 2.0) * h,
                (cx + bw / 2.0) * w,
                (cy + bh / 2.0) * h,
            ]
            preds.append({"score": float(vals[j].item()), "box": box})
        gt_boxes = [xywh_to_xyxy(t["bbox_xywh"]) for t in sample["positive_targets"]]
        top1_iou = max((iou_xyxy(preds[0]["box"], g) for g in gt_boxes), default=0.0) if preds else 0.0
        top5_iou = max((iou_xyxy(p["box"], g) for p in preds for g in gt_boxes), default=0.0)
        rows.append(
            {
                "benchmark_id": sample["benchmark_id"],
                "image_id": int(sample["image_id"]),
                "affordance_id": sample["affordance_id"],
                "top1_iou": float(top1_iou),
                "top5_iou": float(top5_iou),
                "top1_box": preds[0]["box"] if preds else None,
                "top1_score": float(preds[0]["score"]) if preds else 0.0,
                "gate_mean": float(aux.get("gate_mean", 0.0)) if isinstance(aux, dict) else 0.0,
            }
        )
    return rows, summarize_delta(delta_bucket)


def prediction_change_summary(base_rows, trained_rows):
    by_id = {r["benchmark_id"]: r for r in base_rows}
    changed_box = 0
    changed_hit = 0
    score_delta = []
    for row in trained_rows:
        base = by_id[row["benchmark_id"]]
        if base["top1_box"] is None or row["top1_box"] is None:
            changed_box += int(base["top1_box"] != row["top1_box"])
        else:
            changed_box += int(iou_xyxy(base["top1_box"], row["top1_box"]) < 0.95)
        changed_hit += int((base["top1_iou"] >= 0.5) != (row["top1_iou"] >= 0.5))
        score_delta.append(abs(float(row["top1_score"]) - float(base["top1_score"])))
    n = max(len(trained_rows), 1)
    return {
        "top1_box_changed_rate": changed_box / n,
        "top1_hit_changed_count": changed_hit,
        "mean_abs_top1_score_delta": float(sum(score_delta) / n),
    }


def save_checkpoint(model, path, args, extra=None):
    state = {
        name: p.detach().cpu()
        for name, p in model.named_parameters()
        if p.requires_grad
    }
    torch.save(
        {
            "stage": "tiny_affordance_overfit",
            "extract_layers": sorted(set(int(x) for x in args.layers)),
            "layer_fusion": args.layer_fusion,
            "injection_layers": [1, 3, 5],
            "fusion_mode": "residual",
            "tma_m": int(args.tma_m),
            "tma_n_heads": 8,
            "tma_alpha_init": float(args.alpha_init),
            "gate_after_delta": bool(args.gate_after_delta),
            "lambda_delta": float(args.lambda_delta),
            "lambda_rank": float(args.lambda_rank),
            "lambda_protect": float(args.lambda_protect),
            "rank_margin": float(args.rank_margin),
            "selection_metric": args.selection_metric,
            "official_internvl_extraction": bool(args.official_internvl_extraction),
            "trainable_state_dict": state,
            **(extra or {}),
        },
        path,
    )


def load_trainable_checkpoint(model, path, device):
    payload = torch.load(path, map_location=device)
    missing, unexpected = model.load_state_dict(payload["trainable_state_dict"], strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected trainable checkpoint keys: {unexpected}")
    return {
        "missing_keys": missing,
        "checkpoint_payload": {k: v for k, v in payload.items() if k != "trainable_state_dict"},
    }


def regression_summary(base_rows, trained_rows):
    by_id = {r["benchmark_id"]: r for r in base_rows}
    top1_regressions = 0
    top5_regressions = 0
    for row in trained_rows:
        base = by_id[row["benchmark_id"]]
        if base["top1_iou"] >= 0.5 and row["top1_iou"] < 0.5:
            top1_regressions += 1
        if base["top5_iou"] >= 0.5 and row["top5_iou"] < 0.5:
            top5_regressions += 1
    n = max(len(trained_rows), 1)
    return {
        "n": len(trained_rows),
        "top1_regressions": int(top1_regressions),
        "top5_regressions": int(top5_regressions),
        "regression_count": int(top1_regressions + top5_regressions),
        "regression_penalty": float((top1_regressions + top5_regressions) / n),
    }


def selection_key(summary, metric="top1_top5", regressions=None):
    top1 = float(summary["top1_hit@0.5"])
    top5 = float(summary["top5_hit@0.5"])
    mean_top1 = float(summary["mean_top1_iou"])
    mean_top5 = float(summary["mean_top5_iou"])
    if metric == "combined_regression":
        regressions = regressions or {"regression_penalty": 0.0, "regression_count": 0}
        score = top1 + top5 - float(regressions["regression_penalty"])
        return (
            score,
            top1,
            top5,
            -float(regressions["regression_count"]),
            mean_top1,
            mean_top5,
        )
    if metric == "combined_top1":
        return (
            top1 + top5,
            top1,
            top5,
            mean_top1,
            mean_top5,
        )
    return (
        top1,
        top5,
        mean_top1,
        mean_top5,
    )


def main():
    args = parse_args()
    set_seed(args.seed)
    patch_groundingdino_ms_deform_attn()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    with open(args.benchmark, "r") as f:
        bench = json.load(f)
    train_samples = choose_unique_samples(bench["samples"], args.train_split, args.train_n, args.seed)
    train_image_ids = {int(s["image_id"]) for s in train_samples}
    unseen_samples = choose_unique_samples(
        bench["samples"],
        args.unseen_split,
        args.unseen_n,
        args.seed + 1,
        exclude_ids=train_image_ids,
    )

    gd = load_gd_model(GD_CONFIG, GD_WEIGHTS, device="cpu")
    model = ThinkDetModel(
        grounding_dino=gd,
        internvl_path=INTERNVL_PATH,
        extract_layer=max(args.layers),
        extract_layers=sorted(set(int(x) for x in args.layers)),
        layer_fusion=args.layer_fusion,
        injection_layers=[1, 3, 5],
        tma_m=int(args.tma_m),
        tma_n_heads=8,
        tma_alpha_init=float(args.alpha_init),
        fusion_mode="residual",
        gate_after_delta=bool(args.gate_after_delta),
        use_official_internvl_extraction=bool(args.official_internvl_extraction),
    ).to(device)
    model.set_tma_only()
    tokenizer, special_tokens = resolve_query_scoring_assets(model)
    dino_tf = build_dino_transform()
    ivl_tf = build_internvl_transform()

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=float(args.lr), weight_decay=0.0)

    baseline_train, _ = eval_samples(
        model, train_samples, dino_tf, ivl_tf, tokenizer, special_tokens, device, force_gate0=True
    )
    baseline_unseen, _ = eval_samples(
        model, unseen_samples, dino_tf, ivl_tf, tokenizer, special_tokens, device, force_gate0=True
    )
    baseline_train_by_id = {r["benchmark_id"]: r for r in baseline_train}

    losses = []
    det_losses = []
    rank_losses = []
    protect_losses = []
    delta_losses = []
    checkpoint_history = []
    best_key = None
    best_checkpoint_path = None
    best_step = None
    tag = layer_tag(args.layers)
    model.train()
    for step in range(1, int(args.steps) + 1):
        sample = train_samples[(step - 1) % len(train_samples)]
        _image_pil, ivl_t, dino_inputs, query_text, gt = make_inputs(sample, dino_tf, ivl_tf, device)
        pmap = build_positive_map_for_query(tokenizer, special_tokens, query_text)
        outputs, aux = model(ivl_t, [sample["prompt"]], dino_inputs, force_gate0=False)
        det_loss = affordance_loss(outputs, gt, pmap)
        base_row = baseline_train_by_id[sample["benchmark_id"]]
        dino_top1_correct = bool(base_row["top1_iou"] >= 0.5)
        if float(args.lambda_rank) > 0.0 and not dino_top1_correct:
            rank_loss = proposal_ranking_loss(
                outputs,
                gt,
                pmap,
                margin=float(args.rank_margin),
            )
        else:
            rank_loss = det_loss.new_zeros(())
        if float(args.lambda_protect) > 0.0 and dino_top1_correct:
            protect_loss = proposal_ranking_loss(
                outputs,
                gt,
                pmap,
                margin=float(args.rank_margin),
            )
        else:
            protect_loss = det_loss.new_zeros(())
        delta_terms = [
            layer.last_delta_l2
            for layer in model.adapted_layers
            if getattr(layer, "last_delta_l2", None) is not None
        ]
        if delta_terms:
            delta_loss = torch.stack(delta_terms).mean()
        else:
            delta_loss = det_loss.new_zeros(())
        loss = (
            det_loss
            + float(args.lambda_rank) * rank_loss
            + float(args.lambda_protect) * protect_loss
            + float(args.lambda_delta) * delta_loss
        )
        if torch.isnan(loss) or torch.isinf(loss):
            raise RuntimeError(f"Non-finite loss at step {step}: {loss.item()}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        losses.append(float(loss.item()))
        det_losses.append(float(det_loss.item()))
        rank_losses.append(float(rank_loss.item()))
        protect_losses.append(float(protect_loss.item()))
        delta_losses.append(float(delta_loss.item()))
        if step % max(1, args.log_every) == 0 or step == 1:
            gates = [float(torch.tanh(layer.augmenter.alpha).item()) for layer in model.adapted_layers]
            print(
                f"[overfit {layer_tag(args.layers)}] step={step}/{args.steps} "
                f"loss={loss.item():.4f} det={det_loss.item():.4f} "
                f"rank={rank_loss.item():.4f} protect={protect_loss.item():.4f} "
                f"delta={delta_loss.item():.4f} "
                f"gate_mean={sum(gates)/max(len(gates),1):.5f}"
            )
        should_eval = int(args.eval_every) > 0 and (
            step % int(args.eval_every) == 0 or step == int(args.steps)
        )
        if should_eval:
            dev_rows, _ = eval_samples(
                model,
                unseen_samples,
                dino_tf,
                ivl_tf,
                tokenizer,
                special_tokens,
                device,
                force_gate0=False,
            )
            dev_summary = summarize_rows(dev_rows)
            dev_regressions = regression_summary(baseline_unseen, dev_rows)
            ckpt_path = os.path.join(args.out_dir, f"tiny_overfit_{tag}_step{step:04d}.pth")
            save_checkpoint(
                model,
                ckpt_path,
                args,
                extra={
                    "selection_split": args.unseen_split,
                    "selection_step": int(step),
                    "selection_summary": dev_summary,
                    "selection_regressions": dev_regressions,
                },
            )
            cur_key = selection_key(dev_summary, args.selection_metric, dev_regressions)
            is_best = best_key is None or cur_key > best_key
            checkpoint_history.append(
                {
                    "step": int(step),
                    "checkpoint": ckpt_path,
                    "selection_summary": dev_summary,
                    "selection_regressions": dev_regressions,
                    "selection_metric": args.selection_metric,
                    "selection_key": list(cur_key),
                    "is_best": bool(is_best),
                }
            )
            if is_best:
                best_key = cur_key
                best_step = int(step)
                best_checkpoint_path = os.path.join(args.out_dir, f"tiny_overfit_{tag}_best.pth")
                save_checkpoint(
                    model,
                    best_checkpoint_path,
                    args,
                    extra={
                        "selection_split": args.unseen_split,
                        "selection_step": int(step),
                        "selection_summary": dev_summary,
                        "selection_regressions": dev_regressions,
                        "selection_metric": args.selection_metric,
                        "selection_key": list(cur_key),
                    },
                )
            print(
                f"[select {layer_tag(args.layers)}] step={step} "
                f"dev_top1={dev_summary['top1_hit@0.5']:.4f} "
                f"dev_top5={dev_summary['top5_hit@0.5']:.4f} "
                f"combined={dev_summary['top1_hit@0.5'] + dev_summary['top5_hit@0.5']:.4f} "
                f"regress={dev_regressions['regression_count']} "
                f"score={cur_key[0]:.4f} "
                f"best_step={best_step}"
            )
            model.train()

    final_checkpoint_path = os.path.join(args.out_dir, f"tiny_overfit_{tag}_final.pth")
    save_checkpoint(
        model,
        final_checkpoint_path,
        args,
        extra={"selection_split": args.unseen_split, "selection_step": int(args.steps)},
    )
    if best_checkpoint_path is None:
        best_checkpoint_path = os.path.join(args.out_dir, f"tiny_overfit_{tag}_best.pth")
        trained_unseen_probe, _ = eval_samples(
            model, unseen_samples, dino_tf, ivl_tf, tokenizer, special_tokens, device, force_gate0=False
        )
        best_summary = summarize_rows(trained_unseen_probe)
        best_regressions = regression_summary(baseline_unseen, trained_unseen_probe)
        best_key = selection_key(best_summary, args.selection_metric, best_regressions)
        best_step = int(args.steps)
        checkpoint_history.append(
            {
                "step": int(args.steps),
                "checkpoint": best_checkpoint_path,
                "selection_summary": best_summary,
                "selection_regressions": best_regressions,
                "selection_metric": args.selection_metric,
                "selection_key": list(best_key),
                "is_best": True,
            }
        )
        save_checkpoint(
            model,
            best_checkpoint_path,
            args,
            extra={
                "selection_split": args.unseen_split,
                "selection_step": int(args.steps),
                "selection_summary": best_summary,
                "selection_regressions": best_regressions,
                "selection_metric": args.selection_metric,
                "selection_key": list(best_key),
            },
        )
    load_trainable_checkpoint(model, best_checkpoint_path, device)

    trained_train, delta_train = eval_samples(
        model, train_samples, dino_tf, ivl_tf, tokenizer, special_tokens, device, force_gate0=False
    )
    trained_unseen, delta_unseen = eval_samples(
        model, unseen_samples, dino_tf, ivl_tf, tokenizer, special_tokens, device, force_gate0=False
    )

    payload = {
        "status": "ok",
        "benchmark": args.benchmark,
        "layer_setup": sorted(set(int(x) for x in args.layers)),
        "layer_fusion": args.layer_fusion,
        "train_n": len(train_samples),
        "unseen_n": len(unseen_samples),
        "train_split": args.train_split,
        "unseen_split": args.unseen_split,
        "steps": int(args.steps),
        "lr": float(args.lr),
        "alpha_init": float(args.alpha_init),
        "gate_after_delta": bool(args.gate_after_delta),
        "lambda_delta": float(args.lambda_delta),
        "lambda_rank": float(args.lambda_rank),
        "lambda_protect": float(args.lambda_protect),
        "rank_margin": float(args.rank_margin),
        "eval_every": int(args.eval_every),
        "selection_metric": args.selection_metric,
        "official_internvl_extraction": bool(args.official_internvl_extraction),
        "checkpoint": best_checkpoint_path,
        "best_checkpoint": best_checkpoint_path,
        "final_checkpoint": final_checkpoint_path,
        "best_step": best_step,
        "checkpoint_history": checkpoint_history,
        "train_affordance_counts": dict(sorted(defaultdict(int, {
            k: sum(1 for s in train_samples if s["affordance_id"] == k)
            for k in sorted({s["affordance_id"] for s in train_samples})
        }).items())),
        "unseen_affordance_counts": dict(sorted(defaultdict(int, {
            k: sum(1 for s in unseen_samples if s["affordance_id"] == k)
            for k in sorted({s["affordance_id"] for s in unseen_samples})
        }).items())),
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "loss_mean_last10": float(sum(losses[-10:]) / max(len(losses[-10:]), 1)) if losses else None,
        "det_loss_mean_last10": float(sum(det_losses[-10:]) / max(len(det_losses[-10:]), 1)) if det_losses else None,
        "rank_loss_mean_last10": float(sum(rank_losses[-10:]) / max(len(rank_losses[-10:]), 1)) if rank_losses else None,
        "protect_loss_mean_last10": float(sum(protect_losses[-10:]) / max(len(protect_losses[-10:]), 1)) if protect_losses else None,
        "delta_loss_mean_last10": float(sum(delta_losses[-10:]) / max(len(delta_losses[-10:]), 1)) if delta_losses else None,
        "baseline_train": summarize_rows(baseline_train),
        "trained_train": summarize_rows(trained_train),
        "baseline_unseen": summarize_rows(baseline_unseen),
        "trained_unseen": summarize_rows(trained_unseen),
        "prediction_change_train": prediction_change_summary(baseline_train, trained_train),
        "prediction_change_unseen": prediction_change_summary(baseline_unseen, trained_unseen),
        "delta_norm_train": delta_train,
        "delta_norm_unseen": delta_unseen,
        "train_rows_baseline": baseline_train,
        "train_rows_trained": trained_train,
        "unseen_rows_baseline": baseline_unseen,
        "unseen_rows_trained": trained_unseen,
    }
    out_path = os.path.join(args.out_dir, f"tiny_overfit_{tag}.json")
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print("Saved:", out_path)
    print(json.dumps({
        "layer_setup": payload["layer_setup"],
        "baseline_train": payload["baseline_train"],
        "trained_train": payload["trained_train"],
        "baseline_unseen": payload["baseline_unseen"],
        "trained_unseen": payload["trained_unseen"],
    }, indent=2))


if __name__ == "__main__":
    main()
