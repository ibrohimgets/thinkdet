"""
ThinkDet — Unified Training (replaces Stage 1 + Stage 2)

Design principles (see analysis doc for why):
  1. No "adapter-only warmup" stage. Adapters + detection heads train jointly
     from epoch 1. Heads at very low LR (1e-5) so they barely move and act
     as soft regularisers.

  2. Knowledge Distillation (KD) loss every step (single forward):
         adapted_out = model(imgs, queries, dino_inputs)
         kd = mean_l MSE(memory_text_after_adapter_l, detach(memory_text_before_adapter_l))
         loss = det_loss + lambda_kd * kd

  3. Residual fusion ONLY. concat mode is permanently disabled.

  4. LR groups:
       adapters : 2e-4
       heads    : 1e-5

  5. Gate: tanh(alpha), alpha_init=0. Optional L1 penalty on tanh(alpha).

  6. Eval uses official protocol: num_select=300, NO confidence threshold.
     Report every epoch on the full COCO val2017.

  7. Success criterion: first reproduce baseline AP ±0.2.
     Only then measure gains.

Launch (single node, 8 GPUs):
    torchrun --nproc_per_node=8 \\
        thinkdet/scripts/training/train_unified.py
"""

import os
import sys
import time
import json
import math
import argparse
import tempfile
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler
from scipy.optimize import linear_sum_assignment

os.environ.setdefault("OMP_NUM_THREADS", "1")

ROOT = "/home/iibrohimm/project/next_step"
sys.path.insert(0, ROOT)
GROUNDING_DINO_ROOT = os.path.join(ROOT, "GroundingDINO")
sys.path.insert(0, GROUNDING_DINO_ROOT)

from thinkdet.models.arch import ThinkDetModel, DEFAULT_INJECTION_LAYERS
from thinkdet.data.coco_grounding import (
    COCOGroundingDataset,
    collate_fn,
    build_positive_map,
)
from groundingdino.util.inference import load_model as load_gd_model
from groundingdino.util.misc import NestedTensor


# ===================================================================
# CONFIG
# ===================================================================
class Cfg:
    # ── Paths ────────────────────────────────────────────────────────
    coco_root     = f"{ROOT}/dataSets/coco"
    train_ann     = f"{coco_root}/annotations/instances_train2017.json"
    train_img_dir = f"{coco_root}/train2017"
    val_ann       = f"{coco_root}/annotations/instances_val2017.json"
    val_img_dir   = f"{coco_root}/val2017"

    gd_config  = (
        f"{ROOT}/GroundingDINO/groundingdino/config/"
        "GroundingDINO_SwinT_OGC.py"
    )
    gd_weights = (
        f"{ROOT}/GroundingDINO/weights/groundingdino_swint_ogc.pth"
    )
    internvl_path = f"{ROOT}/InternVL3_5-1B"

    # ── Model ─────────────────────────────────────────────────────────
    extract_layer    = 9
    extract_layers   = [9]
    layer_fusion     = "mean"
    injection_layers = [1, 3, 5]   # DINO decoder layers to augment
    d_model          = 256
    tma_m            = 8
    tma_n_heads      = 8
    tma_alpha_init   = 0.0         # gate starts closed (tanh(0)=0)
    fusion_mode      = "residual"  # ONLY residual; concat is disabled

    # ── Training ──────────────────────────────────────────────────────
    epochs        = 5
    per_gpu_batch = 1              # InternVL + DINO are memory-heavy
    grad_accum    = 8              # effective batch = 1 * 8 * 8 GPUs = 64
    lr_adapters   = 2e-4
    lr_heads      = 1e-5           # very low; heads barely move
    weight_decay  = 0.01
    warmup_ratio  = 0.05           # slightly longer warmup for joint training
    max_grad_norm = 1.0
    num_workers   = 4
    debug_limit   = 0

    # ── Loss ──────────────────────────────────────────────────────────
    lambda_kd      = 0.0           # disabled: preserve-KD fights adapter changes
    lambda_gate_l1 = 1e-4          # L1 penalty on tanh(alpha): keeps gate sparse
    preserve_kd_enabled = False

    # ── Eval — official protocol ──────────────────────────────────────
    num_select    = 300            # top-K predictions (no conf filter)
    val_every_ep  = 1              # evaluate every epoch

    # ── Output ────────────────────────────────────────────────────────
    output_dir   = f"{ROOT}/thinkdet/checkpoints/unified"
    log_interval = 100
    save_interval = 2000
    seed = 1337


def format_layer_tag(extract_layers):
    layers = sorted(set(int(x) for x in extract_layers))
    if len(layers) == 1:
        return f"layer{layers[0]}"
    return "layers_" + "_".join(str(x) for x in layers)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ===================================================================
# BOX HELPERS
# ===================================================================
def box_cxcywh_to_xyxy(x: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = x.unbind(-1)
    return torch.stack([cx - 0.5*w, cy - 0.5*h, cx + 0.5*w, cy + 0.5*h], dim=-1)


def safe_giou(b1: torch.Tensor, b2: torch.Tensor) -> torch.Tensor:
    b1 = torch.stack([
        b1[..., 0], b1[..., 1],
        torch.max(b1[..., 2], b1[..., 0] + 1e-4),
        torch.max(b1[..., 3], b1[..., 1] + 1e-4),
    ], dim=-1)
    b2 = torch.stack([
        b2[..., 0], b2[..., 1],
        torch.max(b2[..., 2], b2[..., 0] + 1e-4),
        torch.max(b2[..., 3], b2[..., 1] + 1e-4),
    ], dim=-1)
    a1 = (b1[:, 2] - b1[:, 0]) * (b1[:, 3] - b1[:, 1])
    a2 = (b2[:, 2] - b2[:, 0]) * (b2[:, 3] - b2[:, 1])
    lt   = torch.max(b1[:, None, :2], b2[None, :, :2])
    rb   = torch.min(b1[:, None, 2:], b2[None, :, 2:])
    inter = (rb - lt).clamp(min=0).prod(dim=-1)
    union = a1[:, None] + a2[None, :] - inter
    iou  = inter / (union + 1e-6)
    lt2  = torch.min(b1[:, None, :2], b2[None, :, :2])
    rb2  = torch.max(b1[:, None, 2:], b2[None, :, 2:])
    ae   = (rb2 - lt2).clamp(min=0).prod(dim=-1)
    return iou - (ae - union) / (ae + 1e-6)


# ===================================================================
# DETECTION LOSS  (focal + L1 + GIoU with Hungarian matching)
# ===================================================================
def sigmoid_focal_loss(inputs, targets, num_boxes, alpha=0.25, gamma=2.0):
    inputs = inputs.clamp(-50, 50)
    prob   = inputs.sigmoid()
    ce     = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t    = prob * targets + (1 - prob) * (1 - targets)
    loss   = ce * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss.mean(1).sum() / max(num_boxes, 1)


def compute_detection_loss(
    pred_logits, pred_boxes,
    gt_boxes_list, gt_cat_names_list,
    cat_name_to_idx, positive_map, positive_map_norm,
    device,
):
    B = pred_logits.shape[0]
    total_loss = torch.tensor(0.0, device=device, requires_grad=True)

    for b in range(B):
        logits    = pred_logits[b]
        boxes     = pred_boxes[b]
        gt        = gt_boxes_list[b].to(device)
        cat_names = gt_cat_names_list[b]

        if gt.shape[0] == 0:
            bg = torch.zeros_like(logits)
            total_loss = total_loss + sigmoid_focal_loss(logits, bg, 1) * 0.1
            continue

        num_gt = gt.shape[0]
        ci_list = [cat_name_to_idx.get(n, 0) for n in cat_names]
        ci_t    = torch.tensor(ci_list, device=device, dtype=torch.long)

        with torch.no_grad():
            probs = logits.clamp(-50, 50).sigmoid()
            cp = probs @ positive_map_norm.T
            cc = torch.zeros(logits.shape[0], num_gt, device=device)
            for j in range(num_gt):
                ci = ci_t[j].item()
                if ci < cp.shape[-1]:
                    cc[:, j] = -cp[:, ci]
            cl1 = torch.cdist(boxes, gt, p=1)
            cg  = 1.0 - safe_giou(
                box_cxcywh_to_xyxy(boxes), box_cxcywh_to_xyxy(gt),
            )
            C = (5 * cl1 + 2 * cg + cc).nan_to_num(100.0, 100.0, -100.0)
            ri, coli = linear_sum_assignment(C.cpu().numpy())

        ri   = torch.tensor(ri,   dtype=torch.long, device=device)
        coli = torch.tensor(coli, dtype=torch.long, device=device)

        mp, mg = boxes[ri], gt[coli]
        l1 = F.l1_loss(mp, mg, reduction="sum") / max(num_gt, 1)
        gv = safe_giou(box_cxcywh_to_xyxy(mp), box_cxcywh_to_xyxy(mg))
        gl = (1 - gv.diag()).sum() / max(num_gt, 1)

        tgt = torch.zeros_like(logits)
        for qi, gi in zip(ri, coli):
            ci = ci_t[gi].item()
            if ci < positive_map.shape[0]:
                tgt[qi] = positive_map[ci]
        fl = sigmoid_focal_loss(logits, tgt, max(num_gt, 1))

        total_loss = total_loss + 5 * l1 + 2 * gl + fl

    return total_loss / max(B, 1)


# ===================================================================
# KNOWLEDGE DISTILLATION LOSS
# ===================================================================
def compute_kd_loss(aux, fallback_tensor):
    """
    Single-forward preservation KD:
    average per-adapted-layer MSE between pre-adapter decoder text memory
    (detached target) and post-adapter memory_text.
    """
    losses = aux.get("pre_adapter_kd_losses") or []
    if not losses:
        return fallback_tensor.new_zeros(())
    return torch.stack(losses).mean()


# ===================================================================
# LR SCHEDULE
# ===================================================================
def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        p = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * p)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ===================================================================
# OFFICIAL COCO EVAL  (num_select=300, NO confidence threshold)
# ===================================================================
@torch.no_grad()
def evaluate_coco_official(
    model_raw,
    val_dataset,
    positive_map_norm,
    cat_name_to_idx,
    all_query_text,
    device,
    num_select: int = 300,
):
    """
    Official GroundingDINO evaluation protocol:
      - Take top num_select predictions per image (by max category score)
      - No confidence threshold filter
    This matches the 48.53 AP baseline number.
    """
    from pycocotools.cocoeval import COCOeval

    model_raw.eval()
    loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    idx_to_cat_name = {v: k for k, v in cat_name_to_idx.items()}
    name_to_coco_id = {v: k for k, v in val_dataset.cat_id_to_name.items()}
    predictions = []
    t0 = time.time()

    for n, batch in enumerate(loader):
        internvl_images = batch["internvl_images"].to(device)
        dino_nested = NestedTensor(
            batch["dino_images"].tensors.to(device),
            batch["dino_images"].mask.to(device),
        )
        dino_inputs = {"samples": dino_nested, "captions": [all_query_text]}

        try:
            outputs, _ = model_raw(internvl_images, batch["query_texts"], dino_inputs)
        except Exception:
            continue

        image_id = batch["image_ids"][0]
        img_info = val_dataset.coco.loadImgs(image_id)[0]
        img_w, img_h = img_info["width"], img_info["height"]

        logits = outputs["pred_logits"][0].clamp(-50, 50)   # [Q, text_len]
        boxes  = outputs["pred_boxes"][0]                   # [Q, 4]

        # Score per category, take max over text tokens
        cat_scores          = logits.sigmoid() @ positive_map_norm.T   # [Q, C]
        max_scores, max_cats = cat_scores.max(dim=-1)                  # [Q]

        # Take top num_select (no conf threshold)
        k = min(num_select, max_scores.shape[0])
        topk_idx    = max_scores.topk(k).indices
        top_scores  = max_scores[topk_idx]
        top_cats    = max_cats[topk_idx]
        top_boxes   = boxes[topk_idx]

        pred_xyxy = box_cxcywh_to_xyxy(top_boxes)
        pred_xyxy[:, 0] *= img_w
        pred_xyxy[:, 2] *= img_w
        pred_xyxy[:, 1] *= img_h
        pred_xyxy[:, 3] *= img_h

        for i in range(k):
            cat_name  = idx_to_cat_name.get(top_cats[i].item())
            coco_cat  = name_to_coco_id.get(cat_name) if cat_name else None
            if coco_cat is None:
                continue
            x1, y1, x2, y2 = pred_xyxy[i].tolist()
            predictions.append({
                "image_id":    image_id,
                "category_id": coco_cat,
                "bbox": [round(x1, 2), round(y1, 2),
                         round(max(0.0, x2 - x1), 2),
                         round(max(0.0, y2 - y1), 2)],
                "score": round(float(top_scores[i].item()), 4),
            })

        if (n + 1) % 1000 == 0:
            print(f"    [eval] {n+1}/{len(loader)} ({time.time()-t0:.0f}s) "
                  f"{len(predictions)} preds so far")

    print(f"  [eval] Inference done: {len(predictions)} total predictions")

    if not predictions:
        return {"AP": 0.0, "AP_50": 0.0, "AP_75": 0.0, "num_predictions": 0}

    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(predictions, tmp)
    tmp.close()

    coco_dt   = val_dataset.coco.loadRes(tmp.name)
    coco_eval = COCOeval(val_dataset.coco, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    os.unlink(tmp.name)

    return {
        "AP":              float(coco_eval.stats[0]),
        "AP_50":           float(coco_eval.stats[1]),
        "AP_75":           float(coco_eval.stats[2]),
        "AP_S":            float(coco_eval.stats[3]),
        "AP_M":            float(coco_eval.stats[4]),
        "AP_L":            float(coco_eval.stats[5]),
        "AR_1":            float(coco_eval.stats[6]),
        "AR_10":           float(coco_eval.stats[7]),
        "AR_100":          float(coco_eval.stats[8]),
        "num_predictions": len(predictions),
        "eval_time_s":     time.time() - t0,
    }


# ===================================================================
# CHECKPOINT
# ===================================================================
def save_checkpoint(model, optimizer, scheduler, epoch, step, cfg, tag=""):
    trainable_state = {
        name: param.detach().cpu()
        for name, param in model.named_parameters()
        if param.requires_grad
    }
    fname = f"thinkdet_unified_{tag}.pth"
    path  = os.path.join(cfg.output_dir, fname)
    torch.save({
        "epoch":           epoch + 1,
        "global_step":     step,
        "training":        "unified",
        "extract_layers":  cfg.extract_layers,
        "layer_fusion":    cfg.layer_fusion,
        "injection_layers": cfg.injection_layers,
        "fusion_mode":     cfg.fusion_mode,
        "tma_m":           cfg.tma_m,
        "lambda_kd":       cfg.lambda_kd,
        "lambda_gate_l1":  cfg.lambda_gate_l1,
        "trainable_state_dict":  trainable_state,
        "optimizer_state_dict":  optimizer.state_dict(),
        "scheduler_state_dict":  scheduler.state_dict(),
    }, path)
    print(f"  [SAVE] {path}  ({len(trainable_state)} tensors)")


# ===================================================================
# GRADIENT HEALTH CHECK
# ===================================================================
def gradient_health_check(model_raw, is_main):
    if not is_main:
        return
    print("\n  [GRAD CHECK] ─────────────────────────────────────────")
    ok, issues = 0, 0
    for name, param in model_raw.named_parameters():
        if not param.requires_grad:
            if param.grad is not None:
                print(f"    WARNING: frozen param {name} has gradient!")
                issues += 1
            continue
        if param.grad is None:
            print(f"    WARNING: {name} — no gradient!")
            issues += 1
        elif param.grad.abs().max() == 0:
            print(f"    WARNING: {name} — zero gradient!")
            issues += 1
        else:
            ok += 1
            if "alpha" in name:
                gn = param.grad.norm().item()
                gate = float(torch.tanh(param).item())
                print(f"    OK gate: {name} = tanh({param.item():.4f}) = {gate:.4f}  "
                      f"grad_norm={gn:.6f}")
    print(f"    Summary: {ok} ok, {issues} issues")
    print("  ─────────────────────────────────────────────────────\n")


# ===================================================================
# DISTRIBUTED INIT
# ===================================================================
def init_distributed():
    ddp = int(os.environ.get("WORLD_SIZE", 1)) > 1
    if ddp:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device     = torch.device(f"cuda:{local_rank}")
        world_size = dist.get_world_size()
    else:
        local_rank = 0
        device     = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        world_size = 1
    torch.backends.cudnn.benchmark = True
    return ddp, local_rank, device, world_size


# ===================================================================
# MAIN
# ===================================================================
def main():
    cfg = Cfg()

    parser = argparse.ArgumentParser(description="ThinkDet Unified Training")
    parser.add_argument("--epochs",         type=int,   default=cfg.epochs)
    parser.add_argument("--lr_adapters",    type=float, default=cfg.lr_adapters)
    parser.add_argument("--lr_heads",       type=float, default=cfg.lr_heads)
    parser.add_argument("--warmup_ratio",   type=float, default=cfg.warmup_ratio)
    parser.add_argument("--lambda_kd",      type=float, default=cfg.lambda_kd)
    parser.add_argument("--lambda_gate_l1", type=float, default=cfg.lambda_gate_l1)
    parser.add_argument("--num_select",     type=int,   default=cfg.num_select)
    parser.add_argument("--tma_m",          type=int,   default=cfg.tma_m)
    parser.add_argument("--debug_limit",    type=int,   default=cfg.debug_limit)
    parser.add_argument("--log_interval",   type=int,   default=cfg.log_interval)
    parser.add_argument("--output_dir",     type=str,   default=None)
    parser.add_argument("--num_workers",    type=int,   default=cfg.num_workers)
    parser.add_argument("--val_every_ep",   type=int,   default=cfg.val_every_ep)
    parser.add_argument("--seed",           type=int,   default=cfg.seed)
    parser.add_argument("--resume",         type=str,   default=None)
    parser.add_argument(
        "--resume_weights_only",
        action="store_true",
        help="Resume model weights + epoch/step, but reset optimizer/scheduler "
             "to CLI LR settings.",
    )
    parser.add_argument(
        "--injection_layers",
        type=int, nargs="+",
        default=cfg.injection_layers,
        help="DINO decoder layer indices to inject TMA (default: 1 3 5)",
    )
    parser.add_argument("--extract_layer", type=int, default=None)
    parser.add_argument("--extract_layers", type=int, nargs="+", default=None)
    parser.add_argument(
        "--layer_fusion",
        type=str,
        default=cfg.layer_fusion,
        choices=["mean", "last", "learned"],
    )
    args = parser.parse_args()

    cfg.epochs         = args.epochs
    cfg.lr_adapters    = args.lr_adapters
    cfg.lr_heads       = args.lr_heads
    cfg.warmup_ratio   = args.warmup_ratio
    cfg.lambda_kd      = args.lambda_kd
    cfg.lambda_gate_l1 = args.lambda_gate_l1
    cfg.num_select     = args.num_select
    cfg.tma_m          = args.tma_m
    cfg.debug_limit    = args.debug_limit
    cfg.log_interval   = args.log_interval
    cfg.num_workers    = args.num_workers
    cfg.val_every_ep   = args.val_every_ep
    cfg.seed           = int(args.seed)
    cfg.injection_layers = list(args.injection_layers)
    if args.extract_layers:
        cfg.extract_layers = sorted(set(int(x) for x in args.extract_layers))
        cfg.extract_layer = max(cfg.extract_layers)
    elif args.extract_layer is not None:
        cfg.extract_layer = int(args.extract_layer)
        cfg.extract_layers = [cfg.extract_layer]
    cfg.layer_fusion = str(args.layer_fusion)
    layer_tag = format_layer_tag(cfg.extract_layers)
    if args.output_dir:
        cfg.output_dir = args.output_dir
    else:
        ts = time.strftime("%Y%m%d_%H%M%S")
        cfg.output_dir = os.path.join(
            cfg.output_dir,
            f"{layer_tag}_unified_kd{cfg.lambda_kd}_l1{cfg.lambda_gate_l1}_{ts}",
        )

    ddp, local_rank, device, world_size = init_distributed()
    is_main = (local_rank == 0)
    set_seed(cfg.seed)

    if is_main:
        os.makedirs(cfg.output_dir, exist_ok=True)
        print("=" * 70)
        print("  ThinkDet — Unified Training (no Stage 1 / no Stage 2 split)")
        print("=" * 70)
        print(f"  extract_layers:   {cfg.extract_layers}")
        print(f"  injection_layers: {cfg.injection_layers}")
        print(f"  fusion_mode:      {cfg.fusion_mode}")
        print(f"  tma_m (tokens):   {cfg.tma_m}")
        print(f"  alpha_init:       {cfg.tma_alpha_init}  (gate=tanh({cfg.tma_alpha_init})=0)")
        print(f"  lr_adapters:      {cfg.lr_adapters}")
        print(f"  lr_heads:         {cfg.lr_heads}")
        print(f"  lambda_kd:        {cfg.lambda_kd}  (KD: logits + boxes)")
        print(f"  preserve_kd:      {cfg.preserve_kd_enabled}")
        if cfg.lambda_kd <= 0 or not cfg.preserve_kd_enabled:
            print("  KD mode:          disabled (no preserve-KD loss applied)")
        print(f"  lambda_gate_l1:   {cfg.lambda_gate_l1}  (L1 on tanh(alpha))")
        print(f"  num_select:       {cfg.num_select}  (official eval, no conf thresh)")
        print(f"  GPUs:             {world_size}")
        print(f"  Effective batch:  {cfg.per_gpu_batch * cfg.grad_accum * world_size}")
        print(f"  Epochs:           {cfg.epochs}")
        print(f"  Seed:             {cfg.seed}")
        print(f"  Warmup:           {cfg.warmup_ratio:.0%} of total steps")
        print(f"  Output:           {cfg.output_dir}")
        print()

    # ── 1. Build model ─────────────────────────────────────────────
    if is_main:
        print("[1/5] Loading GroundingDINO ...")
    gd = load_gd_model(cfg.gd_config, cfg.gd_weights, device="cpu")

    if is_main:
        print("[1/5] Building ThinkDetModel ...")
    model = ThinkDetModel(
        grounding_dino   = gd,
        internvl_path    = cfg.internvl_path,
        extract_layer    = cfg.extract_layer,
        extract_layers   = cfg.extract_layers,
        layer_fusion     = cfg.layer_fusion,
        injection_layers = cfg.injection_layers,
        d_model          = cfg.d_model,
        tma_m            = cfg.tma_m,
        tma_n_heads      = cfg.tma_n_heads,
        tma_alpha_init   = cfg.tma_alpha_init,
        fusion_mode      = cfg.fusion_mode,   # "residual" always
        preserve_kd_enabled = cfg.preserve_kd_enabled,
    )

    # Joint training from epoch 1: adapters + heads, everything else frozen
    model.set_tma_and_heads()
    model = model.to(device)

    if ddp:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    model_raw = model.module if ddp else model

    if is_main:
        total_p   = sum(p.numel() for p in model_raw.parameters())
        trainable = sum(p.numel() for p in model_raw.parameters() if p.requires_grad)
        print(f"  Trainable: {trainable:,} / {total_p:,} "
              f"({100*trainable/total_p:.2f}%)")

    # ── 2. Data ────────────────────────────────────────────────────
    if is_main:
        print("\n[2/5] Loading COCO train2017 ...")
    train_ds = COCOGroundingDataset(
        img_dir    = cfg.train_img_dir,
        ann_file   = cfg.train_ann,
        query_mode = "fixed",   # fixed vocab for consistent KD reference
    )
    if cfg.debug_limit > 0:
        train_ds.img_ids = train_ds.img_ids[: cfg.debug_limit]
        cfg.log_interval = 10
        if is_main:
            print(f"  [DEBUG] limited to {cfg.debug_limit} images")

    sampler      = DistributedSampler(train_ds, shuffle=True, seed=cfg.seed) if ddp else None
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size  = cfg.per_gpu_batch,
        shuffle     = (sampler is None),
        num_workers = cfg.num_workers,
        collate_fn  = collate_fn,
        pin_memory  = True,
        drop_last   = True,
        sampler     = sampler,
    )

    # Positive map for detection loss
    all_cat_names = sorted(train_ds.cat_id_to_name.values())
    gd_tok  = model_raw.grounding_dino.tokenizer
    gd_spec = model_raw.grounding_dino.specical_tokens
    all_query_text, pmap, pmap_norm, cat_name_to_idx = build_positive_map(
        gd_tok, gd_spec, all_cat_names,
    )
    pmap      = pmap.to(device)
    pmap_norm = pmap_norm.to(device)

    if is_main:
        print(f"  Categories: {len(all_cat_names)}")
        print(f"  Query: '{all_query_text[:80]}...'")

    # Val dataset (full, for official eval)
    val_ds = COCOGroundingDataset(
        img_dir    = cfg.val_img_dir,
        ann_file   = cfg.val_ann,
        query_mode = "fixed",
    )

    # ── 3. Optimizer ───────────────────────────────────────────────
    if is_main:
        print("\n[3/5] Setting up optimiser ...")

    aug_ids     = {id(p) for p in model_raw.get_augmenter_params()}
    adp_params, head_params = [], []
    for name, p in model_raw.named_parameters():
        if not p.requires_grad:
            continue
        if id(p) in aug_ids:
            adp_params.append(p)
        else:
            head_params.append(p)

    if is_main:
        print(f"  Adapter params: {sum(p.numel() for p in adp_params):,}")
        print(f"  Head params:    {sum(p.numel() for p in head_params):,}")

    optimizer = torch.optim.AdamW([
        {"params": adp_params,  "lr": cfg.lr_adapters, "name": "adapters"},
        {"params": head_params, "lr": cfg.lr_heads,    "name": "heads"},
    ], weight_decay=cfg.weight_decay)

    steps_per_epoch = len(train_loader) // cfg.grad_accum
    total_steps     = steps_per_epoch * cfg.epochs
    warmup_steps    = int(total_steps * cfg.warmup_ratio)
    scheduler       = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    if is_main:
        print(f"  steps/epoch: {steps_per_epoch}  total: {total_steps}  "
              f"warmup: {warmup_steps}")

    # ── Resume ─────────────────────────────────────────────────────
    start_epoch  = 0
    global_step  = 0
    best_ap      = -1.0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location="cpu")
        model_raw.load_state_dict(ckpt["trainable_state_dict"], strict=False)
        if not args.resume_weights_only:
            if "optimizer_state_dict" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if "scheduler_state_dict" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt.get("epoch", 0)
        global_step = ckpt.get("global_step", 0)
        if is_main:
            resume_mode = "weights-only" if args.resume_weights_only else "full-state"
            print(
                f"  Resumed from {args.resume} ({resume_mode})  "
                f"epoch={start_epoch}  step={global_step}"
            )

    # ── 4. Training ────────────────────────────────────────────────
    if is_main:
        print("\n[4/5] Training ...\n")

    log_path     = os.path.join(cfg.output_dir, "train_log.jsonl")
    val_log_path = os.path.join(cfg.output_dir, "val_log.jsonl")
    if is_main and not args.resume:
        for p in (log_path, val_log_path):
            with open(p, "w"):
                pass

    scaler        = GradScaler()
    grad_checked  = False
    t0            = time.time()

    for epoch in range(start_epoch, cfg.epochs):
        if ddp and sampler:
            sampler.set_epoch(epoch)

        model.train()
        optimizer.zero_grad()
        epoch_losses      = []
        epoch_det_losses  = []
        epoch_kd_losses   = []

        for batch_idx, batch in enumerate(train_loader):
            internvl_images = batch["internvl_images"].to(device)
            dino_nested = NestedTensor(
                batch["dino_images"].tensors.to(device),
                batch["dino_images"].mask.to(device),
            )
            B           = internvl_images.shape[0]
            dino_inputs = {"samples": dino_nested, "captions": [all_query_text] * B}

            try:
                with autocast(dtype=torch.float16):
                    # ── Adapted forward (InternVL + DINO with TMA) ──
                    adapted_out, aux = model(
                        internvl_images, batch["query_texts"], dino_inputs,
                    )

                    # Detection loss (Hungarian matching)
                    det_loss = compute_detection_loss(
                        adapted_out["pred_logits"], adapted_out["pred_boxes"],
                        batch["boxes"], batch["category_names"],
                        cat_name_to_idx, pmap, pmap_norm, device,
                    )

                    # Single-forward preservation KD on pre-adapter decoder text
                    # memory (detached targets), avoiding dual-forward
                    # interference in the shared decoder path.
                    if cfg.lambda_kd > 0:
                        kd_loss = compute_kd_loss(aux, det_loss)
                    else:
                        kd_loss = det_loss.new_zeros(())

                    # Gate L1 regularisation: keep gates small unless useful
                    gate_l1 = torch.stack([
                        torch.tanh(layer.augmenter.alpha).abs()
                        for layer in model_raw.adapted_layers
                    ]).sum()

                    loss = (
                        det_loss
                        + cfg.lambda_kd * kd_loss
                        + cfg.lambda_gate_l1 * gate_l1
                    )

            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    if is_main:
                        print(f"  [OOM] step {global_step}, skipping")
                    torch.cuda.empty_cache()
                    optimizer.zero_grad()
                    continue
                raise

            if torch.isnan(loss) or torch.isinf(loss):
                if is_main:
                    print(f"  [NaN/Inf] step {global_step}, det={det_loss.item():.4f} "
                          f"kd={kd_loss.item():.4f}, skipping")
                optimizer.zero_grad()
                continue

            scaler.scale(loss / cfg.grad_accum).backward()
            epoch_losses.append(loss.item())
            epoch_det_losses.append(det_loss.item())
            epoch_kd_losses.append(kd_loss.item())

            if (batch_idx + 1) % cfg.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model_raw.parameters() if p.requires_grad],
                    cfg.max_grad_norm,
                )

                if not grad_checked and global_step >= 50:
                    gradient_health_check(model_raw, is_main)
                    grad_checked = True

                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if is_main and global_step % cfg.log_interval == 0:
                    sl    = slice(-cfg.log_interval * cfg.grad_accum, None)
                    avg   = float(np.mean(epoch_losses[sl]))
                    avg_d = float(np.mean(epoch_det_losses[sl]))
                    avg_k = float(np.mean(epoch_kd_losses[sl]))
                    lr_now = scheduler.get_last_lr()[0]

                    elapsed = time.time() - t0
                    progress = max(global_step - start_epoch * steps_per_epoch, 1)
                    eta     = elapsed / progress * (total_steps - global_step)

                    # Gate values (tanh(alpha)) for each adapted layer
                    gates = aux.get("gate_values", [])
                    gate_str = "  ".join(
                        f"L{cfg.injection_layers[i]}_gate={g:.4f}"
                        for i, g in enumerate(gates)
                    ) or "no adapters"

                    hvlm = aux.get("h_vlm_norm_mean", float("nan"))

                    print(
                        f"  [E{epoch+1}/{cfg.epochs}] step {global_step}/{total_steps}  "
                        f"loss={avg:.4f}  det={avg_d:.4f}  kd={avg_k:.4f}  "
                        f"{gate_str}  h_vlm={hvlm:.2f}  "
                        f"lr={lr_now:.2e}  ETA={eta/3600:.1f}h"
                    )

                    with open(log_path, "a") as f:
                        f.write(json.dumps({
                            "step":         global_step,
                            "epoch":        epoch + 1,
                            "loss":         avg,
                            "det_loss":     avg_d,
                            "kd_loss":      avg_k,
                            "gate_values":  gates,
                            "h_vlm_norm":   float(hvlm),
                            "lr":           float(lr_now),
                        }) + "\n")

                if is_main and global_step % cfg.save_interval == 0:
                    save_checkpoint(
                        model_raw, optimizer, scheduler,
                        epoch, global_step, cfg, tag=f"step{global_step}",
                    )

        # ── End of epoch ────────────────────────────────────────────
        epoch_avg = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        if is_main:
            elapsed = time.time() - t0
            print(f"\n{'='*65}")
            print(f"  Epoch {epoch+1}/{cfg.epochs}  loss={epoch_avg:.4f}  "
                  f"det={np.mean(epoch_det_losses):.4f}  "
                  f"kd={np.mean(epoch_kd_losses):.4f}  "
                  f"time={elapsed/3600:.2f}h")
            print(f"{'='*65}")
            save_checkpoint(
                model_raw, optimizer, scheduler,
                epoch, global_step, cfg, tag=f"epoch{epoch+1}",
            )

        # ── Official COCO val eval (main process only) ─────────────
        should_eval = (
            is_main
            and cfg.val_every_ep > 0
            and (epoch + 1) % cfg.val_every_ep == 0
        )
        if should_eval:
            print(f"\n[5/5] Official COCO val eval (epoch {epoch+1}) ...")
            val_res = evaluate_coco_official(
                model_raw          = model_raw,
                val_dataset        = val_ds,
                positive_map_norm  = pmap_norm,
                cat_name_to_idx    = cat_name_to_idx,
                all_query_text     = all_query_text,
                device             = device,
                num_select         = cfg.num_select,
            )
            val_res["epoch"]      = epoch + 1
            val_res["global_step"] = global_step
            val_res["train_loss"] = epoch_avg

            with open(val_log_path, "a") as f:
                f.write(json.dumps(val_res) + "\n")

            ap = val_res["AP"]
            print(
                f"  [EVAL] AP={ap:.4f}  AP50={val_res['AP_50']:.4f}  "
                f"AP75={val_res['AP_75']:.4f}  "
                f"preds={val_res['num_predictions']}"
            )
            # Baseline is 48.53 — flag clearly if we're close
            diff = ap - 0.4853
            status = (
                "✅ WITHIN ±0.002 OF BASELINE"
                if abs(diff) < 0.002
                else f"{'⬆' if diff > 0 else '⬇'} Δ={diff:+.4f} vs baseline 0.4853"
            )
            print(f"  {status}")

            if ap > best_ap:
                best_ap = ap
                save_checkpoint(
                    model_raw, optimizer, scheduler,
                    epoch, global_step, cfg, tag="best",
                )
                print(f"  [BEST] New best AP={best_ap:.4f}")

            model.train()

        if ddp:
            dist.barrier()

    # ── Done ────────────────────────────────────────────────────────
    if is_main:
        total_time = time.time() - t0
        print(f"\n{'='*70}")
        print("  ThinkDet Unified Training complete!")
        print(f"  Total time:  {total_time/3600:.2f}h")
        print(f"  Best val AP: {best_ap:.4f}  (baseline target: 0.4853)")
        print(f"  Checkpoints: {cfg.output_dir}")
        print(f"{'='*70}")

    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
