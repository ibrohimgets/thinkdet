"""
ThinkDet TMA — RefCOCO Stage 2: Fine-tune TMA + Detection Heads on RefCOCO

Protocol:
    - Load Stage-1 TMA checkpoint (COCO-trained TextAugmenters)
    - Unfreeze TextAugmenters + detection heads (class_embed, bbox_embed)
    - InternVL + DINO backbone remain frozen
    - Train on joint RefCOCO + RefCOCO+ + RefCOCOg
    - Loss: REC loss (5*L1 + 2*GIoU + FocalLoss) for 1 GT box per sample
    - AMP (fp16) for efficiency

Launch:
    torchrun --nproc_per_node=8 thinkdet/scripts/training/train_refcoco_stage2_tma.py
"""

import os
import sys
import time
import json
import math
import argparse
import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler

ROOT = "/home/iibrohimm/project/next_step"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "GroundingDINO", "GroundingDINO"))

from thinkdet.models.arch import ThinkDetModel, DEFAULT_INJECTION_LAYERS
from thinkdet.data.refcoco_grounding import (
    build_joint_refcoco_train,
    build_refcoco_eval,
    refcoco_collate_fn,
    build_rec_positive_map_batch,
)
from groundingdino.util.inference import load_model as load_gd_model
from groundingdino.util.misc import NestedTensor


# ===================================================================
# Stage 1 checkpoint (default)
# ===================================================================
STAGE1_CKPT = (
    f"{ROOT}/thinkdet/checkpoints/stage1_tma/"
    "layer9_tma_m8_full_2ep_7gpu_20260218_152649/"
    "thinkdet_tma_stage1_epoch2.pth"
)


# ===================================================================
# BOX UTILITIES
# ===================================================================
def box_cxcywh_to_xyxy(x):
    cx, cy, w, h = x.unbind(-1)
    return torch.stack([cx - 0.5*w, cy - 0.5*h, cx + 0.5*w, cy + 0.5*h], -1)


def safe_generalized_box_iou(b1, b2):
    b1 = torch.stack([b1[...,0], b1[...,1],
                      torch.max(b1[...,2], b1[...,0]+1e-4),
                      torch.max(b1[...,3], b1[...,1]+1e-4)], -1)
    b2 = torch.stack([b2[...,0], b2[...,1],
                      torch.max(b2[...,2], b2[...,0]+1e-4),
                      torch.max(b2[...,3], b2[...,1]+1e-4)], -1)
    a1 = (b1[:,2]-b1[:,0])*(b1[:,3]-b1[:,1])
    a2 = (b2[:,2]-b2[:,0])*(b2[:,3]-b2[:,1])
    lt = torch.max(b1[:,None,:2], b2[None,:,:2])
    rb = torch.min(b1[:,None,2:], b2[None,:,2:])
    wh = (rb-lt).clamp(min=0); inter = wh[:,:,0]*wh[:,:,1]
    union = a1[:,None]+a2[None,:]-inter
    iou = inter/(union+1e-6)
    lt2 = torch.min(b1[:,None,:2], b2[None,:,:2])
    rb2 = torch.max(b1[:,None,2:], b2[None,:,2:])
    wh2 = (rb2-lt2).clamp(min=0); ae = wh2[:,:,0]*wh2[:,:,1]
    return iou - (ae-union)/(ae+1e-6)


def compute_iou(box1, box2):
    lt = torch.max(box1[:, :2], box2[:, :2])
    rb = torch.min(box1[:, 2:], box2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]
    a1 = (box1[:, 2]-box1[:, 0]) * (box1[:, 3]-box1[:, 1])
    a2 = (box2[:, 2]-box2[:, 0]) * (box2[:, 3]-box2[:, 1])
    return inter / (a1 + a2 - inter + 1e-6)


# ===================================================================
# LOSS
# ===================================================================
def sigmoid_focal_loss(inputs, targets, num_boxes, alpha=0.25, gamma=2.0):
    inputs = inputs.clamp(min=-50, max=50)
    prob = inputs.sigmoid()
    ce = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob*targets + (1-prob)*(1-targets)
    loss = ce * ((1-p_t)**gamma)
    if alpha >= 0:
        alpha_t = alpha*targets + (1-alpha)*(1-targets)
        loss = alpha_t*loss
    return loss.mean(1).sum() / max(num_boxes, 1)


def compute_rec_loss(pred_logits, pred_boxes, gt_boxes_list,
                     positive_maps, positive_maps_norm, device):
    """
    REC loss: each sample has exactly 1 GT box.
    Hungarian matching to find best query, then L1 + GIoU + focal.
    """
    from scipy.optimize import linear_sum_assignment
    B = pred_logits.shape[0]
    text_len = pred_logits.shape[2]
    total_loss = torch.tensor(0.0, device=device, requires_grad=True)

    pmaps = positive_maps[:, :, :text_len].to(device)
    pmaps_n = positive_maps_norm[:, :, :text_len].to(device)

    for b in range(B):
        logits = pred_logits[b]
        boxes  = pred_boxes[b]
        gt     = gt_boxes_list[b].to(device)
        pmap   = pmaps[b]
        pmap_n = pmaps_n[b]

        with torch.no_grad():
            probs = logits.clamp(-50, 50).sigmoid()
            cp = (probs * pmap_n).sum(dim=-1, keepdim=True)
            cc = -cp
            cl1 = torch.cdist(boxes, gt, p=1)
            cg = 1.0 - safe_generalized_box_iou(
                box_cxcywh_to_xyxy(boxes), box_cxcywh_to_xyxy(gt))
            C = (5*cl1 + 2*cg + cc).nan_to_num(100., 100., -100.)
            ri, coli = linear_sum_assignment(C.cpu().numpy())

        ri = torch.tensor(ri, dtype=torch.long, device=device)
        coli = torch.tensor(coli, dtype=torch.long, device=device)

        mp, mg = boxes[ri], gt[coli]
        l1 = F.l1_loss(mp, mg, reduction='sum')
        gv = safe_generalized_box_iou(
            box_cxcywh_to_xyxy(mp), box_cxcywh_to_xyxy(mg))
        gl = (1 - gv.diag()).sum()

        tgt = torch.zeros_like(logits)
        tgt[ri[0]] = pmap[0]
        fl = sigmoid_focal_loss(logits, tgt, 1)

        total_loss = total_loss + 5*l1 + 2*gl + fl

    return total_loss / max(B, 1)


# ===================================================================
# REC EVALUATION (Acc@0.5)
# ===================================================================
@torch.no_grad()
def evaluate_rec(model_raw, eval_dataset, tokenizer, special_tokens,
                 device, max_samples=None):
    model_raw.eval()
    loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=1, shuffle=False,
        num_workers=4, collate_fn=refcoco_collate_fn, pin_memory=True,
    )

    correct = 0
    total = 0
    t0 = time.time()

    for batch_idx, batch in enumerate(loader):
        if max_samples and total >= max_samples:
            break

        internvl_images = batch['internvl_images'].to(device)
        dino_nested = NestedTensor(
            batch['dino_images'].tensors.to(device),
            batch['dino_images'].mask.to(device),
        )
        expression = batch['expressions'][0]
        query_text = batch['query_texts'][0]
        gt_box = batch['boxes'][0].to(device)

        dino_inputs = {'samples': dino_nested, 'captions': [query_text]}

        try:
            outputs, aux = model_raw(internvl_images, [expression], dino_inputs)
        except Exception:
            total += 1
            continue

        _, _, pmap_norm = build_rec_positive_map_batch(
            tokenizer, special_tokens, [expression],
        )

        logits = outputs['pred_logits'][0]
        boxes  = outputs['pred_boxes'][0]

        text_len = logits.shape[-1]
        pmap_n = pmap_norm[0, :, :text_len].to(device)

        probs = logits.clamp(-50, 50).sigmoid()
        scores = (probs * pmap_n).sum(dim=-1)

        best_idx = scores.argmax()
        pred_box = boxes[best_idx].unsqueeze(0)

        pred_xyxy = box_cxcywh_to_xyxy(pred_box)
        gt_xyxy   = box_cxcywh_to_xyxy(gt_box)
        iou = compute_iou(pred_xyxy, gt_xyxy)

        if iou.item() > 0.5:
            correct += 1
        total += 1

        if total % 1000 == 0:
            elapsed = time.time() - t0
            print("    eval: {}/{} acc={:.4f} ({:.0f}s)".format(
                total, len(eval_dataset), correct/total, elapsed))

    accuracy = correct / max(total, 1)
    return accuracy, correct, total


def evaluate_all_splits(model_raw, tokenizer, special_tokens, device,
                        data_root, image_dir):
    eval_configs = [
        ('refcoco',  'unc', 'val'),
        ('refcoco',  'unc', 'testA'),
        ('refcoco',  'unc', 'testB'),
        ('refcoco+', 'unc', 'val'),
        ('refcoco+', 'unc', 'testA'),
        ('refcoco+', 'unc', 'testB'),
        ('refcocog', 'umd', 'val'),
        ('refcocog', 'umd', 'test'),
    ]

    results = {}
    for ds_name, split_by, split in eval_configs:
        ds = build_refcoco_eval(
            data_root=data_root, image_dir=image_dir,
            dataset_name=ds_name, split_by=split_by, split=split,
        )
        acc, correct, total = evaluate_rec(
            model_raw, ds, tokenizer, special_tokens, device,
        )
        key = "{}_{}".format(ds_name, split)
        results[key] = {
            'accuracy': acc,
            'correct': correct,
            'total': total,
        }
        print("  {}: {:.4f} ({}/{})".format(key, acc, correct, total))

    return results


# ===================================================================
# COSINE SCHEDULE
# ===================================================================
def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        p = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * p)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ===================================================================
# CONFIG
# ===================================================================
class Cfg:
    # Data (RefCOCO)
    data_root     = "{}/dataSets/refer/data".format(ROOT)
    image_dir     = "{}/dataSets/coco/train2017".format(ROOT)

    # Models
    gd_config  = (
        "{}/GroundingDINO/GroundingDINO/groundingdino/config/"
        "GroundingDINO_SwinT_OGC.py".format(ROOT)
    )
    gd_weights = (
        "{}/GroundingDINO/GroundingDINO/weights/groundingdino_swint_ogc.pth".format(ROOT)
    )
    internvl_path = "{}/InternVL3_5-1B".format(ROOT)
    stage1_ckpt   = STAGE1_CKPT

    # Layer setup (must match stage1)
    extract_layer    = 9
    extract_layers   = [9]
    layer_fusion     = "mean"
    d_model          = 256
    tma_m            = 8
    tma_n_heads      = 8
    # Safer default for baseline-preserving experiments.
    injection_layers = [3]
    tma_fusion_mode = "residual"  # concat | residual

    # Training
    epochs        = 3
    per_gpu_batch = 2
    grad_accum    = 4     # effective batch = 2 * 4 * 8 = 64
    lr_tma        = 3e-5
    lr_heads      = 5e-5
    weight_decay  = 0.01
    warmup_ratio  = 0.05
    max_grad_norm = 1.0
    num_workers   = 4
    debug_limit   = 0

    # Save / log
    output_dir    = "{}/thinkdet/checkpoints/refcoco_stage2_tma".format(ROOT)
    log_interval  = 100
    save_interval = 2000
    grad_check_step = 100
    eval_epochs   = [1, 2, 3]


# ===================================================================
# DISTRIBUTED
# ===================================================================
def init_distributed():
    ddp = int(os.environ.get("WORLD_SIZE", 1)) > 1
    if ddp:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda:{}".format(local_rank))
        world_size = dist.get_world_size()
    else:
        local_rank = 0
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        world_size = 1
    torch.backends.cudnn.benchmark = True
    return ddp, local_rank, device, world_size


# ===================================================================
# GRADIENT HEALTH CHECK
# ===================================================================
def gradient_health_check(model_raw, is_main):
    if not is_main:
        return
    print("\n  [GRAD CHECK] ──────────────────────────────────")
    tma_ok, head_ok, issues = 0, 0, 0
    for name, param in model_raw.named_parameters():
        if param.requires_grad:
            if param.grad is None:
                print("    WARNING: {} — no gradient!".format(name))
                issues += 1
            elif param.grad.abs().max().item() == 0:
                print("    WARNING: {} — zero gradient!".format(name))
                issues += 1
            else:
                grad_norm = param.grad.norm().item()
                if "alpha" in name:
                    print("    OK: {} — grad_norm={:.6f}  value={:.6f}".format(
                        name, grad_norm, param.item()))
                    tma_ok += 1
                elif "augmenter" in name or "residual_fuser" in name:
                    tma_ok += 1
                else:
                    head_ok += 1
        elif param.grad is not None:
            print("    WARNING: {} — frozen but has gradient!".format(name))
            issues += 1
    print("    Summary: TMA={} ok, Heads={} ok, Issues={}".format(
        tma_ok, head_ok, issues))
    print("  ─────────────────────────────────────────────\n")


# ===================================================================
# CHECKPOINT
# ===================================================================
def load_stage1_weights(model, ckpt_path, is_main):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError("Stage-1 checkpoint not found: {}".format(ckpt_path))
    ckpt = torch.load(ckpt_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(
        ckpt["trainable_state_dict"], strict=False
    )
    if is_main:
        print("  Loaded Stage-1 weights: {}".format(ckpt_path))
        print("  Missing keys:     {}".format(len(missing)))
        print("  Unexpected keys:  {}".format(len(unexpected)))


def save_checkpoint(model, optimizer, scheduler, epoch, step, cfg, tag=""):
    trainable_state = {
        name: param.detach().cpu()
        for name, param in model.named_parameters()
        if param.requires_grad
    }
    path = os.path.join(cfg.output_dir, "thinkdet_refcoco_stage2_tma_{}.pth".format(tag))
    torch.save(
        {
            "epoch": epoch + 1,
            "global_step": step,
            "stage": "refcoco_stage2_tma",
            "stage1_ckpt": cfg.stage1_ckpt,
            "extract_layer": cfg.extract_layer,
            "extract_layers": cfg.extract_layers,
            "layer_fusion": cfg.layer_fusion,
            "injection_layers": cfg.injection_layers,
            "fusion_mode": cfg.tma_fusion_mode,
            "tma_m": cfg.tma_m,
            "tma_n_heads": cfg.tma_n_heads,
            "trainable_state_dict": trainable_state,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        },
        path,
    )
    print("  [SAVE] {}  ({} param tensors)".format(path, len(trainable_state)))


# ===================================================================
# ARGS
# ===================================================================
def parse_args():
    cfg = Cfg()
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1_ckpt", type=str, default=cfg.stage1_ckpt)
    parser.add_argument("--epochs", type=int, default=cfg.epochs)
    parser.add_argument("--debug_limit", type=int, default=cfg.debug_limit)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--lr_tma", type=float, default=cfg.lr_tma)
    parser.add_argument("--lr_heads", type=float, default=cfg.lr_heads)
    parser.add_argument("--per_gpu_batch", type=int, default=cfg.per_gpu_batch)
    parser.add_argument("--tma_m", type=int, default=cfg.tma_m)
    parser.add_argument(
        "--tma_fusion_mode",
        type=str,
        default=cfg.tma_fusion_mode,
        choices=["concat", "residual"],
    )
    parser.add_argument("--resume", type=str, default=None)
    return parser.parse_args()


# ===================================================================
# MAIN
# ===================================================================
def main():
    cfg = Cfg()
    args = parse_args()
    cfg.stage1_ckpt = args.stage1_ckpt
    cfg.epochs = args.epochs
    cfg.debug_limit = args.debug_limit
    cfg.lr_tma = float(args.lr_tma)
    cfg.lr_heads = float(args.lr_heads)
    cfg.per_gpu_batch = int(args.per_gpu_batch)
    cfg.tma_m = int(args.tma_m)
    cfg.tma_fusion_mode = str(args.tma_fusion_mode)
    if args.output_dir:
        cfg.output_dir = args.output_dir

    ddp, local_rank, device, world_size = init_distributed()
    is_main = local_rank == 0

    if is_main:
        ts = time.strftime("%Y%m%d_%H%M%S")
        cfg.output_dir = "{}_{}gpu_{}".format(cfg.output_dir, world_size, ts)
        os.makedirs(cfg.output_dir, exist_ok=True)
        print("=" * 70)
        print("  ThinkDet TMA — RefCOCO Stage 2: TMA + Detection Heads")
        print("=" * 70)
        print("  stage1_ckpt:      {}".format(cfg.stage1_ckpt))
        print("  extract_layer:    {}".format(cfg.extract_layer))
        print("  injection_layers: {}".format(cfg.injection_layers))
        print("  tma_m:            {}".format(cfg.tma_m))
        print("  tma_fusion_mode:  {}".format(cfg.tma_fusion_mode))
        print("  lr_tma:           {}".format(cfg.lr_tma))
        print("  lr_heads:         {}".format(cfg.lr_heads))
        print("  epochs:           {}".format(cfg.epochs))
        print("  GPUs:             {}".format(world_size))
        print("  per_gpu_batch:    {}".format(cfg.per_gpu_batch))
        print("  grad_accum:       {}".format(cfg.grad_accum))
        print("  effective batch:  {}".format(
            cfg.per_gpu_batch * cfg.grad_accum * world_size))
        print("  AMP:              fp16")
        print("  output:           {}".format(cfg.output_dir))
        print()

    # Broadcast output_dir from rank 0
    if ddp:
        if is_main:
            dir_bytes = cfg.output_dir.encode('utf-8')
            dir_len = torch.tensor([len(dir_bytes)], dtype=torch.long, device=device)
        else:
            dir_len = torch.tensor([0], dtype=torch.long, device=device)
        dist.broadcast(dir_len, src=0)
        if is_main:
            dir_tensor = torch.tensor(list(dir_bytes), dtype=torch.uint8, device=device)
        else:
            dir_tensor = torch.zeros(dir_len.item(), dtype=torch.uint8, device=device)
        dist.broadcast(dir_tensor, src=0)
        if not is_main:
            cfg.output_dir = bytes(dir_tensor.cpu().tolist()).decode('utf-8')

    # ── 1. Build model ──
    if is_main:
        print("[1/4] Building model ...")
    gd = load_gd_model(cfg.gd_config, cfg.gd_weights, device="cpu")
    model = ThinkDetModel(
        grounding_dino=gd,
        internvl_path=cfg.internvl_path,
        extract_layer=cfg.extract_layer,
        extract_layers=cfg.extract_layers,
        layer_fusion=cfg.layer_fusion,
        injection_layers=cfg.injection_layers,
        d_model=cfg.d_model,
        tma_m=cfg.tma_m,
        tma_n_heads=cfg.tma_n_heads,
        fusion_mode=cfg.tma_fusion_mode,
    )
    model.set_tma_and_heads()
    load_stage1_weights(model, cfg.stage1_ckpt, is_main=is_main)
    model = model.to(device)

    if ddp:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    model_raw = model.module if ddp else model

    total_p = sum(p.numel() for p in model_raw.parameters())
    train_p = sum(p.numel() for p in model_raw.parameters() if p.requires_grad)
    if is_main:
        print("  Total params:     {:,}".format(total_p))
        print("  Trainable params: {:,} ({:.2f}%)".format(
            train_p, 100*train_p/total_p))

    # ── 2. Data ──
    if is_main:
        print("\n[2/4] Loading RefCOCO/+/g train ...")
    train_ds = build_joint_refcoco_train(
        data_root=cfg.data_root,
        image_dir=cfg.image_dir,
    )
    if cfg.debug_limit > 0:
        # Limit for sanity check
        from torch.utils.data import Subset
        indices = list(range(min(cfg.debug_limit, len(train_ds))))
        train_ds = Subset(train_ds, indices)
        cfg.log_interval = 10
        if is_main:
            print("  [DEBUG] Limiting to {} samples".format(cfg.debug_limit))

    sampler = DistributedSampler(train_ds, shuffle=True) if ddp else None
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=cfg.per_gpu_batch,
        shuffle=(sampler is None),
        num_workers=cfg.num_workers,
        collate_fn=refcoco_collate_fn,
        pin_memory=True,
        drop_last=True,
        sampler=sampler,
    )

    gd_tok = model_raw.grounding_dino.tokenizer
    gd_spec = model_raw.grounding_dino.specical_tokens

    if is_main:
        print("  Train samples: {}".format(len(train_ds)))

    # ── 3. Optimizer ──
    if is_main:
        print("\n[3/4] Setting up optimizer ...")

    aug_params, head_params = [], []
    aug_ids = {id(p) for p in model_raw.get_augmenter_params()}
    for name, param in model_raw.named_parameters():
        if not param.requires_grad:
            continue
        if id(param) in aug_ids:
            aug_params.append(param)
        else:
            head_params.append(param)

    if not head_params and not aug_params:
        raise RuntimeError("No trainable parameters found.")

    param_groups = []
    if aug_params:
        param_groups.append({"params": aug_params, "lr": cfg.lr_tma, "name": "tma"})
        if is_main:
            print("  TMA params:  {:,}".format(sum(p.numel() for p in aug_params)))
    if head_params:
        param_groups.append({"params": head_params, "lr": cfg.lr_heads, "name": "heads"})
        if is_main:
            print("  Head params: {:,}".format(sum(p.numel() for p in head_params)))

    optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.weight_decay)

    steps_per_epoch = len(train_loader) // cfg.grad_accum
    total_steps = steps_per_epoch * cfg.epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    if is_main:
        print("  steps/epoch: {}  total: {}  warmup: {}".format(
            steps_per_epoch, total_steps, warmup_steps))

    start_epoch = 0
    global_step = 0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location="cpu")
        model_raw.load_state_dict(ckpt["trainable_state_dict"], strict=False)
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt.get("epoch", 0)
        global_step = ckpt.get("global_step", 0)
        if is_main:
            print("  Resumed at epoch {}, step {}".format(start_epoch, global_step))

    # ── 4. Train ──
    if is_main:
        print("\n[4/4] Training ...\n")

    log_path = os.path.join(cfg.output_dir, "train_log.jsonl")
    if is_main and not args.resume:
        with open(log_path, "w"):
            pass

    scaler = GradScaler()
    grad_checked = False
    best_acc = 0.0

    t0 = time.time()
    for epoch in range(start_epoch, cfg.epochs):
        if ddp and sampler:
            sampler.set_epoch(epoch)

        model.train()
        optimizer.zero_grad()
        epoch_losses = []

        for batch_idx, batch in enumerate(train_loader):
            internvl_images = batch["internvl_images"].to(device)
            dino_nested = NestedTensor(
                batch["dino_images"].tensors.to(device),
                batch["dino_images"].mask.to(device),
            )
            B = internvl_images.shape[0]

            # Per-sample captions and positive maps
            query_texts = batch["query_texts"]
            dino_inputs = {"samples": dino_nested, "captions": query_texts}

            _, pmaps, pmaps_norm = build_rec_positive_map_batch(
                gd_tok, gd_spec, batch["expressions"],
            )

            try:
                with autocast(dtype=torch.float16):
                    outputs, aux = model(
                        internvl_images, batch["expressions"], dino_inputs
                    )

                    loss = compute_rec_loss(
                        outputs["pred_logits"],
                        outputs["pred_boxes"],
                        batch["boxes"],
                        pmaps,
                        pmaps_norm,
                        device,
                    )
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    if is_main:
                        print("  [OOM] step {}, skipping".format(global_step))
                    torch.cuda.empty_cache()
                    optimizer.zero_grad()
                    continue
                raise

            if torch.isnan(loss) or torch.isinf(loss):
                if is_main:
                    print("  [NaN/Inf] step {}, skipping".format(global_step))
                optimizer.zero_grad()
                continue

            scaler.scale(loss / cfg.grad_accum).backward()
            epoch_losses.append(loss.item())

            if (batch_idx + 1) % cfg.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model_raw.parameters() if p.requires_grad],
                    cfg.max_grad_norm,
                )

                if not grad_checked and global_step + 1 >= cfg.grad_check_step:
                    gradient_health_check(model_raw, is_main)
                    grad_checked = True

                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if is_main and global_step % cfg.log_interval == 0:
                    avg = np.mean(epoch_losses[-cfg.log_interval * cfg.grad_accum:])
                    lr_now = scheduler.get_last_lr()[0]
                    elapsed = time.time() - t0
                    done = max(global_step - start_epoch * steps_per_epoch, 1)
                    eta = elapsed / done * (total_steps - global_step)
                    hvlm = aux.get("h_vlm_norm_mean", float("nan"))
                    alpha_vals = aux.get("alpha_values", [])
                    alpha_by_layer = {
                        int(layer): float(alpha_vals[i]) if i < len(alpha_vals) else float("nan")
                        for i, layer in enumerate(cfg.injection_layers)
                    }
                    layer1_alpha = alpha_by_layer.get(1, float("nan"))
                    layer3_alpha = alpha_by_layer.get(3, float("nan"))
                    layer5_alpha = alpha_by_layer.get(5, float("nan"))
                    alpha_msg = "  ".join(
                        "L{}_a={:.4f}".format(
                            int(layer), alpha_by_layer.get(int(layer), float("nan"))
                        )
                        for layer in cfg.injection_layers
                    ) or "no_tma_layers"
                    print(
                        "  [E{}/{}] step {}/{}  "
                        "loss={:.4f}  h_vlm={:.2f}  "
                        "{}  "
                        "lr={:.2e}  ETA={:.1f}h".format(
                            epoch+1, cfg.epochs, global_step, total_steps,
                            avg, hvlm,
                            alpha_msg,
                            lr_now, eta/3600)
                    )
                    with open(log_path, "a") as f:
                        f.write(
                            json.dumps({
                                "step": global_step,
                                "epoch": epoch + 1,
                                "loss": float(avg),
                                "h_vlm_norm_mean": float(hvlm),
                                "Layer1_alpha": float(layer1_alpha),
                                "Layer3_alpha": float(layer3_alpha),
                                "Layer5_alpha": float(layer5_alpha),
                                "lr": float(lr_now),
                            }) + "\n"
                        )

                if is_main and global_step % cfg.save_interval == 0:
                    save_checkpoint(
                        model_raw, optimizer, scheduler,
                        epoch, global_step, cfg, tag="step{}".format(global_step)
                    )

        # ── Epoch summary ──
        epoch_avg = np.mean(epoch_losses) if epoch_losses else 0.0
        if is_main:
            elapsed = time.time() - t0
            print("\n" + "=" * 60)
            print("  Epoch {}/{}  loss={:.4f}  time={:.2f}h".format(
                epoch+1, cfg.epochs, epoch_avg, elapsed/3600))
            print("=" * 60)
            save_checkpoint(
                model_raw, optimizer, scheduler,
                epoch, global_step, cfg, tag="epoch{}".format(epoch+1)
            )

        if ddp:
            dist.barrier()

        # ── Evaluate at scheduled epochs (single-GPU only) ──
        should_eval = ((epoch + 1) in cfg.eval_epochs) and (not ddp)
        if is_main and (epoch + 1) in cfg.eval_epochs and ddp:
            print("\n  [EVAL] Skipping in-train eval at epoch {} under DDP.".format(
                epoch+1))
            print("         Run eval_refcoco.py after training.")

        if is_main and should_eval:
            print("\n  [EVAL] Evaluating REC (epoch {}) ...".format(epoch+1))
            results = evaluate_all_splits(
                model_raw, gd_tok, gd_spec, device,
                cfg.data_root, cfg.image_dir,
            )
            results["epoch"] = epoch + 1
            results["global_step"] = global_step
            results["train_loss"] = float(epoch_avg)

            eval_path = os.path.join(cfg.output_dir,
                                      "eval_epoch{}.json".format(epoch+1))
            with open(eval_path, "w") as f:
                json.dump(results, f, indent=2)

            print("\n  " + "=" * 60)
            print("  REC Results — Epoch {}".format(epoch+1))
            print("  " + "=" * 60)
            for key, val in results.items():
                if isinstance(val, dict) and "accuracy" in val:
                    print("  {:20s}: {:.4f} ({}/{})".format(
                        key, val["accuracy"], val["correct"], val["total"]))
            print("  " + "=" * 60 + "\n")

            val_acc = results.get("refcoco_val", {}).get("accuracy", 0)
            if val_acc > best_acc:
                best_acc = val_acc
                save_checkpoint(model_raw, optimizer, scheduler,
                                epoch, global_step, cfg, tag="best")
                print("  New best refcoco_val acc: {:.4f}".format(best_acc))

            model.train()

        if ddp:
            dist.barrier()

    # ── Final ──
    if is_main:
        total_time = time.time() - t0
        print("\n" + "=" * 70)
        print("  RefCOCO Stage 2 TMA training complete!")
        print("  Total time:  {:.2f}h".format(total_time/3600))
        print("  Best val acc: {:.4f}".format(best_acc))
        print("  Checkpoints: {}".format(cfg.output_dir))
        print("=" * 70)

    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
