"""
ThinkDet v2 — REC Training on RefCOCO / RefCOCO+ / RefCOCOg

    extract layers = {9, 10, [9,10], [9,10,11]} (controlled sweep only)
    Trainable:  adapters + detection heads  (LLM frozen)
    Init:       COCO Stage 3 checkpoint (epoch4)
    Data:       Joint RefCOCO + RefCOCO+ + RefCOCOg train
    Eval:       Acc@0.5 on all val/test splits
    Metric:     Pick highest-scoring query box, IoU > 0.5 = correct

Launch:
    torchrun --nproc_per_node=8 train_refcoco.py
"""

import os
import sys
import time
import json
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

ROOT = '/home/iibrohimm/project/next_step'
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'GroundingDINO', 'GroundingDINO'))

from thinkdet.models.arch import ThinkDetModel
from thinkdet.data.refcoco_grounding import (
    build_joint_refcoco_train,
    build_refcoco_eval,
    refcoco_collate_fn,
    build_rec_positive_map_batch,
)
from groundingdino.util.inference import load_model as load_gd_model
from groundingdino.util.misc import NestedTensor

ALLOWED_LAYER_SETUPS = {
    'layer9': [9],
    'layer10': [10],
    'fusion_9_10': [9, 10],
    'fusion_9_10_11': [9, 10, 11],
}
DEFAULT_LAYER_SETUP = 'fusion_9_10'


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
    """IoU between two [1, 4] xyxy boxes."""
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

    # Slice positive maps to match actual text length
    pmaps = positive_maps[:, :, :text_len].to(device)
    pmaps_n = positive_maps_norm[:, :, :text_len].to(device)

    for b in range(B):
        logits = pred_logits[b]          # [900, text_len]
        boxes  = pred_boxes[b]           # [900, 4]
        gt     = gt_boxes_list[b].to(device)  # [1, 4]
        pmap   = pmaps[b]               # [1, text_len]
        pmap_n = pmaps_n[b]             # [1, text_len]

        # Hungarian matching with 1 GT box
        with torch.no_grad():
            probs = logits.clamp(-50, 50).sigmoid()
            cp = (probs * pmap_n).sum(dim=-1, keepdim=True)  # [900, 1]
            cc = -cp
            cl1 = torch.cdist(boxes, gt, p=1)   # [900, 1]
            cg = 1.0 - safe_generalized_box_iou(
                box_cxcywh_to_xyxy(boxes), box_cxcywh_to_xyxy(gt))
            C = (5*cl1 + 2*cg + cc).nan_to_num(100., 100., -100.)
            ri, coli = linear_sum_assignment(C.cpu().numpy())

        ri = torch.tensor(ri, dtype=torch.long, device=device)
        coli = torch.tensor(coli, dtype=torch.long, device=device)

        # Box losses
        mp, mg = boxes[ri], gt[coli]
        l1 = F.l1_loss(mp, mg, reduction='sum')
        gv = safe_generalized_box_iou(
            box_cxcywh_to_xyxy(mp), box_cxcywh_to_xyxy(mg))
        gl = (1 - gv.diag()).sum()

        # Classification: matched query gets all expression tokens positive
        tgt = torch.zeros_like(logits)
        tgt[ri[0]] = pmap[0]
        fl = sigmoid_focal_loss(logits, tgt, 1)

        total_loss = total_loss + 5*l1 + 2*gl + fl

    return total_loss / max(B, 1)


def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        p = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * p)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ===================================================================
# REC EVALUATION (Acc@0.5)
# ===================================================================
@torch.no_grad()
def evaluate_rec(model_raw, eval_dataset, tokenizer, special_tokens,
                 device, max_samples=None):
    """
    Evaluate REC: pick highest-scoring box, check IoU > 0.5 with GT.
    Returns accuracy as float [0, 1].
    """
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
        gt_box = batch['boxes'][0].to(device)  # [1, 4]

        dino_inputs = {'samples': dino_nested, 'captions': [query_text]}

        try:
            outputs, aux = model_raw(internvl_images, [expression], dino_inputs)
        except Exception:
            total += 1
            continue

        # Build positive map for this expression
        _, _, pmap_norm = build_rec_positive_map_batch(
            tokenizer, special_tokens, [expression],
        )

        logits = outputs['pred_logits'][0]  # [nq, text_len]
        boxes  = outputs['pred_boxes'][0]   # [nq, 4]

        text_len = logits.shape[-1]
        pmap_n = pmap_norm[0, :, :text_len].to(device)  # [1, text_len]

        probs = logits.clamp(-50, 50).sigmoid()
        scores = (probs * pmap_n).sum(dim=-1)  # [nq]

        best_idx = scores.argmax()
        pred_box = boxes[best_idx].unsqueeze(0)  # [1, 4]

        pred_xyxy = box_cxcywh_to_xyxy(pred_box)
        gt_xyxy   = box_cxcywh_to_xyxy(gt_box)
        iou = compute_iou(pred_xyxy, gt_xyxy)

        if iou.item() > 0.5:
            correct += 1
        total += 1

        if total % 1000 == 0:
            elapsed = time.time() - t0
            print(f"    eval: {total}/{len(eval_dataset)} "
                  f"acc={correct/total:.4f} ({elapsed:.0f}s)")

    accuracy = correct / max(total, 1)
    return accuracy, correct, total


def evaluate_all_splits(model_raw, tokenizer, special_tokens, device,
                        data_root, image_dir):
    """Evaluate on all RefCOCO/+/g val/test splits."""
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
        key = f"{ds_name}_{split}"
        results[key] = {
            'accuracy': acc,
            'correct': correct,
            'total': total,
        }
        print(f"  {key}: {acc:.4f} ({correct}/{total})")

    return results


# ===================================================================
# CONFIG
# ===================================================================
class Cfg:
    # Data
    data_root     = f"{ROOT}/dataSets/refer/data"
    image_dir     = f"{ROOT}/dataSets/coco/train2017"

    # Models
    gd_config  = f"{ROOT}/GroundingDINO/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    gd_weights = f"{ROOT}/GroundingDINO/GroundingDINO/weights/groundingdino_swint_ogc.pth"
    internvl_path = f"{ROOT}/InternVL3_5-1B"

    # Init from COCO detection checkpoint
    coco_checkpoint = f"{ROOT}/thinkdet/checkpoints/stage3/thinkdet_stage3_epoch4.pth"

    # ThinkDet — restricted to validated layer settings.
    layer_setup = DEFAULT_LAYER_SETUP
    extract_layers = ALLOWED_LAYER_SETUPS[DEFAULT_LAYER_SETUP]
    extract_layer = extract_layers[-1]
    layer_fusion = 'mean'
    d_model       = 256
    n_heads       = 8
    uncertainty_enabled = True
    uncertainty_num_variants = 2
    uncertainty_beta = 8.0
    uncertainty_min_reliability = 0.05

    # Training — fine-tuned from COCO pretrained
    epochs           = 10
    per_gpu_batch    = 4
    grad_accum       = 2       # effective batch = 4*2*8 = 64
    lr_adapter       = 5e-5    # lower than Stage 3 (adapters pre-trained)
    lr_heads         = 2e-5    # lower than Stage 3 (heads pre-trained)
    weight_decay     = 0.01
    warmup_ratio     = 0.05
    max_grad_norm    = 0.1
    conf_loss_weight = 1e-3
    num_workers      = 4

    # Checkpoints
    output_dir    = f"{ROOT}/thinkdet/checkpoints/refcoco"
    log_interval  = 50
    save_interval = 2000
    eval_epochs   = [2, 4, 6, 8, 10]


# ===================================================================
# CHECKPOINT
# ===================================================================
def save_checkpoint(model, optimizer, scheduler, epoch, step, cfg, tag=""):
    trainable_state = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_state[name] = param.data.cpu()
    path = os.path.join(cfg.output_dir, f"thinkdet_refcoco_{tag}.pth")
    torch.save({
        'epoch': epoch + 1,
        'global_step': step,
        'layer_setup': cfg.layer_setup,
        'extract_layer': cfg.extract_layer,
        'extract_layers': cfg.extract_layers,
        'layer_fusion': cfg.layer_fusion,
        'uncertainty_enabled': cfg.uncertainty_enabled,
        'uncertainty_num_variants': cfg.uncertainty_num_variants,
        'uncertainty_beta': cfg.uncertainty_beta,
        'uncertainty_min_reliability': cfg.uncertainty_min_reliability,
        'trainable_state_dict': trainable_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, path)
    print(f"  [SAVE] {path}  ({len(trainable_state)} param tensors)")


# ===================================================================
# MAIN
# ===================================================================
def main():
    cfg = Cfg()

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=cfg.epochs)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--coco_checkpoint', type=str, default=cfg.coco_checkpoint)
    parser.add_argument('--conf_loss_weight', type=float, default=cfg.conf_loss_weight)
    parser.add_argument('--lr_adapter', type=float, default=cfg.lr_adapter)
    parser.add_argument('--lr_heads', type=float, default=cfg.lr_heads)
    parser.add_argument('--per_gpu_batch', type=int, default=cfg.per_gpu_batch)
    parser.add_argument('--adapters_only', action='store_true',
                        help='Train adapters only (keep GroundingDINO heads frozen)')
    parser.add_argument(
        '--layer_setup',
        type=str,
        default=cfg.layer_setup,
        choices=sorted(ALLOWED_LAYER_SETUPS.keys()),
        help='Restricted layer setup for this sweep (no other layers allowed)',
    )
    parser.add_argument('--disable_uncertainty', action='store_true',
                        help='Disable prompt-perturbation uncertainty weighting')
    parser.add_argument('--uncertainty_variants', type=int,
                        default=cfg.uncertainty_num_variants,
                        help='Number of prompt perturbations for uncertainty estimation')
    parser.add_argument('--uncertainty_beta', type=float, default=cfg.uncertainty_beta,
                        help='Scale for converting variance into reliability')
    parser.add_argument('--uncertainty_min_reliability', type=float,
                        default=cfg.uncertainty_min_reliability,
                        help='Lower bound for layer/sample reliability')
    args = parser.parse_args()
    cfg.epochs = args.epochs
    cfg.conf_loss_weight = args.conf_loss_weight
    cfg.lr_adapter = args.lr_adapter
    cfg.lr_heads = args.lr_heads
    cfg.per_gpu_batch = args.per_gpu_batch
    cfg.coco_checkpoint = args.coco_checkpoint
    cfg.layer_setup = args.layer_setup
    cfg.extract_layers = list(ALLOWED_LAYER_SETUPS[cfg.layer_setup])
    cfg.extract_layer = cfg.extract_layers[-1]
    cfg.layer_fusion = 'mean'
    cfg.uncertainty_enabled = not args.disable_uncertainty
    cfg.uncertainty_num_variants = max(0, int(args.uncertainty_variants))
    cfg.uncertainty_beta = float(args.uncertainty_beta)
    cfg.uncertainty_min_reliability = float(args.uncertainty_min_reliability)
    cfg.output_dir = os.path.join(cfg.output_dir, cfg.layer_setup)

    # ── DDP ──
    ddp = int(os.environ.get('WORLD_SIZE', 1)) > 1
    if ddp:
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
        world_size = dist.get_world_size()
    else:
        local_rank = 0
        device = torch.device('cuda:0')
        world_size = 1
    is_main = (local_rank == 0)

    if is_main:
        os.makedirs(cfg.output_dir, exist_ok=True)
        print("=" * 70)
        print("  ThinkDet v2 — REC Training on RefCOCO/+/g")
        print("=" * 70)
        print(f"  layer_setup:     {cfg.layer_setup}")
        print(f"  extract_layers:  {cfg.extract_layers} (fusion={cfg.layer_fusion})")
        print(f"  GPUs:            {world_size}")
        print(f"  Per-GPU batch:   {cfg.per_gpu_batch}")
        print(f"  Grad accum:      {cfg.grad_accum}")
        print(f"  Effective batch: {cfg.per_gpu_batch * cfg.grad_accum * world_size}")
        print(f"  Epochs:          {cfg.epochs}")
        print(f"  LR (adapter):    {cfg.lr_adapter}")
        print(f"  LR (heads):      {cfg.lr_heads}")
        print(f"  Conf weight:     {cfg.conf_loss_weight}")
        print(f"  Adapters only:   {args.adapters_only}")
        print(f"  Uncertainty:     {cfg.uncertainty_enabled}")
        print(f"  Unc variants:    {cfg.uncertainty_num_variants}")
        print(f"  Unc beta:        {cfg.uncertainty_beta}")
        print(f"  Min reliability: {cfg.uncertainty_min_reliability}")
        print(f"  COCO init:       {cfg.coco_checkpoint}")
        print(f"  Eval at epochs:  {cfg.eval_epochs}")
        print(f"  Output:          {cfg.output_dir}")
        if ddp and len(cfg.eval_epochs) > 0:
            print("  Eval mode:       disabled during DDP train (run offline eval script)")
        print()

    # ── 1. Build model ──
    if is_main: print("[1/5] Loading GroundingDINO ...")
    gd = load_gd_model(cfg.gd_config, cfg.gd_weights, device='cpu')

    if is_main: print("[1/5] Building ThinkDetModel ...")
    model = ThinkDetModel(
        grounding_dino=gd,
        internvl_path=cfg.internvl_path,
        extract_layer=cfg.extract_layer,
        extract_layers=cfg.extract_layers,
        layer_fusion=cfg.layer_fusion,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        uncertainty_enabled=cfg.uncertainty_enabled,
        uncertainty_num_variants=cfg.uncertainty_num_variants,
        uncertainty_beta=cfg.uncertainty_beta,
        uncertainty_min_reliability=cfg.uncertainty_min_reliability,
    )

    # Adapters only by default for this REC run. Optionally unfreeze
    # detection heads when adapters_only is not requested.
    model.set_stage_a()
    if not args.adapters_only:
        model._unfreeze_detection_heads()

    # ── Load COCO checkpoint ──
    if cfg.coco_checkpoint and os.path.exists(cfg.coco_checkpoint) and not args.resume:
        if is_main: print(f"\n  Loading COCO checkpoint: {cfg.coco_checkpoint}")
        ckpt = torch.load(cfg.coco_checkpoint, map_location='cpu')
        missing, unexpected = model.load_state_dict(
            ckpt['trainable_state_dict'], strict=False,
        )
        if is_main:
            print(f"  COCO epoch: {ckpt.get('epoch', 'N/A')}, "
                  f"step: {ckpt.get('global_step', 'N/A')}")
            print(f"  Missing: {len(missing)}, Unexpected: {len(unexpected)}")

    model = model.to(device)

    if ddp:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    model_raw = model.module if ddp else model

    total_p = sum(p.numel() for p in model_raw.parameters())
    train_p = sum(p.numel() for p in model_raw.parameters() if p.requires_grad)
    if is_main:
        print(f"  Total params:     {total_p:,}")
        print(f"  Trainable params: {train_p:,} ({100*train_p/total_p:.2f}%)")

    # ── 2. Train dataset (joint) ──
    if is_main: print("\n[2/5] Loading RefCOCO/+/g train ...")
    train_ds = build_joint_refcoco_train(
        data_root=cfg.data_root,
        image_dir=cfg.image_dir,
    )
    sampler = DistributedSampler(train_ds, shuffle=True) if ddp else None
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=cfg.per_gpu_batch, shuffle=(sampler is None),
        num_workers=cfg.num_workers, collate_fn=refcoco_collate_fn,
        pin_memory=True, drop_last=True, sampler=sampler,
    )

    # ── 3. Tokenizer for positive maps ──
    gd_tok = model_raw.grounding_dino.tokenizer
    gd_spec = model_raw.grounding_dino.specical_tokens

    if is_main:
        print(f"  Joint train refs: {len(train_ds)}")

    # ── 4. Optimizer ──
    if is_main: print("\n[3/5] Setting up optimizer ...")
    adapter_params, head_params = [], []
    for name, p in model_raw.named_parameters():
        if not p.requires_grad:
            continue
        if 'adapter' in name or 'adapted_layers' in name:
            adapter_params.append(p)
        else:
            head_params.append(p)

    if is_main:
        print(f"  Adapter params: {sum(p.numel() for p in adapter_params):,}")
        print(f"  Head params:    {sum(p.numel() for p in head_params):,}")

    optim_groups = [
        {'params': adapter_params, 'lr': cfg.lr_adapter, 'name': 'adapter'},
    ]
    if len(head_params) > 0:
        optim_groups.append({'params': head_params, 'lr': cfg.lr_heads, 'name': 'heads'})

    optimizer = torch.optim.AdamW(optim_groups, weight_decay=cfg.weight_decay)

    steps_per_epoch = len(train_loader) // cfg.grad_accum
    total_steps = steps_per_epoch * cfg.epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    if is_main:
        print(f"  Steps/epoch: {steps_per_epoch}")
        print(f"  Total steps: {total_steps}")
        print(f"  Warmup:      {warmup_steps}")

    # ── Resume ──
    start_epoch = 0
    global_step = 0
    if args.resume and os.path.exists(args.resume):
        if is_main: print(f"\n  Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location='cpu')
        missing, unexpected = model_raw.load_state_dict(
            ckpt['trainable_state_dict'], strict=False)
        if 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt.get('epoch', 0)
        global_step = ckpt.get('global_step', 0)
        if is_main:
            print(f"  Resumed at epoch {start_epoch}, step {global_step}")

    if is_main:
        print(f"\n  InternVL:  {cfg.internvl_path}")
        print(f"  Setup:     {cfg.layer_setup}")
        print(f"  Layers:    {cfg.extract_layers} (fusion={cfg.layer_fusion})")
        print(f"  Uncertainty enabled: {cfg.uncertainty_enabled} "
              f"(variants={cfg.uncertainty_num_variants}, beta={cfg.uncertainty_beta})")

    # ── 5. Training loop ──
    if is_main: print(f"\n[4/5] Training ...\n")

    log_path = os.path.join(cfg.output_dir, 'train_log.jsonl')
    if is_main and not args.resume:
        with open(log_path, 'w'):
            pass

    best_acc = 0.0
    t0 = time.time()

    for epoch in range(start_epoch, cfg.epochs):
        if ddp and sampler:
            sampler.set_epoch(epoch)

        model.train()
        optimizer.zero_grad()
        epoch_losses = []

        for batch_idx, batch in enumerate(train_loader):
            internvl_imgs = batch['internvl_images'].to(device)
            dino_nested = NestedTensor(
                batch['dino_images'].tensors.to(device),
                batch['dino_images'].mask.to(device),
            )
            B = internvl_imgs.shape[0]

            # Per-sample captions for GroundingDINO
            query_texts = batch['query_texts']
            dino_inputs = {'samples': dino_nested, 'captions': query_texts}

            # Build per-sample positive maps
            _, pmaps, pmaps_norm = build_rec_positive_map_batch(
                gd_tok, gd_spec, batch['expressions'],
            )

            try:
                outputs, aux = model(
                    internvl_imgs, batch['expressions'], dino_inputs,
                )
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    if is_main: print(f"  [OOM] step {global_step}, skipping")
                    torch.cuda.empty_cache()
                    optimizer.zero_grad()
                    continue
                raise

            det_loss = compute_rec_loss(
                outputs['pred_logits'], outputs['pred_boxes'],
                batch['boxes'], pmaps, pmaps_norm, device,
            )
            conf_loss = model_raw.get_confidence_loss(aux['confidences'])
            if isinstance(conf_loss, torch.Tensor) and conf_loss.device != device:
                conf_loss = conf_loss.to(device)

            loss = det_loss + cfg.conf_loss_weight * conf_loss

            if torch.isnan(loss) or torch.isinf(loss):
                if is_main: print(f"  [NaN] step {global_step}, skipping")
                optimizer.zero_grad()
                continue

            (loss / cfg.grad_accum).backward()
            epoch_losses.append(loss.item())

            if (batch_idx + 1) % cfg.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model_raw.parameters() if p.requires_grad],
                    cfg.max_grad_norm,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if is_main and global_step % cfg.log_interval == 0:
                    avg = np.mean(epoch_losses[-cfg.log_interval*cfg.grad_accum:])
                    lr_now = scheduler.get_last_lr()[0]
                    elapsed = time.time() - t0
                    eta = elapsed / max(global_step - (start_epoch * steps_per_epoch), 1) \
                          * (total_steps - global_step)
                    sample_rel = aux.get('sample_reliability', None)
                    rel_mean = float(sample_rel.mean().item()) if sample_rel is not None else -1.0
                    print(f"  [E{epoch+1}/{cfg.epochs}] step {global_step}/{total_steps}  "
                          f"loss={avg:.4f}  det={det_loss.item():.4f}  "
                          f"conf={conf_loss.item():.4f}  "
                          f"rel={rel_mean:.4f}  "
                          f"lr={lr_now:.2e}  ETA={eta/3600:.1f}h")

                    with open(log_path, 'a') as f:
                        f.write(json.dumps({
                            'step': global_step, 'epoch': epoch+1,
                            'loss': float(avg),
                            'det_loss': float(det_loss.item()),
                            'conf_loss': float(conf_loss.item()),
                            'sample_reliability': rel_mean,
                            'lr': float(lr_now),
                        }) + '\n')

                if is_main and global_step % cfg.save_interval == 0:
                    save_checkpoint(model_raw, optimizer, scheduler,
                                    epoch, global_step, cfg, tag=f"step{global_step}")

        # ── Epoch summary ──
        epoch_avg = np.mean(epoch_losses) if epoch_losses else 0
        if is_main:
            elapsed = time.time() - t0
            print(f"\n  {'='*60}")
            print(f"  Epoch {epoch+1}/{cfg.epochs}  loss={epoch_avg:.4f}  "
                  f"time={elapsed/3600:.2f}h")
            print(f"  {'='*60}")

        if is_main:
            save_checkpoint(model_raw, optimizer, scheduler,
                            epoch, global_step, cfg, tag=f"epoch{epoch+1}")

        if ddp: dist.barrier()

        # ── Evaluate at scheduled epochs (single-GPU only to avoid DDP stalls) ──
        should_eval = ((epoch + 1) in cfg.eval_epochs) and (not ddp)
        if is_main and (epoch + 1) in cfg.eval_epochs and ddp:
            print(f"\n[5/5] Skipping in-train REC eval at epoch {epoch+1} under DDP.")
            print("      Use thinkdet/results/eval/eval_refcoco.py after checkpoint save.")

        if is_main and should_eval:
            print(f"\n[5/5] Evaluating REC (epoch {epoch+1}) ...")
            results = evaluate_all_splits(
                model_raw, gd_tok, gd_spec, device,
                cfg.data_root, cfg.image_dir,
            )
            results['epoch'] = epoch + 1
            results['global_step'] = global_step
            results['train_loss'] = float(epoch_avg)
            results['train_time_h'] = (time.time() - t0) / 3600

            eval_path = os.path.join(cfg.output_dir, f"eval_epoch{epoch+1}.json")
            with open(eval_path, 'w') as f:
                json.dump(results, f, indent=2)

            print(f"\n  {'='*60}")
            print(f"  REC Results — Epoch {epoch+1}")
            print(f"  {'='*60}")
            for key, val in results.items():
                if isinstance(val, dict) and 'accuracy' in val:
                    print(f"  {key:20s}: {val['accuracy']:.4f} "
                          f"({val['correct']}/{val['total']})")
            print(f"  {'='*60}\n")

            # Track best by refcoco_val accuracy
            val_acc = results.get('refcoco_val', {}).get('accuracy', 0)
            if val_acc > best_acc:
                best_acc = val_acc
                save_checkpoint(model_raw, optimizer, scheduler,
                                epoch, global_step, cfg, tag="best")
                print(f"  New best refcoco_val acc: {best_acc:.4f}")

            model.train()

        if ddp: dist.barrier()

    # ── Final ──
    if is_main:
        total_time = time.time() - t0
        print(f"\n{'='*70}")
        print(f"  Training complete!")
        print(f"  Total time:  {total_time/3600:.2f}h")
        print(f"  Best val acc: {best_acc:.4f}")
        print(f"  Checkpoints: {cfg.output_dir}")
        print(f"{'='*70}")

    if ddp:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
