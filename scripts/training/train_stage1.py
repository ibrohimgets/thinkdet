"""
ThinkDet v2 

    extract layers = {9, 10, [9,10], [9,10,11]} (controlled sweep only)
    Trainable:  adapters  (LLM frozen)
    Data:       COCO 2017 train (118k images)
    Eval:       COCO 2017 val  (5k images) — mAP via pycocotools

Launch:
    torchrun --nproc_per_node=8 train_stage1.py
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
from thinkdet.data.coco_grounding import (
    COCOGroundingDataset, collate_fn, build_positive_map,
)
from groundingdino.util.inference import load_model as load_gd_model
from groundingdino.util.misc import NestedTensor

ALLOWED_LAYER_SETUPS = {
    'layer9': [9],
    'layer10': [10],
    'fusion_9_10': [9, 10],
    'fusion_9_10_11': [9, 10, 11],
}
DEFAULT_LAYER_SETUP = 'layer9'


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


def compute_detection_loss(pred_logits, pred_boxes, gt_boxes_list,
                           gt_cat_names_list, cat_name_to_idx,
                           positive_map, positive_map_norm, device):
    from scipy.optimize import linear_sum_assignment
    B = pred_logits.shape[0]
    total_loss = torch.tensor(0.0, device=device, requires_grad=True)

    for b in range(B):
        logits = pred_logits[b]
        boxes  = pred_boxes[b]
        gt     = gt_boxes_list[b].to(device)
        cat_names = gt_cat_names_list[b]

        if gt.shape[0] == 0:
            bg = torch.zeros_like(logits)
            total_loss = total_loss + sigmoid_focal_loss(logits, bg, 1) * 0.1
            continue

        num_gt = gt.shape[0]
        ci_list = [cat_name_to_idx.get(n, 0) for n in cat_names]
        ci_t = torch.tensor(ci_list, device=device, dtype=torch.long)

        with torch.no_grad():
            probs = logits.clamp(-50, 50).sigmoid()
            cp = probs @ positive_map_norm.T
            cc = torch.zeros(logits.shape[0], num_gt, device=device)
            for j in range(num_gt):
                ci = ci_t[j].item()
                if ci < cp.shape[-1]:
                    cc[:, j] = -cp[:, ci]
            cl1 = torch.cdist(boxes, gt, p=1)
            cg = 1.0 - safe_generalized_box_iou(
                box_cxcywh_to_xyxy(boxes), box_cxcywh_to_xyxy(gt))
            C = (5*cl1 + 2*cg + cc).nan_to_num(100., 100., -100.)
            ri, coli = linear_sum_assignment(C.cpu().numpy())

        ri = torch.tensor(ri, dtype=torch.long, device=device)
        coli = torch.tensor(coli, dtype=torch.long, device=device)

        mp, mg = boxes[ri], gt[coli]
        l1 = F.l1_loss(mp, mg, reduction='sum') / max(num_gt, 1)
        gv = safe_generalized_box_iou(
            box_cxcywh_to_xyxy(mp), box_cxcywh_to_xyxy(mg))
        gl = (1 - gv.diag()).sum() / max(num_gt, 1)

        tgt = torch.zeros_like(logits)
        for qi, gi in zip(ri, coli):
            ci = ci_t[gi].item()
            if ci < positive_map.shape[0]:
                tgt[qi] = positive_map[ci]
        fl = sigmoid_focal_loss(logits, tgt, max(num_gt, 1))

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
# COCO EVALUATION  (mAP via pycocotools)
# ===================================================================
@torch.no_grad()
def evaluate_coco(model_raw, val_dataset, positive_map_norm,
                  cat_name_to_idx, all_cat_names, all_query_text,
                  device, conf_thresh=0.3, max_dets=100):
    """
    Run inference on COCO val, collect predictions in COCO format,
    compute mAP with pycocotools.
    """
    from pycocotools.cocoeval import COCOeval
    from pycocotools.coco import COCO
    import tempfile

    model_raw.eval()
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=4, collate_fn=collate_fn, pin_memory=True,
    )

    # Reverse map: cat_name -> COCO cat_id
    idx_to_cat_name = {v: k for k, v in cat_name_to_idx.items()}
    name_to_coco_id = {v: k for k, v in val_dataset.cat_id_to_name.items()}

    predictions = []
    num_processed = 0

    print(f"  [eval] Running on {len(val_dataset)} val images ...")
    t0 = time.time()

    for batch_idx, batch in enumerate(val_loader):
        internvl_images = batch['internvl_images'].to(device)
        dino_nested = NestedTensor(
            batch['dino_images'].tensors.to(device),
            batch['dino_images'].mask.to(device),
        )
        B = internvl_images.shape[0]
        dino_inputs = {'samples': dino_nested, 'captions': [all_query_text] * B}

        try:
            outputs, aux = model_raw(internvl_images, batch['query_texts'], dino_inputs)
        except Exception as e:
            continue

        image_id = batch['image_ids'][0]
        img_info = val_dataset.coco.loadImgs(image_id)[0]
        img_w, img_h = img_info['width'], img_info['height']

        logits = outputs['pred_logits'][0].clamp(-50, 50)
        boxes  = outputs['pred_boxes'][0]
        probs  = logits.sigmoid()

        # Map to category scores
        cat_scores = probs @ positive_map_norm.T  # [nq, num_cats]
        max_scores, max_cats = cat_scores.max(dim=-1)

        keep = max_scores > conf_thresh
        if keep.sum() == 0:
            num_processed += 1
            if (num_processed) % 500 == 0:
                print(f"  [eval] {num_processed}/{len(val_dataset)} images "
                      f"({time.time()-t0:.0f}s)")
            continue

        kept_scores = max_scores[keep]
        kept_cats   = max_cats[keep]
        kept_boxes  = boxes[keep]

        # Sort by score, limit to max_dets
        if kept_scores.shape[0] > max_dets:
            topk_idx = kept_scores.argsort(descending=True)[:max_dets]
            kept_scores = kept_scores[topk_idx]
            kept_cats   = kept_cats[topk_idx]
            kept_boxes  = kept_boxes[topk_idx]

        # Convert cxcywh normalized -> xyxy absolute
        pred_xyxy = box_cxcywh_to_xyxy(kept_boxes)
        pred_xyxy[:, 0] *= img_w
        pred_xyxy[:, 1] *= img_h
        pred_xyxy[:, 2] *= img_w
        pred_xyxy[:, 3] *= img_h

        for i in range(kept_scores.shape[0]):
            cat_idx = kept_cats[i].item()
            cat_name = idx_to_cat_name.get(cat_idx)
            if cat_name is None:
                continue
            coco_cat_id = name_to_coco_id.get(cat_name)
            if coco_cat_id is None:
                continue

            x1, y1, x2, y2 = pred_xyxy[i].tolist()
            w = max(0, x2 - x1)
            h = max(0, y2 - y1)

            predictions.append({
                'image_id': image_id,
                'category_id': coco_cat_id,
                'bbox': [round(x1, 2), round(y1, 2), round(w, 2), round(h, 2)],
                'score': round(kept_scores[i].item(), 4),
            })

        num_processed += 1
        if (num_processed) % 500 == 0:
            print(f"  [eval] {num_processed}/{len(val_dataset)} images "
                  f"({time.time()-t0:.0f}s)")

    eval_time = time.time() - t0
    print(f"  [eval] Inference done: {num_processed} images, "
          f"{len(predictions)} detections, {eval_time:.0f}s")

    if len(predictions) == 0:
        print("  [eval] WARNING: No predictions! Returning zeros.")
        return {
            'mAP': 0.0, 'mAP_50': 0.0, 'mAP_75': 0.0,
            'mAP_S': 0.0, 'mAP_M': 0.0, 'mAP_L': 0.0,
            'mAR_1': 0.0, 'mAR_10': 0.0, 'mAR_100': 0.0,
            'num_predictions': 0, 'num_images': num_processed,
            'eval_time_min': eval_time / 60,
        }

    # Write predictions to temp file for COCOeval
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    json.dump(predictions, tmp)
    tmp.close()

    coco_gt = val_dataset.coco
    coco_dt = coco_gt.loadRes(tmp.name)

    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    os.unlink(tmp.name)

    results = {
        'mAP':     float(coco_eval.stats[0]),
        'mAP_50':  float(coco_eval.stats[1]),
        'mAP_75':  float(coco_eval.stats[2]),
        'mAP_S':   float(coco_eval.stats[3]),
        'mAP_M':   float(coco_eval.stats[4]),
        'mAP_L':   float(coco_eval.stats[5]),
        'mAR_1':   float(coco_eval.stats[6]),
        'mAR_10':  float(coco_eval.stats[7]),
        'mAR_100': float(coco_eval.stats[8]),
        'num_predictions': len(predictions),
        'num_images': num_processed,
        'eval_time_min': eval_time / 60,
    }
    return results


# ===================================================================
# CONFIG
# ===================================================================
class Cfg:
    # Data
    coco_root     = f"{ROOT}/dataSets/coco"
    train_ann     = f"{coco_root}/annotations/instances_train2017.json"
    train_img_dir = f"{coco_root}/train2017"
    val_ann       = f"{coco_root}/annotations/instances_val2017.json"
    val_img_dir   = f"{coco_root}/val2017"

    # Models
    gd_config  = f"{ROOT}/GroundingDINO/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    gd_weights = f"{ROOT}/GroundingDINO/GroundingDINO/weights/groundingdino_swint_ogc.pth"
    internvl_path = f"{ROOT}/InternVL3_5-1B"

    # ThinkDet — restricted to the probing winners.
    layer_setup   = DEFAULT_LAYER_SETUP
    extract_layers = ALLOWED_LAYER_SETUPS[DEFAULT_LAYER_SETUP]
    extract_layer = extract_layers[-1]
    layer_fusion  = 'mean'
    d_model       = 256
    n_heads       = 8
    uncertainty_enabled = False
    uncertainty_num_variants = 2
    uncertainty_beta = 8.0
    uncertainty_min_reliability = 0.05
    gate_warmup_enabled = True
    gate_warmup_steps = 500
    gate_warmup_value = 1.0
    gate_unfreeze_value = 0.7

    # Training — adapters + heads only (LLM frozen)
    epochs           = 12
    per_gpu_batch    = 1       # InternVL + DINO need more memory
    grad_accum       = 8       # effective batch = 1*8*8 = 64
    lr_adapter       = 2e-4
    lr_heads         = 1e-4
    weight_decay     = 0.01
    warmup_ratio     = 0.03
    max_grad_norm    = 1.0
    conf_loss_weight = 0.0     # Disabled: entropy loss fights detection loss
    num_workers      = 4

    # Checkpoints
    output_dir    = f"{ROOT}/thinkdet/checkpoints/stage1"
    log_interval  = 50
    save_interval = 2000
    eval_epochs   = [4, 8, 12]


# ===================================================================
# CHECKPOINT
# ===================================================================
def save_checkpoint(model, optimizer, scheduler, epoch, step, cfg, tag=""):
    trainable_state = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_state[name] = param.data.cpu()
    path = os.path.join(cfg.output_dir, f"thinkdet_stage1_{tag}.pth")
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
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--conf_loss_weight', type=float, default=cfg.conf_loss_weight,
                        help='Weight for confidence entropy regularization')
    parser.add_argument('--debug_limit', type=int, default=0,
                        help='Limit dataset size for debugging (0 = full)')
    parser.add_argument('--baseline', action='store_true',
                        help='Run baseline (no adapters) for comparison')
    parser.add_argument(
        '--layer_setup',
        type=str,
        default=cfg.layer_setup,
        choices=sorted(ALLOWED_LAYER_SETUPS.keys()),
        help='Restricted layer setup for this sweep (no other layers allowed)',
    )
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for checkpoints and logs')
    parser.add_argument('--disable_uncertainty', action='store_true',
                        help='Disable prompt-perturbation uncertainty weighting')
    parser.add_argument('--enable_uncertainty', action='store_true',
                        help='Enable prompt-perturbation uncertainty weighting')
    parser.add_argument('--uncertainty_variants', type=int,
                        default=cfg.uncertainty_num_variants,
                        help='Number of prompt perturbations for uncertainty estimation')
    parser.add_argument('--uncertainty_beta', type=float, default=cfg.uncertainty_beta,
                        help='Scale for converting variance into reliability')
    parser.add_argument('--uncertainty_min_reliability', type=float,
                        default=cfg.uncertainty_min_reliability,
                        help='Lower bound for layer/sample reliability')
    parser.add_argument('--gate_warmup_steps', type=int, default=cfg.gate_warmup_steps,
                        help='Number of optimizer steps to keep adapter gates forced open')
    parser.add_argument('--gate_warmup_value', type=float, default=cfg.gate_warmup_value,
                        help='Target gate value during warmup (post-tanh target)')
    parser.add_argument('--gate_unfreeze_value', type=float, default=cfg.gate_unfreeze_value,
                        help='Gate value to reset to when warmup ends (keeps tanh unsaturated)')
    parser.add_argument('--disable_gate_warmup', action='store_true',
                        help='Disable forcing adapter gates open at startup')
    args = parser.parse_args()
    cfg.epochs = args.epochs
    cfg.conf_loss_weight = args.conf_loss_weight
    cfg.debug_limit = args.debug_limit
    cfg.baseline = args.baseline
    cfg.layer_setup = args.layer_setup
    cfg.extract_layers = list(ALLOWED_LAYER_SETUPS[cfg.layer_setup])
    cfg.extract_layer = cfg.extract_layers[-1]
    cfg.layer_fusion = 'mean'
    if args.enable_uncertainty and args.disable_uncertainty:
        raise ValueError("Choose at most one of --enable_uncertainty / --disable_uncertainty")
    cfg.uncertainty_enabled = cfg.uncertainty_enabled
    if args.enable_uncertainty:
        cfg.uncertainty_enabled = True
    if args.disable_uncertainty:
        cfg.uncertainty_enabled = False
    cfg.uncertainty_num_variants = max(0, int(args.uncertainty_variants))
    cfg.uncertainty_beta = float(args.uncertainty_beta)
    cfg.uncertainty_min_reliability = float(args.uncertainty_min_reliability)
    cfg.gate_warmup_steps = max(0, int(args.gate_warmup_steps))
    cfg.gate_warmup_value = float(args.gate_warmup_value)
    cfg.gate_unfreeze_value = float(args.gate_unfreeze_value)
    cfg.gate_warmup_enabled = not args.disable_gate_warmup
    if args.output_dir:
        cfg.output_dir = args.output_dir
    else:
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
        print("  ThinkDet v2 — Stage 1: Adapter Learning")
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
        print(f"  Grad clip:       {cfg.max_grad_norm}")
        print(f"  Mean-center:     enabled (batch)")
        if cfg.baseline:
            print(f"  Gate warmup:     skipped (baseline mode)")
        elif cfg.gate_warmup_enabled and cfg.gate_warmup_steps > 0:
            print(
                f"  Gate warmup:     enabled ({cfg.gate_warmup_steps} steps, "
                f"value={cfg.gate_warmup_value} -> unfreeze={cfg.gate_unfreeze_value})"
            )
        else:
            print(f"  Gate warmup:     disabled")
        print(f"  LLM:             frozen")
        print(f"  Uncertainty:     {cfg.uncertainty_enabled}")
        print(f"  Unc variants:    {cfg.uncertainty_num_variants}")
        print(f"  Unc beta:        {cfg.uncertainty_beta}")
        print(f"  Min reliability: {cfg.uncertainty_min_reliability}")
        print(f"  Eval at epochs:  {cfg.eval_epochs}")
        print(f"  Output:          {cfg.output_dir}")
        if ddp and len(cfg.eval_epochs) > 0:
            print("  Eval mode:       disabled during DDP train (run offline eval script)")
        print()

    # ── 1. Build model (live InternVL + GroundingDINO) ──
    if is_main: print("[1/5] Loading GroundingDINO ...")
    gd = load_gd_model(cfg.gd_config, cfg.gd_weights, device='cpu')

    if is_main: print(f"[1/5] Building ThinkDetModel (live InternVL) ...")
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
        injection_layers=[] if cfg.baseline else None,
    )

    # Protocol: Start with Stage A (adapters only, heads frozen)
    model.set_stage_a()
    
    # Baseline: unfreeze heads immediately (no adapters to warm up)
    if cfg.baseline:
        if is_main: print("  [Baseline] Unfreezing detection heads immediately.")
        model._unfreeze_detection_heads()

    model = model.to(device)

    if ddp:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    model_raw = model.module if ddp else model

    total_p = sum(p.numel() for p in model_raw.parameters())
    train_p = sum(p.numel() for p in model_raw.parameters() if p.requires_grad)
    if is_main:
        print(f"  Total params:     {total_p:,}")
        print(f"  Trainable params: {train_p:,} ({100*train_p/total_p:.2f}%)")

    # ── 2. Train dataset ──
    if is_main: print(f"\n[2/5] Loading COCO train2017 ...")
    train_ds = COCOGroundingDataset(
        img_dir=cfg.train_img_dir,
        ann_file=cfg.train_ann,
    )
    if cfg.debug_limit > 0:
        if is_main: print(f"  [DEBUG] Limiting dataset to {cfg.debug_limit} images")
        train_ds.img_ids = train_ds.img_ids[:cfg.debug_limit]
        cfg.log_interval = 10
    sampler = DistributedSampler(train_ds, shuffle=True) if ddp else None
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=cfg.per_gpu_batch, shuffle=(sampler is None),
        num_workers=cfg.num_workers, collate_fn=collate_fn,
        pin_memory=True, drop_last=True, sampler=sampler,
    )

    # ── 3. Positive map ──
    all_cat_names = sorted(train_ds.cat_id_to_name.values())
    gd_tok = model_raw.grounding_dino.tokenizer
    gd_spec = model_raw.grounding_dino.specical_tokens
    all_query_text, pmap, pmap_norm, cat_name_to_idx = build_positive_map(
        gd_tok, gd_spec, all_cat_names,
    )
    pmap = pmap.to(device)
    pmap_norm = pmap_norm.to(device)

    if is_main:
        print(f"  Categories: {len(all_cat_names)}")
        print(f"  Query: '{all_query_text[:80]}...'")

    # ── 4. Optimizer — 3 param groups ──
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

    optimizer = torch.optim.AdamW([
        {'params': adapter_params, 'lr': cfg.lr_adapter, 'name': 'adapter'},
        {'params': head_params,    'lr': cfg.lr_heads,    'name': 'heads'},
    ], weight_decay=cfg.weight_decay)

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

    gate_warmup_active = (
        (not cfg.baseline)
        and cfg.gate_warmup_enabled
        and cfg.gate_warmup_steps > 0
        and global_step < cfg.gate_warmup_steps
    )
    if gate_warmup_active:
        model_raw.freeze_gate(value=cfg.gate_warmup_value)
        if is_main:
            print(
                f"  [GateWarmup] Gates forced open until step {cfg.gate_warmup_steps} "
                f"(start step={global_step})"
            )

    # ── 5. Training loop ──
    if is_main: print(f"\n[4/5] Training ...\n")

    log_path = os.path.join(cfg.output_dir, 'train_log.jsonl')
    if is_main and not args.resume:
        # Fresh runs should not append to stale logs from older experiments.
        with open(log_path, 'w'):
            pass
        print(f"  [log] Reset {log_path}")
    best_map = 0.0
    t0 = time.time()

    for epoch in range(start_epoch, cfg.epochs):
        if ddp and sampler:
            sampler.set_epoch(epoch)

        model.train()
        optimizer.zero_grad()
        epoch_losses = []

        for batch_idx, batch in enumerate(train_loader):
            internvl_images = batch['internvl_images'].to(device)
            dino_nested = NestedTensor(
                batch['dino_images'].tensors.to(device),
                batch['dino_images'].mask.to(device),
            )
            B = internvl_images.shape[0]
            dino_inputs = {'samples': dino_nested, 'captions': [all_query_text]*B}

            try:
                outputs, aux = model(internvl_images, batch['query_texts'], dino_inputs)
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    if is_main: print(f"  [OOM] step {global_step}, skipping")
                    torch.cuda.empty_cache()
                    optimizer.zero_grad()
                    continue
                raise

            det_loss = compute_detection_loss(
                outputs['pred_logits'], outputs['pred_boxes'],
                batch['boxes'], batch['category_names'],
                cat_name_to_idx, pmap, pmap_norm, device,
            )
            if cfg.conf_loss_weight > 0:
                conf_loss = model_raw.get_confidence_loss(aux['confidences'])
                if isinstance(conf_loss, torch.Tensor) and conf_loss.device != device:
                    conf_loss = conf_loss.to(device)
                loss = det_loss + cfg.conf_loss_weight * conf_loss
            else:
                conf_loss = torch.tensor(0.0, device=device)
                loss = det_loss

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

                if gate_warmup_active and global_step >= cfg.gate_warmup_steps:
                    model_raw.unfreeze_gate(reset_value=cfg.gate_unfreeze_value)
                    gate_warmup_active = False
                    if is_main:
                        print(
                            f"  [GateWarmup] Completed at step {global_step}; "
                            f"gates reset to {cfg.gate_unfreeze_value} and unfrozen."
                        )

                if is_main and global_step % cfg.log_interval == 0:
                    # Log gate values
                    gates = [round(layer.adapter.gate.item(), 4) for layer in model_raw.adapted_layers]
                    gate_tanh = [round(torch.tanh(layer.adapter.gate).item(), 4) for layer in model_raw.adapted_layers]
                    sample_rel = aux.get('sample_reliability', None)
                    rel_mean = float(sample_rel.mean().item()) if sample_rel is not None else -1.0
                    print(f"  [Gate] raw={gates}  tanh={gate_tanh}")

                    avg = np.mean(epoch_losses[-cfg.log_interval*cfg.grad_accum:])
                    lr_now = scheduler.get_last_lr()[0]
                    elapsed = time.time() - t0
                    eta = elapsed / max(global_step - (start_epoch * steps_per_epoch), 1) \
                          * (total_steps - global_step)
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

        # Save epoch checkpoint
        if is_main:
            save_checkpoint(model_raw, optimizer, scheduler,
                            epoch, global_step, cfg, tag=f"epoch{epoch+1}")

        if ddp: dist.barrier()

        # ── Evaluate at scheduled epochs (single-GPU only to avoid DDP stalls) ──
        should_eval = ((epoch + 1) in cfg.eval_epochs) and (not ddp)
        if is_main and (epoch + 1) in cfg.eval_epochs and ddp:
            print(f"\n[5/5] Skipping in-train COCO eval at epoch {epoch+1} under DDP.")
            print("      Use thinkdet/results/eval/eval_checkpoint.py after checkpoint save.")

        if is_main and should_eval:
            print(f"\n[5/5] Evaluating on COCO val2017 (epoch {epoch+1}) ...")
            val_ds = COCOGroundingDataset(
                img_dir=cfg.val_img_dir,
                ann_file=cfg.val_ann,
            )
            results = evaluate_coco(
                model_raw, val_ds, pmap_norm,
                cat_name_to_idx, all_cat_names, all_query_text,
                device,
            )
            results['epoch'] = epoch + 1
            results['global_step'] = global_step
            results['train_loss'] = float(epoch_avg)
            results['train_time_h'] = (time.time() - t0) / 3600

            eval_path = os.path.join(cfg.output_dir, f"eval_epoch{epoch+1}.json")
            with open(eval_path, 'w') as f:
                json.dump(results, f, indent=2)

            print(f"\n  {'='*60}")
            print(f"  COCO Val Results — Epoch {epoch+1}")
            print(f"  {'='*60}")
            print(f"  mAP:       {results['mAP']:.4f}")
            print(f"  mAP@50:    {results['mAP_50']:.4f}")
            print(f"  mAP@75:    {results['mAP_75']:.4f}")
            print(f"  mAP (S):   {results['mAP_S']:.4f}")
            print(f"  mAP (M):   {results['mAP_M']:.4f}")
            print(f"  mAP (L):   {results['mAP_L']:.4f}")
            print(f"  mAR@1:     {results['mAR_1']:.4f}")
            print(f"  mAR@10:    {results['mAR_10']:.4f}")
            print(f"  mAR@100:   {results['mAR_100']:.4f}")
            print(f"  {'='*60}\n")

            if results['mAP'] > best_map:
                best_map = results['mAP']
                save_checkpoint(model_raw, optimizer, scheduler,
                                epoch, global_step, cfg, tag="best")
                print(f"  New best mAP: {best_map:.4f}")

            model.train()

        if ddp: dist.barrier()

    # ── Final ──
    if is_main:
        total_time = time.time() - t0
        print(f"\n{'='*70}")
        print(f"  Training complete!")
        print(f"  Total time:  {total_time/3600:.2f}h")
        print(f"  Best mAP:    {best_map:.4f}")
        print(f"  Checkpoints: {cfg.output_dir}")
        print(f"{'='*70}")

    if ddp:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
