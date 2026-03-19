"""
ThinkDet - Full RefCOCO+ Training with Final Acc@0.5 Eval

This script is aligned with the current ThinkDetModel API and focuses on
RefCOCO+ only:
    - training split: refcoco+ / unc / train
    - eval splits:    val, testA, testB
    - metric:         Acc@0.5

It supports:
    - arbitrary extract_layers lists
    - layer_fusion in {"mean", "last", "learned"}
    - residual text-memory injection
    - KD disabled by default
"""

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

ROOT = "/home/iibrohimm/project/next_step"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "GroundingDINO", "GroundingDINO"))

from groundingdino.util.inference import load_model as load_gd_model
from groundingdino.util.misc import NestedTensor

from thinkdet.data.refcoco_grounding import (
    RefCOCOGroundingDataset,
    build_rec_positive_map_batch,
    build_refcoco_eval,
    refcoco_collate_fn,
)
from thinkdet.models.arch import ThinkDetModel


def box_cxcywh_to_xyxy(x):
    cx, cy, w, h = x.unbind(-1)
    return torch.stack([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], -1)


def safe_generalized_box_iou(b1, b2):
    b1 = torch.stack(
        [
            b1[..., 0],
            b1[..., 1],
            torch.max(b1[..., 2], b1[..., 0] + 1e-4),
            torch.max(b1[..., 3], b1[..., 1] + 1e-4),
        ],
        -1,
    )
    b2 = torch.stack(
        [
            b2[..., 0],
            b2[..., 1],
            torch.max(b2[..., 2], b2[..., 0] + 1e-4),
            torch.max(b2[..., 3], b2[..., 1] + 1e-4),
        ],
        -1,
    )
    a1 = (b1[:, 2] - b1[:, 0]) * (b1[:, 3] - b1[:, 1])
    a2 = (b2[:, 2] - b2[:, 0]) * (b2[:, 3] - b2[:, 1])
    lt = torch.max(b1[:, None, :2], b2[None, :, :2])
    rb = torch.min(b1[:, None, 2:], b2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = a1[:, None] + a2[None, :] - inter
    iou = inter / (union + 1e-6)
    lt2 = torch.min(b1[:, None, :2], b2[None, :, :2])
    rb2 = torch.max(b1[:, None, 2:], b2[None, :, 2:])
    wh2 = (rb2 - lt2).clamp(min=0)
    ae = wh2[:, :, 0] * wh2[:, :, 1]
    return iou - (ae - union) / (ae + 1e-6)


def compute_iou(box1, box2):
    lt = torch.max(box1[:, :2], box2[:, :2])
    rb = torch.min(box1[:, 2:], box2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]
    a1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    a2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    return inter / (a1 + a2 - inter + 1e-6)


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha=0.25, gamma=2.0):
    inputs = inputs.clamp(min=-50, max=50)
    prob = inputs.sigmoid()
    ce = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss.mean(1).sum() / max(num_boxes, 1)


def compute_rec_loss(pred_logits, pred_boxes, gt_boxes_list, positive_maps, positive_maps_norm, device):
    batch_size = pred_logits.shape[0]
    text_len = pred_logits.shape[2]
    total_loss = torch.tensor(0.0, device=device, requires_grad=True)

    pmaps = positive_maps[:, :, :text_len].to(device)
    pmaps_norm = positive_maps_norm[:, :, :text_len].to(device)

    for batch_idx in range(batch_size):
        logits = pred_logits[batch_idx]
        boxes = pred_boxes[batch_idx]
        gt = gt_boxes_list[batch_idx].to(device)
        pmap = pmaps[batch_idx]
        pmap_norm = pmaps_norm[batch_idx]

        with torch.no_grad():
            probs = logits.clamp(-50, 50).sigmoid()
            cls_cost = -(probs * pmap_norm).sum(dim=-1, keepdim=True)
            l1_cost = torch.cdist(boxes, gt, p=1)
            giou_cost = 1.0 - safe_generalized_box_iou(
                box_cxcywh_to_xyxy(boxes), box_cxcywh_to_xyxy(gt)
            )
            cost = (5 * l1_cost + 2 * giou_cost + cls_cost).nan_to_num(100.0, 100.0, -100.0)
            row_ind, col_ind = linear_sum_assignment(cost.cpu().numpy())

        row_ind = torch.tensor(row_ind, dtype=torch.long, device=device)
        col_ind = torch.tensor(col_ind, dtype=torch.long, device=device)

        matched_pred = boxes[row_ind]
        matched_gt = gt[col_ind]
        l1 = F.l1_loss(matched_pred, matched_gt, reduction="sum")
        giou = safe_generalized_box_iou(
            box_cxcywh_to_xyxy(matched_pred), box_cxcywh_to_xyxy(matched_gt)
        )
        giou_loss = (1 - giou.diag()).sum()

        cls_target = torch.zeros_like(logits)
        cls_target[row_ind[0]] = pmap[0]
        focal = sigmoid_focal_loss(logits, cls_target, 1)

        total_loss = total_loss + 5 * l1 + 2 * giou_loss + focal

    return total_loss / max(batch_size, 1)


def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@dataclass
class Cfg:
    data_root: str = f"{ROOT}/dataSets/refer/data"
    image_dir: str = f"{ROOT}/dataSets/coco/train2017"
    gd_config: str = f"{ROOT}/GroundingDINO/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    gd_weights: str = f"{ROOT}/GroundingDINO/GroundingDINO/weights/groundingdino_swint_ogc.pth"
    internvl_path: str = f"{ROOT}/InternVL3_5-1B"
    init_checkpoint: str = f"{ROOT}/thinkdet/checkpoints/stage3/thinkdet_stage3_epoch4.pth"
    output_root: str = f"{ROOT}/thinkdet/checkpoints/refcocoplus_full"
    dataset_name: str = "refcoco+"
    split_by: str = "unc"
    train_split: str = "train"
    eval_splits: Optional[List[str]] = None
    extract_layers: Optional[List[int]] = None
    layer_fusion: str = "mean"
    injection_layers: Optional[List[int]] = None
    fusion_mode: str = "residual"
    preserve_kd_enabled: bool = False
    d_model: int = 256
    tma_m: int = 8
    tma_n_heads: int = 8
    tma_alpha_init: float = 0.0
    epochs: int = 10
    per_gpu_batch: int = 1
    grad_accum: int = 8
    lr_adapter: float = 5e-5
    lr_heads: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    max_grad_norm: float = 0.1
    num_workers: int = 2
    log_interval: int = 20
    output_dir: str = ""
    run_name: str = ""
    debug_limit: int = 0

    def __post_init__(self):
        if self.eval_splits is None:
            self.eval_splits = ["val", "testA", "testB"]
        if self.extract_layers is None:
            self.extract_layers = [8]
        if self.injection_layers is None:
            self.injection_layers = [1, 3, 5]


def get_ddp_context():
    use_ddp = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if use_ddp:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        world_size = dist.get_world_size()
        rank = dist.get_rank()
    else:
        local_rank = 0
        rank = 0
        world_size = 1
        device = torch.device("cuda:0")
    return use_ddp, local_rank, rank, world_size, device


def is_main_process(rank):
    return rank == 0


def unique_trainable_param_groups(model_raw):
    adapter_params = []
    adapter_ids = set()
    for param in model_raw.get_augmenter_params():
        if param.requires_grad and id(param) not in adapter_ids:
            adapter_params.append(param)
            adapter_ids.add(id(param))

    head_params = []
    for param in model_raw.parameters():
        if param.requires_grad and id(param) not in adapter_ids:
            head_params.append(param)

    return adapter_params, head_params


def get_layer_fusion_weights(model_raw):
    logits = getattr(model_raw.feature_extractor, "layer_fusion_logits", None)
    if logits is None:
        return None
    weights = torch.softmax(logits.detach().cpu(), dim=0).tolist()
    return {
        str(layer): float(weight)
        for layer, weight in zip(model_raw.feature_extractor.extract_layers, weights)
    }


def save_checkpoint(model_raw, optimizer, scheduler, epoch, global_step, cfg, tag):
    trainable_state = {}
    for name, param in model_raw.named_parameters():
        if param.requires_grad:
            trainable_state[name] = param.detach().cpu()

    payload = {
        "epoch": epoch + 1,
        "global_step": global_step,
        "dataset_name": cfg.dataset_name,
        "split_by": cfg.split_by,
        "extract_layers": cfg.extract_layers,
        "extract_layer": cfg.extract_layers[-1],
        "layer_fusion": cfg.layer_fusion,
        "injection_layers": cfg.injection_layers,
        "fusion_mode": cfg.fusion_mode,
        "preserve_kd_enabled": cfg.preserve_kd_enabled,
        "layer_fusion_weights": get_layer_fusion_weights(model_raw),
        "trainable_state_dict": trainable_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }
    path = os.path.join(cfg.output_dir, f"thinkdet_refcocoplus_{tag}.pth")
    torch.save(payload, path)
    return path


@torch.no_grad()
def evaluate_split(model_raw, dataset, tokenizer, special_tokens, device, rank, world_size):
    model_raw.eval()
    local_indices = list(range(rank, len(dataset), world_size))
    correct = torch.zeros(1, device=device, dtype=torch.long)
    total = torch.zeros(1, device=device, dtype=torch.long)
    start = time.time()

    for local_pos, ds_index in enumerate(local_indices):
        sample = dataset[ds_index]
        internvl_images = sample["internvl_image"].unsqueeze(0).to(device)
        dino_nested = NestedTensor(
            sample["dino_image"].unsqueeze(0).to(device),
            torch.zeros(
                1,
                sample["dino_image"].shape[1],
                sample["dino_image"].shape[2],
                dtype=torch.bool,
                device=device,
            ),
        )
        expression = sample["expression"]
        query_text = sample["query_text"]
        gt_box = sample["box"].to(device)
        dino_inputs = {"samples": dino_nested, "captions": [query_text]}

        outputs, _ = model_raw(internvl_images, [expression], dino_inputs)
        _, _, positive_map_norm = build_rec_positive_map_batch(
            tokenizer, special_tokens, [expression]
        )
        logits = outputs["pred_logits"][0]
        boxes = outputs["pred_boxes"][0]
        text_len = logits.shape[-1]
        pmap_norm = positive_map_norm[0, :, :text_len].to(device)
        probs = logits.clamp(-50, 50).sigmoid()
        scores = (probs * pmap_norm).sum(dim=-1)
        best_idx = scores.argmax()
        pred_box = boxes[best_idx].unsqueeze(0)
        iou = compute_iou(box_cxcywh_to_xyxy(pred_box), box_cxcywh_to_xyxy(gt_box))
        if iou.item() > 0.5:
            correct += 1
        total += 1

        if rank == 0 and (local_pos + 1) % 500 == 0:
            elapsed = time.time() - start
            print(f"    eval progress: {local_pos + 1}/{len(local_indices)} ({elapsed:.0f}s)")

    if world_size > 1:
        dist.all_reduce(correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(total, op=dist.ReduceOp.SUM)

    return {
        "accuracy": float(correct.item() / max(total.item(), 1)),
        "correct": int(correct.item()),
        "total": int(total.item()),
    }


def parse_args():
    cfg = Cfg()
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=cfg.epochs)
    parser.add_argument("--per_gpu_batch", type=int, default=cfg.per_gpu_batch)
    parser.add_argument("--grad_accum", type=int, default=cfg.grad_accum)
    parser.add_argument("--lr_adapter", type=float, default=cfg.lr_adapter)
    parser.add_argument("--lr_heads", type=float, default=cfg.lr_heads)
    parser.add_argument("--weight_decay", type=float, default=cfg.weight_decay)
    parser.add_argument("--warmup_ratio", type=float, default=cfg.warmup_ratio)
    parser.add_argument("--max_grad_norm", type=float, default=cfg.max_grad_norm)
    parser.add_argument("--num_workers", type=int, default=cfg.num_workers)
    parser.add_argument("--log_interval", type=int, default=cfg.log_interval)
    parser.add_argument("--debug_limit", type=int, default=cfg.debug_limit)
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--init_checkpoint", type=str, default=cfg.init_checkpoint)
    parser.add_argument("--internvl_path", type=str, default=cfg.internvl_path)
    parser.add_argument("--layer_fusion", type=str, default=cfg.layer_fusion, choices=["mean", "last", "learned"])
    parser.add_argument("--fusion_mode", type=str, default=cfg.fusion_mode, choices=["concat", "residual"])
    parser.add_argument("--tma_alpha_init", type=float, default=cfg.tma_alpha_init)
    parser.add_argument("--extract_layers", type=int, nargs="+", default=cfg.extract_layers)
    parser.add_argument("--injection_layers", type=int, nargs="+", default=cfg.injection_layers)
    args = parser.parse_args()

    cfg.epochs = args.epochs
    cfg.per_gpu_batch = args.per_gpu_batch
    cfg.grad_accum = args.grad_accum
    cfg.lr_adapter = args.lr_adapter
    cfg.lr_heads = args.lr_heads
    cfg.weight_decay = args.weight_decay
    cfg.warmup_ratio = args.warmup_ratio
    cfg.max_grad_norm = args.max_grad_norm
    cfg.num_workers = args.num_workers
    cfg.log_interval = args.log_interval
    cfg.debug_limit = args.debug_limit
    cfg.init_checkpoint = args.init_checkpoint
    cfg.internvl_path = args.internvl_path
    cfg.layer_fusion = args.layer_fusion
    cfg.fusion_mode = args.fusion_mode
    cfg.tma_alpha_init = args.tma_alpha_init
    cfg.extract_layers = [int(x) for x in args.extract_layers]
    cfg.injection_layers = [int(x) for x in args.injection_layers]
    cfg.run_name = args.run_name

    layer_tag = "layers_" + "_".join(str(x) for x in cfg.extract_layers)
    if cfg.run_name:
        layer_tag = f"{cfg.run_name}_{layer_tag}"
    if args.output_dir:
        cfg.output_dir = args.output_dir
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        cfg.output_dir = os.path.join(
            cfg.output_root,
            f"{layer_tag}_{cfg.layer_fusion}_{cfg.fusion_mode}_e{cfg.epochs}_{timestamp}",
        )
    return cfg


def main():
    cfg = parse_args()
    use_ddp, local_rank, rank, world_size, device = get_ddp_context()
    main_process = is_main_process(rank)

    if main_process:
        os.makedirs(cfg.output_dir, exist_ok=True)
        print("=" * 72)
        print("  ThinkDet - Full RefCOCO+ Training")
        print("=" * 72)
        print(f"  dataset:          {cfg.dataset_name}/{cfg.split_by}")
        print(f"  train split:      {cfg.train_split}")
        print(f"  eval splits:      {cfg.eval_splits}")
        print(f"  extract_layers:   {cfg.extract_layers}")
        print(f"  layer_fusion:     {cfg.layer_fusion}")
        print(f"  injection_layers: {cfg.injection_layers}")
        print(f"  fusion_mode:      {cfg.fusion_mode}")
        print(f"  preserve_kd:      {cfg.preserve_kd_enabled}")
        print(f"  epochs:           {cfg.epochs}")
        print(f"  per_gpu_batch:    {cfg.per_gpu_batch}")
        print(f"  grad_accum:       {cfg.grad_accum}")
        print(f"  effective batch:  {cfg.per_gpu_batch * cfg.grad_accum * world_size}")
        print(f"  GPUs:             {world_size}")
        print(f"  debug_limit:      {cfg.debug_limit}")
        print(f"  init checkpoint:  {cfg.init_checkpoint}")
        print(f"  output:           {cfg.output_dir}")
        print()

    if main_process:
        print("[1/5] Loading GroundingDINO ...")
    grounding_dino = load_gd_model(cfg.gd_config, cfg.gd_weights, device="cpu")

    if main_process:
        print("[1/5] Building ThinkDetModel ...")
    model = ThinkDetModel(
        grounding_dino=grounding_dino,
        internvl_path=cfg.internvl_path,
        extract_layer=cfg.extract_layers[-1],
        extract_layers=cfg.extract_layers,
        layer_fusion=cfg.layer_fusion,
        injection_layers=cfg.injection_layers,
        d_model=cfg.d_model,
        tma_m=cfg.tma_m,
        tma_n_heads=cfg.tma_n_heads,
        tma_alpha_init=cfg.tma_alpha_init,
        fusion_mode=cfg.fusion_mode,
        preserve_kd_enabled=cfg.preserve_kd_enabled,
    )
    model.set_tma_and_heads()

    if cfg.init_checkpoint and os.path.exists(cfg.init_checkpoint):
        if main_process:
            print(f"  Loading init checkpoint: {cfg.init_checkpoint}")
        checkpoint = torch.load(cfg.init_checkpoint, map_location="cpu")
        missing, unexpected = model.load_state_dict(
            checkpoint["trainable_state_dict"], strict=False
        )
        if main_process:
            print(
                f"  Init load complete: missing={len(missing)} unexpected={len(unexpected)} "
                f"epoch={checkpoint.get('epoch', 'N/A')} step={checkpoint.get('global_step', 'N/A')}"
            )

    model = model.to(device)
    if use_ddp:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
    model_raw = model.module if use_ddp else model

    total_params = sum(p.numel() for p in model_raw.parameters())
    trainable_params = sum(p.numel() for p in model_raw.parameters() if p.requires_grad)
    if main_process:
        print(f"  Total params:     {total_params:,}")
        print(f"  Trainable params: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")

    if main_process:
        print("\n[2/5] Loading RefCOCO+ train ...")
    train_ds = RefCOCOGroundingDataset(
        data_root=cfg.data_root,
        dataset_name=cfg.dataset_name,
        split_by=cfg.split_by,
        split=cfg.train_split,
        image_dir=cfg.image_dir,
        train_mode=True,
        use_hflip=False,
    )
    if cfg.debug_limit > 0:
        train_ds.samples = train_ds.samples[: cfg.debug_limit]
        if main_process:
            print(f"  [DEBUG] limited train samples to {len(train_ds.samples)}")

    sampler = DistributedSampler(train_ds, shuffle=True) if use_ddp else None
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.per_gpu_batch,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=refcoco_collate_fn,
    )

    gd_tok = model_raw.grounding_dino.tokenizer
    gd_spec = model_raw.grounding_dino.specical_tokens

    if main_process:
        print(f"  Train refs: {len(train_ds)}")
        print("\n[3/5] Setting up optimizer ...")

    adapter_params, head_params = unique_trainable_param_groups(model_raw)
    if main_process:
        print(f"  Adapter params: {sum(p.numel() for p in adapter_params):,}")
        print(f"  Head params:    {sum(p.numel() for p in head_params):,}")

    optimizer = torch.optim.AdamW(
        [
            {"params": adapter_params, "lr": cfg.lr_adapter, "name": "adapter"},
            {"params": head_params, "lr": cfg.lr_heads, "name": "heads"},
        ],
        weight_decay=cfg.weight_decay,
    )
    steps_per_epoch = max(1, len(train_loader) // cfg.grad_accum)
    total_steps = steps_per_epoch * cfg.epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    if main_process:
        print(f"  Steps/epoch: {steps_per_epoch}")
        print(f"  Total steps: {total_steps}")
        print(f"  Warmup:      {warmup_steps}")
        print("\n[4/5] Training ...\n")

    log_path = os.path.join(cfg.output_dir, "train_log.jsonl")
    eval_path = os.path.join(cfg.output_dir, "eval_results.json")
    if main_process:
        with open(log_path, "w", encoding="utf-8"):
            pass

    global_step = 0
    train_start = time.time()
    epoch_losses: List[float] = []
    last_epoch_loss = 0.0

    for epoch in range(cfg.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)

        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_losses = []

        for batch_idx, batch in enumerate(train_loader):
            internvl_images = batch["internvl_images"].to(device, non_blocking=True)
            dino_nested = NestedTensor(
                batch["dino_images"].tensors.to(device, non_blocking=True),
                batch["dino_images"].mask.to(device, non_blocking=True),
            )
            dino_inputs = {"samples": dino_nested, "captions": batch["query_texts"]}
            _, positive_maps, positive_maps_norm = build_rec_positive_map_batch(
                gd_tok, gd_spec, batch["expressions"]
            )

            outputs, aux = model(internvl_images, batch["expressions"], dino_inputs)
            det_loss = compute_rec_loss(
                outputs["pred_logits"],
                outputs["pred_boxes"],
                batch["boxes"],
                positive_maps,
                positive_maps_norm,
                device,
            )

            if torch.isnan(det_loss) or torch.isinf(det_loss):
                if main_process:
                    print(f"  [NaN] batch {batch_idx}, skipping")
                optimizer.zero_grad(set_to_none=True)
                continue

            (det_loss / cfg.grad_accum).backward()
            epoch_losses.append(float(det_loss.item()))

            if (batch_idx + 1) % cfg.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model_raw.parameters() if p.requires_grad],
                    cfg.max_grad_norm,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if main_process and global_step % cfg.log_interval == 0:
                    recent = epoch_losses[-cfg.log_interval * cfg.grad_accum :]
                    avg_loss = float(np.mean(recent)) if recent else float(det_loss.item())
                    lr_now = float(scheduler.get_last_lr()[0])
                    elapsed = time.time() - train_start
                    eta = elapsed / max(global_step, 1) * (total_steps - global_step)
                    line = {
                        "step": global_step,
                        "epoch": epoch + 1,
                        "loss": avg_loss,
                        "det_loss": float(det_loss.item()),
                        "lr": lr_now,
                        "gate_mean": float(aux.get("gate_mean", 0.0)),
                        "h_vlm_norm_mean": float(aux.get("h_vlm_norm_mean", 0.0)),
                        "layer_fusion_weights": aux.get("layer_fusion_weights"),
                    }
                    print(
                        f"  [E{epoch + 1}/{cfg.epochs}] step {global_step}/{total_steps} "
                        f"loss={avg_loss:.4f} det={det_loss.item():.4f} "
                        f"lr={lr_now:.2e} gate={line['gate_mean']:.4f} ETA={eta/3600:.2f}h"
                    )
                    with open(log_path, "a", encoding="utf-8") as handle:
                        handle.write(json.dumps(line) + "\n")

        last_epoch_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        if main_process:
            print("\n" + "=" * 60)
            print(f"  Epoch {epoch + 1}/{cfg.epochs} loss={last_epoch_loss:.4f}")
            print("=" * 60)
            save_checkpoint(model_raw, optimizer, scheduler, epoch, global_step, cfg, tag=f"epoch{epoch + 1}")

        if use_ddp:
            dist.barrier()

    if main_process:
        print("\n[5/5] Evaluating RefCOCO+ val/testA/testB ...")

    results = {}
    for split in cfg.eval_splits:
        eval_ds = build_refcoco_eval(
            data_root=cfg.data_root,
            image_dir=cfg.image_dir,
            dataset_name=cfg.dataset_name,
            split_by=cfg.split_by,
            split=split,
        )
        split_result = evaluate_split(model_raw, eval_ds, gd_tok, gd_spec, device, rank, world_size)
        results[split] = split_result
        if main_process:
            print(
                f"  {split}: {split_result['accuracy']:.4f} "
                f"({split_result['correct']}/{split_result['total']})"
            )
        if use_ddp:
            dist.barrier()

    final_fusion_weights = get_layer_fusion_weights(model_raw)
    final_ckpt = None
    if main_process:
        final_ckpt = save_checkpoint(
            model_raw, optimizer, scheduler, cfg.epochs - 1, global_step, cfg, tag="final"
        )
        summary = {
            "dataset_name": cfg.dataset_name,
            "split_by": cfg.split_by,
            "extract_layers": cfg.extract_layers,
            "extract_layer": cfg.extract_layers[-1],
            "layer_fusion": cfg.layer_fusion,
            "layer_fusion_weights": final_fusion_weights,
            "injection_layers": cfg.injection_layers,
            "fusion_mode": cfg.fusion_mode,
            "preserve_kd_enabled": cfg.preserve_kd_enabled,
            "epochs": cfg.epochs,
            "per_gpu_batch": cfg.per_gpu_batch,
            "grad_accum": cfg.grad_accum,
            "world_size": world_size,
            "global_step": global_step,
            "final_epoch_loss": last_epoch_loss,
            "results": results,
            "train_log": log_path,
            "checkpoint": final_ckpt,
        }
        with open(eval_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        print(f"  Saved eval: {eval_path}")

    if use_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
