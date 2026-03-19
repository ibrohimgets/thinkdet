"""
Diagnostic RefCOCO+ layer sweep without touching the core ThinkDet architecture.

This script trains one shared detector with six independent adapter branches,
one per InternVL extraction layer in [8, 9, 10, 11, 12, 13].

Each branch:
  - receives its own extracted H_vlm layer
  - injects into DINO memory_text at decoder layers [1, 3, 5]
  - gets its own REC loss signal

Training loss is the mean of the six branch losses. Detection heads are shared
and trainable so the setup stays close to the current residual TMA path while
keeping the layer-specific adapters independent.
"""

import argparse
import copy
import json
import math
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
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
from thinkdet.models.cross_attention import ThinkDetTextAugmenter
from thinkdet.models.decoder_layer import ThinkDetDecoderLayer
from thinkdet.models.projector import InternVLFeatureExtractor


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
    bsz = pred_logits.shape[0]
    text_len = pred_logits.shape[2]
    total_loss = torch.tensor(0.0, device=device, requires_grad=True)

    pmaps = positive_maps[:, :, :text_len].to(device)
    pmaps_n = positive_maps_norm[:, :, :text_len].to(device)

    for b in range(bsz):
        logits = pred_logits[b]
        boxes = pred_boxes[b]
        gt = gt_boxes_list[b].to(device)
        pmap = pmaps[b]
        pmap_n = pmaps_n[b]

        with torch.no_grad():
            probs = logits.clamp(-50, 50).sigmoid()
            cp = (probs * pmap_n).sum(dim=-1, keepdim=True)
            cc = -cp
            cl1 = torch.cdist(boxes, gt, p=1)
            cg = 1.0 - safe_generalized_box_iou(
                box_cxcywh_to_xyxy(boxes), box_cxcywh_to_xyxy(gt)
            )
            cost = (5 * cl1 + 2 * cg + cc).nan_to_num(100.0, 100.0, -100.0)
            ri, coli = linear_sum_assignment(cost.cpu().numpy())

        ri = torch.tensor(ri, dtype=torch.long, device=device)
        coli = torch.tensor(coli, dtype=torch.long, device=device)
        matched_pred = boxes[ri]
        matched_gt = gt[coli]

        l1 = F.l1_loss(matched_pred, matched_gt, reduction="sum")
        giou = safe_generalized_box_iou(
            box_cxcywh_to_xyxy(matched_pred), box_cxcywh_to_xyxy(matched_gt)
        )
        giou_loss = (1 - giou.diag()).sum()

        target = torch.zeros_like(logits)
        target[ri[0]] = pmap[0]
        focal = sigmoid_focal_loss(logits, target, 1)

        total_loss = total_loss + 5 * l1 + 2 * giou_loss + focal

    return total_loss / max(bsz, 1)


def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        p = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * p)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class LayerBranch(nn.Module):
    def __init__(
        self,
        base_layers,
        injection_layers,
        mllm_hidden_dim,
        d_model=256,
        tma_m=8,
        tma_n_heads=8,
        tma_alpha_init=0.0,
        fusion_mode="residual",
        residual_hidden_mult=2,
    ):
        super().__init__()
        self.injection_layers = list(injection_layers)
        self.wrappers = nn.ModuleDict()

        for layer_idx in self.injection_layers:
            original_layer = copy.deepcopy(base_layers[layer_idx])
            for param in original_layer.parameters():
                param.requires_grad = False

            augmenter = ThinkDetTextAugmenter(
                mllm_hidden_dim=mllm_hidden_dim,
                d_model=d_model,
                M=tma_m,
                n_heads=tma_n_heads,
                alpha_init=tma_alpha_init,
            )
            wrapped = ThinkDetDecoderLayer(
                original_layer=original_layer,
                augmenter=augmenter,
                fusion_mode=fusion_mode,
                residual_hidden_mult=residual_hidden_mult,
                preserve_kd_enabled=False,
            )
            self.wrappers[str(layer_idx)] = wrapped

    def install(self, decoder):
        for layer_idx in self.injection_layers:
            decoder.layers[layer_idx] = self.wrappers[str(layer_idx)]

    def clear(self):
        for wrapped in self.wrappers.values():
            wrapped.clear_h_vlm()

    def set_h_vlm(self, h_vlm):
        for wrapped in self.wrappers.values():
            wrapped.set_h_vlm(h_vlm)

    def branch_params(self):
        params = []
        for wrapped in self.wrappers.values():
            params.extend(wrapped.tma_parameters())
        return params

    def gate_values(self):
        return {
            str(layer_idx): float(torch.tanh(self.wrappers[str(layer_idx)].augmenter.alpha).item())
            for layer_idx in self.injection_layers
        }


class RefCOCOPlusLayerSweepModel(nn.Module):
    def __init__(
        self,
        grounding_dino,
        internvl_path,
        extract_layers,
        injection_layers,
        d_model=256,
        tma_m=8,
        tma_n_heads=8,
        tma_alpha_init=0.0,
        fusion_mode="residual",
    ):
        super().__init__()
        self.grounding_dino = grounding_dino
        self.extract_layers = [int(x) for x in extract_layers]
        self.injection_layers = [int(x) for x in injection_layers]
        self.fusion_mode = fusion_mode

        for param in self.grounding_dino.parameters():
            param.requires_grad = False

        self.feature_extractor = InternVLFeatureExtractor(
            model_path=internvl_path,
            extract_layers=self.extract_layers,
            layer_fusion="mean",
            freeze=True,
        )

        self.mllm_hidden_dim = self.feature_extractor.llm_hidden_dim
        self.branches = nn.ModuleDict()

        decoder = self._get_decoder()
        self._base_decoder_layers = {idx: decoder.layers[idx] for idx in self.injection_layers}

        for layer_idx in self.extract_layers:
            self.branches[str(layer_idx)] = LayerBranch(
                base_layers=self._base_decoder_layers,
                injection_layers=self.injection_layers,
                mllm_hidden_dim=self.mllm_hidden_dim,
                d_model=d_model,
                tma_m=tma_m,
                tma_n_heads=tma_n_heads,
                tma_alpha_init=tma_alpha_init,
                fusion_mode=fusion_mode,
            )

        self._unfreeze_detection_heads()

    def _get_decoder(self):
        model = self.grounding_dino
        if hasattr(model, "model"):
            model = model.model
        if hasattr(model, "transformer"):
            return model.transformer.decoder
        if hasattr(model, "decoder"):
            return model.decoder
        raise AttributeError("Cannot find decoder in GroundingDINO model")

    def _restore_base_decoder(self):
        decoder = self._get_decoder()
        for layer_idx in self.injection_layers:
            decoder.layers[layer_idx] = self._base_decoder_layers[layer_idx]

    def _unfreeze_detection_heads(self):
        model = self.grounding_dino
        if hasattr(model, "model"):
            model = model.model
        for name in ["class_embed", "bbox_embed", "enc_output", "enc_score_head", "enc_bbox_head"]:
            module = getattr(model, name, None)
            if module is not None:
                for param in module.parameters():
                    param.requires_grad = True

    def get_adapter_params(self):
        params = []
        for branch in self.branches.values():
            params.extend(branch.branch_params())
        return params

    def forward(self, images, expressions, dino_inputs):
        selected = self.feature_extractor.extract_selected_layers(images, expressions)
        decoder = self._get_decoder()

        outputs_by_head = {}
        aux_by_head = {}

        for layer_idx in self.extract_layers:
            branch = self.branches[str(layer_idx)]
            h_vlm = selected[layer_idx]
            branch.set_h_vlm(h_vlm)
            branch.install(decoder)
            try:
                outputs = self.grounding_dino(**dino_inputs)
            finally:
                branch.clear()
                self._restore_base_decoder()

            outputs_by_head[str(layer_idx)] = outputs
            aux_by_head[str(layer_idx)] = {
                "h_vlm_norm": float(h_vlm.flatten(1).norm(dim=1).mean().item()),
                "gate_values": branch.gate_values(),
            }

        return outputs_by_head, aux_by_head


class Cfg:
    data_root = f"{ROOT}/dataSets/refer/data"
    image_dir = f"{ROOT}/dataSets/coco/train2017"

    gd_config = f"{ROOT}/GroundingDINO/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    gd_weights = f"{ROOT}/GroundingDINO/GroundingDINO/weights/groundingdino_swint_ogc.pth"
    internvl_path = f"{ROOT}/InternVL3_5-1B"

    dataset_name = "refcoco+"
    split_by = "unc"
    train_split = "train"
    eval_splits = ["val", "testA", "testB"]

    extract_layers = [8, 9, 10, 11, 12, 13]
    injection_layers = [1, 3, 5]
    fusion_mode = "residual"
    tma_m = 8
    tma_n_heads = 8
    tma_alpha_init = 0.0
    d_model = 256

    epochs = 1
    per_gpu_batch = 1
    grad_accum = 8
    lr_adapters = 2e-4
    lr_heads = 1e-5
    weight_decay = 0.01
    warmup_ratio = 0.05
    max_grad_norm = 1.0
    num_workers = 4
    debug_limit = 2000
    log_interval = 1

    output_root = f"{ROOT}/thinkdet/checkpoints/refcocoplus_layer_head_diag"


def parse_args():
    cfg = Cfg()
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=cfg.epochs)
    parser.add_argument("--per_gpu_batch", type=int, default=cfg.per_gpu_batch)
    parser.add_argument("--grad_accum", type=int, default=cfg.grad_accum)
    parser.add_argument("--debug_limit", type=int, default=cfg.debug_limit)
    parser.add_argument("--log_interval", type=int, default=cfg.log_interval)
    parser.add_argument("--num_workers", type=int, default=cfg.num_workers)
    parser.add_argument("--output_dir", type=str, default="")
    return parser.parse_args(), cfg


def setup_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    ddp = world_size > 1
    if ddp:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        rank = dist.get_rank()
    else:
        local_rank = 0
        rank = 0
        device = torch.device("cuda:0")
    return ddp, rank, world_size, local_rank, device


def save_checkpoint(model_raw, optimizer, scheduler, epoch, step, cfg, path):
    trainable_state = {}
    for name, param in model_raw.named_parameters():
        if param.requires_grad:
            trainable_state[name] = param.detach().cpu()
    torch.save(
        {
            "epoch": epoch + 1,
            "global_step": step,
            "dataset_name": cfg.dataset_name,
            "split_by": cfg.split_by,
            "extract_layers": list(cfg.extract_layers),
            "injection_layers": list(cfg.injection_layers),
            "fusion_mode": cfg.fusion_mode,
            "trainable_state_dict": trainable_state,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        },
        path,
    )


@torch.no_grad()
def evaluate_split(model_raw, dataset, tokenizer, special_tokens, device, rank, world_size):
    model_raw.eval()

    correct = torch.zeros(len(model_raw.extract_layers), device=device, dtype=torch.long)
    total = torch.zeros(1, device=device, dtype=torch.long)
    layer_keys = [str(layer) for layer in model_raw.extract_layers]

    indices = list(range(rank, len(dataset), world_size))
    t0 = time.time()

    for local_idx, sample_idx in enumerate(indices):
        batch = refcoco_collate_fn([dataset[sample_idx]])
        internvl_images = batch["internvl_images"].to(device)
        dino_nested = NestedTensor(
            batch["dino_images"].tensors.to(device),
            batch["dino_images"].mask.to(device),
        )
        expression = batch["expressions"][0]
        query_text = batch["query_texts"][0]
        gt_box = batch["boxes"][0].to(device)
        dino_inputs = {"samples": dino_nested, "captions": [query_text]}

        outputs_by_head, _ = model_raw(internvl_images, [expression], dino_inputs)
        _, _, pmap_norm = build_rec_positive_map_batch(tokenizer, special_tokens, [expression])

        for layer_pos, layer_key in enumerate(layer_keys):
            outputs = outputs_by_head[layer_key]
            logits = outputs["pred_logits"][0]
            boxes = outputs["pred_boxes"][0]
            text_len = logits.shape[-1]
            pmap_n = pmap_norm[0, :, :text_len].to(device)
            probs = logits.clamp(-50, 50).sigmoid()
            scores = (probs * pmap_n).sum(dim=-1)
            best_idx = scores.argmax()
            pred_box = boxes[best_idx].unsqueeze(0)
            pred_xyxy = box_cxcywh_to_xyxy(pred_box)
            gt_xyxy = box_cxcywh_to_xyxy(gt_box)
            iou = compute_iou(pred_xyxy, gt_xyxy)
            if iou.item() > 0.5:
                correct[layer_pos] += 1

        total += 1

        if rank == 0 and (local_idx + 1) % 500 == 0:
            elapsed = time.time() - t0
            print(f"    eval progress: {local_idx + 1}/{len(indices)} ({elapsed:.0f}s)")

    if world_size > 1:
        dist.all_reduce(correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(total, op=dist.ReduceOp.SUM)

    total_count = int(total.item())
    results = {}
    for layer_pos, layer_key in enumerate(layer_keys):
        layer_correct = int(correct[layer_pos].item())
        results[layer_key] = {
            "accuracy": float(layer_correct / max(total_count, 1)),
            "correct": layer_correct,
            "total": total_count,
        }
    return results


def main():
    args, cfg = parse_args()
    cfg.epochs = args.epochs
    cfg.per_gpu_batch = args.per_gpu_batch
    cfg.grad_accum = args.grad_accum
    cfg.debug_limit = args.debug_limit
    cfg.log_interval = args.log_interval
    cfg.num_workers = args.num_workers

    ddp, rank, world_size, local_rank, device = setup_distributed()
    is_main = rank == 0

    layer_tag = "layers_" + "_".join(str(x) for x in cfg.extract_layers)
    if args.output_dir:
        output_dir = args.output_dir
    else:
        ts = time.strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(cfg.output_root, f"{layer_tag}_diag_e{cfg.epochs}_dbg{cfg.debug_limit}_{ts}")
    cfg.output_dir = output_dir

    if is_main:
        os.makedirs(cfg.output_dir, exist_ok=True)
        print("=" * 70)
        print("  RefCOCO+ Layer Head Diagnostic")
        print("=" * 70)
        print(f"  dataset:          {cfg.dataset_name}/{cfg.split_by}")
        print(f"  train split:      {cfg.train_split}")
        print(f"  eval splits:      {cfg.eval_splits}")
        print(f"  extract_layers:   {cfg.extract_layers}")
        print(f"  injection_layers: {cfg.injection_layers}")
        print(f"  fusion_mode:      {cfg.fusion_mode}")
        print(f"  epochs:           {cfg.epochs}")
        print(f"  per_gpu_batch:    {cfg.per_gpu_batch}")
        print(f"  grad_accum:       {cfg.grad_accum}")
        print(f"  debug_limit:      {cfg.debug_limit}")
        print(f"  log_interval:     {cfg.log_interval}")
        print(f"  GPUs:             {world_size}")
        print(f"  output:           {cfg.output_dir}")

    if is_main:
        print("\n[1/5] Loading GroundingDINO ...")
    gd = load_gd_model(cfg.gd_config, cfg.gd_weights, device="cpu")

    if is_main:
        print("[1/5] Building diagnostic model ...")
    model = RefCOCOPlusLayerSweepModel(
        grounding_dino=gd,
        internvl_path=cfg.internvl_path,
        extract_layers=cfg.extract_layers,
        injection_layers=cfg.injection_layers,
        d_model=cfg.d_model,
        tma_m=cfg.tma_m,
        tma_n_heads=cfg.tma_n_heads,
        tma_alpha_init=cfg.tma_alpha_init,
        fusion_mode=cfg.fusion_mode,
    ).to(device)

    if ddp:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
    model_raw = model.module if ddp else model

    total_params = sum(p.numel() for p in model_raw.parameters())
    trainable_params = sum(p.numel() for p in model_raw.parameters() if p.requires_grad)
    if is_main:
        print(f"  Total params:     {total_params:,}")
        print(f"  Trainable params: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")

    if is_main:
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
        if is_main:
            print(f"  [DEBUG] limited train samples to {len(train_ds.samples)}")

    train_sampler = DistributedSampler(train_ds, shuffle=True) if ddp else None
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.per_gpu_batch,
        shuffle=(train_sampler is None),
        num_workers=cfg.num_workers,
        collate_fn=refcoco_collate_fn,
        pin_memory=True,
        drop_last=True,
        sampler=train_sampler,
    )

    gd_tok = model_raw.grounding_dino.tokenizer
    gd_spec = model_raw.grounding_dino.specical_tokens

    if is_main:
        print("\n[3/5] Setting up optimizer ...")
    adapter_params = model_raw.get_adapter_params()
    adapter_ids = {id(p) for p in adapter_params}
    head_params = [
        p for p in model_raw.parameters()
        if p.requires_grad and id(p) not in adapter_ids
    ]

    if is_main:
        print(f"  Adapter params: {sum(p.numel() for p in adapter_params):,}")
        print(f"  Head params:    {sum(p.numel() for p in head_params):,}")

    optimizer = torch.optim.AdamW(
        [
            {"params": adapter_params, "lr": cfg.lr_adapters, "name": "adapters"},
            {"params": head_params, "lr": cfg.lr_heads, "name": "heads"},
        ],
        weight_decay=cfg.weight_decay,
    )

    steps_per_epoch = len(train_loader) // cfg.grad_accum
    total_steps = steps_per_epoch * cfg.epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    if is_main:
        print(f"  Steps/epoch: {steps_per_epoch}")
        print(f"  Total steps: {total_steps}")
        print(f"  Warmup:      {warmup_steps}")

    log_path = os.path.join(cfg.output_dir, "train_log.jsonl")
    eval_path = os.path.join(cfg.output_dir, "eval_results.json")
    ckpt_path = os.path.join(cfg.output_dir, "diag_checkpoint_epoch1.pth")
    if is_main:
        with open(log_path, "w"):
            pass

    if is_main:
        print("\n[4/5] Training ...\n")

    global_step = 0
    t0 = time.time()
    layer_keys = [str(layer) for layer in cfg.extract_layers]
    epoch_total_losses = []
    epoch_head_losses = defaultdict(list)

    for epoch in range(cfg.epochs):
        if ddp and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        optimizer.zero_grad()

        step_total_window = []
        step_head_window = defaultdict(list)
        step_hvlm_window = defaultdict(list)

        for batch_idx, batch in enumerate(train_loader):
            internvl_imgs = batch["internvl_images"].to(device)
            dino_nested = NestedTensor(
                batch["dino_images"].tensors.to(device),
                batch["dino_images"].mask.to(device),
            )
            dino_inputs = {"samples": dino_nested, "captions": batch["query_texts"]}
            _, pmaps, pmaps_norm = build_rec_positive_map_batch(
                gd_tok, gd_spec, batch["expressions"]
            )

            outputs_by_head, aux_by_head = model(
                internvl_imgs, batch["expressions"], dino_inputs
            )

            head_losses = {}
            for layer_key in layer_keys:
                outputs = outputs_by_head[layer_key]
                det_loss = compute_rec_loss(
                    outputs["pred_logits"],
                    outputs["pred_boxes"],
                    batch["boxes"],
                    pmaps,
                    pmaps_norm,
                    device,
                )
                head_losses[layer_key] = det_loss
                step_head_window[layer_key].append(float(det_loss.item()))
                step_hvlm_window[layer_key].append(float(aux_by_head[layer_key]["h_vlm_norm"]))
                epoch_head_losses[layer_key].append(float(det_loss.item()))

            loss = torch.stack(list(head_losses.values())).mean()
            (loss / cfg.grad_accum).backward()
            step_total_window.append(float(loss.item()))
            epoch_total_losses.append(float(loss.item()))

            if (batch_idx + 1) % cfg.grad_accum != 0:
                continue

            torch.nn.utils.clip_grad_norm_(
                [p for p in model_raw.parameters() if p.requires_grad],
                cfg.max_grad_norm,
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            if is_main and global_step % cfg.log_interval == 0:
                total_avg = float(np.mean(step_total_window)) if step_total_window else 0.0
                head_avg = {
                    layer_key: float(np.mean(step_head_window[layer_key]))
                    for layer_key in layer_keys
                }
                hvlm_avg = {
                    layer_key: float(np.mean(step_hvlm_window[layer_key]))
                    for layer_key in layer_keys
                }
                gate_means = {
                    layer_key: float(np.mean(list(aux_by_head[layer_key]["gate_values"].values())))
                    for layer_key in layer_keys
                }
                lr_now = float(scheduler.get_last_lr()[0])
                elapsed = time.time() - t0
                eta = elapsed / max(global_step, 1) * (total_steps - global_step)
                print(
                    f"  [E{epoch+1}/{cfg.epochs}] step {global_step}/{total_steps} "
                    f"loss={total_avg:.4f} lr={lr_now:.2e} ETA={eta/3600:.2f}h"
                )
                print(
                    "    head_losses: "
                    + " ".join(f"L{layer_key}={head_avg[layer_key]:.4f}" for layer_key in layer_keys)
                )
                with open(log_path, "a") as f:
                    f.write(
                        json.dumps(
                            {
                                "step": global_step,
                                "epoch": epoch + 1,
                                "loss": total_avg,
                                "head_losses": head_avg,
                                "head_h_vlm_norms": hvlm_avg,
                                "head_gate_means": gate_means,
                                "lr": lr_now,
                            }
                        )
                        + "\n"
                    )

            step_total_window = []
            step_head_window = defaultdict(list)
            step_hvlm_window = defaultdict(list)

        if ddp:
            dist.barrier()

    epoch_avg = float(np.mean(epoch_total_losses)) if epoch_total_losses else 0.0
    head_epoch_avg = {
        layer_key: float(np.mean(epoch_head_losses[layer_key])) if epoch_head_losses[layer_key] else 0.0
        for layer_key in layer_keys
    }

    if is_main:
        print(f"\n{'=' * 60}")
        print(f"  Epoch {cfg.epochs}/{cfg.epochs} loss={epoch_avg:.4f}")
        print("  " + " ".join(f"L{layer_key}={head_epoch_avg[layer_key]:.4f}" for layer_key in layer_keys))
        print(f"{'=' * 60}")
        save_checkpoint(model_raw, optimizer, scheduler, cfg.epochs - 1, global_step, cfg, ckpt_path)

    if ddp:
        dist.barrier()

    if is_main:
        print("\n[5/5] Evaluating RefCOCO+ val/testA/testB ...")

    model_raw.eval()
    all_results = {}
    for split in cfg.eval_splits:
        dataset = build_refcoco_eval(
            data_root=cfg.data_root,
            image_dir=cfg.image_dir,
            dataset_name=cfg.dataset_name,
            split_by=cfg.split_by,
            split=split,
        )
        split_results = evaluate_split(model_raw, dataset, gd_tok, gd_spec, device, rank, world_size)
        all_results[split] = split_results
        if is_main:
            print(
                f"  {split}: "
                + " ".join(
                    f"L{layer_key}={split_results[layer_key]['accuracy']:.4f}"
                    for layer_key in layer_keys
                )
            )

    if is_main:
        payload = {
            "dataset_name": cfg.dataset_name,
            "split_by": cfg.split_by,
            "extract_layers": cfg.extract_layers,
            "injection_layers": cfg.injection_layers,
            "fusion_mode": cfg.fusion_mode,
            "epochs": cfg.epochs,
            "debug_limit": cfg.debug_limit,
            "global_step": global_step,
            "epoch_loss": epoch_avg,
            "head_epoch_losses": head_epoch_avg,
            "results": all_results,
            "train_log": log_path,
            "checkpoint": ckpt_path,
        }
        with open(eval_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"  Saved eval: {eval_path}")

    if ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
