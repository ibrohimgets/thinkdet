"""
ThinkDet TMA — Stage 2: TextAugmenter + Detection Heads

Protocol:
    - Load Stage-1 TMA checkpoint (trained TextAugmenters)
    - Unfreeze TextAugmenters + detection heads (class_embed, bbox_embed)
    - InternVL + DINO backbone remain frozen
    - Loss: detection loss only (no KD)
    - AMP (fp16) for efficiency

Launch:
    torchrun --nproc_per_node=7 thinkdet/scripts/training/train_stage2_tma.py \
        --stage1_ckpt /path/to/thinkdet_tma_stage1_epoch2.pth
"""

import os
import sys
import time
import json
import argparse
import tempfile
import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler

ROOT = "/home/iibrohimm/project/next_step"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "GroundingDINO", "GroundingDINO"))

from thinkdet.models.arch import ThinkDetModel, DEFAULT_INJECTION_LAYERS
from thinkdet.data.coco_grounding import (
    COCOGroundingDataset,
    collate_fn,
    build_positive_map,
)
from groundingdino.util.inference import load_model as load_gd_model
from groundingdino.util.misc import NestedTensor
from thinkdet.scripts.training.train_stage1 import (
    compute_detection_loss,
    get_cosine_schedule_with_warmup,
)


STAGE1_CKPT = (
    f"{ROOT}/thinkdet/checkpoints/stage1_tma/"
    "layer9_tma_m8_full_2ep_7gpu_20260218_152649/"
    "thinkdet_tma_stage1_epoch2.pth"
)


class Cfg:
    # Data
    coco_root     = f"{ROOT}/dataSets/coco"
    train_ann     = f"{coco_root}/annotations/instances_train2017.json"
    train_img_dir = f"{coco_root}/train2017"
    val_ann       = f"{coco_root}/annotations/instances_val2017.json"
    val_img_dir   = f"{coco_root}/val2017"

    # Models
    gd_config  = (
        f"{ROOT}/GroundingDINO/GroundingDINO/groundingdino/config/"
        "GroundingDINO_SwinT_OGC.py"
    )
    gd_weights = (
        f"{ROOT}/GroundingDINO/GroundingDINO/weights/groundingdino_swint_ogc.pth"
    )
    internvl_path = f"{ROOT}/InternVL3_5-1B"
    stage1_ckpt   = STAGE1_CKPT

    # Layer setup (must match stage1)
    extract_layer    = 9
    extract_layers   = [9]
    layer_fusion     = "mean"
    d_model          = 256
    tma_m            = 8
    tma_n_heads      = 8
    tma_alpha_init   = 0.3
    alpha_floor      = 0.2
    # Safer default for baseline-preserving tuning.
    injection_layers = [3]
    tma_fusion_mode = "residual"  # concat | residual

    # InternVL text-query conditioning during training
    train_query_mode = "mixed"   # fixed | gt | mixed
    dynamic_query_ratio = 0.7
    max_query_categories = 12

    # Training
    epochs        = 2
    per_gpu_batch = 1
    grad_accum    = 8
    lr_tma        = 5e-5   # lower than stage1
    lr_heads      = 5e-5
    weight_decay  = 0.01
    warmup_ratio  = 0.03
    max_grad_norm = 1.0
    num_workers   = 4
    debug_limit   = 0
    val_interval_epochs = 1
    val_max_images = 1000
    val_conf_thresh = 0.1

    # Save
    output_dir    = f"{ROOT}/thinkdet/checkpoints/stage2_tma/layer9_tma_m8"
    log_interval  = 200
    save_interval = 2000
    grad_check_step = 100


def format_layer_tag(extract_layers):
    layers = sorted(set(int(x) for x in extract_layers))
    if len(layers) == 1:
        return f"layer{layers[0]}"
    return "layers_" + "_".join(str(x) for x in layers)


def init_distributed():
    ddp = int(os.environ.get("WORLD_SIZE", 1)) > 1
    if ddp:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        world_size = dist.get_world_size()
    else:
        local_rank = 0
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        world_size = 1
    torch.backends.cudnn.benchmark = True
    return ddp, local_rank, device, world_size


def gradient_health_check(model_raw, is_main):
    """Verify gradient flow after first N steps."""
    if not is_main:
        return
    print("\n  [GRAD CHECK] ──────────────────────────────────")
    tma_ok, head_ok, issues = 0, 0, 0
    for name, param in model_raw.named_parameters():
        if param.requires_grad:
            if param.grad is None:
                print(f"    WARNING: {name} — no gradient!")
                issues += 1
            elif param.grad.abs().max().item() == 0:
                print(f"    WARNING: {name} — zero gradient!")
                issues += 1
            else:
                grad_norm = param.grad.norm().item()
                if "alpha" in name:
                    print(f"    OK: {name} — grad_norm={grad_norm:.6f}  value={param.item():.6f}")
                    tma_ok += 1
                elif "augmenter" in name or "residual_fuser" in name:
                    tma_ok += 1
                else:
                    head_ok += 1
        elif param.grad is not None:
            print(f"    WARNING: {name} — frozen but has gradient!")
            issues += 1
    print(f"    Summary: TMA={tma_ok} ok, Heads={head_ok} ok, Issues={issues}")
    print("  ─────────────────────────────────────────────\n")


def load_stage1_weights(model, ckpt_path, is_main):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Stage-1 checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(
        ckpt["trainable_state_dict"], strict=False
    )
    if is_main:
        print(f"  Loaded Stage-1 weights: {ckpt_path}")
        print(f"  Missing keys:     {len(missing)}")
        print(f"  Unexpected keys:  {len(unexpected)}")


def enforce_alpha_floor(model, alpha_floor, is_main):
    floor = float(alpha_floor)
    adjusted = 0
    for layer in getattr(model, "adapted_layers", []):
        alpha = getattr(layer.augmenter, "alpha", None)
        if alpha is None:
            continue
        if alpha.item() < floor:
            alpha.data.fill_(floor)
            adjusted += 1
    if is_main:
        print(f"  Applied alpha_floor={floor:.3f} on {adjusted} augmenter layers")


def box_cxcywh_to_xyxy(x):
    cx, cy, w, h = x.unbind(-1)
    return torch.stack([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], dim=-1)


@torch.no_grad()
def evaluate_coco_map(
    model_raw,
    val_dataset,
    positive_map_norm,
    cat_name_to_idx,
    all_query_text,
    device,
    conf_thresh=0.1,
    max_dets=100,
):
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

    for batch in loader:
        internvl_images = batch["internvl_images"].to(device)
        dino_nested = NestedTensor(
            batch["dino_images"].tensors.to(device),
            batch["dino_images"].mask.to(device),
        )
        dino_inputs = {"samples": dino_nested, "captions": [all_query_text]}
        outputs, _ = model_raw(internvl_images, batch["query_texts"], dino_inputs)

        image_id = batch["image_ids"][0]
        img_info = val_dataset.coco.loadImgs(image_id)[0]
        img_w, img_h = img_info["width"], img_info["height"]

        logits = outputs["pred_logits"][0].clamp(-50, 50)
        boxes = outputs["pred_boxes"][0]
        probs = logits.sigmoid()
        cat_scores = probs @ positive_map_norm.T
        max_scores, max_cats = cat_scores.max(dim=-1)

        keep = max_scores > conf_thresh
        if keep.sum().item() == 0:
            continue

        kept_scores = max_scores[keep]
        kept_cats = max_cats[keep]
        kept_boxes = boxes[keep]

        if kept_scores.shape[0] > max_dets:
            topk = kept_scores.argsort(descending=True)[:max_dets]
            kept_scores = kept_scores[topk]
            kept_cats = kept_cats[topk]
            kept_boxes = kept_boxes[topk]

        pred_xyxy = box_cxcywh_to_xyxy(kept_boxes)
        pred_xyxy[:, 0] *= img_w
        pred_xyxy[:, 2] *= img_w
        pred_xyxy[:, 1] *= img_h
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
            predictions.append(
                {
                    "image_id": image_id,
                    "category_id": coco_cat_id,
                    "bbox": [
                        round(x1, 2),
                        round(y1, 2),
                        round(max(0.0, x2 - x1), 2),
                        round(max(0.0, y2 - y1), 2),
                    ],
                    "score": round(float(kept_scores[i].item()), 4),
                }
            )

    if not predictions:
        return {"mAP": 0.0, "mAP_50": 0.0, "mAP_75": 0.0, "num_predictions": 0}

    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(predictions, tmp)
    tmp.close()
    coco_dt = val_dataset.coco.loadRes(tmp.name)
    coco_eval = COCOeval(val_dataset.coco, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    os.unlink(tmp.name)

    return {
        "mAP": float(coco_eval.stats[0]),
        "mAP_50": float(coco_eval.stats[1]),
        "mAP_75": float(coco_eval.stats[2]),
        "num_predictions": int(len(predictions)),
    }


def save_checkpoint(model, optimizer, scheduler, epoch, step, cfg, tag=""):
    trainable_state = {
        name: param.detach().cpu()
        for name, param in model.named_parameters()
        if param.requires_grad
    }
    path = os.path.join(cfg.output_dir, f"thinkdet_tma_stage2_{tag}.pth")
    torch.save(
        {
            "epoch": epoch + 1,
            "global_step": step,
            "stage": "tma_stage2",
            "stage1_ckpt": cfg.stage1_ckpt,
            "extract_layer": cfg.extract_layer,
            "extract_layers": cfg.extract_layers,
            "layer_fusion": cfg.layer_fusion,
            "injection_layers": cfg.injection_layers,
            "fusion_mode": cfg.tma_fusion_mode,
            "tma_m": cfg.tma_m,
            "tma_n_heads": cfg.tma_n_heads,
            "tma_alpha_init": cfg.tma_alpha_init,
            "alpha_floor": cfg.alpha_floor,
            "train_query_mode": cfg.train_query_mode,
            "dynamic_query_ratio": cfg.dynamic_query_ratio,
            "trainable_state_dict": trainable_state,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        },
        path,
    )
    print(f"  [SAVE] {path}  ({len(trainable_state)} param tensors)")


def parse_args():
    cfg = Cfg()
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1_ckpt", type=str, default=cfg.stage1_ckpt)
    parser.add_argument("--epochs", type=int, default=cfg.epochs)
    parser.add_argument("--debug_limit", type=int, default=cfg.debug_limit)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=cfg.num_workers)
    parser.add_argument("--lr_tma", type=float, default=cfg.lr_tma)
    parser.add_argument("--lr_heads", type=float, default=cfg.lr_heads)
    parser.add_argument("--tma_m", type=int, default=cfg.tma_m)
    parser.add_argument("--tma_alpha_init", type=float, default=cfg.tma_alpha_init)
    parser.add_argument("--alpha_floor", type=float, default=cfg.alpha_floor)
    parser.add_argument(
        "--tma_fusion_mode",
        type=str,
        default=cfg.tma_fusion_mode,
        choices=["concat", "residual"],
    )
    parser.add_argument(
        "--train_query_mode",
        type=str,
        default=cfg.train_query_mode,
        choices=["fixed", "gt", "mixed"],
    )
    parser.add_argument("--dynamic_query_ratio", type=float, default=cfg.dynamic_query_ratio)
    parser.add_argument("--max_query_categories", type=int, default=cfg.max_query_categories)
    parser.add_argument("--val_interval_epochs", type=int, default=cfg.val_interval_epochs)
    parser.add_argument("--val_max_images", type=int, default=cfg.val_max_images)
    parser.add_argument("--val_conf_thresh", type=float, default=cfg.val_conf_thresh)
    parser.add_argument("--extract_layer", type=int, default=None)
    parser.add_argument("--extract_layers", type=int, nargs="+", default=None)
    parser.add_argument(
        "--layer_fusion",
        type=str,
        default=cfg.layer_fusion,
        choices=["mean", "last"],
    )
    parser.add_argument("--resume", type=str, default=None)
    return parser.parse_args()


def main():
    cfg = Cfg()
    args = parse_args()
    cfg.stage1_ckpt = args.stage1_ckpt
    cfg.epochs = args.epochs
    cfg.debug_limit = args.debug_limit
    cfg.lr_tma = float(args.lr_tma)
    cfg.lr_heads = float(args.lr_heads)
    cfg.tma_m = int(args.tma_m)
    cfg.num_workers = int(args.num_workers)
    cfg.tma_alpha_init = float(args.tma_alpha_init)
    cfg.alpha_floor = float(args.alpha_floor)
    cfg.tma_fusion_mode = str(args.tma_fusion_mode)
    cfg.train_query_mode = str(args.train_query_mode)
    cfg.dynamic_query_ratio = float(args.dynamic_query_ratio)
    cfg.max_query_categories = int(args.max_query_categories)
    cfg.val_interval_epochs = int(args.val_interval_epochs)
    cfg.val_max_images = int(args.val_max_images)
    cfg.val_conf_thresh = float(args.val_conf_thresh)
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
        cfg.output_dir = f"{ROOT}/thinkdet/checkpoints/stage2_tma/{layer_tag}_tma_m{cfg.tma_m}"

    ddp, local_rank, device, world_size = init_distributed()
    is_main = local_rank == 0

    if is_main:
        os.makedirs(cfg.output_dir, exist_ok=True)
        print("=" * 70)
        print("  ThinkDet TMA — Stage 2: TMA + Detection Heads")
        print("=" * 70)
        print(f"  stage1_ckpt:      {cfg.stage1_ckpt}")
        print(f"  extract_layer:    {cfg.extract_layer}")
        print(f"  extract_layers:   {cfg.extract_layers}")
        print(f"  layer_fusion:     {cfg.layer_fusion}")
        print(f"  injection_layers: {cfg.injection_layers}")
        print(f"  tma_m:            {cfg.tma_m}")
        print(f"  tma_alpha_init:   {cfg.tma_alpha_init}")
        print(f"  alpha_floor:      {cfg.alpha_floor}")
        print(f"  tma_fusion_mode:  {cfg.tma_fusion_mode}")
        print(f"  train_query_mode: {cfg.train_query_mode}")
        print(f"  dynamic_q_ratio:  {cfg.dynamic_query_ratio}")
        print(f"  lr_tma:           {cfg.lr_tma}")
        print(f"  lr_heads:         {cfg.lr_heads}")
        print(f"  val_interval_ep:  {cfg.val_interval_epochs}")
        print(f"  val_max_images:   {cfg.val_max_images}")
        print(f"  val_conf_thresh:  {cfg.val_conf_thresh}")
        print(f"  epochs:           {cfg.epochs}")
        print(f"  GPUs:             {world_size}")
        print(f"  effective batch:  {cfg.per_gpu_batch * cfg.grad_accum * world_size}")
        print(f"  AMP:              fp16")
        print(f"  cudnn.benchmark:  True")
        print(f"  OMP_NUM_THREADS:  {os.environ.get('OMP_NUM_THREADS', 'unset')}")
        print(f"  num_workers:      {cfg.num_workers}")
        print(f"  log_interval:     {cfg.log_interval}")
        print(f"  output:           {cfg.output_dir}")
        print()

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
        tma_alpha_init=cfg.tma_alpha_init,
        fusion_mode=cfg.tma_fusion_mode,
    )
    model.set_tma_and_heads()
    load_stage1_weights(model, cfg.stage1_ckpt, is_main=is_main)
    enforce_alpha_floor(model, cfg.alpha_floor, is_main=is_main)
    model = model.to(device)

    if ddp:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    model_raw = model.module if ddp else model

    if is_main:
        print("\n[2/4] Loading COCO train2017 ...")
    train_ds = COCOGroundingDataset(
        img_dir=cfg.train_img_dir,
        ann_file=cfg.train_ann,
        query_mode=cfg.train_query_mode,
        dynamic_query_ratio=cfg.dynamic_query_ratio,
        max_query_categories=cfg.max_query_categories,
    )
    if cfg.debug_limit > 0:
        train_ds.img_ids = train_ds.img_ids[: cfg.debug_limit]
        cfg.log_interval = 10
        if is_main:
            print(f"  [DEBUG] Limiting to {cfg.debug_limit} images")

    sampler = DistributedSampler(train_ds, shuffle=True) if ddp else None
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=cfg.per_gpu_batch,
        shuffle=(sampler is None),
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
        sampler=sampler,
    )

    all_cat_names = sorted(train_ds.cat_id_to_name.values())
    gd_tok = model_raw.grounding_dino.tokenizer
    gd_spec = model_raw.grounding_dino.specical_tokens
    all_query_text, pmap, pmap_norm, cat_name_to_idx = build_positive_map(
        gd_tok, gd_spec, all_cat_names
    )
    pmap = pmap.to(device)
    pmap_norm = pmap_norm.to(device)

    if is_main:
        print(f"  Categories: {len(all_cat_names)}")

    val_ds = None
    if is_main and cfg.val_interval_epochs > 0:
        val_ds = COCOGroundingDataset(
            img_dir=cfg.val_img_dir,
            ann_file=cfg.val_ann,
            query_mode="fixed",
        )
        if cfg.val_max_images > 0:
            val_ds.img_ids = val_ds.img_ids[: cfg.val_max_images]
        print(f"  Val images: {len(val_ds.img_ids)}")

    if is_main:
        print("\n[3/4] Setting up optimizer ...")

    # Separate param groups: TMA vs heads
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
        raise RuntimeError("No trainable parameters found for Stage-2.")

    param_groups = []
    if aug_params:
        param_groups.append({"params": aug_params, "lr": cfg.lr_tma, "name": "tma"})
        if is_main:
            print(f"  TMA params:  {sum(p.numel() for p in aug_params):,}")
    if head_params:
        param_groups.append({"params": head_params, "lr": cfg.lr_heads, "name": "heads"})
        if is_main:
            print(f"  Head params: {sum(p.numel() for p in head_params):,}")

    optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.weight_decay)

    steps_per_epoch = len(train_loader) // cfg.grad_accum
    total_steps = steps_per_epoch * cfg.epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    if is_main:
        print(f"  steps/epoch: {steps_per_epoch}  total: {total_steps}  warmup: {warmup_steps}")

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
            print(f"  Resumed at epoch {start_epoch}, step {global_step}")

    if is_main:
        print("\n[4/4] Training ...\n")

    log_path = os.path.join(cfg.output_dir, "train_log.jsonl")
    if is_main and not args.resume:
        with open(log_path, "w"):
            pass

    scaler = GradScaler()
    grad_checked = False
    best_map = -1.0
    val_log_path = os.path.join(cfg.output_dir, "val_log.jsonl")
    if is_main and not args.resume and cfg.val_interval_epochs > 0:
        with open(val_log_path, "w"):
            pass

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
            dino_inputs = {"samples": dino_nested, "captions": [all_query_text] * B}

            try:
                with autocast(dtype=torch.float16):
                    outputs, aux = model(internvl_images, batch["query_texts"], dino_inputs)

                    loss = compute_detection_loss(
                        outputs["pred_logits"],
                        outputs["pred_boxes"],
                        batch["boxes"],
                        batch["category_names"],
                        cat_name_to_idx,
                        pmap,
                        pmap_norm,
                        device,
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
                    print(f"  [NaN/Inf] step {global_step}, skipping")
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

                # Gradient health check after first N steps
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
                        f"L{int(layer)}_a={alpha_by_layer.get(int(layer), float('nan')):.4f}"
                        for layer in cfg.injection_layers
                    ) or "no_tma_layers"
                    print(
                        f"  [E{epoch+1}/{cfg.epochs}] step {global_step}/{total_steps}  "
                        f"loss={avg:.4f}  h_vlm={hvlm:.2f}  "
                        f"{alpha_msg}  "
                        f"lr={lr_now:.2e}  ETA={eta/3600:.1f}h"
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
                        epoch, global_step, cfg, tag=f"step{global_step}"
                    )

        epoch_avg = np.mean(epoch_losses) if epoch_losses else 0.0
        if is_main:
            elapsed = time.time() - t0
            print(f"\n{'='*60}")
            print(f"  Epoch {epoch+1}/{cfg.epochs}  loss={epoch_avg:.4f}  time={elapsed/3600:.2f}h")
            print(f"{'='*60}")
            save_checkpoint(
                model_raw, optimizer, scheduler,
                epoch, global_step, cfg, tag=f"epoch{epoch+1}"
            )
            should_eval = (
                cfg.val_interval_epochs > 0
                and val_ds is not None
                and ((epoch + 1) % cfg.val_interval_epochs == 0)
            )
            if should_eval:
                print(f"\n  [VAL] COCO mAP eval at epoch {epoch+1} ...")
                val_res = evaluate_coco_map(
                    model_raw=model_raw,
                    val_dataset=val_ds,
                    positive_map_norm=pmap_norm,
                    cat_name_to_idx=cat_name_to_idx,
                    all_query_text=all_query_text,
                    device=device,
                    conf_thresh=cfg.val_conf_thresh,
                )
                val_res["epoch"] = epoch + 1
                val_res["global_step"] = global_step
                val_res["train_loss"] = float(epoch_avg)
                with open(val_log_path, "a") as f:
                    f.write(json.dumps(val_res) + "\n")
                print(
                    f"  [VAL] mAP={val_res['mAP']:.4f} "
                    f"mAP50={val_res['mAP_50']:.4f} "
                    f"preds={val_res['num_predictions']}"
                )
                if val_res["mAP"] > best_map:
                    best_map = val_res["mAP"]
                    save_checkpoint(
                        model_raw, optimizer, scheduler,
                        epoch, global_step, cfg, tag="best"
                    )
                    print(f"  [BEST] Updated best mAP to {best_map:.4f}")
                model.train()

        if ddp:
            dist.barrier()

    if is_main:
        total_time = time.time() - t0
        print(f"\n{'='*70}")
        print("  Stage 2 TMA training complete!")
        print(f"  Total time:  {total_time/3600:.2f}h")
        if cfg.val_interval_epochs > 0:
            print(f"  Best val mAP: {best_map:.4f}")
        print(f"  Checkpoints: {cfg.output_dir}")
        print(f"{'='*70}")

    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
