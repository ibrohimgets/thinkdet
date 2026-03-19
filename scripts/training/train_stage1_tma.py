"""
ThinkDet TMA — Stage 1: TextAugmenter Training

Protocol:
    - Load fresh GroundingDINO + InternVL (both frozen)
    - Train only the ThinkDetTextAugmenter at injection layers [1,3,5]
    - Loss: detection loss only (no KD, no gate, no delta)
    - AMP (fp16) for efficiency
    - 2 epochs recommended for initial validation

Launch:
    torchrun --nproc_per_node=7 thinkdet/scripts/training/train_stage1_tma.py
    torchrun --nproc_per_node=7 thinkdet/scripts/training/train_stage1_tma.py --epochs 1 --debug_limit 500
"""

import os
import sys
import time
import json
import math
import argparse
import numpy as np

# Keep CPU thread contention low for multi-GPU torchrun.
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


class Cfg:
    # Data
    coco_root     = f"{ROOT}/dataSets/coco"
    train_ann     = f"{coco_root}/annotations/instances_train2017.json"
    train_img_dir = f"{coco_root}/train2017"

    # Models
    gd_config  = (
        f"{ROOT}/GroundingDINO/GroundingDINO/groundingdino/config/"
        "GroundingDINO_SwinT_OGC.py"
    )
    gd_weights = (
        f"{ROOT}/GroundingDINO/GroundingDINO/weights/groundingdino_swint_ogc.pth"
    )
    internvl_path = f"{ROOT}/InternVL3_5-1B"

    # Layer setup (fixed to ablation winner)
    extract_layer  = 9
    extract_layers = [9]
    layer_fusion   = "mean"
    d_model        = 256
    tma_m          = 8
    tma_n_heads    = 8
    tma_alpha_init = 0.3
    # Safer default for baseline preservation experiments.
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
    lr_tma        = 2e-4
    weight_decay  = 0.01
    warmup_ratio  = 0.03
    max_grad_norm = 1.0
    num_workers   = 4
    debug_limit   = 0

    # Save
    output_dir    = f"{ROOT}/thinkdet/checkpoints/stage1_tma/layer9_tma_m8"
    log_interval  = 200
    save_interval = 2000
    grad_check_step = 100  # gradient health check after this many steps


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
    has_issues = False
    for name, param in model_raw.named_parameters():
        if param.requires_grad:
            if param.grad is None:
                print(f"    WARNING: {name} — no gradient!")
                has_issues = True
            elif param.grad.abs().max().item() == 0:
                print(f"    WARNING: {name} — zero gradient!")
                has_issues = True
            else:
                grad_norm = param.grad.norm().item()
                if "alpha" in name:
                    print(f"    OK: {name} — grad_norm={grad_norm:.6f}  value={param.item():.6f}")
                else:
                    print(f"    OK: {name} — grad_norm={grad_norm:.6f}")
        elif param.grad is not None:
            print(f"    WARNING: {name} — frozen but has gradient!")
            has_issues = True
    if not has_issues:
        print("    All augmenter params have healthy gradients.")
    print("  ─────────────────────────────────────────────\n")


def save_checkpoint(model, optimizer, scheduler, epoch, step, cfg, tag=""):
    trainable_state = {
        name: param.detach().cpu()
        for name, param in model.named_parameters()
        if param.requires_grad
    }
    path = os.path.join(cfg.output_dir, f"thinkdet_tma_stage1_{tag}.pth")
    torch.save(
        {
            "epoch": epoch + 1,
            "global_step": step,
            "stage": "tma_stage1",
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
    print(f"  [SAVE] {path}  ({len(trainable_state)} param tensors)")


def parse_args():
    cfg = Cfg()
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=cfg.epochs)
    parser.add_argument("--debug_limit", type=int, default=cfg.debug_limit)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=cfg.num_workers)
    parser.add_argument("--lr_tma", type=float, default=cfg.lr_tma)
    parser.add_argument("--tma_m", type=int, default=cfg.tma_m)
    parser.add_argument("--tma_alpha_init", type=float, default=cfg.tma_alpha_init)
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
    cfg.epochs = args.epochs
    cfg.debug_limit = args.debug_limit
    cfg.lr_tma = float(args.lr_tma)
    cfg.tma_m = int(args.tma_m)
    cfg.num_workers = int(args.num_workers)
    cfg.tma_alpha_init = float(args.tma_alpha_init)
    cfg.tma_fusion_mode = str(args.tma_fusion_mode)
    cfg.train_query_mode = str(args.train_query_mode)
    cfg.dynamic_query_ratio = float(args.dynamic_query_ratio)
    cfg.max_query_categories = int(args.max_query_categories)
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
        cfg.output_dir = f"{ROOT}/thinkdet/checkpoints/stage1_tma/{layer_tag}_tma_m{cfg.tma_m}"

    ddp, local_rank, device, world_size = init_distributed()
    is_main = local_rank == 0

    if is_main:
        os.makedirs(cfg.output_dir, exist_ok=True)
        print("=" * 70)
        print("  ThinkDet TMA — Stage 1: TextAugmenter Training")
        print("=" * 70)
        print(f"  extract_layer:    {cfg.extract_layer}")
        print(f"  extract_layers:   {cfg.extract_layers}")
        print(f"  layer_fusion:     {cfg.layer_fusion}")
        print(f"  injection_layers: {cfg.injection_layers}")
        print(f"  tma_m:            {cfg.tma_m}")
        print(f"  tma_n_heads:      {cfg.tma_n_heads}")
        print(f"  tma_alpha_init:   {cfg.tma_alpha_init}")
        print(f"  tma_fusion_mode:  {cfg.tma_fusion_mode}")
        print(f"  train_query_mode: {cfg.train_query_mode}")
        print(f"  dynamic_q_ratio:  {cfg.dynamic_query_ratio}")
        print(f"  lr_tma:           {cfg.lr_tma}")
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
    model.set_tma_only()
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

    if is_main:
        print("\n[3/4] Setting up optimizer ...")
    aug_params = model_raw.get_augmenter_params()
    if not aug_params:
        raise RuntimeError("No trainable augmenter parameters found.")
    if is_main:
        print(f"  TMA params: {sum(p.numel() for p in aug_params):,}")

    optimizer = torch.optim.AdamW(
        [{"params": aug_params, "lr": cfg.lr_tma, "name": "tma"}],
        weight_decay=cfg.weight_decay,
    )

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
                    scaler.update()
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
                    print(
                        f"  [E{epoch+1}/{cfg.epochs}] step {global_step}/{total_steps}  "
                        f"loss={avg:.4f}  h_vlm={hvlm:.2f}  "
                        f"Layer1_alpha={layer1_alpha:.4f}  "
                        f"Layer3_alpha={layer3_alpha:.4f}  "
                        f"Layer5_alpha={layer5_alpha:.4f}  "
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
                                "alpha_L1": float(layer1_alpha),  # backward-compatible field
                                "alpha_L3": float(layer3_alpha),  # backward-compatible field
                                "alpha_L5": float(layer5_alpha),  # backward-compatible field
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

        if ddp:
            dist.barrier()

    if is_main:
        total_time = time.time() - t0
        print(f"\n{'='*70}")
        print("  Stage 1 TMA training complete!")
        print(f"  Total time:  {total_time/3600:.2f}h")
        print(f"  Checkpoints: {cfg.output_dir}")
        print(f"{'='*70}")

    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
