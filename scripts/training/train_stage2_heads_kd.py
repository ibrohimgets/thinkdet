"""
ThinkDet v2 — Stage B (Head Recalibration) with Logit Distillation

Goal:
    Recalibrate GroundingDINO detection heads to adapter-shifted features
    without changing backbone/encoder/MLLM.

Protocol:
    - Load Stage-1 adapter checkpoint.
    - Keep adapters fixed (default) at moderate gate value.
    - Unfreeze detection heads only.
    - Train with detection loss + small logit MSE distillation to baseline teacher.

Launch:
    torchrun --nproc_per_node=8 thinkdet/scripts/training/train_stage2_heads_kd.py
"""

import os
import sys
import time
import json
import math
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

ROOT = "/home/iibrohimm/project/next_step"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "GroundingDINO", "GroundingDINO"))

from thinkdet.models.arch import ThinkDetModel
from thinkdet.data.coco_grounding import (
    COCOGroundingDataset,
    collate_fn,
    build_positive_map,
)
from groundingdino.util.inference import load_model as load_gd_model
from groundingdino.util.misc import NestedTensor
from thinkdet.scripts.training.train_stage1 import (
    ALLOWED_LAYER_SETUPS,
    DEFAULT_LAYER_SETUP,
    compute_detection_loss,
    get_cosine_schedule_with_warmup,
)


class Cfg:
    # Data
    coco_root = f"{ROOT}/dataSets/coco"
    train_ann = f"{coco_root}/annotations/instances_train2017.json"
    train_img_dir = f"{coco_root}/train2017"

    # Models
    gd_config = (
        f"{ROOT}/GroundingDINO/GroundingDINO/groundingdino/config/"
        "GroundingDINO_SwinT_OGC.py"
    )
    gd_weights = f"{ROOT}/GroundingDINO/GroundingDINO/weights/groundingdino_swint_ogc.pth"
    internvl_path = f"{ROOT}/InternVL3_5-1B"
    stage1_ckpt = (
        f"{ROOT}/thinkdet/checkpoints/stage1/layer9_e12_fix/thinkdet_stage1_epoch12.pth"
    )

    # Layer setup (restricted)
    layer_setup = DEFAULT_LAYER_SETUP
    extract_layers = ALLOWED_LAYER_SETUPS[DEFAULT_LAYER_SETUP]
    extract_layer = extract_layers[-1]
    layer_fusion = "mean"
    d_model = 256
    n_heads = 8

    # Stage-B settings
    freeze_adapters = True
    gate_value = 0.25
    kd_weight = 0.10
    delta_gain = 1.0

    # Training
    epochs = 4
    per_gpu_batch = 1
    grad_accum = 8
    lr_heads = 5e-5
    weight_decay = 0.01
    warmup_ratio = 0.03
    max_grad_norm = 1.0
    num_workers = 4
    debug_limit = 0

    # Save
    output_dir = f"{ROOT}/thinkdet/checkpoints/stage2/heads_kd"
    log_interval = 50
    save_interval = 2000


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
        device = torch.device("cuda:0")
        world_size = 1
    return ddp, local_rank, device, world_size


def load_stage1_weights(student, ckpt_path, is_main):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Stage-1 checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    missing, unexpected = student.load_state_dict(ckpt["trainable_state_dict"], strict=False)
    if is_main:
        print(f"  Loaded Stage-1 weights: {ckpt_path}")
        print(f"  Missing keys: {len(missing)}")
        print(f"  Unexpected keys: {len(unexpected)}")


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch,
    step,
    cfg,
    adapter_param_ids,
    head_param_ids,
    tag="",
):
    keep_ids = adapter_param_ids.union(head_param_ids)
    trainable_state = {}
    for name, param in model.named_parameters():
        if id(param) in keep_ids:
            trainable_state[name] = param.detach().cpu()

    path = os.path.join(cfg.output_dir, f"thinkdet_stage2_{tag}.pth")
    torch.save(
        {
            "epoch": epoch + 1,
            "global_step": step,
            "stage": "stage_b_heads_kd",
            "stage1_checkpoint": cfg.stage1_ckpt,
            "layer_setup": cfg.layer_setup,
            "extract_layer": cfg.extract_layer,
            "extract_layers": cfg.extract_layers,
            "layer_fusion": cfg.layer_fusion,
            "freeze_adapters": cfg.freeze_adapters,
            "gate_value": cfg.gate_value,
            "kd_weight": cfg.kd_weight,
            "delta_gain": cfg.delta_gain,
            "trainable_state_dict": trainable_state,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        },
        path,
    )
    print(f"  [SAVE] {path}  ({len(trainable_state)} param tensors)")


def freeze_all_adapters(model):
    for layer in model.adapted_layers:
        for param in layer.adapter.parameters():
            param.requires_grad = False


def collect_param_groups(model):
    adapter_params = []
    head_params = []
    for name, param in model.named_parameters():
        if "adapter" in name or "adapted_layers" in name:
            adapter_params.append(param)
        if param.requires_grad:
            if "adapter" in name or "adapted_layers" in name:
                adapter_params.append(param)
            else:
                head_params.append(param)
    adapter_param_ids = {id(p) for p in model.get_adapter_params()}
    head_param_ids = {id(p) for p in head_params}
    return adapter_params, head_params, adapter_param_ids, head_param_ids


def parse_args():
    cfg = Cfg()
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1_ckpt", type=str, default=cfg.stage1_ckpt)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=cfg.epochs)
    parser.add_argument(
        "--layer_setup",
        type=str,
        default=cfg.layer_setup,
        choices=sorted(ALLOWED_LAYER_SETUPS.keys()),
    )
    parser.add_argument("--kd_weight", type=float, default=cfg.kd_weight)
    parser.add_argument("--gate_value", type=float, default=cfg.gate_value)
    parser.add_argument("--delta_gain", type=float, default=cfg.delta_gain)
    parser.add_argument("--lr_heads", type=float, default=cfg.lr_heads)
    parser.add_argument("--debug_limit", type=int, default=cfg.debug_limit)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--freeze_adapters", action="store_true", default=True)
    parser.add_argument("--disable_freeze_adapters", action="store_true")
    return parser.parse_args()


def main():
    cfg = Cfg()
    args = parse_args()
    cfg.stage1_ckpt = args.stage1_ckpt
    cfg.epochs = args.epochs
    cfg.layer_setup = args.layer_setup
    cfg.extract_layers = list(ALLOWED_LAYER_SETUPS[cfg.layer_setup])
    cfg.extract_layer = cfg.extract_layers[-1]
    cfg.kd_weight = float(args.kd_weight)
    cfg.gate_value = float(args.gate_value)
    cfg.delta_gain = float(args.delta_gain)
    cfg.lr_heads = float(args.lr_heads)
    cfg.debug_limit = int(args.debug_limit)
    cfg.freeze_adapters = bool(args.freeze_adapters) and not bool(args.disable_freeze_adapters)
    if args.output_dir:
        cfg.output_dir = args.output_dir
    else:
        suffix = f"{cfg.layer_setup}_gate{str(cfg.gate_value).replace('.', 'p')}_kd{str(cfg.kd_weight).replace('.', 'p')}"
        cfg.output_dir = os.path.join(cfg.output_dir, suffix)

    ddp, local_rank, device, world_size = init_distributed()
    is_main = local_rank == 0

    if is_main:
        os.makedirs(cfg.output_dir, exist_ok=True)
        print("=" * 72)
        print("  ThinkDet v2 — Stage B (Heads + KD)")
        print("=" * 72)
        print(f"  layer_setup:      {cfg.layer_setup}")
        print(f"  extract_layers:   {cfg.extract_layers} (fusion={cfg.layer_fusion})")
        print(f"  Stage-1 ckpt:     {cfg.stage1_ckpt}")
        print(f"  freeze adapters:  {cfg.freeze_adapters}")
        print(f"  fixed gate value: {cfg.gate_value}")
        print(f"  kd weight:        {cfg.kd_weight}")
        print(f"  delta gain:       {cfg.delta_gain}")
        print(f"  lr heads:         {cfg.lr_heads}")
        print(f"  GPUs:             {world_size}")
        print(f"  Per-GPU batch:    {cfg.per_gpu_batch}")
        print(f"  Grad accum:       {cfg.grad_accum}")
        print(f"  Effective batch:  {cfg.per_gpu_batch * cfg.grad_accum * world_size}")
        print(f"  Epochs:           {cfg.epochs}")
        print(f"  Output:           {cfg.output_dir}")
        print()

    if is_main:
        print("[1/5] Loading GroundingDINO (student) ...")
    gd_student = load_gd_model(cfg.gd_config, cfg.gd_weights, device="cpu")
    if is_main:
        print("[1/5] Building ThinkDet student ...")
    student = ThinkDetModel(
        grounding_dino=gd_student,
        internvl_path=cfg.internvl_path,
        extract_layer=cfg.extract_layer,
        extract_layers=cfg.extract_layers,
        layer_fusion=cfg.layer_fusion,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        delta_gain=cfg.delta_gain,
        uncertainty_enabled=False,
        injection_layers=None,
    )
    student.set_stage_b()
    load_stage1_weights(student, cfg.stage1_ckpt, is_main=is_main)

    if cfg.freeze_adapters:
        freeze_all_adapters(student)
        student.freeze_gate(value=cfg.gate_value)
        if is_main:
            print("  [StageB] Adapters frozen; gates fixed.")

    student = student.to(device)

    if ddp:
        student = DDP(student, device_ids=[local_rank], find_unused_parameters=True)
    student_raw = student.module if ddp else student

    if is_main:
        print("[1/5] Loading GroundingDINO (teacher baseline) ...")
    teacher = load_gd_model(cfg.gd_config, cfg.gd_weights, device="cpu")
    teacher = teacher.to(device)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    total_p = sum(p.numel() for p in student_raw.parameters())
    train_p = sum(p.numel() for p in student_raw.parameters() if p.requires_grad)
    if is_main:
        print(f"  Total params:     {total_p:,}")
        print(f"  Trainable params: {train_p:,} ({100 * train_p / total_p:.2f}%)")

    if is_main:
        print("\n[2/5] Loading COCO train2017 ...")
    train_ds = COCOGroundingDataset(img_dir=cfg.train_img_dir, ann_file=cfg.train_ann)
    if cfg.debug_limit > 0:
        if is_main:
            print(f"  [DEBUG] Limiting dataset to {cfg.debug_limit} images")
        train_ds.img_ids = train_ds.img_ids[: cfg.debug_limit]
        cfg.log_interval = 10

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
    gd_tok = student_raw.grounding_dino.tokenizer
    gd_spec = student_raw.grounding_dino.specical_tokens
    all_query_text, pmap, pmap_norm, cat_name_to_idx = build_positive_map(
        gd_tok, gd_spec, all_cat_names
    )
    pmap = pmap.to(device)
    pmap_norm = pmap_norm.to(device)

    if is_main:
        print(f"  Categories: {len(all_cat_names)}")
        print(f"  Query: '{all_query_text[:80]}...'")

    if is_main:
        print("\n[3/5] Setting up optimizer ...")
    _, head_params, adapter_param_ids, head_param_ids = collect_param_groups(student_raw)
    if len(head_params) == 0:
        raise RuntimeError("No trainable head parameters found for Stage-B training.")
    if is_main:
        print(f"  Head params: {sum(p.numel() for p in head_params):,}")
        print(f"  Adapter params tracked for save: {len(adapter_param_ids):,}")

    optimizer = torch.optim.AdamW(
        [{"params": head_params, "lr": cfg.lr_heads, "name": "heads"}],
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

    start_epoch = 0
    global_step = 0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location="cpu")
        student_raw.load_state_dict(ckpt["trainable_state_dict"], strict=False)
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt.get("epoch", 0)
        global_step = ckpt.get("global_step", 0)
        if is_main:
            print(f"  Resumed at epoch {start_epoch}, step {global_step}")

    if is_main:
        print("\n[4/5] Training ...\n")

    log_path = os.path.join(cfg.output_dir, "train_log.jsonl")
    delta_log_path = os.path.join(cfg.output_dir, "delta_diag_first100.jsonl")
    delta_diag_count = 0
    delta_prompt_vals = []
    delta_norm_vals = []
    delta_ratio_vals = []
    if is_main and not args.resume:
        with open(log_path, "w"):
            pass
        with open(delta_log_path, "w"):
            pass
        print(f"  [log] Reset {log_path}")
        print(f"  [log] Reset {delta_log_path}")

    t0 = time.time()
    for epoch in range(start_epoch, cfg.epochs):
        if ddp and sampler:
            sampler.set_epoch(epoch)

        student.train()
        teacher.eval()
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

            with torch.no_grad():
                teacher_outputs = teacher(**dino_inputs)

            try:
                outputs, aux = student(internvl_images, batch["query_texts"], dino_inputs)
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    if is_main:
                        print(f"  [OOM] step {global_step}, skipping")
                    torch.cuda.empty_cache()
                    optimizer.zero_grad()
                    continue
                raise

            if is_main and delta_diag_count < 100:
                hp = float(aux.get("h_prompt_fro_norm_mean", float("nan")))
                hd = float(aux.get("h_delta_fro_norm_mean", float("nan")))
                hr = float(aux.get("h_delta_over_prompt_ratio_mean", float("nan")))
                hnz = float(aux.get("h_delta_nonzero_frac", float("nan")))
                delta_diag_count += 1
                delta_prompt_vals.append(hp)
                delta_norm_vals.append(hd)
                delta_ratio_vals.append(hr)
                print(
                    f"  [DeltaDiag {delta_diag_count:03d}/100] "
                    f"H_prompt_F={hp:.6f} "
                    f"H_delta_F={hd:.6f} "
                    f"delta/raw_ratio={hr:.6f} "
                    f"delta_nonzero_frac={hnz:.6f}"
                )
                with open(delta_log_path, "a") as f:
                    f.write(
                        json.dumps(
                            {
                                "epoch": epoch + 1,
                                "batch_idx": batch_idx + 1,
                                "h_prompt_fro_norm_mean": hp,
                                "h_delta_fro_norm_mean": hd,
                                "h_delta_over_prompt_ratio_mean": hr,
                                "h_delta_nonzero_frac": hnz,
                            }
                        )
                        + "\n"
                    )

            det_loss = compute_detection_loss(
                outputs["pred_logits"],
                outputs["pred_boxes"],
                batch["boxes"],
                batch["category_names"],
                cat_name_to_idx,
                pmap,
                pmap_norm,
                device,
            )
            kd_loss = F.mse_loss(
                outputs["pred_logits"].clamp(-50, 50),
                teacher_outputs["pred_logits"].detach().clamp(-50, 50),
            )
            loss = det_loss + cfg.kd_weight * kd_loss

            if torch.isnan(loss) or torch.isinf(loss):
                if is_main:
                    print(f"  [NaN/Inf] step {global_step}, skipping")
                optimizer.zero_grad()
                continue

            (loss / cfg.grad_accum).backward()
            epoch_losses.append(loss.item())

            if (batch_idx + 1) % cfg.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in student_raw.parameters() if p.requires_grad],
                    cfg.max_grad_norm,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if is_main and global_step % cfg.log_interval == 0:
                    gate_tanh = [
                        round(torch.tanh(layer.adapter.gate).item(), 4)
                        for layer in student_raw.adapted_layers
                    ]
                    avg = np.mean(epoch_losses[-cfg.log_interval * cfg.grad_accum :])
                    lr_now = scheduler.get_last_lr()[0]
                    elapsed = time.time() - t0
                    done = max(global_step - (start_epoch * steps_per_epoch), 1)
                    eta = elapsed / done * (total_steps - global_step)
                    print(f"  [Gate] tanh={gate_tanh}")
                    print(
                        f"  [E{epoch+1}/{cfg.epochs}] step {global_step}/{total_steps}  "
                        f"loss={avg:.4f} det={det_loss.item():.4f} "
                        f"kd={kd_loss.item():.4f} lr={lr_now:.2e} ETA={eta/3600:.1f}h"
                    )
                    with open(log_path, "a") as f:
                        f.write(
                            json.dumps(
                                {
                                    "step": global_step,
                                    "epoch": epoch + 1,
                                    "loss": float(avg),
                                    "det_loss": float(det_loss.item()),
                                    "kd_loss": float(kd_loss.item()),
                                    "lr": float(lr_now),
                                }
                            )
                            + "\n"
                        )

                if is_main and global_step % cfg.save_interval == 0:
                    save_checkpoint(
                        student_raw,
                        optimizer,
                        scheduler,
                        epoch,
                        global_step,
                        cfg,
                        adapter_param_ids,
                        head_param_ids,
                        tag=f"step{global_step}",
                    )

        epoch_avg = np.mean(epoch_losses) if epoch_losses else 0.0
        if is_main:
            elapsed = time.time() - t0
            print("\n" + "=" * 60)
            print(
                f"  Epoch {epoch+1}/{cfg.epochs} "
                f"loss={epoch_avg:.4f} time={elapsed/3600:.2f}h"
            )
            if delta_prompt_vals:
                p_arr = np.array(delta_prompt_vals, dtype=np.float64)
                d_arr = np.array(delta_norm_vals, dtype=np.float64)
                r_arr = np.array(delta_ratio_vals, dtype=np.float64)
                print(
                    f"  [DeltaDiag Summary] n={len(delta_prompt_vals)} "
                    f"H_prompt_F mean/std={p_arr.mean():.6f}/{p_arr.std():.6f} "
                    f"H_delta_F mean/std={d_arr.mean():.6f}/{d_arr.std():.6f} "
                    f"delta/raw_ratio mean/std={r_arr.mean():.6f}/{r_arr.std():.6f}"
                )
            print("=" * 60)
            save_checkpoint(
                student_raw,
                optimizer,
                scheduler,
                epoch,
                global_step,
                cfg,
                adapter_param_ids,
                head_param_ids,
                tag=f"epoch{epoch+1}",
            )

        if ddp:
            dist.barrier()

    if is_main:
        total_time = time.time() - t0
        print("\n" + "=" * 72)
        print("  Stage-B training complete!")
        print(f"  Total time:  {total_time/3600:.2f}h")
        print(f"  Checkpoints: {cfg.output_dir}")
        print("=" * 72)

    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
