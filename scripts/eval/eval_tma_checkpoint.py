"""
ThinkDet TMA — COCO Evaluator

Runs baseline vs TMA checkpoint side-by-side and reports:
    mAP, mAP@50, mAP@10, recall, num_predictions, mean_logit_magnitude

Usage (single GPU):
    python thinkdet/scripts/eval/eval_tma_checkpoint.py \
        --checkpoint /path/to/thinkdet_tma_stage1_epoch1.pth \
        --output /path/to/result.json

Usage (multi-GPU):
    torchrun --nproc_per_node=8 thinkdet/scripts/eval/eval_tma_checkpoint.py \
        --checkpoint /path/to/thinkdet_tma_stage1_epoch1.pth \
        --output /path/to/result.json
"""

import argparse
import json
import os
import sys
import tempfile
import time

import torch
import torch.distributed as dist

ROOT = "/home/iibrohimm/project/next_step"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "GroundingDINO", "GroundingDINO"))

from thinkdet.models.arch import ThinkDetModel, DEFAULT_INJECTION_LAYERS
from thinkdet.data.coco_grounding import (
    COCOGroundingDataset,
    collate_fn,
    build_positive_map,
)
from thinkdet.scripts.training.train_stage1 import box_cxcywh_to_xyxy
from groundingdino.util.inference import load_model as load_gd_model
from groundingdino.util.misc import NestedTensor


GD_CONFIG = (
    f"{ROOT}/GroundingDINO/GroundingDINO/groundingdino/config/"
    "GroundingDINO_SwinT_OGC.py"
)
GD_WEIGHTS = (
    f"{ROOT}/GroundingDINO/GroundingDINO/weights/groundingdino_swint_ogc.pth"
)
INTERNVL_PATH = f"{ROOT}/InternVL3_5-1B"
COCO_VAL_IMG  = f"{ROOT}/dataSets/coco/val2017"
COCO_VAL_ANN  = f"{ROOT}/dataSets/coco/annotations/instances_val2017.json"


def init_distributed():
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda:0")
    return rank, world_size, local_rank, device


@torch.no_grad()
def run_inference(model_fn, val_dataset, all_query_text, pmap_norm,
                  conf_thresh, device, world_size, rank, max_dets=100):
    """
    Run inference on COCO val, returning:
        predictions (list of COCO-format dicts)
        logit_magnitudes (list of mean abs logit per image)
    """
    from thinkdet.data.coco_grounding import collate_fn

    loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
        sampler=torch.utils.data.distributed.DistributedSampler(
            val_dataset, num_replicas=world_size, rank=rank, shuffle=False
        ) if world_size > 1 else None,
    )

    idx_to_cat = {v: k for k, v in
                  {n: i for i, n in enumerate(sorted(val_dataset.cat_id_to_name.values()))}.items()}
    name_to_coco = {v: k for k, v in val_dataset.cat_id_to_name.items()}

    predictions = []
    logit_magnitudes = []

    for batch in loader:
        internvl_images = batch["internvl_images"].to(device)
        dino_nested = NestedTensor(
            batch["dino_images"].tensors.to(device),
            batch["dino_images"].mask.to(device),
        )
        B = internvl_images.shape[0]
        dino_inputs = {"samples": dino_nested, "captions": [all_query_text] * B}

        try:
            out = model_fn(internvl_images, batch["query_texts"], dino_inputs)
        except Exception:
            continue

        # model_fn returns (outputs, aux) for TMA, just outputs for baseline
        if isinstance(out, tuple):
            outputs = out[0]
        else:
            outputs = out

        image_id = batch["image_ids"][0]
        img_info = val_dataset.coco.loadImgs(image_id)[0]
        img_w, img_h = img_info["width"], img_info["height"]

        logits = outputs["pred_logits"][0].clamp(-50, 50)
        boxes  = outputs["pred_boxes"][0]

        # track logit magnitude
        logit_magnitudes.append(float(logits.abs().mean().item()))

        probs = logits.sigmoid()
        cat_scores = probs @ pmap_norm.T
        max_scores, max_cats = cat_scores.max(dim=-1)

        keep = max_scores > conf_thresh
        if not keep.any():
            continue

        kept_scores = max_scores[keep]
        kept_cats   = max_cats[keep]
        kept_boxes  = boxes[keep]

        if kept_scores.shape[0] > max_dets:
            idx = kept_scores.argsort(descending=True)[:max_dets]
            kept_scores = kept_scores[idx]
            kept_cats   = kept_cats[idx]
            kept_boxes  = kept_boxes[idx]

        pred_xyxy = box_cxcywh_to_xyxy(kept_boxes)
        pred_xyxy[:, 0] *= img_w; pred_xyxy[:, 2] *= img_w
        pred_xyxy[:, 1] *= img_h; pred_xyxy[:, 3] *= img_h

        cat_names = sorted(val_dataset.cat_id_to_name.values())
        for i in range(kept_scores.shape[0]):
            ci = kept_cats[i].item()
            if ci >= len(cat_names):
                continue
            cname = cat_names[ci]
            coco_id = name_to_coco.get(cname)
            if coco_id is None:
                continue
            x1, y1, x2, y2 = pred_xyxy[i].tolist()
            predictions.append({
                "image_id":    image_id,
                "category_id": coco_id,
                "bbox":        [round(x1, 2), round(y1, 2),
                                round(max(0, x2-x1), 2), round(max(0, y2-y1), 2)],
                "score":       round(float(kept_scores[i].item()), 4),
            })

    return predictions, logit_magnitudes


def gather_predictions(predictions, world_size):
    if world_size == 1:
        return predictions
    gathered = [None] * world_size
    dist.all_gather_object(gathered, predictions)
    merged = []
    for p in gathered:
        merged.extend(p)
    return merged


def gather_floats(vals, world_size):
    if world_size == 1:
        return vals
    gathered = [None] * world_size
    dist.all_gather_object(gathered, vals)
    merged = []
    for v in gathered:
        merged.extend(v)
    return merged


def coco_eval(predictions, coco_gt):
    from pycocotools.cocoeval import COCOeval
    if not predictions:
        return {"mAP": 0.0, "mAP_50": 0.0, "mAP_75": 0.0,
                "mAP_S": 0.0, "mAP_M": 0.0, "mAP_L": 0.0,
                "mAR_1": 0.0, "mAR_10": 0.0, "mAR_100": 0.0}
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(predictions, tmp); tmp.close()
    coco_dt = coco_gt.loadRes(tmp.name)
    ev = COCOeval(coco_gt, coco_dt, "bbox")
    ev.evaluate(); ev.accumulate(); ev.summarize()
    os.unlink(tmp.name)
    return {
        "mAP":    round(float(ev.stats[0]), 4),
        "mAP_50": round(float(ev.stats[1]), 4),
        "mAP_75": round(float(ev.stats[2]), 4),
        "mAP_S":  round(float(ev.stats[3]), 4),
        "mAP_M":  round(float(ev.stats[4]), 4),
        "mAP_L":  round(float(ev.stats[5]), 4),
        "mAR_1":  round(float(ev.stats[6]), 4),
        "mAR_10": round(float(ev.stats[7]), 4),
        "mAR_100":round(float(ev.stats[8]), 4),
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--conf_thresh", type=float, default=0.30)
    p.add_argument("--conf_thresh_low", type=float, default=0.10)
    p.add_argument("--extract_layer", type=int, default=9)
    p.add_argument("--tma_m", type=int, default=8)
    p.add_argument("--tma_n_heads", type=int, default=8)
    return p.parse_args()


def main():
    args = parse_args()
    rank, world_size, local_rank, device = init_distributed()
    is_main = rank == 0

    if is_main:
        print("=" * 60)
        print("  ThinkDet TMA — COCO Evaluation")
        print("=" * 60)
        print(f"  checkpoint:  {args.checkpoint}")
        print(f"  conf_thresh: {args.conf_thresh} / {args.conf_thresh_low}")
        print()

    # ── Load dataset ──
    val_ds = COCOGroundingDataset(img_dir=COCO_VAL_IMG, ann_file=COCO_VAL_ANN)
    all_cat_names = sorted(val_ds.cat_id_to_name.values())

    # Build pmap using a fresh GD model just for tokenizer
    gd_tok_model = load_gd_model(GD_CONFIG, GD_WEIGHTS, device="cpu")
    gd_tok  = gd_tok_model.tokenizer
    gd_spec = gd_tok_model.specical_tokens
    all_query_text, _, pmap_norm, _ = build_positive_map(gd_tok, gd_spec, all_cat_names)
    pmap_norm = pmap_norm.to(device)
    del gd_tok_model

    # ── BASELINE ──
    if is_main:
        print("[1/2] Running BASELINE ...")
    gd_base = load_gd_model(GD_CONFIG, GD_WEIGHTS, device="cpu")
    gd_base = gd_base.to(device).eval()

    def baseline_fn(images, queries, dino_inputs):
        return gd_base(**dino_inputs), {}

    preds_base_hi, lmags_base_hi = run_inference(
        baseline_fn, val_ds, all_query_text, pmap_norm,
        args.conf_thresh, device, world_size, rank,
    )
    preds_base_lo, _ = run_inference(
        baseline_fn, val_ds, all_query_text, pmap_norm,
        args.conf_thresh_low, device, world_size, rank,
    )
    del gd_base

    # ── TMA ──
    if is_main:
        print("[2/2] Running TMA checkpoint ...")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    inject_layers = ckpt.get("injection_layers", DEFAULT_INJECTION_LAYERS)
    tma_m = ckpt.get("tma_m", args.tma_m)
    tma_n_heads = ckpt.get("tma_n_heads", args.tma_n_heads)
    extract_layers = ckpt.get("extract_layers") or [ckpt.get("extract_layer", args.extract_layer)]
    fusion_mode = ckpt.get("fusion_mode", "concat")

    gd_tma = load_gd_model(GD_CONFIG, GD_WEIGHTS, device="cpu")
    model_tma = ThinkDetModel(
        grounding_dino=gd_tma,
        internvl_path=INTERNVL_PATH,
        extract_layer=max(extract_layers),
        extract_layers=extract_layers,
        injection_layers=inject_layers,
        tma_m=tma_m,
        tma_n_heads=tma_n_heads,
        fusion_mode=fusion_mode,
    )
    missing, unexpected = model_tma.load_state_dict(
        ckpt["trainable_state_dict"], strict=False
    )
    if is_main:
        print(
            f"  Missing keys: {len(missing)}  Unexpected: {len(unexpected)}"
            f"  fusion_mode={fusion_mode}"
        )

    model_tma = model_tma.to(device).eval()

    preds_tma_hi, lmags_tma_hi = run_inference(
        model_tma, val_ds, all_query_text, pmap_norm,
        args.conf_thresh, device, world_size, rank,
    )
    preds_tma_lo, _ = run_inference(
        model_tma, val_ds, all_query_text, pmap_norm,
        args.conf_thresh_low, device, world_size, rank,
    )

    # ── Gather across GPUs ──
    preds_base_hi = gather_predictions(preds_base_hi, world_size)
    preds_base_lo = gather_predictions(preds_base_lo, world_size)
    preds_tma_hi  = gather_predictions(preds_tma_hi, world_size)
    preds_tma_lo  = gather_predictions(preds_tma_lo, world_size)
    lmags_base = gather_floats(lmags_base_hi, world_size)
    lmags_tma  = gather_floats(lmags_tma_hi, world_size)

    if is_main:
        metrics_base_hi = coco_eval(preds_base_hi, val_ds.coco)
        metrics_base_lo = coco_eval(preds_base_lo, val_ds.coco)
        metrics_tma_hi  = coco_eval(preds_tma_hi,  val_ds.coco)
        metrics_tma_lo  = coco_eval(preds_tma_lo,  val_ds.coco)

        mean_lmag_base = round(sum(lmags_base) / max(len(lmags_base), 1), 4)
        mean_lmag_tma  = round(sum(lmags_tma)  / max(len(lmags_tma), 1), 4)

        def delta(a, b):
            return round(a - b, 4)

        print()
        print("=" * 60)
        print("  RESULTS")
        print("=" * 60)
        print(f"  {'Metric':<30} {'Baseline':>10} {'TMA':>10} {'Delta':>10}")
        print(f"  {'-'*60}")
        rows = [
            (f"mAP         @conf={args.conf_thresh}",
             metrics_base_hi["mAP"], metrics_tma_hi["mAP"]),
            (f"mAP_50      @conf={args.conf_thresh}",
             metrics_base_hi["mAP_50"], metrics_tma_hi["mAP_50"]),
            (f"mAP         @conf={args.conf_thresh_low}",
             metrics_base_lo["mAP"], metrics_tma_lo["mAP"]),
            (f"mAP_50      @conf={args.conf_thresh_low}",
             metrics_base_lo["mAP_50"], metrics_tma_lo["mAP_50"]),
            (f"mAR_100     @conf={args.conf_thresh}",
             metrics_base_hi["mAR_100"], metrics_tma_hi["mAR_100"]),
            (f"num_preds   @conf={args.conf_thresh}",
             len(preds_base_hi), len(preds_tma_hi)),
            (f"num_preds   @conf={args.conf_thresh_low}",
             len(preds_base_lo), len(preds_tma_lo)),
            (f"mean_logit_magnitude",
             mean_lmag_base, mean_lmag_tma),
        ]
        for label, bv, tv in rows:
            d = delta(tv, bv) if isinstance(tv, float) else tv - bv
            sign = "+" if d > 0 else ""
            print(f"  {label:<30} {bv:>10} {tv:>10} {sign}{d:>9}")

        print("=" * 60)

        result = {
            "checkpoint":   args.checkpoint,
            "conf_thresh":  args.conf_thresh,
            "conf_thresh_low": args.conf_thresh_low,
            "baseline": {
                f"conf{args.conf_thresh}": metrics_base_hi,
                f"conf{args.conf_thresh_low}": metrics_base_lo,
                "num_predictions_hi": len(preds_base_hi),
                "num_predictions_lo": len(preds_base_lo),
                "mean_logit_magnitude": mean_lmag_base,
            },
            "tma": {
                f"conf{args.conf_thresh}": metrics_tma_hi,
                f"conf{args.conf_thresh_low}": metrics_tma_lo,
                "num_predictions_hi": len(preds_tma_hi),
                "num_predictions_lo": len(preds_tma_lo),
                "mean_logit_magnitude": mean_lmag_tma,
            },
            "delta": {
                f"mAP_conf{args.conf_thresh}":
                    delta(metrics_tma_hi["mAP"], metrics_base_hi["mAP"]),
                f"mAP_conf{args.conf_thresh_low}":
                    delta(metrics_tma_lo["mAP"], metrics_base_lo["mAP"]),
                "num_predictions_hi":
                    len(preds_tma_hi) - len(preds_base_hi),
                "mean_logit_magnitude":
                    delta(mean_lmag_tma, mean_lmag_base),
            },
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n  Saved: {args.output}")

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
