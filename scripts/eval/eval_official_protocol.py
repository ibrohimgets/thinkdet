"""
ThinkDet — Official-Protocol COCO Evaluation (Multi-GPU)

Matches GroundingDINO's official test_ap_on_coco.py protocol:
  - NO confidence threshold pre-filtering
  - Top-300 detections per image (not 100)
  - All detections fed to pycocotools COCOeval
  - Same transforms: RandomResize([800], max_size=1333)

Reports: AP, AP50, AP75, AP_S, AP_M, AP_L, AR_1, AR_10, AR_100

Usage (7 GPUs — both baseline + TMA):
    torchrun --nproc_per_node=7 thinkdet/scripts/eval/eval_official_protocol.py \
        --mode both \
        --checkpoint /path/to/thinkdet_tma_stage1_epoch2.pth

Usage (7 GPUs — baseline only):
    torchrun --nproc_per_node=7 thinkdet/scripts/eval/eval_official_protocol.py \
        --mode baseline
"""

import argparse
from collections import defaultdict
import gc
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


def gather_predictions(predictions, world_size):
    if world_size == 1:
        return predictions
    gathered = [None] * world_size
    dist.all_gather_object(gathered, predictions)
    merged = []
    for p in gathered:
        merged.extend(p)
    return merged


@torch.no_grad()
def run_inference_official(model_fn, val_dataset, all_query_text, pmap_norm,
                           device, world_size, rank, num_select=300, max_images=0,
                           shortlist_top_cats=0, tokenizer=None, special_tokens=None,
                           num_workers=4, empty_cache_every=0):
    """
    Official-protocol inference: NO conf_thresh, top-K by score.
    Distributed across GPUs via DistributedSampler.
    Returns COCO-format predictions list (local shard).
    """
    eval_len = len(val_dataset) if int(max_images) <= 0 else min(len(val_dataset), int(max_images))
    eval_dataset = (
        torch.utils.data.Subset(val_dataset, list(range(eval_len)))
        if eval_len < len(val_dataset)
        else val_dataset
    )

    sampler = (
        torch.utils.data.distributed.DistributedSampler(
            eval_dataset, num_replicas=world_size, rank=rank, shuffle=False
        ) if world_size > 1 else None
    )

    loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=int(num_workers),
        collate_fn=collate_fn,
        pin_memory=True,
        sampler=sampler,
    )

    cat_names = sorted(val_dataset.cat_id_to_name.values())
    name_to_coco = {v: k for k, v in val_dataset.cat_id_to_name.items()}

    predictions = []
    processed_image_ids = []
    t0 = time.time()

    for i, batch in enumerate(loader):
        out_short = None
        image_id = int(batch["image_ids"][0])
        processed_image_ids.append(image_id)
        internvl_images = batch["internvl_images"].to(device)
        dino_nested = NestedTensor(
            batch["dino_images"].tensors.to(device),
            batch["dino_images"].mask.to(device),
        )
        B = internvl_images.shape[0]
        dino_inputs = {"samples": dino_nested, "captions": [all_query_text] * B}

        try:
            out = model_fn(internvl_images, batch["query_texts"], dino_inputs)
        except Exception as e:
            if i < 5 and rank == 0:
                print(f"  Warning: image {i} failed: {e}")
            continue

        if isinstance(out, tuple):
            outputs = out[0]
        else:
            outputs = out

        img_info = val_dataset.coco.loadImgs(image_id)[0]
        img_w, img_h = img_info["width"], img_info["height"]

        logits = outputs["pred_logits"][0].clamp(-50, 50)
        boxes  = outputs["pred_boxes"][0]

        # Positive map projection: logits -> category scores
        probs = logits.sigmoid()                    # [N_queries, 256]
        cat_scores = probs @ pmap_norm.T            # [N_queries, 80]

        active_outputs = outputs
        active_cat_names = cat_names
        active_pmap_norm = pmap_norm

        if (
            int(shortlist_top_cats) > 0
            and int(shortlist_top_cats) < len(cat_names)
            and tokenizer is not None
            and special_tokens is not None
        ):
            shortlist_k = min(int(shortlist_top_cats), cat_scores.shape[1])
            shortlist_idx = cat_scores.max(dim=0).values.topk(shortlist_k).indices.tolist()
            shortlist_names = [cat_names[idx] for idx in shortlist_idx]
            short_query_text, _, short_pmap_norm, _ = build_positive_map(
                tokenizer,
                special_tokens,
                shortlist_names,
            )
            short_pmap_norm = short_pmap_norm.to(device)
            short_dino_inputs = {"samples": dino_nested, "captions": [short_query_text] * B}
            try:
                out_short = model_fn(internvl_images, [short_query_text], short_dino_inputs)
            except Exception:
                out_short = None
            if out_short is not None:
                active_outputs = out_short[0] if isinstance(out_short, tuple) else out_short
                active_cat_names = shortlist_names
                active_pmap_norm = short_pmap_norm

        # Flatten and take top-K (same as official PostProcessCocoGrounding)
        logits = active_outputs["pred_logits"][0].clamp(-50, 50)
        boxes  = active_outputs["pred_boxes"][0]
        probs = logits.sigmoid()
        cat_scores = probs @ active_pmap_norm.T
        flat_scores = cat_scores.flatten()           # [N_queries * 80]
        topk_vals, topk_idx = flat_scores.topk(min(num_select, flat_scores.shape[0]))
        topk_box_idx = topk_idx // cat_scores.shape[1]
        topk_cat_idx = topk_idx % cat_scores.shape[1]

        sel_boxes = boxes[topk_box_idx]
        pred_xyxy = box_cxcywh_to_xyxy(sel_boxes)
        pred_xyxy[:, 0] *= img_w; pred_xyxy[:, 2] *= img_w
        pred_xyxy[:, 1] *= img_h; pred_xyxy[:, 3] *= img_h

        for j in range(topk_vals.shape[0]):
            ci = topk_cat_idx[j].item()
            if ci >= len(active_cat_names):
                continue
            cname = active_cat_names[ci]
            coco_id = name_to_coco.get(cname)
            if coco_id is None:
                continue
            x1, y1, x2, y2 = pred_xyxy[j].tolist()
            predictions.append({
                "image_id":    image_id,
                "category_id": coco_id,
                "bbox":        [round(x1, 2), round(y1, 2),
                                round(max(0, x2-x1), 2), round(max(0, y2-y1), 2)],
                "score":       round(float(topk_vals[j].item()), 5),
            })

        if rank == 0 and (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(loader) - i - 1)
            print(f"  [{i+1}/{len(loader)}] preds={len(predictions)}, "
                  f"time={elapsed:.0f}s, ETA={eta:.0f}s")

        del out, outputs, logits, boxes, probs, cat_scores, flat_scores
        del topk_vals, topk_idx, topk_box_idx, topk_cat_idx, sel_boxes, pred_xyxy
        del dino_inputs, dino_nested, internvl_images, batch
        if out_short is not None:
            del out_short
        if torch.cuda.is_available() and int(empty_cache_every) > 0 and ((i + 1) % int(empty_cache_every) == 0):
            torch.cuda.empty_cache()
            gc.collect()

    elapsed = time.time() - t0
    if rank == 0:
        print(f"  Done (rank 0): {len(loader)} images, {len(predictions)} local preds, {elapsed:.0f}s")
    return predictions, processed_image_ids


def bbox_xywh_to_xyxy(box):
    x, y, w, h = box
    return [float(x), float(y), float(x + w), float(y + h)]


def iou_xyxy(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-9)


def greedy_nms(detections, iou_thresh):
    ordered = sorted(detections, key=lambda d: float(d["score"]), reverse=True)
    kept = []
    kept_boxes = []
    for det in ordered:
        box = bbox_xywh_to_xyxy(det["bbox"])
        if any(iou_xyxy(box, prev_box) >= float(iou_thresh) for prev_box in kept_boxes):
            continue
        kept.append(det)
        kept_boxes.append(box)
    return kept


def fuse_prediction_lists(
    preds_base,
    preds_tma,
    num_select=300,
    match_iou=0.6,
    consensus_lambda=0.35,
    add_unmatched=False,
    unmatched_score_scale=0.5,
    nms_iou=0.7,
):
    by_image_base = defaultdict(list)
    by_image_tma = defaultdict(list)
    for pred in preds_base:
        by_image_base[int(pred["image_id"])].append(pred)
    for pred in preds_tma:
        by_image_tma[int(pred["image_id"])].append(pred)

    fused = []
    all_image_ids = sorted(set(by_image_base.keys()) | set(by_image_tma.keys()))

    for image_id in all_image_ids:
        base_list = sorted(by_image_base.get(image_id, []), key=lambda d: float(d["score"]), reverse=True)
        tma_list = sorted(by_image_tma.get(image_id, []), key=lambda d: float(d["score"]), reverse=True)
        used_tma = set()
        fused_img = []

        for base_det in base_list:
            best_idx = None
            best_iou = 0.0
            best_tma_score = -1.0
            base_box = bbox_xywh_to_xyxy(base_det["bbox"])

            for idx, tma_det in enumerate(tma_list):
                if idx in used_tma:
                    continue
                if int(tma_det["category_id"]) != int(base_det["category_id"]):
                    continue
                overlap = iou_xyxy(base_box, bbox_xywh_to_xyxy(tma_det["bbox"]))
                if overlap < float(match_iou):
                    continue
                tma_score = float(tma_det["score"])
                if overlap > best_iou or (abs(overlap - best_iou) < 1e-9 and tma_score > best_tma_score):
                    best_idx = idx
                    best_iou = overlap
                    best_tma_score = tma_score

            fused_det = dict(base_det)
            base_score = float(base_det["score"])
            if best_idx is not None:
                used_tma.add(best_idx)
                tma_det = tma_list[best_idx]
                tma_score = float(tma_det["score"])
                boost = float(consensus_lambda) * tma_score * best_iou * (1.0 - base_score)
                fused_det["score"] = round(min(1.0, base_score + boost), 5)
                fused_det["fusion_source"] = "base+tma"
                fused_det["tma_score"] = round(tma_score, 5)
                fused_det["match_iou"] = round(best_iou, 4)
            else:
                fused_det["fusion_source"] = "base"
            fused_img.append(fused_det)

        if add_unmatched:
            for idx, tma_det in enumerate(tma_list):
                if idx in used_tma:
                    continue
                det = dict(tma_det)
                det["score"] = round(min(1.0, float(tma_det["score"]) * float(unmatched_score_scale)), 5)
                det["fusion_source"] = "tma_only"
                fused_img.append(det)

        by_cat = defaultdict(list)
        for det in fused_img:
            by_cat[int(det["category_id"])].append(det)

        fused_img_nms = []
        for dets in by_cat.values():
            fused_img_nms.extend(greedy_nms(dets, nms_iou))

        fused_img_nms.sort(key=lambda d: float(d["score"]), reverse=True)
        fused.extend(fused_img_nms[: int(num_select)])

    return fused


def coco_eval(predictions, coco_gt, img_ids=None):
    from pycocotools.cocoeval import COCOeval
    if not predictions:
        print("  WARNING: No predictions!")
        return {"AP": 0.0, "AP_50": 0.0, "AP_75": 0.0,
                "AP_S": 0.0, "AP_M": 0.0, "AP_L": 0.0,
                "AR_1": 0.0, "AR_10": 0.0, "AR_100": 0.0}
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(predictions, tmp); tmp.close()
    coco_dt = coco_gt.loadRes(tmp.name)
    ev = COCOeval(coco_gt, coco_dt, "bbox")
    if img_ids:
        ev.params.imgIds = list(img_ids)
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    os.unlink(tmp.name)
    return {
        "AP":     round(float(ev.stats[0]), 4),
        "AP_50":  round(float(ev.stats[1]), 4),
        "AP_75":  round(float(ev.stats[2]), 4),
        "AP_S":   round(float(ev.stats[3]), 4),
        "AP_M":   round(float(ev.stats[4]), 4),
        "AP_L":   round(float(ev.stats[5]), 4),
        "AR_1":   round(float(ev.stats[6]), 4),
        "AR_10":  round(float(ev.stats[7]), 4),
        "AR_100": round(float(ev.stats[8]), 4),
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["baseline", "tma", "both"], default="both")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--output", type=str,
                   default=f"{ROOT}/thinkdet/results/eval/official_protocol_eval.json")
    p.add_argument("--dump_pred_prefix", type=str, default="",
                   help="If set, dump raw prediction lists as <prefix>_baseline|tma|fused.json")
    p.add_argument("--num_select", type=int, default=300)
    p.add_argument("--max_images", type=int, default=0,
                   help="If > 0, evaluate only the first N val images.")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--empty_cache_every", type=int, default=0,
                   help="If > 0, call torch.cuda.empty_cache() every N images.")
    p.add_argument("--extract_layer", type=int, default=9)
    p.add_argument("--tma_m", type=int, default=8)
    p.add_argument("--tma_n_heads", type=int, default=8)
    p.add_argument("--baseline_shortlist_top_cats", type=int, default=0)
    p.add_argument("--tma_shortlist_top_cats", type=int, default=0)
    p.add_argument("--eval_fused", action="store_true",
                   help="After baseline and TMA runs, evaluate a fused prediction set.")
    p.add_argument("--fuse_match_iou", type=float, default=0.6)
    p.add_argument("--fuse_consensus_lambda", type=float, default=0.35)
    p.add_argument("--fuse_add_unmatched", action="store_true")
    p.add_argument("--fuse_unmatched_score_scale", type=float, default=0.5)
    p.add_argument("--fuse_nms_iou", type=float, default=0.7)
    return p.parse_args()


def main():
    args = parse_args()
    rank, world_size, local_rank, device = init_distributed()
    is_main = rank == 0

    if is_main:
        print("=" * 60)
        print("  Official-Protocol COCO Evaluation")
        print("=" * 60)
        print(f"  Mode:       {args.mode}")
        print(f"  GPUs:       {world_size}")
        print(f"  num_select: {args.num_select} (top-K, NO conf threshold)")
        print(f"  num_workers:{args.num_workers}")
        if args.max_images > 0:
            print(f"  max_images: {args.max_images}")
        if args.empty_cache_every > 0:
            print(f"  empty_cache_every: {args.empty_cache_every}")
        if args.checkpoint:
            print(f"  checkpoint: {args.checkpoint}")
        if args.baseline_shortlist_top_cats > 0 or args.tma_shortlist_top_cats > 0:
            print(
                f"  shortlist: baseline={args.baseline_shortlist_top_cats} "
                f"tma={args.tma_shortlist_top_cats}"
            )
        if args.eval_fused:
            print(
                f"  fused:      True "
                f"(match_iou={args.fuse_match_iou}, lambda={args.fuse_consensus_lambda}, "
                f"add_unmatched={args.fuse_add_unmatched})"
            )
        print()

    # Load dataset
    val_ds = COCOGroundingDataset(img_dir=COCO_VAL_IMG, ann_file=COCO_VAL_ANN)
    all_cat_names = sorted(val_ds.cat_id_to_name.values())
    if is_main:
        print(f"  Dataset: {len(val_ds)} images, {len(all_cat_names)} categories")

    # Build positive map
    gd_tok_model = load_gd_model(GD_CONFIG, GD_WEIGHTS, device="cpu")
    gd_tok  = gd_tok_model.tokenizer
    gd_spec = gd_tok_model.specical_tokens
    all_query_text, _, pmap_norm, _ = build_positive_map(gd_tok, gd_spec, all_cat_names)
    pmap_norm = pmap_norm.to(device)
    del gd_tok_model

    result = {"num_select": args.num_select, "protocol": "official (no conf_thresh)",
              "world_size": world_size, "max_images": int(args.max_images)}
    eval_img_ids = None

    # ── BASELINE ──
    preds_base = None
    preds_tma = None
    metrics_base = None
    metrics_tma = None
    if args.mode in ("baseline", "both"):
        if is_main:
            print("\n[BASELINE] Running official-protocol evaluation...")
        gd_base = load_gd_model(GD_CONFIG, GD_WEIGHTS, device="cpu")
        gd_base = gd_base.to(device).eval()

        def baseline_fn(images, queries, dino_inputs):
            return gd_base(**dino_inputs), {}

        preds_base_local, base_img_ids_local = run_inference_official(
            baseline_fn, val_ds, all_query_text, pmap_norm,
            device, world_size, rank, num_select=args.num_select, max_images=args.max_images,
            shortlist_top_cats=args.baseline_shortlist_top_cats,
            tokenizer=gd_tok,
            special_tokens=gd_spec,
            num_workers=args.num_workers,
            empty_cache_every=args.empty_cache_every,
        )
        del gd_base
        torch.cuda.empty_cache()

        # Gather across GPUs
        preds_base = gather_predictions(preds_base_local, world_size)
        eval_img_ids = sorted(set(gather_predictions(base_img_ids_local, world_size)))

        if is_main:
            print(f"\n[BASELINE] Total predictions: {len(preds_base)}")
            if args.dump_pred_prefix:
                with open(f"{args.dump_pred_prefix}_baseline.json", "w") as f:
                    json.dump(preds_base, f)
            print("[BASELINE] COCO eval:")
            metrics_base = coco_eval(preds_base, val_ds.coco, img_ids=eval_img_ids)
            result["baseline"] = {
                "metrics": metrics_base,
                "num_predictions": len(preds_base),
                "shortlist_top_cats": int(args.baseline_shortlist_top_cats),
            }
            print(f"\n  >>> BASELINE AP = {metrics_base['AP']} <<<")
            print(f"  >>> BASELINE AP50 = {metrics_base['AP_50']} <<<\n")

        if world_size > 1:
            dist.barrier()

    # ── TMA ──
    if args.mode in ("tma", "both"):
        if not args.checkpoint:
            if is_main:
                print("ERROR: --checkpoint required for tma/both mode")
            return
        if is_main:
            print(f"\n[TMA] Running official-protocol evaluation...")
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        inject_layers = ckpt.get("injection_layers", DEFAULT_INJECTION_LAYERS)
        tma_m = ckpt.get("tma_m", args.tma_m)
        tma_n_heads = ckpt.get("tma_n_heads", args.tma_n_heads)
        extract_layers = ckpt.get("extract_layers") or [ckpt.get("extract_layer", args.extract_layer)]
        layer_fusion = ckpt.get("layer_fusion", "mean")
        fusion_mode = ckpt.get("fusion_mode", "concat")

        gd_tma = load_gd_model(GD_CONFIG, GD_WEIGHTS, device="cpu")
        model_tma = ThinkDetModel(
            grounding_dino=gd_tma,
            internvl_path=INTERNVL_PATH,
            extract_layer=max(extract_layers),
            extract_layers=extract_layers,
            layer_fusion=layer_fusion,
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
                f"  fusion_mode={fusion_mode}  layer_fusion={layer_fusion}"
            )
        model_tma = model_tma.to(device).eval()

        preds_tma_local, tma_img_ids_local = run_inference_official(
            model_tma, val_ds, all_query_text, pmap_norm,
            device, world_size, rank, num_select=args.num_select, max_images=args.max_images,
            shortlist_top_cats=args.tma_shortlist_top_cats,
            tokenizer=gd_tok,
            special_tokens=gd_spec,
            num_workers=args.num_workers,
            empty_cache_every=args.empty_cache_every,
        )
        del model_tma
        torch.cuda.empty_cache()

        # Gather across GPUs
        preds_tma = gather_predictions(preds_tma_local, world_size)
        if eval_img_ids is None:
            eval_img_ids = sorted(set(gather_predictions(tma_img_ids_local, world_size)))

        if is_main:
            print(f"\n[TMA] Total predictions: {len(preds_tma)}")
            if args.dump_pred_prefix:
                with open(f"{args.dump_pred_prefix}_tma.json", "w") as f:
                    json.dump(preds_tma, f)
            print("[TMA] COCO eval:")
            metrics_tma = coco_eval(preds_tma, val_ds.coco, img_ids=eval_img_ids)
            result["tma"] = {
                "checkpoint": args.checkpoint,
                "metrics": metrics_tma,
                "num_predictions": len(preds_tma),
                "shortlist_top_cats": int(args.tma_shortlist_top_cats),
            }
            print(f"\n  >>> TMA AP = {metrics_tma['AP']} <<<")
            print(f"  >>> TMA AP50 = {metrics_tma['AP_50']} <<<\n")

        if world_size > 1:
            dist.barrier()

    if args.eval_fused:
        if args.mode != "both":
            if is_main:
                print("ERROR: --eval_fused requires --mode both")
            return
        if is_main:
            print("\n[FUSED] Building consensus fusion predictions...")
            preds_fused = fuse_prediction_lists(
                preds_base,
                preds_tma,
                num_select=args.num_select,
                match_iou=args.fuse_match_iou,
                consensus_lambda=args.fuse_consensus_lambda,
                add_unmatched=args.fuse_add_unmatched,
                unmatched_score_scale=args.fuse_unmatched_score_scale,
                nms_iou=args.fuse_nms_iou,
            )
            print(f"[FUSED] Total predictions: {len(preds_fused)}")
            if args.dump_pred_prefix:
                with open(f"{args.dump_pred_prefix}_fused.json", "w") as f:
                    json.dump(preds_fused, f)
            print("[FUSED] COCO eval:")
            metrics_fused = coco_eval(preds_fused, val_ds.coco, img_ids=eval_img_ids)
            result["fused"] = {
                "metrics": metrics_fused,
                "num_predictions": len(preds_fused),
                "config": {
                    "match_iou": float(args.fuse_match_iou),
                    "consensus_lambda": float(args.fuse_consensus_lambda),
                    "add_unmatched": bool(args.fuse_add_unmatched),
                    "unmatched_score_scale": float(args.fuse_unmatched_score_scale),
                    "nms_iou": float(args.fuse_nms_iou),
                },
            }
            print(f"\n  >>> FUSED AP = {metrics_fused['AP']} <<<")
            print(f"  >>> FUSED AP50 = {metrics_fused['AP_50']} <<<\n")

    # ── Summary ──
    if is_main and args.mode == "both":
        print("=" * 60)
        print("  COMPARISON (Official Protocol)")
        print("=" * 60)
        if args.eval_fused and "fused" in result:
            print(f"  {'Metric':<15} {'Baseline':>10} {'TMA':>10} {'Fused':>10}")
            print(f"  {'-'*60}")
            for key in ["AP", "AP_50", "AP_75", "AP_S", "AP_M", "AP_L", "AR_100"]:
                print(
                    f"  {key:<15} "
                    f"{metrics_base[key]:>10.4f} "
                    f"{metrics_tma[key]:>10.4f} "
                    f"{metrics_fused[key]:>10.4f}"
                )
            print(
                f"  {'num_preds':<15} {len(preds_base):>10} {len(preds_tma):>10} "
                f"{len(preds_fused):>10}"
            )
        else:
            print(f"  {'Metric':<15} {'Baseline':>10} {'TMA':>10} {'Delta':>10}")
            print(f"  {'-'*50}")
            for key in ["AP", "AP_50", "AP_75", "AP_S", "AP_M", "AP_L", "AR_100"]:
                bv = metrics_base[key]
                tv = metrics_tma[key]
                d = round(tv - bv, 4)
                sign = "+" if d > 0 else ""
                print(f"  {key:<15} {bv:>10.4f} {tv:>10.4f} {sign}{d:>9.4f}")
            print(f"  {'num_preds':<15} {len(preds_base):>10} {len(preds_tma):>10} "
                  f"{len(preds_tma)-len(preds_base):>+10}")
        print("=" * 60)

    if is_main:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved: {args.output}")

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
