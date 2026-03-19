#!/usr/bin/env python3
"""
Official ThinkDet Stage-1 COCO evaluator.

Key guarantees:
- Runs exactly one mode per process: baseline / trained / trained_gate0.
- Captures and reports all runtime errors (no silent skip).
- Fails the run by default if any runtime errors occur.
- Works with torchrun multi-GPU sharding.

Example:
  torchrun --nproc_per_node=8 thinkdet/scripts/eval/eval_stage1_checkpoint.py \
    --mode trained \
    --checkpoint /path/to/thinkdet_stage1_epoch12.pth \
    --output /path/to/results.json
"""

import argparse
import json
import math
import os
import sys
import tempfile
import time
from collections import Counter
from typing import Dict, List, Tuple

import torch
import torch.distributed as dist

ROOT = "/home/iibrohimm/project/next_step"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "GroundingDINO", "GroundingDINO"))

from thinkdet.scripts.training.train_stage1 import (  # noqa: E402
    ALLOWED_LAYER_SETUPS,
    Cfg,
    box_cxcywh_to_xyxy,
)
from thinkdet.models.arch import ThinkDetModel  # noqa: E402
from thinkdet.data.coco_grounding import (  # noqa: E402
    COCOGroundingDataset,
    build_positive_map,
    collate_fn,
)
from groundingdino.util.inference import load_model as load_gd_model  # noqa: E402
from groundingdino.util.misc import NestedTensor  # noqa: E402


def init_distributed() -> Tuple[int, int, int, torch.device]:
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for this evaluator.")
        device = torch.device("cuda:0")
    return rank, world_size, local_rank, device


def cleanup_distributed():
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def target_to_raw_gate(value: float) -> float:
    if value >= 0.99:
        return 5.0
    if value <= -0.99:
        return -5.0
    return math.atanh(value)


def resolve_layer_setup(args, checkpoint_meta: Dict) -> str:
    if args.layer_setup:
        return args.layer_setup
    ck_setup = checkpoint_meta.get("layer_setup")
    if ck_setup in ALLOWED_LAYER_SETUPS:
        return ck_setup
    return "layer9"


def build_model(
    cfg: Cfg,
    mode: str,
    layer_setup: str,
    checkpoint_path: str,
    gate_override: float,
    device: torch.device,
):
    extract_layers = list(ALLOWED_LAYER_SETUPS[layer_setup])
    extract_layer = extract_layers[-1]
    baseline = mode == "baseline"
    ckpt = None
    delta_gain = 1.0
    checkpoint_meta = {}
    if not baseline:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        delta_gain = float(ckpt.get("delta_gain", 1.0))
        checkpoint_meta = {
            "epoch": ckpt.get("epoch"),
            "global_step": ckpt.get("global_step"),
            "layer_setup": ckpt.get("layer_setup"),
            "extract_layers": ckpt.get("extract_layers"),
            "layer_fusion": ckpt.get("layer_fusion"),
            "delta_gain": delta_gain,
        }

    gd = load_gd_model(cfg.gd_config, cfg.gd_weights, device="cpu")
    model = ThinkDetModel(
        grounding_dino=gd,
        internvl_path=cfg.internvl_path,
        extract_layer=extract_layer,
        extract_layers=extract_layers,
        layer_fusion="mean",
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        delta_gain=delta_gain,
        uncertainty_enabled=False,
        injection_layers=[] if baseline else None,
    )
    model.set_stage_a()
    if baseline:
        model._unfreeze_detection_heads()

    missing = []
    unexpected = []
    if not baseline:
        missing, unexpected = model.load_state_dict(
            ckpt["trainable_state_dict"], strict=False
        )

        if mode == "trained_gate0":
            raw_gate = target_to_raw_gate(gate_override)
            for layer in model.adapted_layers:
                layer.adapter.gate.data.fill_(raw_gate)

    model = model.to(device)
    model.eval()

    load_info = {
        "missing_keys_count": len(missing),
        "unexpected_keys_count": len(unexpected),
        "missing_keys_head": missing[:20],
        "unexpected_keys_head": unexpected[:20],
    }
    return model, extract_layers, checkpoint_meta, load_info


@torch.no_grad()
def run_local_shard(
    model,
    shard_dataset,
    positive_map_norm: torch.Tensor,
    cat_name_to_idx: Dict[str, int],
    all_query_text: str,
    device: torch.device,
    conf_thresh: float,
    max_dets: int,
    num_workers: int,
):
    loader = torch.utils.data.DataLoader(
        shard_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    idx_to_cat_name = {v: k for k, v in cat_name_to_idx.items()}
    name_to_coco_id = {v: k for k, v in shard_dataset.cat_id_to_name.items()}

    predictions = []
    processed = 0
    error_count = 0
    error_types = Counter()
    sample_errors = []
    t0 = time.time()

    for batch in loader:
        image_id = batch["image_ids"][0]
        try:
            internvl_images = batch["internvl_images"].to(device, non_blocking=True)
            dino_nested = NestedTensor(
                batch["dino_images"].tensors.to(device, non_blocking=True),
                batch["dino_images"].mask.to(device, non_blocking=True),
            )
            dino_inputs = {"samples": dino_nested, "captions": [all_query_text]}
            outputs, _ = model(internvl_images, batch["query_texts"], dino_inputs)
        except Exception as exc:
            error_count += 1
            error_type = type(exc).__name__
            error_types[error_type] += 1
            if len(sample_errors) < 10:
                sample_errors.append(
                    {
                        "image_id": int(image_id),
                        "error_type": error_type,
                        "message": str(exc).split("\n")[0][:300],
                    }
                )
            if "out of memory" in str(exc).lower() and torch.cuda.is_available():
                torch.cuda.empty_cache()
            processed += 1
            continue

        img_info = shard_dataset.coco.loadImgs(image_id)[0]
        img_w, img_h = img_info["width"], img_info["height"]

        logits = outputs["pred_logits"][0].clamp(-50, 50)
        boxes = outputs["pred_boxes"][0]
        probs = logits.sigmoid()

        cat_scores = probs @ positive_map_norm.T
        max_scores, max_cats = cat_scores.max(dim=-1)

        keep = max_scores > conf_thresh
        if keep.sum() > 0:
            kept_scores = max_scores[keep]
            kept_cats = max_cats[keep]
            kept_boxes = boxes[keep]

            if kept_scores.shape[0] > max_dets:
                topk_idx = kept_scores.argsort(descending=True)[:max_dets]
                kept_scores = kept_scores[topk_idx]
                kept_cats = kept_cats[topk_idx]
                kept_boxes = kept_boxes[topk_idx]

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
                predictions.append(
                    {
                        "image_id": int(image_id),
                        "category_id": int(coco_cat_id),
                        "bbox": [
                            round(x1, 2),
                            round(y1, 2),
                            round(max(0.0, x2 - x1), 2),
                            round(max(0.0, y2 - y1), 2),
                        ],
                        "score": round(kept_scores[i].item(), 4),
                    }
                )

        processed += 1

    elapsed = time.time() - t0
    local_runtime = {
        "processed_images": processed,
        "error_count": error_count,
        "error_types": dict(error_types),
        "sample_errors": sample_errors,
        "elapsed_sec": elapsed,
        "num_predictions": len(predictions),
    }
    return predictions, local_runtime


def coco_eval_from_predictions(coco_gt, predictions, num_images: int, eval_time_min: float):
    from pycocotools.cocoeval import COCOeval

    if len(predictions) == 0:
        return {
            "mAP": 0.0,
            "mAP_50": 0.0,
            "mAP_75": 0.0,
            "mAP_S": 0.0,
            "mAP_M": 0.0,
            "mAP_L": 0.0,
            "mAR_1": 0.0,
            "mAR_10": 0.0,
            "mAR_100": 0.0,
            "num_predictions": 0,
            "num_images": int(num_images),
            "eval_time_min": float(eval_time_min),
        }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
        json.dump(predictions, tmp)
        tmp_path = tmp.name

    coco_dt = coco_gt.loadRes(tmp_path)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    os.unlink(tmp_path)

    return {
        "mAP": float(coco_eval.stats[0]),
        "mAP_50": float(coco_eval.stats[1]),
        "mAP_75": float(coco_eval.stats[2]),
        "mAP_S": float(coco_eval.stats[3]),
        "mAP_M": float(coco_eval.stats[4]),
        "mAP_L": float(coco_eval.stats[5]),
        "mAR_1": float(coco_eval.stats[6]),
        "mAR_10": float(coco_eval.stats[7]),
        "mAR_100": float(coco_eval.stats[8]),
        "num_predictions": len(predictions),
        "num_images": int(num_images),
        "eval_time_min": float(eval_time_min),
    }


def merge_error_types(counter_dicts: List[Dict[str, int]]) -> Dict[str, int]:
    merged = Counter()
    for item in counter_dicts:
        merged.update(item)
    return dict(merged)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["baseline", "trained", "trained_gate0"],
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Required for trained/trained_gate0 mode.",
    )
    parser.add_argument(
        "--layer_setup",
        type=str,
        default=None,
        choices=sorted(ALLOWED_LAYER_SETUPS.keys()),
        help="Override layer setup. If omitted, uses checkpoint metadata or default layer9.",
    )
    parser.add_argument("--conf_thresh", type=float, default=0.3)
    parser.add_argument("--max_dets", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument(
        "--max_images",
        type=int,
        default=0,
        help="Optional eval subset size for debugging (0 = full val split).",
    )
    parser.add_argument(
        "--gate_override",
        type=float,
        default=0.0,
        help="Post-tanh gate value used for trained_gate0 mode.",
    )
    parser.add_argument(
        "--max_allowed_errors",
        type=int,
        default=0,
        help="Evaluator exits non-zero if runtime errors exceed this threshold.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path. Defaults to thinkdet/results/eval/...",
    )
    parser.add_argument(
        "--tmp_dir",
        type=str,
        default=None,
        help="Directory for per-rank temporary artifacts.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.mode != "baseline" and not args.checkpoint:
        raise ValueError("--checkpoint is required for trained/trained_gate0 mode.")
    if args.checkpoint and not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    rank, world_size, _, device = init_distributed()
    is_main = rank == 0

    cfg = Cfg()
    checkpoint_meta = {}
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        checkpoint_meta = {
            "epoch": ckpt.get("epoch"),
            "global_step": ckpt.get("global_step"),
            "layer_setup": ckpt.get("layer_setup"),
            "extract_layers": ckpt.get("extract_layers"),
            "layer_fusion": ckpt.get("layer_fusion"),
        }

    layer_setup = resolve_layer_setup(args, checkpoint_meta)

    if is_main:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        default_name = (
            f"stage1_{layer_setup}_{args.mode}_coco_val_w{world_size}_{timestamp}.json"
        )
        output_path = args.output or os.path.join(
            ROOT, "thinkdet", "results", "eval", default_name
        )
        tmp_dir = args.tmp_dir or os.path.join(
            ROOT,
            "thinkdet",
            "results",
            "eval",
            f"tmp_eval_stage1_{args.mode}_{timestamp}",
        )
    else:
        output_path = args.output or ""
        tmp_dir = args.tmp_dir or ""

    if dist.is_initialized():
        shared_paths = [output_path, tmp_dir]
        dist.broadcast_object_list(shared_paths, src=0)
        output_path, tmp_dir = shared_paths

    output_dir = os.path.dirname(output_path)

    if is_main:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(tmp_dir, exist_ok=True)
        print(
            f"[eval] mode={args.mode} layer_setup={layer_setup} world_size={world_size} "
            f"conf={args.conf_thresh} max_dets={args.max_dets}"
        )
    if dist.is_initialized():
        dist.barrier()

    train_ds = COCOGroundingDataset(img_dir=cfg.train_img_dir, ann_file=cfg.train_ann)
    val_full = COCOGroundingDataset(img_dir=cfg.val_img_dir, ann_file=cfg.val_ann)
    if args.max_images > 0:
        val_full.img_ids = val_full.img_ids[: args.max_images]
    val_shard = COCOGroundingDataset(img_dir=cfg.val_img_dir, ann_file=cfg.val_ann)
    if args.max_images > 0:
        val_shard.img_ids = val_shard.img_ids[: args.max_images]
    val_shard.img_ids = val_shard.img_ids[rank::world_size]

    model, extract_layers, ck_meta_loaded, load_info = build_model(
        cfg=cfg,
        mode=args.mode,
        layer_setup=layer_setup,
        checkpoint_path=args.checkpoint,
        gate_override=args.gate_override,
        device=device,
    )

    all_cat_names = sorted(train_ds.cat_id_to_name.values())
    all_query_text, _, positive_map_norm, cat_name_to_idx = build_positive_map(
        model.grounding_dino.tokenizer,
        model.grounding_dino.specical_tokens,
        all_cat_names,
    )
    positive_map_norm = positive_map_norm.to(device)

    predictions, local_runtime = run_local_shard(
        model=model,
        shard_dataset=val_shard,
        positive_map_norm=positive_map_norm,
        cat_name_to_idx=cat_name_to_idx,
        all_query_text=all_query_text,
        device=device,
        conf_thresh=args.conf_thresh,
        max_dets=args.max_dets,
        num_workers=args.num_workers,
    )

    pred_path = os.path.join(tmp_dir, f"pred_rank{rank}.json")
    stat_path = os.path.join(tmp_dir, f"runtime_rank{rank}.json")
    with open(pred_path, "w") as f:
        json.dump(predictions, f)
    with open(stat_path, "w") as f:
        json.dump(local_runtime, f)

    if dist.is_initialized():
        dist.barrier()

    exit_code = 0
    if is_main:
        merged_predictions = []
        rank_runtime = []
        for r in range(world_size):
            with open(os.path.join(tmp_dir, f"pred_rank{r}.json")) as f:
                merged_predictions.extend(json.load(f))
            with open(os.path.join(tmp_dir, f"runtime_rank{r}.json")) as f:
                rank_runtime.append(json.load(f))

        processed_images_sum = int(sum(x["processed_images"] for x in rank_runtime))
        errors_sum = int(sum(x["error_count"] for x in rank_runtime))
        error_types = merge_error_types([x["error_types"] for x in rank_runtime])
        sample_errors = []
        for r, rr in enumerate(rank_runtime):
            for item in rr.get("sample_errors", []):
                if len(sample_errors) >= 20:
                    break
                sample_errors.append({"rank": r, **item})
            if len(sample_errors) >= 20:
                break
        eval_time_min = max(float(x["elapsed_sec"]) for x in rank_runtime) / 60.0

        metrics = coco_eval_from_predictions(
            coco_gt=val_full.coco,
            predictions=merged_predictions,
            num_images=len(val_full),
            eval_time_min=eval_time_min,
        )
        runtime = {
            "processed_images_sum": processed_images_sum,
            "errors_sum": errors_sum,
            "error_rate_pct": 100.0 * errors_sum / max(1, processed_images_sum),
            "error_types": error_types,
            "sample_errors": sample_errors,
        }

        status = "ok"
        if errors_sum > args.max_allowed_errors:
            status = "invalid_errors"
            exit_code = 2

        result = {
            "status": status,
            "mode": args.mode,
            "dataset": "COCO val2017",
            "checkpoint": args.checkpoint,
            "layer_setup": layer_setup,
            "extract_layers": extract_layers,
            "layer_fusion": "mean",
            "world_size": world_size,
            "conf_thresh": args.conf_thresh,
            "max_dets": args.max_dets,
            "max_images": args.max_images if args.max_images > 0 else len(val_full),
            "metrics": metrics,
            "runtime": runtime,
            "checkpoint_meta": ck_meta_loaded if ck_meta_loaded else checkpoint_meta,
            "load_info": load_info,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)

        print(f"[done] Saved: {output_path}")
        print(json.dumps(result, indent=2))
        if status != "ok":
            print(
                f"[error] Runtime errors exceeded threshold: "
                f"errors_sum={errors_sum} max_allowed_errors={args.max_allowed_errors}"
            )
    else:
        result = None

    if dist.is_initialized():
        code_list = [exit_code]
        dist.broadcast_object_list(code_list, src=0)
        exit_code = code_list[0]

    cleanup_distributed()
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
