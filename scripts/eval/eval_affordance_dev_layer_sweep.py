"""
Evaluate InternVL extraction layers on the affordance benchmark dev split.

This is a no-retrain layer override sweep. It loads one ThinkDet checkpoint,
extracts the requested InternVL layers for each sample, then runs the same
GroundingDINO/TMA head once per requested layer and reports Hit@0.5.
"""

import argparse
import datetime
import json
import os
import sys
from collections import defaultdict

import torch
from PIL import Image


ROOT = "/home/iibrohimm/project/next_step"
GROUNDING_DINO_ROOT = f"{ROOT}/GroundingDINO"
sys.path.insert(0, ROOT)
sys.path.insert(0, GROUNDING_DINO_ROOT)

from groundingdino.util.misc import NestedTensor
from thinkdet.scripts.eval.eval_affordance_unified import (
    build_dino_transform,
    build_internvl_transform,
    build_positive_map_for_query,
    iou_xyxy,
    load_unified_model,
    resolve_query_scoring_assets,
    score_outputs_for_query,
    summarize_rows,
    xywh_to_xyxy,
)


DEFAULT_BENCH = f"{ROOT}/thinkdet/data/benchmarks/affordance_coco_val_heldout_v2.json"
DEFAULT_CKPT = (
    f"{ROOT}/thinkdet/checkpoints/unified/"
    "layer9_kd0p05_l1_1e-4_20260224_135540/"
    "thinkdet_unified_epoch5.pth"
)
DEFAULT_OUT = (
    f"{ROOT}/thinkdet/results/layer_ablation_dev320/"
    f"affordance_dev_layer_sweep_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
)
GD_CONFIG = f"{GROUNDING_DINO_ROOT}/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GD_WEIGHTS = f"{GROUNDING_DINO_ROOT}/weights/groundingdino_swint_ogc.pth"
INTERNVL_PATH = f"{ROOT}/InternVL3_5-1B"


def patch_groundingdino_ms_deform_attn():
    """Use the PyTorch attention fallback when custom ops are unavailable."""
    try:
        from groundingdino.models.GroundingDINO import ms_deform_attn
    except Exception:
        return
    if hasattr(ms_deform_attn, "_C"):
        return

    def fallback_apply(
        value,
        value_spatial_shapes,
        value_level_start_index,
        sampling_locations,
        attention_weights,
        im2col_step,
    ):
        return ms_deform_attn.multi_scale_deformable_attn_pytorch(
            value,
            value_spatial_shapes,
            sampling_locations,
            attention_weights,
        )

    ms_deform_attn.MultiScaleDeformableAttnFunction.apply = staticmethod(fallback_apply)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=str, default=DEFAULT_BENCH)
    parser.add_argument("--split", type=str, default="dev", choices=["dev", "test", "all"])
    parser.add_argument("--layers", type=int, nargs="+", required=True)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output", type=str, default=DEFAULT_OUT)
    parser.add_argument("--gd_config", type=str, default=GD_CONFIG)
    parser.add_argument("--gd_weights", type=str, default=GD_WEIGHTS)
    parser.add_argument("--internvl_path", type=str, default=INTERNVL_PATH)
    parser.add_argument("--thinkdet_checkpoint", type=str, default=DEFAULT_CKPT)
    parser.add_argument("--layer_fusion", type=str, default="mean", choices=["mean", "last"])
    parser.add_argument("--force_gate_value", type=float, default=None)
    return parser.parse_args()


def build_model_args(args, layers):
    class ModelArgs:
        pass

    model_args = ModelArgs()
    model_args.thinkdet_checkpoint = args.thinkdet_checkpoint
    model_args.extract_layer = None
    model_args.extract_layers = sorted(set(int(layer) for layer in layers))
    model_args.layer_fusion = args.layer_fusion
    model_args.force_gate_value = args.force_gate_value
    model_args.gd_config = args.gd_config
    model_args.gd_weights = args.gd_weights
    model_args.internvl_path = args.internvl_path
    return model_args


def make_inputs(image_pil, prompt, device, dino_tf, ivl_tf):
    dino_t = dino_tf(image_pil).unsqueeze(0).to(device)
    mask = torch.zeros(
        1,
        dino_t.shape[2],
        dino_t.shape[3],
        dtype=torch.bool,
        device=device,
    )
    nested = NestedTensor(dino_t, mask)
    ivl_t = ivl_tf(image_pil).unsqueeze(0).to(device)
    query_text = prompt if prompt.strip().endswith(".") else (prompt.strip() + " .")
    return ivl_t, {"samples": nested, "captions": [query_text]}, query_text


def score_layer_outputs(outputs, positive_map_norm, image_size, top_k):
    img_w, img_h = image_size
    scores, boxes = score_outputs_for_query(outputs, positive_map_norm)
    vals, idx = scores.topk(min(int(top_k), scores.shape[0]))
    preds = []
    for j in range(len(vals)):
        cx, cy, bw, bh = boxes[idx[j]].tolist()
        preds.append(
            {
                "score": float(vals[j].item()),
                "box_abs_xyxy": [
                    (cx - bw / 2.0) * img_w,
                    (cy - bh / 2.0) * img_h,
                    (cx + bw / 2.0) * img_w,
                    (cy + bh / 2.0) * img_h,
                ],
            }
        )
    return preds


@torch.no_grad()
def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    layers = sorted(set(int(layer) for layer in args.layers))

    patch_groundingdino_ms_deform_attn()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, ckpt_meta = load_unified_model(build_model_args(args, layers), device)
    tokenizer, special_tokens = resolve_query_scoring_assets(model)
    dino_tf = build_dino_transform()
    ivl_tf = build_internvl_transform()

    with open(args.benchmark, "r") as f:
        bench = json.load(f)
    samples = bench["samples"]
    if args.split != "all":
        samples = [sample for sample in samples if sample.get("split") == args.split]
    if args.max_samples > 0:
        samples = samples[: args.max_samples]

    rows_by_layer = {layer: [] for layer in layers}
    per_affordance = {layer: defaultdict(list) for layer in layers}

    for i, sample in enumerate(samples, 1):
        image_pil = Image.open(sample["image_path"]).convert("RGB")
        ivl_t, dino_inputs, query_text = make_inputs(
            image_pil,
            sample["prompt"],
            device,
            dino_tf,
            ivl_tf,
        )
        positive_map_norm = build_positive_map_for_query(
            tokenizer,
            special_tokens,
            query_text,
            max_text_len=512,
        )
        selected = model.feature_extractor.extract_selected_layers(ivl_t, [sample["prompt"]])
        gt_boxes = [xywh_to_xyxy(t["bbox_xywh"]) for t in sample["positive_targets"]]

        for layer in layers:
            model._set_h_vlm(selected[layer])
            outputs = model.grounding_dino(**dino_inputs)
            model._clear_h_vlm()
            preds = score_layer_outputs(outputs, positive_map_norm, image_pil.size, args.top_k)

            top1_box = preds[0]["box_abs_xyxy"] if preds else None
            top1_iou = (
                max((iou_xyxy(top1_box, gt) for gt in gt_boxes), default=0.0)
                if top1_box
                else 0.0
            )
            topk_best_iou = max(
                (iou_xyxy(pred["box_abs_xyxy"], gt) for pred in preds for gt in gt_boxes),
                default=0.0,
            )
            row = {
                "benchmark_id": sample["benchmark_id"],
                "image_id": int(sample["image_id"]),
                "affordance_id": sample["affordance_id"],
                "prompt": sample["prompt"],
                "top1_iou": float(top1_iou),
                "topk_best_iou": float(topk_best_iou),
                "top1_score": float(preds[0]["score"]) if preds else 0.0,
                "top1_correct_iou50": int(top1_iou >= 0.5),
                "topk_correct_iou50": int(topk_best_iou >= 0.5),
            }
            rows_by_layer[layer].append(row)
            per_affordance[layer][sample["affordance_id"]].append(row)

        if i % max(1, args.log_every) == 0 or i == len(samples):
            print(f"[sweep] {i}/{len(samples)} layers={layers}")

    per_layer = {}
    for layer in layers:
        rows = rows_by_layer[layer]
        per_layer[str(layer)] = {
            "summary": summarize_rows(rows, args.top_k),
            "per_affordance": {
                aff: summarize_rows(aff_rows, args.top_k)
                for aff, aff_rows in sorted(per_affordance[layer].items())
            },
            "per_sample": rows,
        }

    payload = {
        "status": "ok",
        "timestamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
        "benchmark": args.benchmark,
        "split": args.split,
        "n_samples": len(samples),
        "layers": layers,
        "top_k": int(args.top_k),
        "checkpoint": args.thinkdet_checkpoint,
        "ckpt_meta": ckpt_meta,
        "per_layer": per_layer,
    }
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)

    print("Saved:", args.output)
    for layer in layers:
        summary = per_layer[str(layer)]["summary"]
        print(
            f"L{layer}: top1={summary['hit@0.5_top1']:.6f} "
            f"top5={summary['hit@0.5_topk']:.6f}"
        )


if __name__ == "__main__":
    main()
