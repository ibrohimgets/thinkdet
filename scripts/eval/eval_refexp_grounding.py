"""
Evaluate a single unified ThinkDet checkpoint on RefCOCO / RefCOCO+ / RefCOCOg.

This is a compact single-checkpoint evaluator that saves per-sample results for
layer ablation analysis.
"""

import argparse
import datetime
import json
import os
import sys

import torch

ROOT = "/home/iibrohimm/project/next_step"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "GroundingDINO", "GroundingDINO"))

from groundingdino.util.misc import nested_tensor_from_tensor_list
from groundingdino.util.inference import load_model as load_gd_model
from thinkdet.data.refcoco_grounding import build_refcoco_eval
from thinkdet.models.arch import ThinkDetModel, DEFAULT_INJECTION_LAYERS


DEFAULT_CKPT = (
    f"{ROOT}/thinkdet/checkpoints/unified/"
    "layer9_kd0p05_l1_1e-4_20260224_135540/"
    "thinkdet_unified_epoch5.pth"
)
DEFAULT_OUT = (
    f"{ROOT}/thinkdet/results/layer_ablation/"
    f"refexp_eval_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
)
GD_CONFIG = f"{ROOT}/GroundingDINO/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GD_WEIGHTS = f"{ROOT}/GroundingDINO/GroundingDINO/weights/groundingdino_swint_ogc.pth"
INTERNVL_PATH = f"{ROOT}/InternVL3_5-1B"


DEFAULT_SPLIT_BY = {
    "refcoco": "unc",
    "refcoco+": "unc",
    "refcocog": "umd",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="refcocog", choices=["refcoco", "refcoco+", "refcocog"])
    parser.add_argument("--split_by", type=str, default=None)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output", type=str, default=DEFAULT_OUT)

    parser.add_argument("--data_root", type=str, default=f"{ROOT}/dataSets/refer/data")
    parser.add_argument("--image_dir", type=str, default=f"{ROOT}/dataSets/coco/train2017")
    parser.add_argument("--gd_config", type=str, default=GD_CONFIG)
    parser.add_argument("--gd_weights", type=str, default=GD_WEIGHTS)
    parser.add_argument("--internvl_path", type=str, default=INTERNVL_PATH)
    parser.add_argument("--thinkdet_checkpoint", type=str, default=DEFAULT_CKPT)
    parser.add_argument("--extract_layer", type=int, default=None)
    parser.add_argument("--extract_layers", type=int, nargs="+", default=None)
    parser.add_argument("--layer_fusion", type=str, default=None, choices=["mean", "last"])
    return parser.parse_args()


def get_special_tokens(model):
    tokens = getattr(model, "special_tokens", None)
    if tokens is not None:
        return tokens
    return model.tokenizer.all_special_ids


def build_positive_map_for_query(tokenizer, special_tokens, query_text, max_text_len=512):
    tokenized = tokenizer(query_text, return_tensors="pt")
    input_ids = tokenized["input_ids"][0]

    positive_map = torch.zeros(1, max_text_len, dtype=torch.float32)
    special_set = set(int(x) for x in special_tokens)

    for pos, tok_id in enumerate(input_ids.tolist()):
        if pos >= max_text_len:
            break
        if tok_id not in special_set:
            positive_map[0, pos] = 1.0

    row_sum = positive_map.sum(dim=-1, keepdim=True).clamp(min=1e-6)
    return positive_map / row_sum


def box_cxcywh_to_xyxy(box):
    cx, cy, w, h = box.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)


def pairwise_iou_xyxy(boxes1, boxes2):
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter
    return inter / (union + 1e-6)


def score_outputs(outputs, positive_map_norm):
    logits = outputs["pred_logits"][0].clamp(-50, 50)
    boxes = outputs["pred_boxes"][0]
    text_len = logits.shape[-1]
    pmap = positive_map_norm[:, :text_len].to(logits.device)
    probs = logits.sigmoid()
    scores = (probs * pmap).sum(dim=-1)
    return scores, boxes


def load_unified_model(args, device):
    ckpt = torch.load(args.thinkdet_checkpoint, map_location="cpu")
    inject_layers = ckpt.get("injection_layers", DEFAULT_INJECTION_LAYERS)
    tma_m = ckpt.get("tma_m", 8)
    tma_nheads = ckpt.get("tma_n_heads", 8) or 8
    checkpoint_extract_layers = ckpt.get("extract_layers") or [ckpt.get("extract_layer", 9)]
    checkpoint_layer_fusion = ckpt.get("layer_fusion", "mean") or "mean"
    if args.extract_layers:
        ext_layers = sorted(set(int(x) for x in args.extract_layers))
    elif args.extract_layer is not None:
        ext_layers = [int(args.extract_layer)]
    else:
        ext_layers = [int(x) for x in checkpoint_extract_layers]
    layer_fusion = args.layer_fusion or checkpoint_layer_fusion
    fusion_mode = ckpt.get("fusion_mode", "concat") or "concat"

    gd = load_gd_model(args.gd_config, args.gd_weights, device="cpu")
    model = ThinkDetModel(
        grounding_dino=gd,
        internvl_path=args.internvl_path,
        extract_layer=max(ext_layers),
        extract_layers=ext_layers,
        layer_fusion=layer_fusion,
        injection_layers=inject_layers,
        tma_m=tma_m,
        tma_n_heads=tma_nheads,
        fusion_mode=fusion_mode,
    )
    model.load_state_dict(ckpt["trainable_state_dict"], strict=False)
    model = model.to(device).eval()
    return model, {
        "checkpoint_extract_layers": [int(x) for x in checkpoint_extract_layers],
        "checkpoint_layer_fusion": checkpoint_layer_fusion,
        "extract_layers": ext_layers,
        "layer_fusion": layer_fusion,
        "injection_layers": inject_layers,
        "tma_m": tma_m,
        "tma_n_heads": tma_nheads,
        "fusion_mode": fusion_mode,
    }


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    split_by = args.split_by or DEFAULT_SPLIT_BY[args.dataset_name]
    dataset = build_refcoco_eval(
        data_root=args.data_root,
        image_dir=args.image_dir,
        dataset_name=args.dataset_name,
        split_by=split_by,
        split=args.split,
    )
    total = len(dataset)
    if args.max_samples > 0:
        total = min(total, args.max_samples)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, ckpt_meta = load_unified_model(args, device)
    tokenizer = model.grounding_dino.tokenizer
    special_tokens = get_special_tokens(model.grounding_dino)

    rows = []
    with torch.no_grad():
        for idx in range(total):
            sample = dataset[idx]
            query_text = sample["query_text"]
            gt_box = sample["box"].to(device)
            internvl_img = sample["internvl_image"].unsqueeze(0).to(device)
            dino_img = sample["dino_image"].to(device)
            dino_nested = nested_tensor_from_tensor_list([dino_img])

            pmap_norm = build_positive_map_for_query(
                tokenizer, special_tokens, query_text, max_text_len=512
            )
            outputs, _ = model(internvl_img, [query_text], {"samples": dino_nested, "captions": [query_text]})
            scores, boxes = score_outputs(outputs, pmap_norm)
            best_idx = int(scores.argmax().item())
            top1_score = float(scores[best_idx].item())
            pred_top = boxes[best_idx].unsqueeze(0)
            top1_iou = float(
                pairwise_iou_xyxy(
                    box_cxcywh_to_xyxy(pred_top),
                    box_cxcywh_to_xyxy(gt_box),
                )[0, 0].item()
            )

            rows.append(
                {
                    "sample_id": f"{args.dataset_name}_{sample['ref_id']}_{idx}",
                    "dataset_name": args.dataset_name,
                    "split_by": split_by,
                    "split": args.split,
                    "ref_id": int(sample["ref_id"]),
                    "image_id": int(sample["image_id"]),
                    "prompt": sample["expression"],
                    "top1_iou": top1_iou,
                    "top1_score": top1_score,
                    "top1_correct_iou50": int(top1_iou >= 0.5),
                }
            )

            if (idx + 1) % 200 == 0 or (idx + 1) == total:
                print(f"[eval] {idx+1}/{total}")

    n = max(len(rows), 1)
    summary = {
        "n_samples": len(rows),
        "top1_acc": sum(r["top1_correct_iou50"] for r in rows) / n,
        "mean_top1_iou": sum(r["top1_iou"] for r in rows) / n,
    }
    payload = {
        "status": "ok",
        "timestamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
        "dataset": {
            "name": args.dataset_name,
            "split_by": split_by,
            "split": args.split,
            "evaluated_samples": len(rows),
            "total_samples": len(dataset),
        },
        "checkpoint": args.thinkdet_checkpoint,
        "ckpt_meta": ckpt_meta,
        "summary": summary,
        "per_sample": rows,
    }
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)

    print("Saved:", args.output)
    print("top1_acc:", summary["top1_acc"])


if __name__ == "__main__":
    main()
