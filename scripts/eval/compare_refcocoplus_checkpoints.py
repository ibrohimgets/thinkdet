#!/usr/bin/env python3
"""
Compare two ThinkDet RefCOCO+ checkpoints on one informative held-out sample.

The script scans a RefCOCO+ split for a disagreement case, then renders:
    - GT box
    - checkpoint A top-1 prediction
    - checkpoint B top-1 prediction

It is intended for qualitative checkpoint-to-checkpoint comparison after
full RefCOCO+ training.
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass

import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = "/home/iibrohimm/project/next_step"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "GroundingDINO", "GroundingDINO"))

from groundingdino.util.inference import load_model as load_gd_model
from groundingdino.util.misc import nested_tensor_from_tensor_list
from thinkdet.data.refcoco_grounding import build_rec_positive_map_batch, build_refcoco_eval
from thinkdet.models.arch import DEFAULT_INJECTION_LAYERS, ThinkDetModel


GD_CONFIG = f"{ROOT}/GroundingDINO/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GD_WEIGHTS = f"{ROOT}/GroundingDINO/GroundingDINO/weights/groundingdino_swint_ogc.pth"
INTERNVL_PATH = f"{ROOT}/InternVL3_5-1B"
DATA_ROOT = f"{ROOT}/dataSets/refer/data"
IMAGE_DIR = f"{ROOT}/dataSets/coco/train2017"

DEFAULT_CKPT_A = (
    f"{ROOT}/thinkdet/checkpoints/refcocoplus_full/"
    "learned_8_9_10_full_layers_8_9_10_learned_residual_e10_20260314_062540/"
    "thinkdet_refcocoplus_final.pth"
)
DEFAULT_CKPT_B = (
    f"{ROOT}/thinkdet/checkpoints/refcocoplus_full/"
    "control_layer8_full_layers_8_mean_residual_e10_20260314_062541/"
    "thinkdet_refcocoplus_final.pth"
)
DEFAULT_OUT_DIR = f"{ROOT}/thinkdet/results/qualitative/refcocoplus_ckpt_compare"


try:
    FONT_BOLD = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18
    )
    FONT_SMALL = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14
    )
except Exception:
    FONT_BOLD = FONT_SMALL = ImageFont.load_default()


@dataclass
class SamplePrediction:
    score: float
    iou: float
    correct: bool
    box_cxcywh: list
    box_xyxy: list


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_a", type=str, default=DEFAULT_CKPT_A)
    parser.add_argument("--checkpoint_b", type=str, default=DEFAULT_CKPT_B)
    parser.add_argument("--label_a", type=str, default="learned_8_9_10")
    parser.add_argument("--label_b", type=str, default="control_8")
    parser.add_argument("--dataset_name", type=str, default="refcoco+")
    parser.add_argument("--split_by", type=str, default="unc")
    parser.add_argument("--split", type=str, default="val", choices=["val", "testA", "testB"])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max_scan", type=int, default=250)
    parser.add_argument(
        "--target",
        type=str,
        default="a_beats_b",
        choices=["a_beats_b", "b_beats_a", "largest_gap"],
    )
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def box_cxcywh_to_xyxy(box):
    cx, cy, w, h = box.unbind(-1)
    return torch.stack([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], dim=-1)


def pairwise_iou_xyxy(boxes1, boxes2):
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter
    return inter / (union + 1e-6)


def load_thinkdet(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    extract_layers = checkpoint.get("extract_layers") or [checkpoint.get("extract_layer", 9)]
    injection_layers = checkpoint.get("injection_layers", DEFAULT_INJECTION_LAYERS)
    tma_m = checkpoint.get("tma_m", 8)
    tma_n_heads = checkpoint.get("tma_n_heads", 8)
    fusion_mode = checkpoint.get("fusion_mode", "concat")
    layer_fusion = checkpoint.get("layer_fusion", "mean")

    grounding_dino = load_gd_model(GD_CONFIG, GD_WEIGHTS, device="cpu")
    model = ThinkDetModel(
        grounding_dino=grounding_dino,
        internvl_path=INTERNVL_PATH,
        extract_layer=max(extract_layers),
        extract_layers=extract_layers,
        layer_fusion=layer_fusion,
        injection_layers=injection_layers,
        tma_m=tma_m,
        tma_n_heads=tma_n_heads,
        fusion_mode=fusion_mode,
        preserve_kd_enabled=False,
    )
    model.load_state_dict(checkpoint["trainable_state_dict"], strict=False)
    model = model.to(device).eval()
    return model, checkpoint


@torch.no_grad()
def predict_single(model, sample, device):
    image_pil = Image.open(os.path.join(IMAGE_DIR, f"{sample['image_id']:012d}.jpg")).convert("RGB")
    internvl_images = sample["internvl_image"].unsqueeze(0).to(device)
    dino_nested = nested_tensor_from_tensor_list([sample["dino_image"]]).to(device)
    dino_inputs = {"samples": dino_nested, "captions": [sample["query_text"]]}

    outputs, aux = model(internvl_images, [sample["expression"]], dino_inputs)
    tokenizer = model.grounding_dino.tokenizer
    special_tokens = model.grounding_dino.specical_tokens
    _, _, positive_map_norm = build_rec_positive_map_batch(
        tokenizer, special_tokens, [sample["expression"]]
    )

    logits = outputs["pred_logits"][0].clamp(-50, 50)
    boxes = outputs["pred_boxes"][0]
    text_len = logits.shape[-1]
    pmap = positive_map_norm[0, :, :text_len].to(device)
    probs = logits.sigmoid()
    scores = (probs * pmap).sum(dim=-1)
    best_idx = int(scores.argmax().item())
    pred_box = boxes[best_idx].unsqueeze(0)
    gt_box = sample["box"].to(device)
    iou = pairwise_iou_xyxy(
        box_cxcywh_to_xyxy(pred_box),
        box_cxcywh_to_xyxy(gt_box),
    )[0, 0].item()

    img_w, img_h = image_pil.size
    cx, cy, bw, bh = pred_box[0].detach().cpu().tolist()
    pred_xyxy = [
        (cx - bw / 2.0) * img_w,
        (cy - bh / 2.0) * img_h,
        (cx + bw / 2.0) * img_w,
        (cy + bh / 2.0) * img_h,
    ]
    return (
        SamplePrediction(
            score=float(scores[best_idx].item()),
            iou=float(iou),
            correct=bool(iou >= 0.5),
            box_cxcywh=[cx, cy, bw, bh],
            box_xyxy=pred_xyxy,
        ),
        image_pil,
        aux,
    )


def gt_box_xyxy(sample, image_size):
    img_w, img_h = image_size
    cx, cy, bw, bh = sample["box"][0].tolist()
    return [
        (cx - bw / 2.0) * img_w,
        (cy - bh / 2.0) * img_h,
        (cx + bw / 2.0) * img_w,
        (cy + bh / 2.0) * img_h,
    ]


def choose_case(samples, model_a, model_b, device, target):
    best = None

    for idx, sample in enumerate(samples):
        pred_a, image_pil, aux_a = predict_single(model_a, sample, device)
        pred_b, _, aux_b = predict_single(model_b, sample, device)
        delta = pred_a.iou - pred_b.iou

        record = {
            "sample_idx": idx,
            "sample": sample,
            "image_pil": image_pil,
            "pred_a": pred_a,
            "pred_b": pred_b,
            "aux_a": aux_a,
            "aux_b": aux_b,
            "delta": delta,
        }

        if target == "a_beats_b":
            if pred_a.correct and not pred_b.correct:
                if best is None or delta > best["delta"]:
                    best = record
        elif target == "b_beats_a":
            if pred_b.correct and not pred_a.correct:
                if best is None or (-delta) > (-best["delta"]):
                    best = record
        else:
            if best is None or abs(delta) > abs(best["delta"]):
                best = record

        if target in {"a_beats_b", "b_beats_a"} and best is not None and abs(best["delta"]) > 0.25:
            break

    return best


def draw_panel(image_pil, title, prompt, gt_box, pred_box, pred_color, out_path):
    img = image_pil.copy().convert("RGB")
    draw = ImageDraw.Draw(img)

    for off in range(3):
        draw.rectangle(
            [gt_box[0] - off, gt_box[1] - off, gt_box[2] + off, gt_box[3] + off],
            outline=(70, 160, 255),
        )
    for off in range(3):
        draw.rectangle(
            [pred_box[0] - off, pred_box[1] - off, pred_box[2] + off, pred_box[3] + off],
            outline=pred_color,
        )

    header_h = 56
    panel = Image.new("RGB", (img.width, img.height + header_h), (24, 24, 24))
    panel.paste(img, (0, header_h))
    d = ImageDraw.Draw(panel)
    d.text((10, 8), title, fill=(255, 255, 255), font=FONT_BOLD)
    d.text((10, 29), prompt, fill=(170, 230, 180), font=FONT_SMALL)
    panel.save(out_path, quality=95)
    return panel


def make_comparison(panels, headline, subtitle, out_path):
    gap = 8
    header_h = 60
    width = sum(panel.width for panel in panels) + gap * (len(panels) - 1)
    height = max(panel.height for panel in panels) + header_h
    canvas = Image.new("RGB", (width, height), (18, 18, 18))
    d = ImageDraw.Draw(canvas)
    d.text((10, 8), headline, fill=(255, 255, 255), font=FONT_BOLD)
    d.text((10, 32), subtitle, fill=(180, 180, 180), font=FONT_SMALL)
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, header_h))
        x += panel.width + gap
    canvas.save(out_path, quality=95)


def main():
    args = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    print("[1/4] Loading checkpoints ...")
    model_a, ckpt_a = load_thinkdet(args.checkpoint_a, device)
    model_b, ckpt_b = load_thinkdet(args.checkpoint_b, device)

    print("[2/4] Loading RefCOCO+ split ...")
    dataset = build_refcoco_eval(
        data_root=DATA_ROOT,
        image_dir=IMAGE_DIR,
        dataset_name=args.dataset_name,
        split_by=args.split_by,
        split=args.split,
    )
    samples = [dataset[i] for i in range(min(args.max_scan, len(dataset)))]

    print("[3/4] Searching for informative disagreement case ...")
    chosen = choose_case(samples, model_a, model_b, device, args.target)
    if chosen is None:
        raise RuntimeError("No comparison case found in scanned samples")

    sample = chosen["sample"]
    image_pil = chosen["image_pil"]
    gt_xyxy = gt_box_xyxy(sample, image_pil.size)
    pred_a = chosen["pred_a"]
    pred_b = chosen["pred_b"]

    sample_tag = (
        f"{args.split}_img{sample['image_id']}_ref{sample['ref_id']}_idx{chosen['sample_idx']}"
    )
    panel_a_path = os.path.join(args.output_dir, f"{sample_tag}_{args.label_a}.jpg")
    panel_b_path = os.path.join(args.output_dir, f"{sample_tag}_{args.label_b}.jpg")
    comp_path = os.path.join(args.output_dir, f"{sample_tag}_comparison.jpg")
    meta_path = os.path.join(args.output_dir, f"{sample_tag}_comparison.json")

    title_a = (
        f"{args.label_a} | score={pred_a.score:.3f} | IoU={pred_a.iou:.3f} | "
        f"{'correct' if pred_a.correct else 'wrong'}"
    )
    title_b = (
        f"{args.label_b} | score={pred_b.score:.3f} | IoU={pred_b.iou:.3f} | "
        f"{'correct' if pred_b.correct else 'wrong'}"
    )

    panel_a = draw_panel(
        image_pil,
        title_a,
        sample["expression"],
        gt_xyxy,
        pred_a.box_xyxy,
        (40, 210, 70),
        panel_a_path,
    )
    panel_b = draw_panel(
        image_pil,
        title_b,
        sample["expression"],
        gt_xyxy,
        pred_b.box_xyxy,
        (220, 60, 60),
        panel_b_path,
    )

    headline = f"RefCOCO+ {args.split} checkpoint comparison"
    subtitle = (
        f"GT=blue | {args.label_a}=green | {args.label_b}=red | "
        f"image_id={sample['image_id']} ref_id={sample['ref_id']}"
    )
    make_comparison([panel_a, panel_b], headline, subtitle, comp_path)

    metadata = {
        "dataset_name": args.dataset_name,
        "split_by": args.split_by,
        "split": args.split,
        "target": args.target,
        "max_scan": args.max_scan,
        "sample_idx": chosen["sample_idx"],
        "image_id": sample["image_id"],
        "ref_id": sample["ref_id"],
        "expression": sample["expression"],
        "query_text": sample["query_text"],
        "checkpoint_a": args.checkpoint_a,
        "checkpoint_b": args.checkpoint_b,
        "label_a": args.label_a,
        "label_b": args.label_b,
        "prediction_a": pred_a.__dict__,
        "prediction_b": pred_b.__dict__,
        "delta_iou": chosen["delta"],
        "layer_fusion_weights_a": ckpt_a.get("layer_fusion_weights"),
        "layer_fusion_weights_b": ckpt_b.get("layer_fusion_weights"),
        "panel_a": panel_a_path,
        "panel_b": panel_b_path,
        "comparison": comp_path,
    }
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print("[4/4] Saved comparison")
    print(f"  comparison: {comp_path}")
    print(f"  metadata:   {meta_path}")
    print(f"  prompt:     {sample['expression']}")
    print(
        f"  {args.label_a}: score={pred_a.score:.4f} IoU={pred_a.iou:.4f} correct={pred_a.correct}"
    )
    print(
        f"  {args.label_b}: score={pred_b.score:.4f} IoU={pred_b.iou:.4f} correct={pred_b.correct}"
    )


if __name__ == "__main__":
    main()
