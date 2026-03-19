"""
Evaluate a single unified ThinkDet checkpoint on Flickr30k Entities phrase grounding.

This is a downstream detector eval, not the earlier proxy layer probe.
Each unique (image_id, phrase_id, expression) pair is evaluated as one sample.
"""

import argparse
import datetime
import json
import os
import sys
from collections import defaultdict

import torch
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF

ROOT = "/home/iibrohimm/project/next_step"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "GroundingDINO", "GroundingDINO"))

from thinkdet.models.arch import ThinkDetModel, DEFAULT_INJECTION_LAYERS
from groundingdino.util.inference import load_model as load_gd_model
from groundingdino.util.misc import NestedTensor


DEFAULT_CKPT = (
    f"{ROOT}/thinkdet/checkpoints/unified/"
    "layer9_kd0p05_l1_1e-4_20260224_135540/"
    "thinkdet_unified_epoch5.pth"
)
DEFAULT_OUT = (
    f"{ROOT}/thinkdet/results/layer_ablation/"
    f"flickr_grounding_eval_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
)
GD_CONFIG = f"{ROOT}/GroundingDINO/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GD_WEIGHTS = f"{ROOT}/GroundingDINO/GroundingDINO/weights/groundingdino_swint_ogc.pth"
INTERNVL_PATH = f"{ROOT}/InternVL3_5-1B"
FLICKR_ROOT = f"{ROOT}/dataSets/flickr30k_entities"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--flickr_root", type=str, default=FLICKR_ROOT)
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--log_every", type=int, default=200)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output", type=str, default=DEFAULT_OUT)

    parser.add_argument("--gd_config", type=str, default=GD_CONFIG)
    parser.add_argument("--gd_weights", type=str, default=GD_WEIGHTS)
    parser.add_argument("--internvl_path", type=str, default=INTERNVL_PATH)
    parser.add_argument("--thinkdet_checkpoint", type=str, default=DEFAULT_CKPT)
    parser.add_argument("--extract_layer", type=int, default=None)
    parser.add_argument("--extract_layers", type=int, nargs="+", default=None)
    parser.add_argument("--layer_fusion", type=str, default=None, choices=["mean", "last"])
    return parser.parse_args()


def build_dino_transform():
    normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def transform(image_pil):
        w, h = image_pil.size
        scale = 800 / min(w, h)
        if scale * max(w, h) > 1333:
            scale = 1333 / max(w, h)
        nw, nh = int(w * scale), int(h * scale)
        img = image_pil.resize((nw, nh), Image.BILINEAR)
        t = TF.to_tensor(img)
        return normalize(t)

    return transform


def build_internvl_transform():
    return T.Compose([
        T.Resize((448, 448), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def iou_xyxy(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (aa + bb - inter + 1e-9)


def get_special_tokens(model):
    tokens = getattr(model, "specical_tokens", None)
    if tokens is not None:
        return tokens
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


def score_outputs_for_query(outputs, positive_map_norm):
    logits = outputs["pred_logits"][0].clamp(-50, 50)
    boxes = outputs["pred_boxes"][0]
    text_len = logits.shape[-1]
    pmap = positive_map_norm[:, :text_len].to(logits.device)
    probs = logits.sigmoid()
    scores = (probs * pmap).sum(dim=-1)
    return scores, boxes


def resolve_query_scoring_assets(model_fn):
    if hasattr(model_fn, "grounding_dino"):
        dino_model = model_fn.grounding_dino
    else:
        dino_model = model_fn
    if not hasattr(dino_model, "tokenizer"):
        raise AttributeError("Could not resolve GroundingDINO tokenizer for query scoring")
    return dino_model.tokenizer, get_special_tokens(dino_model)


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


def load_flickr_samples(flickr_root, split):
    ann_file = os.path.join(flickr_root, "annotations.json")
    split_file = os.path.join(flickr_root, "raw_source", f"{split}.txt")

    with open(ann_file, "r") as f:
        all_items = json.load(f)
    with open(split_file, "r") as f:
        split_ids = {line.strip().replace(".jpg", "") for line in f if line.strip()}

    samples = []
    seen = set()
    grouped = defaultdict(lambda: {"image_path": None, "boxes": None, "phrases": []})

    for item in all_items:
        image_id = str(item.get("image_id", "")).strip()
        if not image_id or image_id not in split_ids:
            continue
        image_path = item.get("image_path") or os.path.join(flickr_root, "images", f"{image_id}.jpg")

        for phrase_item in item.get("phrases", []):
            if phrase_item.get("is_nobox", False):
                continue
            boxes = phrase_item.get("boxes") or []
            if not boxes:
                continue
            phrase = (phrase_item.get("phrase") or "").strip()
            if not phrase:
                continue
            phrase_id = phrase_item.get("phrase_id")
            if phrase_id is None:
                continue

            group_key = (image_id, str(phrase_id))
            rec = grouped[group_key]
            rec["image_path"] = image_path
            rec["boxes"] = boxes
            norm = phrase.lower()
            if norm not in rec["phrases"]:
                rec["phrases"].append(norm)

    for (image_id, phrase_id), rec in grouped.items():
        for phrase_norm in sorted(rec["phrases"]):
            sample_key = (image_id, phrase_id, phrase_norm)
            if sample_key in seen:
                continue
            seen.add(sample_key)
            samples.append(
                {
                    "sample_id": f"{image_id}_{phrase_id}_{len(samples)}",
                    "image_id": image_id,
                    "phrase_id": str(phrase_id),
                    "prompt": phrase_norm,
                    "image_path": rec["image_path"],
                    "gt_boxes": rec["boxes"],
                }
            )

    samples.sort(key=lambda x: (x["image_id"], x["phrase_id"], x["prompt"]))
    return samples


@torch.no_grad()
def run_topk(model_fn, image_pil, prompt, device, dino_tf, ivl_tf, top_k, tokenizer, special_tokens):
    img_w, img_h = image_pil.size
    dino_t = dino_tf(image_pil).unsqueeze(0).to(device)
    mask = torch.zeros(1, dino_t.shape[2], dino_t.shape[3], dtype=torch.bool, device=device)
    nested = NestedTensor(dino_t, mask)
    ivl_t = ivl_tf(image_pil).unsqueeze(0).to(device)

    query_text = prompt if prompt.strip().endswith(".") else (prompt.strip() + " .")
    outputs, _ = model_fn(ivl_t, [prompt], {"samples": nested, "captions": [query_text]})

    positive_map_norm = build_positive_map_for_query(
        tokenizer, special_tokens, query_text, max_text_len=512
    )
    scores, boxes = score_outputs_for_query(outputs, positive_map_norm)
    vals, idx = scores.topk(min(int(top_k), scores.shape[0]))

    results = []
    for j in range(len(vals)):
        cx, cy, bw, bh = boxes[idx[j]].tolist()
        x1 = (cx - bw / 2.0) * img_w
        y1 = (cy - bh / 2.0) * img_h
        x2 = (cx + bw / 2.0) * img_w
        y2 = (cy + bh / 2.0) * img_h
        results.append(
            {
                "score": float(vals[j].item()),
                "box_abs_xyxy": [x1, y1, x2, y2],
            }
        )
    return results


def summarize_rows(rows, top_k):
    n = max(len(rows), 1)
    return {
        "n_samples": len(rows),
        "mean_top1_iou": sum(r["top1_iou"] for r in rows) / n,
        "mean_topk_best_iou": sum(r["topk_best_iou"] for r in rows) / n,
        "top1_acc_iou50": sum(1 for r in rows if r["top1_iou"] >= 0.5) / n,
        "topk_acc_iou50": sum(1 for r in rows if r["topk_best_iou"] >= 0.5) / n,
        "top1_acc_iou75": sum(1 for r in rows if r["top1_iou"] >= 0.75) / n,
        "top_k": int(top_k),
    }


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    samples = load_flickr_samples(args.flickr_root, args.split)
    if args.max_samples > 0:
        samples = samples[:args.max_samples]

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dino_tf = build_dino_transform()
    ivl_tf = build_internvl_transform()
    model, ckpt_meta = load_unified_model(args, device)
    tokenizer, special_tokens = resolve_query_scoring_assets(model)

    rows = []
    for i, sample in enumerate(samples, 1):
        image_pil = Image.open(sample["image_path"]).convert("RGB")
        preds = run_topk(
            model,
            image_pil,
            sample["prompt"],
            device,
            dino_tf,
            ivl_tf,
            top_k=args.top_k,
            tokenizer=tokenizer,
            special_tokens=special_tokens,
        )

        gt_boxes = sample["gt_boxes"]
        top1_box = preds[0]["box_abs_xyxy"] if preds else None
        top1_score = float(preds[0]["score"]) if preds else 0.0
        top1_iou = max((iou_xyxy(top1_box, g) for g in gt_boxes), default=0.0) if top1_box else 0.0
        topk_best_iou = max(
            (iou_xyxy(pred["box_abs_xyxy"], g) for pred in preds for g in gt_boxes),
            default=0.0,
        )

        rows.append(
            {
                "sample_id": sample["sample_id"],
                "image_id": sample["image_id"],
                "phrase_id": sample["phrase_id"],
                "prompt": sample["prompt"],
                "top1_iou": float(top1_iou),
                "topk_best_iou": float(topk_best_iou),
                "top1_score": float(top1_score),
                "top1_correct_iou50": int(top1_iou >= 0.5),
                "top1_correct_iou75": int(top1_iou >= 0.75),
            }
        )

        if i % max(1, args.log_every) == 0 or i == len(samples):
            print(f"[eval] {i}/{len(samples)}")

    payload = {
        "status": "ok",
        "timestamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
        "dataset": "flickr30k_entities",
        "split": args.split,
        "n_samples": len(rows),
        "checkpoint": args.thinkdet_checkpoint,
        "ckpt_meta": ckpt_meta,
        "summary": summarize_rows(rows, args.top_k),
        "per_sample": rows,
    }
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)

    print("Saved:", args.output)
    print("top1_acc_iou50:", payload["summary"]["top1_acc_iou50"])


if __name__ == "__main__":
    main()
