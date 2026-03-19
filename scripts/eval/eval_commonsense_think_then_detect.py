#!/usr/bin/env python3
"""
Think-then-Detect commonsense evaluation.

Stage 1: InternVL translates abstract commonsense prompt -> concrete object name.
Stage 2: ThinkDet / baseline GroundingDINO detects using the translated prompt.

This demonstrates ThinkDet's integrated advantage: the LLM *thinks* about what
the commonsense prompt means, then the detector finds it.
"""

import argparse
import csv
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

from groundingdino.util.inference import load_model as load_gd_model
from groundingdino.util.misc import NestedTensor
from thinkdet.models.arch import ThinkDetModel, DEFAULT_INJECTION_LAYERS


DEFAULT_BENCH = (
    f"{ROOT}/thinkdet/data/benchmarks/commonsense_refcoco_objects_coco_val_v1.json"
)
DEFAULT_CKPT = (
    f"{ROOT}/thinkdet/checkpoints/unified/"
    "layer9_kd0p05_l1_1e-4_20260224_135540/"
    "thinkdet_unified_epoch5.pth"
)
DEFAULT_OUT = (
    f"{ROOT}/thinkdet/results/eval/"
    f"commonsense_think_then_detect_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
)
GD_CONFIG = (
    f"{ROOT}/GroundingDINO/GroundingDINO/groundingdino/config/"
    "GroundingDINO_SwinT_OGC.py"
)
GD_WEIGHTS = f"{ROOT}/GroundingDINO/GroundingDINO/weights/groundingdino_swint_ogc.pth"
INTERNVL_PATH = f"{ROOT}/InternVL3_5-1B"

TRANSLATE_PROMPT = (
    "A commonsense description of an object is given: \"{prompt}\"\n"
    "What common object does this description refer to? "
    "Answer with ONLY the object name, nothing else. "
    "For example: cup, chair, dog, bicycle."
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", type=str, default=DEFAULT_BENCH)
    p.add_argument("--split", type=str, default="test", choices=["test", "all"])
    p.add_argument("--shard_id", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--log_every", type=int, default=25)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--output", type=str, default=DEFAULT_OUT)

    p.add_argument("--gd_config", type=str, default=GD_CONFIG)
    p.add_argument("--gd_weights", type=str, default=GD_WEIGHTS)
    p.add_argument("--internvl_path", type=str, default=INTERNVL_PATH)
    p.add_argument("--thinkdet_checkpoint", type=str, default=DEFAULT_CKPT)
    p.add_argument("--extract_layer", type=int, default=None)
    p.add_argument("--extract_layers", type=int, nargs="+", default=None)
    p.add_argument("--layer_fusion", type=str, default=None, choices=["mean", "last"])
    return p.parse_args()


def build_dino_transform():
    normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def transform(image_pil):
        w, h = image_pil.size
        scale = 800 / min(w, h)
        if scale * max(w, h) > 1333:
            scale = 1333 / max(w, h)
        nw, nh = int(w * scale), int(h * scale)
        img = image_pil.resize((nw, nh), Image.BILINEAR)
        return normalize(TF.to_tensor(img))

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


def xywh_to_xyxy(box):
    x, y, w, h = box
    return [x, y, x + w, y + h]


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


def outputs_to_predictions(outputs, positive_map_norm, image_size, top_k):
    img_w, img_h = image_size
    scores, boxes = score_outputs_for_query(outputs, positive_map_norm)
    vals, idx = scores.topk(min(int(top_k), scores.shape[0]))
    preds = []
    for j in range(len(vals)):
        cx, cy, bw, bh = boxes[idx[j]].detach().cpu().tolist()
        preds.append({
            "score": float(vals[j].item()),
            "box_abs_xyxy": [
                (cx - bw / 2.0) * img_w,
                (cy - bh / 2.0) * img_h,
                (cx + bw / 2.0) * img_w,
                (cy + bh / 2.0) * img_h,
            ],
        })
    return preds


def evaluate_predictions(preds, gt_boxes, top_k):
    k = min(int(top_k), len(preds))
    if k == 0:
        return {"top1_iou": 0.0, "topk_best_iou": 0.0,
                "hit50_top1": 0.0, "hit50_topk": 0.0,
                "hit75_top1": 0.0, "hit75_topk": 0.0}
    best_iou_per_pred = []
    for pred in preds[:k]:
        ious = [iou_xyxy(pred["box_abs_xyxy"], g) for g in gt_boxes]
        best_iou_per_pred.append(max(ious) if ious else 0.0)
    top1 = best_iou_per_pred[0]
    topk = max(best_iou_per_pred)
    return {
        "top1_iou": top1, "topk_best_iou": topk,
        "hit50_top1": 1.0 if top1 >= 0.5 else 0.0,
        "hit50_topk": 1.0 if topk >= 0.5 else 0.0,
        "hit75_top1": 1.0 if top1 >= 0.75 else 0.0,
        "hit75_topk": 1.0 if topk >= 0.75 else 0.0,
    }


def summarize_rows(rows, top_k):
    n = max(len(rows), 1)
    return {
        "n_samples": len(rows),
        "mean_best_iou_top1": sum(r["top1_iou"] for r in rows) / n,
        "mean_best_iou_topk": sum(r["topk_best_iou"] for r in rows) / n,
        "hit@0.5_top1": sum(r["hit50_top1"] for r in rows) / n,
        "hit@0.5_topk": sum(r["hit50_topk"] for r in rows) / n,
        "hit@0.75_top1": sum(r["hit75_top1"] for r in rows) / n,
        "hit@0.75_topk": sum(r["hit75_topk"] for r in rows) / n,
        "top_k": int(top_k),
    }


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
    return model


@torch.no_grad()
def translate_prompt(internvl_model, tokenizer, prompt, image_pil, ivl_tf, device):
    """Use InternVL to translate a commonsense prompt into a concrete object name."""
    pixel_values = ivl_tf(image_pil).unsqueeze(0).to(device)
    question = TRANSLATE_PROMPT.format(prompt=prompt)
    response = internvl_model.chat(
        tokenizer=tokenizer,
        pixel_values=pixel_values,
        question=question,
        generation_config={
            "max_new_tokens": 20,
            "do_sample": False,
            "top_p": 1.0,
            "pad_token_id": int(getattr(tokenizer, "eos_token_id", 0) or 0),
        },
        history=None,
        return_history=False,
        verbose=False,
    )
    # Clean up: take first line, strip punctuation, lowercase
    translated = response.strip().split("\n")[0].strip().rstrip(".").lower()
    # If response is too long, it's probably garbage — just use first 3 words
    words = translated.split()
    if len(words) > 5:
        translated = " ".join(words[:3])
    return translated


@torch.no_grad()
def run_baseline_detection(gd_model, image_pil, prompt, device, dino_tf, top_k):
    """Run plain GroundingDINO detection."""
    img_w, img_h = image_pil.size
    dino_t = dino_tf(image_pil).unsqueeze(0).to(device)
    mask = torch.zeros(1, dino_t.shape[2], dino_t.shape[3], dtype=torch.bool, device=device)
    nested = NestedTensor(dino_t, mask)
    query_text = prompt if prompt.strip().endswith(".") else (prompt.strip() + " .")
    outputs = gd_model(samples=nested, captions=[query_text])
    tokenizer = gd_model.tokenizer
    special_tokens = get_special_tokens(gd_model)
    pmap = build_positive_map_for_query(tokenizer, special_tokens, query_text)
    return outputs_to_predictions(outputs, pmap, (img_w, img_h), top_k)


@torch.no_grad()
def run_thinkdet_detection(model, image_pil, prompt, device, dino_tf, ivl_tf, top_k):
    """Run ThinkDet detection."""
    img_w, img_h = image_pil.size
    dino_t = dino_tf(image_pil).unsqueeze(0).to(device)
    mask = torch.zeros(1, dino_t.shape[2], dino_t.shape[3], dtype=torch.bool, device=device)
    nested = NestedTensor(dino_t, mask)
    ivl_t = ivl_tf(image_pil).unsqueeze(0).to(device)
    query_text = prompt if prompt.strip().endswith(".") else (prompt.strip() + " .")
    outputs, aux = model(
        ivl_t,
        [prompt],
        {"samples": nested, "captions": [query_text]},
    )
    tokenizer = model.grounding_dino.tokenizer
    special_tokens = get_special_tokens(model.grounding_dino)
    pmap = build_positive_map_for_query(tokenizer, special_tokens, query_text)
    return outputs_to_predictions(outputs, pmap, (img_w, img_h), top_k)


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    with open(args.benchmark, "r") as f:
        bench = json.load(f)

    samples = bench["samples"]
    if args.split != "all":
        samples = [s for s in samples if s.get("split", "test") == args.split]
    if args.max_samples > 0:
        samples = samples[: args.max_samples]
    if args.num_shards > 1:
        samples = samples[args.shard_id :: args.num_shards]
    if not samples:
        raise RuntimeError("No samples selected for evaluation")

    print("=" * 70)
    print("Think-then-Detect Commonsense Evaluation")
    print(f"benchmark: {args.benchmark}")
    print(f"split: {args.split}  n_samples: {len(samples)}  top_k: {args.top_k}")
    print(f"shard: {args.shard_id}/{args.num_shards}  device: {device}")
    print("=" * 70)

    dino_tf = build_dino_transform()
    ivl_tf = build_internvl_transform()

    # Load ThinkDet (includes both InternVL + GroundingDINO)
    print("\nLoading unified ThinkDet...")
    thinkdet = load_unified_model(args, device)
    internvl = thinkdet.feature_extractor.internvl
    ivl_tokenizer = thinkdet.feature_extractor.tokenizer
    gd_model = thinkdet.grounding_dino

    # Track results for 4 conditions:
    # 1. baseline_direct: GroundingDINO with original commonsense prompt
    # 2. baseline_translated: GroundingDINO with LLM-translated prompt
    # 3. thinkdet_direct: ThinkDet with original commonsense prompt
    # 4. thinkdet_translated: ThinkDet with LLM-translated prompt (think-then-detect)
    conditions = ["baseline_direct", "baseline_translated",
                   "thinkdet_direct", "thinkdet_translated"]
    rows = {c: [] for c in conditions}
    per_object = {c: defaultdict(list) for c in conditions}
    translations_log = []

    print("Evaluating...\n")
    for idx, sample in enumerate(samples, start=1):
        image_pil = Image.open(sample["image_path"]).convert("RGB")
        gt_boxes = [xywh_to_xyxy(t["bbox_xywh"]) for t in sample["positive_targets"]]
        orig_prompt = sample["prompt"]

        # Stage 1: LLM translation
        translated = translate_prompt(
            internvl, ivl_tokenizer, orig_prompt, image_pil, ivl_tf, device
        )
        translations_log.append({
            "benchmark_id": sample["benchmark_id"],
            "object_name": sample["object_name"],
            "original_prompt": orig_prompt,
            "translated_prompt": translated,
        })

        # Stage 2: Run all 4 conditions
        # Baseline direct
        preds = run_baseline_detection(gd_model, image_pil, orig_prompt, device, dino_tf, args.top_k)
        m = evaluate_predictions(preds, gt_boxes, args.top_k)
        rows["baseline_direct"].append(m)
        per_object["baseline_direct"][sample["object_id"]].append(m)

        # Baseline translated
        preds = run_baseline_detection(gd_model, image_pil, translated, device, dino_tf, args.top_k)
        m = evaluate_predictions(preds, gt_boxes, args.top_k)
        rows["baseline_translated"].append(m)
        per_object["baseline_translated"][sample["object_id"]].append(m)

        # ThinkDet direct
        preds = run_thinkdet_detection(thinkdet, image_pil, orig_prompt, device, dino_tf, ivl_tf, args.top_k)
        m = evaluate_predictions(preds, gt_boxes, args.top_k)
        rows["thinkdet_direct"].append(m)
        per_object["thinkdet_direct"][sample["object_id"]].append(m)

        # ThinkDet translated (think-then-detect)
        preds = run_thinkdet_detection(thinkdet, image_pil, translated, device, dino_tf, ivl_tf, args.top_k)
        m = evaluate_predictions(preds, gt_boxes, args.top_k)
        rows["thinkdet_translated"].append(m)
        per_object["thinkdet_translated"][sample["object_id"]].append(m)

        if idx % args.log_every == 0:
            print(f"  [{idx}/{len(samples)}]", end="")
            for c in conditions:
                s = summarize_rows(rows[c], args.top_k)
                print(f"  {c}={s['hit@0.5_top1']:.4f}", end="")
            print()

    # Aggregate
    object_name_map = {}
    for obj in bench.get("objects", []):
        object_name_map[obj["object_id"]] = obj["object_name"]

    results = {}
    for c in conditions:
        po = {}
        for obj_id, obj_rows in sorted(per_object[c].items()):
            po[obj_id] = summarize_rows(obj_rows, args.top_k)
        results[c] = {
            "overall": summarize_rows(rows[c], args.top_k),
            "per_object": po,
        }

    payload = {
        "status": "ok",
        "benchmark_path": args.benchmark,
        "split": args.split,
        "n_samples": len(samples),
        "top_k": args.top_k,
        "device": str(device),
        "thinkdet_checkpoint": args.thinkdet_checkpoint,
        "object_name_map": object_name_map,
        "translate_prompt_template": TRANSLATE_PROMPT,
        "results": results,
        "translations_sample": translations_log[:50],
    }

    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)

    # Write markdown summary
    md_path = os.path.splitext(args.output)[0] + ".md"
    with open(md_path, "w") as f:
        f.write("# Think-then-Detect Commonsense Results\n\n")
        f.write(f"- n_samples: {len(samples)}\n")
        f.write(f"- shard: {args.shard_id}/{args.num_shards}\n\n")
        f.write("## Overall\n\n")
        f.write("| condition | hit@0.5_top1 | hit@0.5_topk | mean_iou_top1 |\n")
        f.write("|---|---:|---:|---:|\n")
        for c in conditions:
            r = results[c]["overall"]
            f.write(f"| {c} | {r['hit@0.5_top1']:.4f} | {r['hit@0.5_topk']:.4f} | "
                    f"{r['mean_best_iou_top1']:.4f} |\n")
        f.write("\n## Sample Translations\n\n")
        f.write("| object | original prompt | translated |\n")
        f.write("|---|---|---|\n")
        for t in translations_log[:20]:
            f.write(f"| {t['object_name']} | {t['original_prompt']} | {t['translated_prompt']} |\n")

    # Write CSV
    csv_path = os.path.splitext(args.output)[0] + ".csv"
    with open(csv_path, "w", newline="") as f:
        wr = csv.writer(f)
        header = ["object_id", "object_name"]
        for c in conditions:
            header += [f"{c}_hit@0.5_top1", f"{c}_mean_iou_top1"]
        header.append("n_samples")
        wr.writerow(header)
        for obj_id in sorted(object_name_map.keys()):
            row = [obj_id, object_name_map.get(obj_id, obj_id)]
            for c in conditions:
                m = results[c]["per_object"].get(obj_id, {})
                row.append(f"{m.get('hit@0.5_top1', 0):.6f}")
                row.append(f"{m.get('mean_best_iou_top1', 0):.6f}")
            n = results[conditions[0]]["per_object"].get(obj_id, {}).get("n_samples", 0)
            row.append(n)
            wr.writerow(row)

    print(f"\n[saved] {args.output}")
    print(f"[saved] {md_path}")
    print(f"[saved] {csv_path}")


if __name__ == "__main__":
    main()
