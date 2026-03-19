#!/usr/bin/env python3
"""
Apples-to-apples single-image comparison:
- HF GroundingDINO (threshold configurable, default 0.0)
- ThinkDet primary

Both outputs are rendered with the same top-k display policy and combined into
one side-by-side image.
"""

import argparse
import json
import os
import sys

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


ROOT = "/home/iibrohimm/project/next_step"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "GroundingDINO", "GroundingDINO"))
sys.path.insert(0, SCRIPT_DIR)

from eval_grounding_benchmark_compare import (  # noqa: E402
    DEFAULT_CKPT,
    GD_CONFIG,
    GD_WEIGHTS,
    INTERNVL_PATH,
    build_dino_transform,
    build_internvl_transform,
    build_positive_map_for_query,
    build_scored_bundle,
    bundle_to_predictions,
    get_special_tokens,
    load_unified_model,
    run_thinkdet_outputs,
)
from thinkdet.inference.fallback import summarize_scores  # noqa: E402


DEFAULT_IMAGE = f"{ROOT}/thinkdet/data/Screenshot 2026-03-12 112130.png"
DEFAULT_PROMPT = "tool to eat with"
DEFAULT_HF_MODEL_ID = "IDEA-Research/grounding-dino-tiny"
DEFAULT_OUTPUT_PREFIX = (
    f"{ROOT}/thinkdet/results/qualitative/"
    "apples_to_apples_screenshot_tool_to_eat_with"
)


try:
    FONT_BOLD = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18
    )
    FONT_SMALL = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13
    )
except Exception:
    FONT_BOLD = FONT_SMALL = ImageFont.load_default()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default=DEFAULT_IMAGE)
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--output_prefix", type=str, default=DEFAULT_OUTPUT_PREFIX)

    parser.add_argument("--hf_model_id", type=str, default=DEFAULT_HF_MODEL_ID)
    parser.add_argument("--hf_threshold", type=float, default=0.0)
    parser.add_argument("--hf_text_threshold", type=float, default=0.0)
    parser.add_argument("--local_files_only", action="store_true")
    parser.set_defaults(local_files_only=True)

    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--gd_config", type=str, default=GD_CONFIG)
    parser.add_argument("--gd_weights", type=str, default=GD_WEIGHTS)
    parser.add_argument("--internvl_path", type=str, default=INTERNVL_PATH)
    parser.add_argument("--thinkdet_checkpoint", type=str, default=DEFAULT_CKPT)
    parser.add_argument("--extract_layer", type=int, default=None)
    parser.add_argument("--extract_layers", type=int, nargs="+", default=None)
    parser.add_argument("--layer_fusion", type=str, default=None, choices=["mean", "last"])
    return parser.parse_args()


def iou_xyxy(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (aa + bb - inter + 1e-9)


def normalize_prompt(prompt):
    return prompt if prompt.strip().endswith(".") else (prompt.strip() + " .")


def draw_predictions(image_pil, preds, title, prompt, out_path):
    img = image_pil.copy().convert("RGB")
    draw = ImageDraw.Draw(img)
    colors = [(0, 220, 60), (255, 160, 20), (220, 40, 40), (70, 160, 255), (180, 120, 255)]
    widths = [4, 2, 1, 1, 1]

    for rank, pred in enumerate(preds, start=1):
        x1, y1, x2, y2 = [float(v) for v in pred["box"]]
        color = colors[min(rank - 1, len(colors) - 1)]
        width = widths[min(rank - 1, len(widths) - 1)]
        for off in range(width):
            draw.rectangle([x1 - off, y1 - off, x2 + off, y2 + off], outline=color)
        label = f"#{rank} {pred['score']:.3f}"
        if pred.get("label"):
            label += f" {pred['label']}"
        label_w = max(88, len(label) * 7)
        bar_y = max(0, int(y1) - 18)
        draw.rectangle([x1, bar_y, x1 + label_w, bar_y + 16], fill=color)
        draw.text((x1 + 2, bar_y + 1), label, fill=(255, 255, 255), font=FONT_SMALL)

    bar_h = 38
    panel = Image.new("RGB", (img.width, img.height + bar_h), (28, 28, 28))
    panel.paste(img, (0, bar_h))
    d = ImageDraw.Draw(panel)
    d.text((10, 7), title, fill=(255, 255, 255), font=FONT_BOLD)
    d.text((10, 22), f"Prompt: {prompt}", fill=(150, 230, 160), font=FONT_SMALL)
    panel.save(out_path, quality=95)
    return panel


def make_comparison(panels, prompt, subtitle, out_path):
    gap = 8
    target_h = max(panel.height for panel in panels)
    resized = []
    for panel in panels:
        if panel.height != target_h:
            scale = target_h / float(panel.height)
            panel = panel.resize((int(panel.width * scale), target_h), Image.BILINEAR)
        resized.append(panel)

    total_w = sum(panel.width for panel in resized) + gap * (len(resized) - 1)
    header_h = 52
    canvas = Image.new("RGB", (total_w, target_h + header_h), (18, 18, 18))
    d = ImageDraw.Draw(canvas)
    d.text((10, 8), f'Prompt: "{prompt}"', fill=(120, 255, 140), font=FONT_BOLD)
    d.text((10, 28), subtitle, fill=(180, 180, 180), font=FONT_SMALL)
    x = 0
    for panel in resized:
        canvas.paste(panel, (x, header_h))
        x += panel.width + gap
    canvas.save(out_path, quality=95)


def run_hf_baseline(args, image_pil, device):
    processor = AutoProcessor.from_pretrained(
        args.hf_model_id,
        local_files_only=args.local_files_only,
    )
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        args.hf_model_id,
        local_files_only=args.local_files_only,
    ).to(device)
    model.eval()

    inputs = processor(
        images=image_pil,
        text=[[args.prompt]],
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=args.hf_threshold,
        text_threshold=args.hf_text_threshold,
        target_sizes=[image_pil.size[::-1]],
    )
    result = results[0]
    labels = result.get("text_labels") or result.get("labels") or []

    detections = []
    for box, score, label in zip(result["boxes"], result["scores"], labels):
        detections.append(
            {
                "label": str(label),
                "score": float(score.item()),
                "box": [float(x) for x in box.tolist()],
            }
        )

    detections.sort(key=lambda x: x["score"], reverse=True)
    return detections[: args.top_k]


def run_thinkdet(args, image_pil, device):
    dino_tf = build_dino_transform()
    ivl_tf = build_internvl_transform()

    thinkdet, ckpt_meta = load_unified_model(args, device)
    tokenizer = thinkdet.grounding_dino.tokenizer
    special_tokens = get_special_tokens(thinkdet.grounding_dino)

    with torch.no_grad():
        query_text, outputs, aux = run_thinkdet_outputs(
            thinkdet,
            image_pil,
            args.prompt,
            device,
            dino_tf,
            ivl_tf,
        )
        pmap = build_positive_map_for_query(
            tokenizer,
            special_tokens,
            query_text,
            max_text_len=512,
        )
        bundle = build_scored_bundle(
            "thinkdet",
            outputs,
            pmap,
            query_text,
            aux=aux,
        )
        preds = bundle_to_predictions(bundle, image_pil.size, args.top_k)
        stats = summarize_scores(bundle.scores, bundle.aux)

    detections = []
    for pred in preds:
        detections.append(
            {
                "label": "",
                "score": float(pred["score"]),
                "box": [float(v) for v in pred["box_abs_xyxy"]],
            }
        )

    return {
        "query_used": query_text,
        "stats": stats,
        "detections": detections,
        "checkpoint_meta": ckpt_meta,
    }


def compare_boxes(hf_detections, thinkdet_detections):
    rankwise = []
    n = min(len(hf_detections), len(thinkdet_detections))
    for idx in range(n):
        rankwise.append(
            {
                "rank": idx + 1,
                "iou": iou_xyxy(hf_detections[idx]["box"], thinkdet_detections[idx]["box"]),
            }
        )

    best_iou = 0.0
    best_pair = None
    for i, hf_det in enumerate(hf_detections):
        for j, td_det in enumerate(thinkdet_detections):
            val = iou_xyxy(hf_det["box"], td_det["box"])
            if val > best_iou:
                best_iou = val
                best_pair = {"hf_rank": i + 1, "thinkdet_rank": j + 1, "iou": val}

    return {
        "rankwise_iou": rankwise,
        "best_cross_match": best_pair,
    }


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_prefix), exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    image_pil = Image.open(args.image).convert("RGB")
    image_pil.filename = args.image

    hf_detections = run_hf_baseline(args, image_pil, device)
    thinkdet_result = run_thinkdet(args, image_pil, device)
    thinkdet_detections = thinkdet_result["detections"]
    overlap = compare_boxes(hf_detections, thinkdet_detections)

    hf_image_path = args.output_prefix + "_hf.jpg"
    thinkdet_image_path = args.output_prefix + "_thinkdet.jpg"
    comparison_path = args.output_prefix + "_comparison.jpg"
    json_path = args.output_prefix + ".json"

    hf_panel = draw_predictions(
        image_pil=image_pil,
        preds=hf_detections,
        title=f"HF GroundingDINO ({args.hf_model_id})",
        prompt=args.prompt,
        out_path=hf_image_path,
    )
    thinkdet_panel = draw_predictions(
        image_pil=image_pil,
        preds=thinkdet_detections,
        title="ThinkDet Primary",
        prompt=args.prompt,
        out_path=thinkdet_image_path,
    )
    subtitle = (
        f"HF threshold={args.hf_threshold:.2f}, text_threshold={args.hf_text_threshold:.2f}, "
        f"top_k={args.top_k}"
    )
    make_comparison(
        panels=[hf_panel, thinkdet_panel],
        prompt=args.prompt,
        subtitle=subtitle,
        out_path=comparison_path,
    )

    payload = {
        "image": args.image,
        "prompt": args.prompt,
        "device": str(device),
        "top_k": args.top_k,
        "hf": {
            "model_id": args.hf_model_id,
            "threshold": args.hf_threshold,
            "text_threshold": args.hf_text_threshold,
            "detections": hf_detections,
            "output_image": hf_image_path,
        },
        "thinkdet": {
            "checkpoint": args.thinkdet_checkpoint,
            "query_used": thinkdet_result["query_used"],
            "stats": thinkdet_result["stats"],
            "detections": thinkdet_detections,
            "checkpoint_meta": thinkdet_result["checkpoint_meta"],
            "output_image": thinkdet_image_path,
        },
        "overlap": overlap,
        "comparison_image": comparison_path,
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
