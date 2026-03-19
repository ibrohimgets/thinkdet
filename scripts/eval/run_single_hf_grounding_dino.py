#!/usr/bin/env python3
"""
Run a single prompt on a single image with Hugging Face GroundingDINO.

This uses the HF `transformers` API style the user requested. The default
model is the cached Swin-T-equivalent checkpoint.
"""

import argparse
import json
import os

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


ROOT = "/home/iibrohimm/project/next_step"
DEFAULT_IMAGE = f"{ROOT}/thinkdet/data/Screenshot 2026-03-12 112130.png"
DEFAULT_MODEL_ID = "IDEA-Research/grounding-dino-tiny"
DEFAULT_PROMPT = "tool to eat with"
DEFAULT_OUTPUT_IMAGE = (
    f"{ROOT}/thinkdet/results/qualitative/"
    "hf_grounding_dino_screenshot_tool_to_eat_with.jpg"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default=DEFAULT_IMAGE)
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--box_threshold", type=float, default=0.4)
    parser.add_argument("--text_threshold", type=float, default=0.3)
    parser.add_argument("--output_image", type=str, default=DEFAULT_OUTPUT_IMAGE)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--local_files_only", action="store_true")
    parser.set_defaults(local_files_only=True)
    return parser.parse_args()


try:
    FONT_BOLD = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18
    )
    FONT_SMALL = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13
    )
except Exception:
    FONT_BOLD = FONT_SMALL = ImageFont.load_default()


def draw_results(image, detections, title, prompt, output_path):
    canvas = image.copy().convert("RGB")
    draw = ImageDraw.Draw(canvas)
    colors = [(0, 220, 60), (255, 160, 20), (220, 40, 40), (70, 160, 255), (180, 120, 255)]
    widths = [4, 2, 1, 1, 1]

    for rank, det in enumerate(detections, start=1):
        x1, y1, x2, y2 = det["box"]
        color = colors[min(rank - 1, len(colors) - 1)]
        width = widths[min(rank - 1, len(widths) - 1)]
        for off in range(width):
            draw.rectangle([x1 - off, y1 - off, x2 + off, y2 + off], outline=color)
        label = f"#{rank} {det['score']:.3f} {det['label']}"
        label_w = max(90, len(label) * 7)
        bar_y = max(0, int(y1) - 18)
        draw.rectangle([x1, bar_y, x1 + label_w, bar_y + 16], fill=color)
        draw.text((x1 + 2, bar_y + 1), label, fill=(255, 255, 255), font=FONT_SMALL)

    bar_h = 38
    panel = Image.new("RGB", (canvas.width, canvas.height + bar_h), (28, 28, 28))
    panel.paste(canvas, (0, bar_h))
    d = ImageDraw.Draw(panel)
    d.text((10, 7), title, fill=(255, 255, 255), font=FONT_BOLD)
    d.text((10, 22), f"Prompt: {prompt}", fill=(150, 230, 160), font=FONT_SMALL)
    panel.save(output_path, quality=95)


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_image), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = Image.open(args.image).convert("RGB")

    processor = AutoProcessor.from_pretrained(
        args.model_id,
        local_files_only=args.local_files_only,
    )
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        args.model_id,
        local_files_only=args.local_files_only,
    ).to(device)
    model.eval()

    text_labels = [[args.prompt]]
    inputs = processor(images=image, text=text_labels, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        target_sizes=[image.size[::-1]],
    )

    detections = []
    result = results[0]
    for box, score, label in zip(result["boxes"], result["scores"], result["labels"]):
        detections.append(
            {
                "label": str(label),
                "score": float(score.item()),
                "box": [round(float(x), 2) for x in box.tolist()],
            }
        )

    detections.sort(key=lambda x: x["score"], reverse=True)
    detections = detections[: args.top_k]

    draw_results(
        image=image,
        detections=detections,
        title=f"HF GroundingDINO: {args.model_id}",
        prompt=args.prompt,
        output_path=args.output_image,
    )

    output_json = os.path.splitext(args.output_image)[0] + ".json"
    payload = {
        "image": args.image,
        "prompt": args.prompt,
        "model_id": args.model_id,
        "device": str(device),
        "box_threshold": args.box_threshold,
        "text_threshold": args.text_threshold,
        "detections": detections,
        "output_image": args.output_image,
    }
    with open(output_json, "w") as f:
        json.dump(payload, f, indent=2)

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
