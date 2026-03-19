#!/usr/bin/env python3
"""
Build a deterministic phrase-grounding benchmark from Flickr30k Entities.

The output keeps only boxed phrases, deduplicates repeated mentions across
captions by (image_id, phrase_id, normalized phrase), and aggregates nearby
caption context so the benchmark can support both positive grounding and
confusing-negative analysis later.
"""

import argparse
import datetime
import json
import os
import random
import re
from collections import Counter, defaultdict


ROOT = "/home/iibrohimm/project/next_step"
DEFAULT_ANNOTATIONS = f"{ROOT}/dataSets/flickr30k_entities/annotations.json"
DEFAULT_SPLIT_FILE = f"{ROOT}/dataSets/flickr30k_entities/raw_source/val.txt"
DEFAULT_OUTPUT = (
    f"{ROOT}/thinkdet/data/benchmarks/"
    "flickr30k_entities_val_grounding_10k_v1.json"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=str, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--split_file", type=str, default=DEFAULT_SPLIT_FILE)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--benchmark_name", type=str, default="")
    parser.add_argument("--target_total_samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260310)
    parser.add_argument("--max_negative_phrases", type=int, default=12)
    parser.add_argument("--max_context_phrases", type=int, default=12)
    parser.add_argument("--max_caption_examples", type=int, default=3)
    return parser.parse_args()


def normalize_text(text):
    text = (text or "").strip().lower().replace("`", "'")
    text = re.sub(r"\s+'s\b", "'s", text)
    text = re.sub(r"\s+'\s+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .,;:!?\"'")


def xyxy_abs_to_norm(box_xyxy, img_w, img_h):
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    if img_w <= 0 or img_h <= 0:
        return [0.0, 0.0, 0.0, 0.0]
    cx = (x1 + 0.5 * w) / float(img_w)
    cy = (y1 + 0.5 * h) / float(img_h)
    return [
        max(0.0, min(1.0, cx)),
        max(0.0, min(1.0, cy)),
        max(0.0, min(1.0, w / float(img_w))),
        max(0.0, min(1.0, h / float(img_h))),
    ]


def load_split_ids(path):
    with open(path, "r") as f:
        return {line.strip().replace(".jpg", "") for line in f if line.strip()}


def build_grouped_samples(
    annotations,
    split_ids,
    max_negative_phrases,
    max_context_phrases,
    max_caption_examples,
):
    grouped = {}
    skipped_missing_images = 0
    skipped_bad_size = 0
    skipped_empty_phrase = 0

    for item in annotations:
        image_id = str(item.get("image_id", "")).strip()
        if image_id not in split_ids:
            continue

        image_path = item.get("image_path") or ""
        width = int(item.get("width") or 0)
        height = int(item.get("height") or 0)
        if width <= 0 or height <= 0:
            skipped_bad_size += 1
            continue
        if not image_path or not os.path.exists(image_path):
            skipped_missing_images += 1
            continue

        caption = (item.get("caption") or "").strip()
        negative_phrases = [
            p for p in (normalize_text(x) for x in item.get("negative_phrases") or []) if p
        ]

        boxed_phrases = []
        for phrase_item in item.get("phrases") or []:
            if phrase_item.get("is_nobox", False):
                continue
            boxes = phrase_item.get("boxes") or []
            phrase = normalize_text(phrase_item.get("phrase"))
            if not boxes:
                continue
            if not phrase:
                skipped_empty_phrase += 1
                continue
            boxed_phrases.append((phrase, phrase_item))

        if not boxed_phrases:
            continue

        caption_existing = sorted({phrase for phrase, _ in boxed_phrases})

        for prompt, phrase_item in boxed_phrases:
            phrase_id = phrase_item.get("phrase_id")
            if phrase_id is None:
                continue

            abs_boxes = []
            norm_boxes = []
            for box in phrase_item.get("boxes") or []:
                if len(box) != 4:
                    continue
                abs_box = [float(v) for v in box]
                norm_box = xyxy_abs_to_norm(abs_box, width, height)
                if norm_box[2] <= 0.0 or norm_box[3] <= 0.0:
                    continue
                abs_boxes.append(abs_box)
                norm_boxes.append(norm_box)
            if not abs_boxes:
                continue

            key = (image_id, str(phrase_id), prompt)
            if key not in grouped:
                grouped[key] = {
                    "image_id": image_id,
                    "image_file": os.path.basename(image_path),
                    "image_path": image_path,
                    "width": width,
                    "height": height,
                    "phrase_id": str(phrase_id),
                    "prompt": prompt,
                    "target_boxes_abs_xyxy": abs_boxes,
                    "target_boxes_norm_cxcywh": norm_boxes,
                    "target_box_count": len(norm_boxes),
                    "is_multi_box": len(norm_boxes) > 1,
                    "_all_captions": set(),
                    "_caption_examples": [],
                    "_negative_phrases": set(),
                    "_context_existing_phrases": set(),
                }

            rec = grouped[key]
            if caption and caption not in rec["_all_captions"]:
                rec["_all_captions"].add(caption)
                if len(rec["_caption_examples"]) < max_caption_examples:
                    rec["_caption_examples"].append(caption)

            rec["_negative_phrases"].update(
                p for p in negative_phrases if p and p != prompt
            )
            rec["_context_existing_phrases"].update(
                p for p in caption_existing if p and p != prompt
            )

    rows = []
    for idx, rec in enumerate(sorted(grouped.values(), key=lambda x: (x["image_id"], x["phrase_id"], x["prompt"]))):
        prompt_word_count = len(rec["prompt"].split())
        negative_phrases = sorted(rec["_negative_phrases"])[:max_negative_phrases]
        context_existing = sorted(rec["_context_existing_phrases"])[:max_context_phrases]
        rows.append(
            {
                "benchmark_id": f"flickr_val_{idx:06d}",
                "image_id": rec["image_id"],
                "image_file": rec["image_file"],
                "image_path": rec["image_path"],
                "width": rec["width"],
                "height": rec["height"],
                "phrase_id": rec["phrase_id"],
                "prompt": rec["prompt"],
                "prompt_word_count": prompt_word_count,
                "split": "val",
                "target_box_count": rec["target_box_count"],
                "is_multi_box": rec["is_multi_box"],
                "target_boxes_abs_xyxy": rec["target_boxes_abs_xyxy"],
                "target_boxes_norm_cxcywh": rec["target_boxes_norm_cxcywh"],
                "source_caption_count": len(rec["_all_captions"]),
                "source_caption_examples": rec["_caption_examples"],
                "negative_phrases": negative_phrases,
                "context_existing_phrases": context_existing,
            }
        )

    stats = {
        "grouped_sample_count": len(rows),
        "skipped_missing_images": skipped_missing_images,
        "skipped_bad_size": skipped_bad_size,
        "skipped_empty_phrase": skipped_empty_phrase,
    }
    return rows, stats


def deterministic_round_robin_by_image(rows, target_total, seed):
    rng = random.Random(seed)
    rows_by_image = defaultdict(list)
    for row in rows:
        rows_by_image[row["image_id"]].append(row)

    image_ids = list(rows_by_image.keys())
    rng.shuffle(image_ids)
    for image_id in image_ids:
        rng.shuffle(rows_by_image[image_id])

    target = min(int(target_total), len(rows))
    chosen = []
    round_idx = 0
    while len(chosen) < target:
        added_any = False
        for image_id in image_ids:
            bucket = rows_by_image[image_id]
            if round_idx >= len(bucket):
                continue
            chosen.append(bucket[round_idx])
            added_any = True
            if len(chosen) >= target:
                break
        if not added_any:
            break
        round_idx += 1
    return chosen


def compute_summary(selected_rows, available_count):
    unique_images = len({row["image_id"] for row in selected_rows})
    unique_prompts = len({row["prompt"] for row in selected_rows})
    multi_box_samples = sum(int(row["is_multi_box"]) for row in selected_rows)
    total_boxes = sum(int(row["target_box_count"]) for row in selected_rows)
    prompt_words = [int(row["prompt_word_count"]) for row in selected_rows]
    per_image_counts = Counter(row["image_id"] for row in selected_rows)
    word_hist = Counter(prompt_words)

    return {
        "available_grouped_samples": int(available_count),
        "selected_samples": len(selected_rows),
        "unique_images": unique_images,
        "unique_prompts": unique_prompts,
        "multi_box_samples": multi_box_samples,
        "single_box_samples": len(selected_rows) - multi_box_samples,
        "average_boxes_per_sample": (float(total_boxes) / float(len(selected_rows))) if selected_rows else 0.0,
        "average_prompt_word_count": (float(sum(prompt_words)) / float(len(prompt_words))) if prompt_words else 0.0,
        "max_samples_per_image": max(per_image_counts.values()) if per_image_counts else 0,
        "prompt_word_histogram": {
            str(k): int(v) for k, v in sorted(word_hist.items(), key=lambda kv: (kv[0], kv[1]))
        },
    }


def write_markdown_summary(path, payload):
    summary = payload["summary"]
    protocol = payload["construction_protocol"]
    source = payload["source_dataset"]
    samples = payload["samples"]

    with open(path, "w") as f:
        f.write(f"# {payload['benchmark_name']}\n\n")
        f.write(f"- source_dataset: `{source['name']}`\n")
        f.write(f"- split: `{source['split']}`\n")
        f.write(f"- annotations: `{source['annotations_file']}`\n")
        f.write(f"- split_file: `{source['split_file']}`\n")
        f.write(f"- target_total_samples: {protocol['requested_target_total_samples']}\n")
        f.write(f"- selected_samples: {summary['selected_samples']}\n")
        f.write(f"- available_grouped_samples: {summary['available_grouped_samples']}\n")
        f.write(f"- unique_images: {summary['unique_images']}\n")
        f.write(f"- unique_prompts: {summary['unique_prompts']}\n")
        f.write(f"- multi_box_samples: {summary['multi_box_samples']}\n")
        f.write(f"- average_boxes_per_sample: {summary['average_boxes_per_sample']:.4f}\n")
        f.write(f"- average_prompt_word_count: {summary['average_prompt_word_count']:.4f}\n")
        f.write(f"- max_samples_per_image: {summary['max_samples_per_image']}\n\n")

        f.write("## Sample Examples\n\n")
        f.write("| benchmark_id | image_id | prompt | boxes | negatives |\n")
        f.write("|---|---:|---|---:|---|\n")
        for row in samples[:15]:
            negatives = ", ".join(row["negative_phrases"][:3])
            f.write(
                f"| {row['benchmark_id']} | {row['image_id']} | {row['prompt']} | "
                f"{row['target_box_count']} | {negatives} |\n"
            )


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    with open(args.annotations, "r") as f:
        annotations = json.load(f)

    split_ids = load_split_ids(args.split_file)
    rows, prep_stats = build_grouped_samples(
        annotations=annotations,
        split_ids=split_ids,
        max_negative_phrases=args.max_negative_phrases,
        max_context_phrases=args.max_context_phrases,
        max_caption_examples=args.max_caption_examples,
    )
    selected = deterministic_round_robin_by_image(
        rows=rows,
        target_total=args.target_total_samples,
        seed=args.seed,
    )

    benchmark_name = (
        args.benchmark_name.strip()
        or os.path.splitext(os.path.basename(args.output))[0]
    )
    split_name = os.path.splitext(os.path.basename(args.split_file))[0]
    payload = {
        "status": "ok",
        "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "benchmark_name": benchmark_name,
        "version": 1,
        "source_dataset": {
            "name": "Flickr30k Entities",
            "split": split_name,
            "annotations_file": args.annotations,
            "split_file": args.split_file,
            "image_root": os.path.join(os.path.dirname(args.annotations), "images"),
        },
        "construction_protocol": {
            "model_inference_used": False,
            "sampling": "deterministic round-robin across images after seeded shuffle",
            "seed": args.seed,
            "dedupe_key": ["image_id", "phrase_id", "normalized_prompt"],
            "requested_target_total_samples": int(args.target_total_samples),
            "max_negative_phrases": int(args.max_negative_phrases),
            "max_context_phrases": int(args.max_context_phrases),
            "max_caption_examples": int(args.max_caption_examples),
        },
        "preparation_stats": prep_stats,
        "summary": compute_summary(selected_rows=selected, available_count=len(rows)),
        "samples": selected,
    }

    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)

    md_path = os.path.splitext(args.output)[0] + ".md"
    write_markdown_summary(md_path, payload)

    print(f"[saved] {args.output}")
    print(f"[saved] {md_path}")
    print(
        f"[summary] benchmark={benchmark_name} "
        f"available={payload['summary']['available_grouped_samples']} "
        f"selected={payload['summary']['selected_samples']} "
        f"unique_images={payload['summary']['unique_images']}"
    )


if __name__ == "__main__":
    main()
