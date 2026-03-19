#!/usr/bin/env python3
"""
Build a deterministic commonsense grounding benchmark from a markdown prompt file.

The benchmark keeps only objects that exactly match COCO category names so the
evaluation target is unambiguous with existing GT annotations.
"""

import argparse
import datetime
import json
import math
import os
import random
import re
from collections import Counter, defaultdict


ROOT = "/home/iibrohimm/project/next_step"
DEFAULT_PROMPTS = (
    f"{ROOT}/thinkdet/results/prompts/refcoco_commonsense_prompts_100.md"
)
DEFAULT_COCO_ANN = f"{ROOT}/dataSets/coco/annotations/instances_val2017.json"
DEFAULT_COCO_IMG = f"{ROOT}/dataSets/coco/val2017"
DEFAULT_OUT = (
    f"{ROOT}/thinkdet/data/benchmarks/"
    "commonsense_refcoco_objects_coco_val_v1.json"
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--prompts_md", type=str, default=DEFAULT_PROMPTS)
    p.add_argument("--coco_ann", type=str, default=DEFAULT_COCO_ANN)
    p.add_argument("--image_dir", type=str, default=DEFAULT_COCO_IMG)
    p.add_argument("--output", type=str, default=DEFAULT_OUT)
    p.add_argument("--benchmark_name", type=str, default="")
    p.add_argument("--images_per_object", type=int, default=5)
    p.add_argument(
        "--target_total_samples",
        type=int,
        default=0,
        help="If > 0, resolve a balanced images-per-object target automatically.",
    )
    p.add_argument("--seed", type=int, default=20260309)
    p.add_argument("--min_total_anns", type=int, default=3)
    p.add_argument("--min_distractor_anns", type=int, default=1)
    p.add_argument("--allow_image_reuse", action="store_true")
    return p.parse_args()


def slugify(text):
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def parse_prompt_markdown(path):
    with open(path, "r") as f:
        text = f.read()

    blocks = re.findall(r"^Object: (.*?)\n\n((?:- .*\n?)+)", text, flags=re.M)
    items = []
    for object_name, bullet_block in blocks:
        prompts = [ln[2:].strip() for ln in bullet_block.splitlines() if ln.startswith("- ")]
        items.append({
            "object_name": object_name.strip(),
            "prompts": prompts,
        })
    return items


def xywh_abs_to_norm(box_xywh, img_w, img_h):
    x, y, w, h = box_xywh
    cx = (x + 0.5 * w) / img_w
    cy = (y + 0.5 * h) / img_h
    return [cx, cy, w / img_w, h / img_h]


def build_candidate_pool(
    coco_data,
    object_name,
    image_dir,
    min_total_anns,
    min_distractor_anns,
):
    cat_id_to_name = {c["id"]: c["name"] for c in coco_data["categories"]}
    img_id_to_info = {im["id"]: im for im in coco_data["images"]}
    img_anns = defaultdict(list)
    for ann in coco_data["annotations"]:
        img_anns[ann["image_id"]].append(ann)

    pool = []
    for img_id, anns in img_anns.items():
        info = img_id_to_info.get(img_id)
        if info is None:
            continue
        if len(anns) < min_total_anns:
            continue

        positives = []
        distractors = []
        for ann in anns:
            cat_name = cat_id_to_name[ann["category_id"]]
            if cat_name == object_name:
                positives.append({
                    "ann_id": ann["id"],
                    "category_name": cat_name,
                    "bbox_xywh": ann["bbox"],
                    "bbox_norm_cxcywh": xywh_abs_to_norm(
                        ann["bbox"], info["width"], info["height"]
                    ),
                    "area": ann["area"],
                    "iscrowd": ann.get("iscrowd", 0),
                })
            else:
                distractors.append(ann)

        if not positives:
            continue
        if len(distractors) < min_distractor_anns:
            continue

        cat_counter = Counter(cat_id_to_name[a["category_id"]] for a in anns)
        pool.append({
            "image_id": img_id,
            "image_file": info["file_name"],
            "image_path": os.path.join(image_dir, info["file_name"]),
            "width": int(info["width"]),
            "height": int(info["height"]),
            "num_annotations": len(anns),
            "num_distractor_annotations": len(distractors),
            "category_histogram": dict(cat_counter),
            "positive_category_histogram": {object_name: len(positives)},
            "positive_targets": positives,
        })
    return pool


def deterministic_pick(pool, count, rng, used_image_ids, allow_image_reuse):
    idxs = list(range(len(pool)))
    rng.shuffle(idxs)
    chosen = []
    for idx in idxs:
        row = pool[idx]
        if (not allow_image_reuse) and row["image_id"] in used_image_ids:
            continue
        chosen.append(row)
        used_image_ids.add(row["image_id"])
        if len(chosen) >= count:
            break
    return chosen


def resolve_images_per_object_target(evaluable_items, pool_by_object, requested_total, fallback_target):
    if requested_total <= 0:
        return int(fallback_target)

    prompt_counts = [len(item["prompts"]) for item in evaluable_items]
    if not prompt_counts:
        raise ValueError("No evaluable prompt/object pairs available to build benchmark")

    estimated = max(1, int(math.ceil(float(requested_total) / float(sum(prompt_counts)))))
    max_pool = max((len(pool_by_object[item["object_name"]]) for item in evaluable_items), default=0)

    def realized_total(images_per_object):
        return sum(
            min(len(pool_by_object[item["object_name"]]), int(images_per_object)) * len(item["prompts"])
            for item in evaluable_items
        )

    target = estimated
    while target <= max_pool and realized_total(target) < requested_total:
        target += 1
    return int(target)


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    prompt_items = parse_prompt_markdown(args.prompts_md)
    with open(args.coco_ann, "r") as f:
        coco = json.load(f)

    coco_categories = {c["name"] for c in coco["categories"]}
    evaluable = []
    skipped = []
    for item in prompt_items:
        if item["object_name"] in coco_categories:
            evaluable.append(item)
        else:
            skipped.append(item["object_name"])

    pool_by_object = {
        item["object_name"]: build_candidate_pool(
            coco,
            item["object_name"],
            args.image_dir,
            args.min_total_anns,
            args.min_distractor_anns,
        )
        for item in evaluable
    }

    benchmark_name = (
        args.benchmark_name.strip()
        or os.path.splitext(os.path.basename(args.output))[0]
    )
    resolved_images_per_object = resolve_images_per_object_target(
        evaluable,
        pool_by_object,
        int(args.target_total_samples),
        int(args.images_per_object),
    )

    rng = random.Random(args.seed)
    used_image_ids = set()

    object_defs = []
    per_object_summary = []
    samples = []

    for item in evaluable:
        object_name = item["object_name"]
        object_id = slugify(object_name)
        pool = pool_by_object[object_name]
        chosen = deterministic_pick(
            pool,
            resolved_images_per_object,
            rng,
            used_image_ids,
            args.allow_image_reuse,
        )
        if not chosen:
            continue

        object_defs.append({
            "object_id": object_id,
            "object_name": object_name,
            "target_categories": [object_name],
            "prompts": item["prompts"],
        })
        per_object_summary.append({
            "object_id": object_id,
            "object_name": object_name,
            "prompt_count": len(item["prompts"]),
            "candidate_pool_size": len(pool),
            "selected_images": len(chosen),
            "selected_samples": len(chosen) * len(item["prompts"]),
        })

        for row in chosen:
            for prompt_index, prompt in enumerate(item["prompts"], start=1):
                sample = dict(row)
                sample.update({
                    "benchmark_id": f"{object_id}_{row['image_id']}_p{prompt_index}",
                    "object_id": object_id,
                    "object_name": object_name,
                    "prompt_index": prompt_index,
                    "prompt": prompt if prompt.endswith(".") else f"{prompt} .",
                    "target_categories": [object_name],
                    "split": "test",
                })
                samples.append(sample)

    payload = {
        "status": "ok",
        "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "benchmark_name": benchmark_name,
        "version": 1,
        "source_dataset": {
            "name": "COCO",
            "split": "val2017",
            "annotation_file": args.coco_ann,
            "image_dir": args.image_dir,
        },
        "prompt_source": {
            "path": args.prompts_md,
            "total_prompt_objects": len(prompt_items),
            "evaluable_object_count": len(object_defs),
            "skipped_object_count": len(skipped),
            "skipped_objects": skipped,
        },
        "construction_protocol": {
            "model_inference_used": False,
            "sampling": "deterministic random (seeded), balanced by object",
            "seed": args.seed,
            "requested_target_total_samples": (
                int(args.target_total_samples) if args.target_total_samples > 0 else None
            ),
            "images_per_object_target": resolved_images_per_object,
            "min_total_anns": args.min_total_anns,
            "min_distractor_anns": args.min_distractor_anns,
            "allow_image_reuse": bool(args.allow_image_reuse),
        },
        "objects": object_defs,
        "summary": {
            "total_samples": len(samples),
            "unique_images": len({s["image_id"] for s in samples}),
            "object_count": len(object_defs),
            "samples_per_object": {
                row["object_id"]: row["selected_samples"] for row in per_object_summary
            },
            "per_object_pool_and_selection": per_object_summary,
        },
        "samples": samples,
    }

    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)

    md_path = os.path.splitext(args.output)[0] + ".md"
    with open(md_path, "w") as f:
        f.write(f"# {benchmark_name}\n\n")
        f.write(f"- prompts: `{args.prompts_md}`\n")
        f.write(f"- evaluable_objects: {len(object_defs)}\n")
        f.write(f"- skipped_objects: {len(skipped)}\n")
        f.write(f"- total_samples: {len(samples)}\n")
        f.write(f"- unique_images: {len({s['image_id'] for s in samples})}\n")
        if args.target_total_samples > 0:
            f.write(f"- requested_target_total_samples: {args.target_total_samples}\n")
        f.write(f"- images_per_object_target: {resolved_images_per_object}\n\n")
        if skipped:
            f.write("## Skipped Objects\n\n")
            for name in skipped:
                f.write(f"- {name}\n")
            f.write("\n")
        f.write("## Per-Object Selection\n\n")
        f.write("| object | pool | selected_images | selected_samples |\n")
        f.write("|---|---:|---:|---:|\n")
        for row in per_object_summary:
            f.write(
                f"| {row['object_name']} | {row['candidate_pool_size']} | "
                f"{row['selected_images']} | {row['selected_samples']} |\n"
            )

    print(f"[saved] {args.output}")
    print(f"[saved] {md_path}")
    print(
        f"[summary] benchmark={benchmark_name} evaluable_objects={len(object_defs)} "
        f"skipped={len(skipped)} images_per_object={resolved_images_per_object} "
        f"total_samples={len(samples)}"
    )


if __name__ == "__main__":
    main()
