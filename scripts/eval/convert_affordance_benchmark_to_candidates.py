#!/usr/bin/env python3
"""
Convert the held-out affordance benchmark into a candidate-centric JSON format.

The output is a plain JSON list where each row has:
- image_id
- affordance_id
- affordance_prompt
- target_categories
- candidate_source
- candidates
- target_candidate_ids
- schema_version

`ann_id` is used as the stable COCO annotation identifier for each candidate.
"""

import argparse
import json
import os
from collections import defaultdict


ROOT = "/home/iibrohimm/project/next_step"
DEFAULT_BENCHMARK = (
    f"{ROOT}/thinkdet/data/benchmarks/affordance_coco_val_heldout_v2.json"
)
DEFAULT_COCO_ANN = f"{ROOT}/dataSets/coco/annotations/instances_val2017.json"
DEFAULT_OUTPUT = (
    f"{ROOT}/thinkdet/data/benchmarks/affordance_coco_val_heldout_candidates_v2.json"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=str, default=DEFAULT_BENCHMARK)
    parser.add_argument(
        "--coco_ann",
        type=str,
        default="",
        help="Optional override for the COCO annotation file.",
    )
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def xywh_to_xyxy(box_xywh):
    x, y, w, h = box_xywh
    return [x, y, x + w, y + h]


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def resolve_samples(payload):
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("samples"), list):
        return payload["samples"]
    raise ValueError("Benchmark JSON must be a list or a dict with a 'samples' list")


def resolve_affordance_prompt_map(payload):
    if not isinstance(payload, dict):
        return {}
    affordances = payload.get("affordances")
    if not isinstance(affordances, list):
        return {}
    return {
        row["id"]: row["prompt"]
        for row in affordances
        if isinstance(row, dict) and "id" in row and "prompt" in row
    }


def resolve_coco_ann_path(payload, override_path):
    if override_path:
        return override_path
    if isinstance(payload, dict):
        source = payload.get("source_dataset", {})
        if isinstance(source, dict) and source.get("annotation_file"):
            return source["annotation_file"]
    return DEFAULT_COCO_ANN


def build_coco_indexes(coco_payload):
    cat_id_to_name = {int(row["id"]): row["name"] for row in coco_payload["categories"]}
    img_id_to_anns = defaultdict(list)
    for ann in coco_payload["annotations"]:
        img_id_to_anns[int(ann["image_id"])].append(ann)
    for image_id in img_id_to_anns:
        img_id_to_anns[image_id].sort(key=lambda row: int(row["id"]))
    return cat_id_to_name, img_id_to_anns


def convert_sample(sample, cat_id_to_name, img_id_to_anns, affordance_prompt_map):
    image_id = int(sample["image_id"])
    anns = img_id_to_anns.get(image_id)
    if anns is None:
        raise KeyError(f"No COCO annotations found for image_id={image_id}")

    affordance_id = sample["affordance_id"]
    affordance_prompt = sample.get("prompt") or affordance_prompt_map.get(affordance_id)
    if not affordance_prompt:
        raise KeyError(f"Missing affordance prompt for affordance_id={affordance_id}")

    target_categories = list(sample["target_categories"])
    target_set = set(target_categories)
    candidates = []
    target_candidate_ids = []

    for ann in anns:
        ann_id = int(ann["id"])
        category = cat_id_to_name[int(ann["category_id"])]
        candidates.append(
            {
                "ann_id": ann_id,
                "category": category,
                "bbox_xyxy": xywh_to_xyxy(ann["bbox"]),
            }
        )
        if category in target_set:
            target_candidate_ids.append(ann_id)

    benchmark_positive_ids = {
        int(row["ann_id"]) for row in sample.get("positive_targets", [])
    }
    if benchmark_positive_ids and set(target_candidate_ids) != benchmark_positive_ids:
        raise ValueError(
            "Target id mismatch for image_id={} affordance_id={}: "
            "converted={} benchmark={}".format(
                image_id,
                affordance_id,
                sorted(target_candidate_ids),
                sorted(benchmark_positive_ids),
            )
        )

    return {
        "image_id": image_id,
        "affordance_id": affordance_id,
        "affordance_prompt": affordance_prompt,
        "target_categories": target_categories,
        "candidate_source": "coco_gt",
        "candidates": candidates,
        "target_candidate_ids": target_candidate_ids,
        "schema_version": "1.0",
    }


def main():
    args = parse_args()
    benchmark_payload = load_json(args.benchmark)
    samples = resolve_samples(benchmark_payload)
    affordance_prompt_map = resolve_affordance_prompt_map(benchmark_payload)

    coco_ann_path = resolve_coco_ann_path(benchmark_payload, args.coco_ann)
    coco_payload = load_json(coco_ann_path)
    cat_id_to_name, img_id_to_anns = build_coco_indexes(coco_payload)

    output_rows = [
        convert_sample(sample, cat_id_to_name, img_id_to_anns, affordance_prompt_map)
        for sample in samples
    ]

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output_rows, f, indent=2)

    print("=" * 72)
    print("Converted affordance benchmark to candidate-centric format")
    print(f"input_benchmark={args.benchmark}")
    print(f"coco_ann={coco_ann_path}")
    print(f"output={args.output}")
    print(f"samples={len(output_rows)}")
    print("=" * 72)


if __name__ == "__main__":
    main()
