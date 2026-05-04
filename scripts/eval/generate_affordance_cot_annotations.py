#!/usr/bin/env python3
"""
Generate structured Chain-of-Thought annotations for the held-out affordance
benchmark from the candidate-centric JSON.

This generator is intentionally benchmark-aligned:
- it uses the benchmark's target categories as the canonical positives
- it produces semantic explanations for each candidate
- it uses bounding-box position/scale descriptors for light grounding

It does not perform true pixel-level image understanding.
"""

import argparse
import json
import os
from collections import Counter


ROOT = "/home/iibrohimm/project/next_step"
DEFAULT_INPUT = (
    f"{ROOT}/thinkdet/data/benchmarks/"
    "affordance_coco_val_heldout_candidates_v2.json"
)
DEFAULT_COCO_ANN = f"{ROOT}/dataSets/coco/annotations/instances_val2017.json"
DEFAULT_OUTPUT = (
    f"{ROOT}/thinkdet/data/benchmarks/affordance_coco_val_heldout_cot_v2.json"
)


AFFORDANCE_SPECS = {
    "drink_from": {
        "goal": "objects used as drinking vessels",
        "properties": [
            "can hold liquid",
            "has an opening suitable for drinking",
            "is normally used as a vessel rather than as furniture or a tool",
        ],
        "look_for": [
            "containers sized and shaped for beverages",
            "vessel-like objects with a hollow interior",
            "canonical drinkware categories in the benchmark",
        ],
        "positive_reason": {
            "cup": "A cup is a standard drinking vessel with an opening and a hollow interior for holding liquid.",
            "bottle": "A bottle is designed to contain liquid and can be used directly for drinking.",
            "wine glass": "A wine glass is purpose-built as drinkware and clearly matches the drinking affordance.",
        },
        "borderline_categories": {"bowl", "vase", "sink"},
        "borderline_reason": "It can contain liquid, but in this benchmark it is not one of the canonical drink-from categories.",
        "negative_reason": "It is not a vessel designed for drinking.",
    },
    "sit_on": {
        "goal": "objects that support a seated person",
        "properties": [
            "stable support surface",
            "appropriate size and shape for sitting",
            "serves as furniture for resting or seating",
        ],
        "look_for": [
            "seat-like furniture",
            "flat or cushioned support surfaces",
            "canonical seating categories in the benchmark",
        ],
        "positive_reason": {
            "chair": "A chair is explicitly designed for a person to sit on.",
            "bench": "A bench is a seating surface intended to support one or more seated people.",
            "couch": "A couch is furniture designed for sitting and resting.",
            "bed": "A bed provides a broad stable surface that can support sitting as well as lying down.",
        },
        "borderline_categories": {"toilet", "dining table"},
        "borderline_reason": "It may physically support sitting in some contexts, but it is outside the benchmark's canonical sit-on categories.",
        "negative_reason": "It is not a seating surface intended for a person to sit on.",
    },
    "ride": {
        "goal": "objects or animals used for riding",
        "properties": [
            "supports a rider on top or in a riding position",
            "is used for movement or transport",
            "has a recognizable rideable form in the benchmark definition",
        ],
        "look_for": [
            "rideable vehicles or animals",
            "structures meant to support a rider",
            "canonical ride categories in the benchmark",
        ],
        "positive_reason": {
            "bicycle": "A bicycle is directly designed to be ridden.",
            "motorcycle": "A motorcycle is a motorized vehicle built for riding.",
            "horse": "A horse is an animal that people can ride.",
        },
        "borderline_categories": {"car", "bus", "train", "truck", "boat", "airplane", "skateboard", "surfboard"},
        "borderline_reason": "It is related to transport or motion, but it is not one of the benchmark's canonical ride categories here.",
        "negative_reason": "It is not an object or animal selected by the benchmark as something to ride.",
    },
    "cut_with": {
        "goal": "handheld tools used for cutting",
        "properties": [
            "has a cutting edge or blades",
            "is manipulated by hand as a cutting tool",
            "is designed to divide or slice materials",
        ],
        "look_for": [
            "sharp tools",
            "blade-like structures",
            "canonical cutting categories in the benchmark",
        ],
        "positive_reason": {
            "knife": "A knife is a cutting tool with a blade and clearly matches the affordance.",
            "scissors": "Scissors are purpose-built cutting tools with blades.",
        },
        "borderline_categories": {"fork"},
        "borderline_reason": "It is a utensil, but it is not primarily a cutting tool in this benchmark.",
        "negative_reason": "It is not a cutting tool with the right functional design for this affordance.",
    },
    "carry_in": {
        "goal": "objects that can contain and carry other items",
        "properties": [
            "has interior capacity or a compartment",
            "has an opening or access point",
            "is used to hold and transport belongings",
        ],
        "look_for": [
            "bags, packs, or cases",
            "container-like objects with storage capacity",
            "canonical carry-in categories in the benchmark",
        ],
        "positive_reason": {
            "backpack": "A backpack is designed to hold items inside a compartment and carry them.",
            "handbag": "A handbag is designed to contain personal items and be carried with a handle or strap.",
            "suitcase": "A suitcase is explicitly meant to contain and transport belongings.",
        },
        "borderline_categories": {"bowl", "cup", "bottle", "car", "truck", "boat"},
        "borderline_reason": "It can contain or transport something in a loose sense, but it is not one of the benchmark's canonical carry-in object categories.",
        "negative_reason": "It is not a container object designed to hold and carry things inside it.",
    },
    "talk_on": {
        "goal": "objects used for voice communication",
        "properties": [
            "supports calling or voice communication",
            "functions as a communication device",
            "matches the benchmark's canonical talk-on category",
        ],
        "look_for": [
            "communication electronics",
            "handheld devices used for calls",
            "the canonical talk-on category in the benchmark",
        ],
        "positive_reason": {
            "cell phone": "A cell phone is a communication device designed for talking on calls.",
        },
        "borderline_categories": {"laptop", "tv", "remote", "keyboard", "mouse"},
        "borderline_reason": "It is electronic or communication-adjacent, but it is not the benchmark's canonical talk-on category.",
        "negative_reason": "It is not a device used for talking on phone calls.",
    },
    "eat_with": {
        "goal": "handheld utensils used while eating",
        "properties": [
            "used to pick up, scoop, or cut food during eating",
            "held in the hand as an eating utensil",
            "matches the benchmark's canonical utensil categories",
        ],
        "look_for": [
            "small utensils",
            "objects used directly with food",
            "canonical eat-with categories in the benchmark",
        ],
        "positive_reason": {
            "fork": "A fork is an eating utensil used to pick up food.",
            "spoon": "A spoon is an eating utensil used to scoop food or liquid.",
            "knife": "A knife can function as an eating utensil for cutting food during a meal.",
        },
        "borderline_categories": {"bowl", "cup", "bottle"},
        "borderline_reason": "It is related to food or drink, but it is not a handheld utensil in the benchmark's eat-with set.",
        "negative_reason": "It is not a handheld utensil used to eat with.",
    },
    "read": {
        "goal": "objects intended to be read",
        "properties": [
            "contains written or printed content",
            "is meant for reading or viewing text",
            "matches the benchmark's canonical readable category",
        ],
        "look_for": [
            "printed materials",
            "book-like objects",
            "the canonical read category in the benchmark",
        ],
        "positive_reason": {
            "book": "A book is explicitly intended to be read and is the benchmark's canonical readable object.",
        },
        "borderline_categories": {"laptop", "cell phone", "tv"},
        "borderline_reason": "It can display text, but it is not the benchmark's canonical read category.",
        "negative_reason": "It is not an object selected by the benchmark as something to read.",
    },
}


ANIMALS = {
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe",
}
PEOPLE = {"person"}
VEHICLES = {
    "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
}
FURNITURE = {"chair", "couch", "bed", "bench", "dining table", "toilet"}
CONTAINERS = {
    "cup", "bottle", "wine glass", "bowl", "vase", "backpack", "handbag", "suitcase",
}
UTENSILS = {"fork", "spoon", "knife", "scissors"}
ELECTRONICS = {
    "cell phone", "laptop", "tv", "remote", "keyboard", "mouse", "microwave",
    "oven", "toaster", "refrigerator",
}
SPORTS = {
    "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT)
    parser.add_argument("--coco_ann", type=str, default=DEFAULT_COCO_ANN)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def bbox_to_region_and_size(box, width, height):
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    frac = area / float(max(width * height, 1))

    if cx < width / 3.0:
        hpos = "left"
    elif cx > 2.0 * width / 3.0:
        hpos = "right"
    else:
        hpos = "center"

    if cy < height / 3.0:
        vpos = "upper"
    elif cy > 2.0 * height / 3.0:
        vpos = "lower"
    else:
        vpos = "middle"

    if frac < 0.02:
        scale = "small"
    elif frac < 0.10:
        scale = "medium"
    else:
        scale = "large"

    return f"{scale} {vpos}-{hpos} region"


def category_group_reason(category):
    if category in PEOPLE:
        return "It is a person rather than a tool, vessel, container, device, or furniture item."
    if category in ANIMALS:
        return "It is an animal, so it does not match this affordance unless the benchmark explicitly treats that animal as a target."
    if category in VEHICLES:
        return "It is a vehicle category, which only matches if the benchmark explicitly includes it for this affordance."
    if category in FURNITURE:
        return "It is furniture, which only matches if the benchmark explicitly treats that furniture category as a target."
    if category in CONTAINERS:
        return "It is a container-like object, but containerhood alone is not enough unless it matches the benchmark's target categories."
    if category in UTENSILS:
        return "It is a utensil or tool, but the functional role does not match this affordance here."
    if category in ELECTRONICS:
        return "It is an electronic device, but it does not match the benchmark's target role for this affordance."
    if category in SPORTS:
        return "It is sports equipment, not one of the benchmark's target object types for this affordance."
    return "Its object category does not fit the benchmark's target function for this affordance."


def join_list(items):
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def resolve_affordance_spec(affordance_id, target_categories):
    spec = AFFORDANCE_SPECS.get(affordance_id)
    if spec is not None:
        return spec

    target_text = join_list(target_categories)
    return {
        "goal": f"objects that satisfy the functional query for {affordance_id}",
        "properties": [
            "the object category must be in the benchmark's target set",
            "the object should plausibly serve the queried everyday function",
            "the object must be localized by its COCO ground-truth box",
        ],
        "look_for": [
            f"canonical target categories: {target_text}",
            "objects whose COCO category matches the affordance definition",
            "distractor objects that are visually present but outside the target set",
        ],
        "positive_reason": {
            category: (
                f"A {category} is one of the benchmark's canonical positive "
                f"categories for {affordance_id}."
            )
            for category in target_categories
        },
        "borderline_categories": set(),
        "borderline_reason": (
            "It may be related to the function, but it is outside the benchmark's "
            "canonical target categories for this affordance."
        ),
        "negative_reason": (
            "It is outside the benchmark's canonical target categories for this affordance."
        ),
    }


def build_planning(affordance_id, affordance_prompt, target_categories):
    spec = resolve_affordance_spec(affordance_id, target_categories)
    props = "; ".join(spec["properties"])
    look_for = "; ".join(spec["look_for"])
    canonical = join_list(target_categories)
    return (
        f'To solve the affordance "{affordance_prompt}", I need to identify '
        f"{spec['goal']}. The important properties are: {props}. In the image, "
        f"I should look for: {look_for}. For benchmark consistency, the canonical "
        f"positive categories for this affordance are {canonical}, so I should "
        f"accept candidates in those categories and reject other categories even "
        f"if they are loosely related."
    )


def build_candidate_reasoning(sample, candidate, image_info):
    affordance_id = sample["affordance_id"]
    target_ids = set(int(x) for x in sample["target_candidate_ids"])
    category = candidate["category"]
    ann_id = int(candidate["ann_id"])
    box = candidate["bbox_xyxy"]
    region = bbox_to_region_and_size(box, image_info["width"], image_info["height"])
    spec = resolve_affordance_spec(affordance_id, sample["target_categories"])

    if ann_id in target_ids:
        base = spec["positive_reason"].get(
            category,
            f"This category is one of the benchmark's positive targets for {affordance_id}.",
        )
        verdict = "matches"
    elif category in spec["borderline_categories"]:
        base = spec["borderline_reason"]
        verdict = "does not match"
    else:
        base = f"{spec['negative_reason']} {category_group_reason(category)}"
        verdict = "does not match"

    return (
        f"Candidate {ann_id} is a {category} in the {region}. {base} "
        f"Therefore it {verdict} the affordance."
    )


def build_action(candidate_rows):
    lines = []
    for row in candidate_rows:
        mark = "[MATCH]" if row["ann_id"] in row["_target_set"] else "[REJECT]"
        lines.append(
            f"{mark} Candidate {row['ann_id']} ({row['category']}): {row['reasoning']}"
        )
    return "\n".join(lines)


def build_summary(sample, candidate_rows):
    target_ids = set(int(x) for x in sample["target_candidate_ids"])
    positives = [row for row in candidate_rows if row["ann_id"] in target_ids]
    negatives = [row for row in candidate_rows if row["ann_id"] not in target_ids]
    pos_desc = [f"{row['ann_id']} ({row['category']})" for row in positives]

    if positives:
        pos_text = join_list(pos_desc)
        positive_sentence = (
            f"The matching candidates are {pos_text} because their categories are "
            f"part of the benchmark's canonical positive set for {sample['affordance_id']}."
        )
    else:
        positive_sentence = (
            "No candidates match the affordance under the benchmark's target-category definition."
        )

    neg_counter = Counter(row["category"] for row in negatives)
    if neg_counter:
        major_negatives = [f"{cat} x{count}" for cat, count in neg_counter.most_common(5)]
        negative_sentence = (
            "The remaining candidates are distractors because they fall outside the "
            f"target categories. Main distractor categories: {', '.join(major_negatives)}."
        )
    else:
        negative_sentence = "There are no distractor candidates in this sample."

    return positive_sentence + " " + negative_sentence


def build_image_info(coco_payload):
    return {
        int(row["id"]): {
            "width": int(row["width"]),
            "height": int(row["height"]),
        }
        for row in coco_payload["images"]
    }


def annotate_sample(sample, image_info_map):
    image_id = int(sample["image_id"])
    image_info = image_info_map[image_id]

    planning = build_planning(
        sample["affordance_id"],
        sample["affordance_prompt"],
        sample["target_categories"],
    )

    target_set = set(int(x) for x in sample["target_candidate_ids"])
    candidate_rows = []
    for candidate in sample["candidates"]:
        row = {
            "ann_id": int(candidate["ann_id"]),
            "category": candidate["category"],
            "bbox_xyxy": candidate["bbox_xyxy"],
            "reasoning": build_candidate_reasoning(sample, candidate, image_info),
            "_target_set": target_set,
        }
        candidate_rows.append(row)

    action = build_action(candidate_rows)
    summary = build_summary(sample, candidate_rows)

    for row in candidate_rows:
        row.pop("_target_set", None)

    return {
        "image_id": image_id,
        "affordance_id": sample["affordance_id"],
        "affordance_prompt": sample["affordance_prompt"],
        "target_categories": list(sample["target_categories"]),
        "candidate_source": sample.get("candidate_source", "coco_gt"),
        "planning": planning,
        "action": action,
        "summary": summary,
        "candidates": candidate_rows,
        "target_candidate_ids": list(sample["target_candidate_ids"]),
        "schema_version": "1.0",
    }


def main():
    args = parse_args()
    samples = load_json(args.input)
    coco_payload = load_json(args.coco_ann)
    image_info_map = build_image_info(coco_payload)

    annotated = [annotate_sample(sample, image_info_map) for sample in samples]

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output, "w") as f:
        json.dump(annotated, f, indent=2)

    print("=" * 72)
    print("Generated affordance CoT annotations")
    print(f"input={args.input}")
    print(f"output={args.output}")
    print(f"samples={len(annotated)}")
    print("=" * 72)


if __name__ == "__main__":
    main()
