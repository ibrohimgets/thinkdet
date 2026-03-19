"""
Build a held-out affordance benchmark from COCO val2017 (model-agnostic).

Key properties:
- No model inference used for selection (not mined by wins/failures)
- Deterministic sampling with fixed seed
- Balanced by affordance
- Excludes previously showcased image_ids
- Stores full GT positives per sample
"""

import argparse
import datetime
import json
import os
import random
from collections import Counter, defaultdict


ROOT = "/home/iibrohimm/project/next_step"
COCO_VAL_ANN = f"{ROOT}/dataSets/coco/annotations/instances_val2017.json"
COCO_VAL_IMG = f"{ROOT}/dataSets/coco/val2017"
DEFAULT_VISUAL_ROOT = f"{ROOT}/thinkdet/results/visual_compare"
DEFAULT_OUT = f"{ROOT}/thinkdet/data/benchmarks/affordance_coco_val_heldout_v1.json"


AFFORDANCES = [
    {
        "id": "drink_from",
        "prompt": "something to drink from .",
        "target_categories": ["cup", "bottle", "wine glass"],
    },
    {
        "id": "sit_on",
        "prompt": "something to sit on .",
        "target_categories": ["chair", "bench", "couch", "bed"],
    },
    {
        "id": "ride",
        "prompt": "something to ride .",
        "target_categories": ["bicycle", "motorcycle", "horse"],
    },
    {
        "id": "cut_with",
        "prompt": "something to cut with .",
        "target_categories": ["knife", "scissors"],
    },
    {
        "id": "carry_in",
        "prompt": "something to carry things in .",
        "target_categories": ["backpack", "handbag", "suitcase"],
    },
    {
        "id": "talk_on",
        "prompt": "something to talk on .",
        "target_categories": ["cell phone"],
    },
    {
        "id": "eat_with",
        "prompt": "something to eat with .",
        "target_categories": ["fork", "spoon", "knife"],
    },
    {
        "id": "read",
        "prompt": "something to read .",
        "target_categories": ["book"],
    },
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco_val_ann", type=str, default=COCO_VAL_ANN)
    parser.add_argument("--coco_val_img", type=str, default=COCO_VAL_IMG)
    parser.add_argument("--visual_root", type=str, default=DEFAULT_VISUAL_ROOT)
    parser.add_argument("--extra_exclude_json", type=str, nargs="*", default=[],
                        help="Optional specific JSON files containing selected cases to exclude.")
    parser.add_argument("--output", type=str, default=DEFAULT_OUT)
    parser.add_argument("--samples_per_affordance", type=int, default=80)
    parser.add_argument("--dev_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260219)
    parser.add_argument("--min_total_anns", type=int, default=5)
    parser.add_argument("--min_distractor_anns", type=int, default=2)
    parser.add_argument("--allow_image_reuse", action="store_true",
                        help="If set, same image_id may appear in multiple affordances.")
    return parser.parse_args()


def safe_load_json(path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def _extract_case_records(payload):
    """
    Return list of case-like dicts that may contain image_id.
    Only extracts known qualitative structures to avoid pulling full COCO
    prediction dumps that contain image_id for almost every image.
    """
    if not isinstance(payload, dict):
        return []
    out = []
    if isinstance(payload.get("selected_cases"), list):
        out.extend(x for x in payload["selected_cases"] if isinstance(x, dict))
    if isinstance(payload.get("cases"), list):
        out.extend(x for x in payload["cases"] if isinstance(x, dict))
    if isinstance(payload.get("selected"), list):
        out.extend(x for x in payload["selected"] if isinstance(x, dict))
    return out


def collect_excluded_image_ids(roots, explicit_jsons):
    """
    Collect known image_id values from prior qualitative/eval artifacts
    to avoid overlap with previously showcased examples.
    """
    excluded = set()
    def collect_from_payload(payload):
        for row in _extract_case_records(payload):
            v = row.get("image_id")
            try:
                excluded.add(int(v))
            except Exception:
                continue

    for root in roots:
        if not root or not os.path.isdir(root):
            continue
        for base, _, files in os.walk(root):
            for fn in files:
                if not fn.endswith(".json"):
                    continue
                payload = safe_load_json(os.path.join(base, fn))
                if payload is None:
                    continue
                collect_from_payload(payload)

    for p in explicit_jsons:
        if not p or not os.path.isfile(p):
            continue
        payload = safe_load_json(p)
        if payload is None:
            continue
        collect_from_payload(payload)
    return excluded


def xywh_to_xyxy(box):
    x, y, w, h = box
    return [x, y, x + w, y + h]


def xywh_abs_to_norm(box_xywh, img_w, img_h):
    x, y, w, h = box_xywh
    cx = (x + 0.5 * w) / img_w
    cy = (y + 0.5 * h) / img_h
    return [cx, cy, w / img_w, h / img_h]


def build_candidate_pool(coco_data, affordance_cfg, excluded_ids, min_total_anns, min_distractor_anns):
    cat_id_to_name = {c["id"]: c["name"] for c in coco_data["categories"]}
    img_id_to_info = {im["id"]: im for im in coco_data["images"]}
    img_anns = defaultdict(list)
    for ann in coco_data["annotations"]:
        img_anns[ann["image_id"]].append(ann)

    target_set = set(affordance_cfg["target_categories"])
    pool = []
    for img_id, anns in img_anns.items():
        if img_id in excluded_ids:
            continue
        info = img_id_to_info.get(img_id)
        if info is None:
            continue
        if len(anns) < min_total_anns:
            continue

        positives = []
        for ann in anns:
            name = cat_id_to_name[ann["category_id"]]
            if name in target_set:
                positives.append({
                    "ann_id": ann["id"],
                    "category_name": name,
                    "bbox_xywh": ann["bbox"],
                    "bbox_norm_cxcywh": xywh_abs_to_norm(ann["bbox"], info["width"], info["height"]),
                    "area": ann["area"],
                    "iscrowd": ann.get("iscrowd", 0),
                })
        if not positives:
            continue

        distractor_anns = [
            ann for ann in anns
            if cat_id_to_name[ann["category_id"]] not in target_set
        ]
        if len(distractor_anns) < min_distractor_anns:
            continue

        cat_counter = Counter(cat_id_to_name[a["category_id"]] for a in anns)
        positives_by_cat = Counter(p["category_name"] for p in positives)
        pool.append({
            "image_id": img_id,
            "image_file": info["file_name"],
            "width": int(info["width"]),
            "height": int(info["height"]),
            "num_annotations": len(anns),
            "num_distractor_annotations": len(distractor_anns),
            "category_histogram": dict(cat_counter),
            "positive_category_histogram": dict(positives_by_cat),
            "positives": positives,
        })
    return pool


def deterministic_pick(pool, count, rng, used_image_ids, allow_image_reuse):
    idxs = list(range(len(pool)))
    rng.shuffle(idxs)
    chosen = []
    for i in idxs:
        row = pool[i]
        if (not allow_image_reuse) and row["image_id"] in used_image_ids:
            continue
        chosen.append(row)
        used_image_ids.add(row["image_id"])
        if len(chosen) >= count:
            break
    return chosen


def assign_splits(samples, dev_ratio, rng):
    """Stratified split by affordance."""
    by_aff = defaultdict(list)
    for s in samples:
        by_aff[s["affordance_id"]].append(s)

    for _, rows in by_aff.items():
        rng.shuffle(rows)

    out = []
    for aff_id, rows in by_aff.items():
        n = len(rows)
        n_dev = int(round(n * dev_ratio))
        n_dev = min(max(n_dev, 1), max(n - 1, 1)) if n > 1 else n
        for i, row in enumerate(rows):
            row = dict(row)
            row["split"] = "dev" if i < n_dev else "test"
            out.append(row)
    return out


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    rng = random.Random(args.seed)
    with open(args.coco_val_ann, "r") as f:
        coco_data = json.load(f)

    excluded_ids = collect_excluded_image_ids([args.visual_root], args.extra_exclude_json)
    print(f"Excluded prior-used image_ids: {len(excluded_ids)}")

    used_image_ids = set()
    all_samples = []
    per_aff_stats = []

    for aff in AFFORDANCES:
        pool = build_candidate_pool(
            coco_data=coco_data,
            affordance_cfg=aff,
            excluded_ids=excluded_ids,
            min_total_anns=args.min_total_anns,
            min_distractor_anns=args.min_distractor_anns,
        )
        chosen = deterministic_pick(
            pool=pool,
            count=args.samples_per_affordance,
            rng=rng,
            used_image_ids=used_image_ids,
            allow_image_reuse=args.allow_image_reuse,
        )

        for item in chosen:
            all_samples.append({
                "benchmark_id": f"{aff['id']}_{item['image_id']}",
                "image_id": item["image_id"],
                "image_file": item["image_file"],
                "image_path": os.path.join(args.coco_val_img, item["image_file"]),
                "width": item["width"],
                "height": item["height"],
                "prompt": aff["prompt"],
                "affordance_id": aff["id"],
                "target_categories": list(aff["target_categories"]),
                "num_annotations": item["num_annotations"],
                "num_distractor_annotations": item["num_distractor_annotations"],
                "category_histogram": item["category_histogram"],
                "positive_category_histogram": item["positive_category_histogram"],
                "positive_targets": item["positives"],
            })

        per_aff_stats.append({
            "affordance_id": aff["id"],
            "prompt": aff["prompt"],
            "target_categories": aff["target_categories"],
            "candidate_pool_size": len(pool),
            "selected": len(chosen),
        })

    all_samples = assign_splits(all_samples, args.dev_ratio, rng)
    all_samples.sort(key=lambda x: (x["affordance_id"], x["split"], x["image_id"]))

    split_counts = Counter(s["split"] for s in all_samples)
    aff_counts = Counter(s["affordance_id"] for s in all_samples)
    unique_images = len(set(s["image_id"] for s in all_samples))

    payload = {
        "status": "ok",
        "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "benchmark_name": "affordance_coco_val_heldout_v1",
        "version": 1,
        "source_dataset": {
            "name": "COCO",
            "split": "val2017",
            "annotation_file": args.coco_val_ann,
            "image_dir": args.coco_val_img,
        },
        "construction_protocol": {
            "model_inference_used": False,
            "sampling": "deterministic random (seeded), balanced by affordance",
            "seed": args.seed,
            "samples_per_affordance_target": args.samples_per_affordance,
            "dev_ratio": args.dev_ratio,
            "min_total_anns": args.min_total_anns,
            "min_distractor_anns": args.min_distractor_anns,
            "allow_image_reuse": args.allow_image_reuse,
            "excluded_prior_visual_ids_count": len(excluded_ids),
        },
        "affordances": AFFORDANCES,
        "summary": {
            "total_samples": len(all_samples),
            "unique_images": unique_images,
            "split_counts": dict(split_counts),
            "affordance_counts": dict(aff_counts),
            "per_affordance_pool_and_selection": per_aff_stats,
        },
        "samples": all_samples,
    }

    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)

    # Also save compact stats markdown next to JSON.
    md_path = os.path.splitext(args.output)[0] + "_stats.md"
    with open(md_path, "w") as f:
        f.write("# Affordance Benchmark Stats\n\n")
        f.write(f"- total_samples: {len(all_samples)}\n")
        f.write(f"- unique_images: {unique_images}\n")
        f.write(f"- split_counts: {dict(split_counts)}\n")
        f.write(f"- excluded_prior_visual_ids_count: {len(excluded_ids)}\n\n")
        f.write("| affordance | pool | selected |\n")
        f.write("|---|---:|---:|\n")
        for row in per_aff_stats:
            f.write(f"| {row['affordance_id']} | {row['candidate_pool_size']} | {row['selected']} |\n")

    print("=" * 72)
    print("Built held-out affordance benchmark")
    print(f"JSON: {args.output}")
    print(f"Stats: {md_path}")
    print(f"total_samples={len(all_samples)} unique_images={unique_images}")
    print(f"split_counts={dict(split_counts)}")
    print("=" * 72)


if __name__ == "__main__":
    main()
