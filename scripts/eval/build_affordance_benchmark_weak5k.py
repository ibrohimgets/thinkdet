"""
Build a larger weakly-labeled affordance benchmark (default 5k samples)
from COCO val2017 by expanding each affordance with prompt variants.

Why this exists:
- The held-out builder produces a compact balanced set for the active taxonomy.
- COCO val candidate coverage is not enough to reach 5k unique
  image-affordance pairs for every affordance.
- This script scales sample count by reusing the same GT category targets across
  multiple prompt variants per affordance (weak prompt-paraphrase expansion).

Important:
- Labels are still COCO category GT-based and deterministic.
- Prompt variants are auto-authored (not human-verified for semantic fidelity).
- Same image_id can appear multiple times across prompt variants / affordances.
"""

import argparse
import datetime
import json
import os
import random
import sys
from collections import Counter, defaultdict

ROOT = "/home/iibrohimm/project/next_step"
sys.path.insert(0, ROOT)

from thinkdet.scripts.eval.build_affordance_benchmark import (
    AFFORDANCES,
    COCO_VAL_ANN,
    COCO_VAL_IMG,
    DEFAULT_VISUAL_ROOT,
    build_candidate_pool,
    collect_excluded_image_ids,
    assign_splits,
)


DEFAULT_OUT = (
    f"{ROOT}/thinkdet/data/benchmarks/"
    "affordance_coco_val_promptvar_weak5k_v2.json"
)


# Prompt variants are weak paraphrases. Keep the original prompt first.
PROMPT_VARIANTS = {
    "drink_from": [
        "something to drink from .",
        "something you can drink from .",
        "a container for drinking .",
        "something used for drinking .",
        "a thing to drink out of .",
    ],
    "sit_on": [
        "something to sit on .",
        "something you can sit on .",
        "a thing used for sitting .",
        "a place to sit .",
        "furniture to sit on .",
    ],
    "ride": [
        "something to ride .",
        "something you can ride .",
        "a thing for riding .",
        "something to ride on .",
        "a vehicle or animal to ride .",
    ],
    "cut_with": [
        "something to cut with .",
        "something used for cutting .",
        "a thing to cut with .",
        "something you can use to cut .",
        "a cutting tool .",
    ],
    "carry_in": [
        "something to carry things in .",
        "something used to carry things .",
        "a thing to carry items in .",
        "something you can carry things in .",
        "a bag for carrying things .",
    ],
    "talk_on": [
        "something to talk on .",
        "something used to talk on .",
        "a device to talk on .",
        "something you can talk on .",
        "a phone for talking .",
    ],
    "eat_with": [
        "something to eat with .",
        "something used for eating .",
        "a utensil to eat with .",
        "something you can use to eat .",
        "a thing for eating food with .",
    ],
    "read": [
        "something to read .",
        "something you can read .",
        "a thing to read .",
        "reading material .",
        "something used for reading .",
    ],
    "eat": [
        "something to eat .",
        "something you can eat .",
        "food to eat .",
        "an edible thing .",
        "something for a meal or snack .",
    ],
    "eat_from": [
        "something to eat from .",
        "something food can be eaten from .",
        "a surface or container for eating .",
        "something used to serve food for eating .",
        "a thing to eat food from .",
    ],
    "cook_with": [
        "something to cook with .",
        "something used for cooking food .",
        "an appliance for cooking .",
        "something you can use to heat food .",
        "a thing for preparing hot food .",
    ],
    "type_on": [
        "something to type on .",
        "something used for typing .",
        "a device to enter text on .",
        "something you can type with .",
        "a thing for keyboard input .",
    ],
    "control_with": [
        "something to control a device with .",
        "something used to control a computer or screen .",
        "a device controller .",
        "something you can use to point or select .",
        "an input device for controlling something .",
    ],
    "watch": [
        "something to watch .",
        "something you can watch video on .",
        "a screen for watching .",
        "something used for viewing shows .",
        "a display to look at .",
    ],
    "tell_time": [
        "something to tell time with .",
        "something that shows the time .",
        "a thing used to know the time .",
        "something you can read time from .",
        "a time-telling object .",
    ],
    "shelter_under": [
        "something to shelter under .",
        "something used for cover from rain .",
        "something you can stand under for shade .",
        "a thing for protection overhead .",
        "something to hold over yourself in bad weather .",
    ],
    "wash_at": [
        "somewhere to wash things .",
        "a place to wash hands or dishes .",
        "something used for washing .",
        "a basin for cleaning things .",
        "somewhere water is used for cleaning .",
    ],
    "sleep_on": [
        "something to sleep on .",
        "something you can lie on to rest .",
        "a place to sleep .",
        "furniture for sleeping .",
        "something used as a bed or resting place .",
    ],
    "travel_in": [
        "something to travel in .",
        "a vehicle for transportation .",
        "something people can ride inside .",
        "something used to move from place to place .",
        "a conveyance for travel .",
    ],
    "play_with": [
        "something to play a sport with .",
        "sports equipment to play with .",
        "something used in a game .",
        "a thing for athletic play .",
        "equipment for playing sports .",
    ],
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco_val_ann", type=str, default=COCO_VAL_ANN)
    parser.add_argument("--coco_val_img", type=str, default=COCO_VAL_IMG)
    parser.add_argument("--visual_root", type=str, default=DEFAULT_VISUAL_ROOT)
    parser.add_argument(
        "--extra_exclude_json",
        type=str,
        nargs="*",
        default=[],
        help="Optional specific JSON files containing selected cases to exclude.",
    )
    parser.add_argument("--output", type=str, default=DEFAULT_OUT)
    parser.add_argument("--target_total_samples", type=int, default=5000)
    parser.add_argument("--dev_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260226)
    parser.add_argument("--min_total_anns", type=int, default=5)
    parser.add_argument("--min_distractor_anns", type=int, default=2)
    return parser.parse_args()


def _dedup_prompts_keep_order(prompts):
    out = []
    seen = set()
    for p in prompts:
        p_norm = " ".join(p.strip().split())
        if p_norm in seen:
            continue
        seen.add(p_norm)
        out.append(p_norm)
    return out


def _validate_prompt_variants():
    aff_ids = {a["id"] for a in AFFORDANCES}
    missing = sorted(aff_ids - set(PROMPT_VARIANTS.keys()))
    if missing:
        raise ValueError(f"Missing prompt variants for affordances: {missing}")

    for aff in AFFORDANCES:
        aff_id = aff["id"]
        variants = _dedup_prompts_keep_order(PROMPT_VARIANTS[aff_id])
        if not variants:
            raise ValueError(f"No prompt variants for {aff_id}")
        if aff["prompt"] not in variants:
            raise ValueError(
                f"Canonical prompt for {aff_id} missing from PROMPT_VARIANTS: {aff['prompt']}"
            )
        PROMPT_VARIANTS[aff_id] = variants


def _expanded_prompt_pool(base_pool, aff):
    variants = PROMPT_VARIANTS[aff["id"]]
    out = []
    for pv_idx, prompt in enumerate(variants):
        for item in base_pool:
            out.append(
                {
                    "prompt": prompt,
                    "prompt_variant_index": pv_idx,
                    "prompt_variant_count": len(variants),
                    "base_item": item,
                }
            )
    return out


def _deterministic_pick(candidates, count, rng):
    idxs = list(range(len(candidates)))
    rng.shuffle(idxs)
    return [candidates[i] for i in idxs[:count]]


def _allocations(total, n_groups):
    base = total // n_groups
    rem = total % n_groups
    return [base + (1 if i < rem else 0) for i in range(n_groups)]


def _assign_splits_allow_zero_dev(samples, dev_ratio, rng):
    if dev_ratio <= 0.0:
        out = []
        for s in samples:
            row = dict(s)
            row["split"] = "test"
            out.append(row)
        return out
    return assign_splits(samples, dev_ratio, rng)


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    _validate_prompt_variants()

    if args.target_total_samples <= 0:
        raise ValueError("--target_total_samples must be > 0")

    rng = random.Random(args.seed)
    with open(args.coco_val_ann, "r") as f:
        coco_data = json.load(f)

    excluded_ids = collect_excluded_image_ids([args.visual_root], args.extra_exclude_json)
    print(f"Excluded prior-used image_ids: {len(excluded_ids)}")

    per_aff_target = _allocations(args.target_total_samples, len(AFFORDANCES))

    all_samples = []
    per_aff_stats = []
    per_aff_variant_counts = {}

    total_capacity = 0
    for aff, aff_target in zip(AFFORDANCES, per_aff_target):
        base_pool = build_candidate_pool(
            coco_data=coco_data,
            affordance_cfg=aff,
            excluded_ids=excluded_ids,
            min_total_anns=args.min_total_anns,
            min_distractor_anns=args.min_distractor_anns,
        )
        expanded_pool = _expanded_prompt_pool(base_pool, aff)
        total_capacity += len(expanded_pool)

        if aff_target > len(expanded_pool):
            raise RuntimeError(
                f"Target for affordance {aff['id']} is {aff_target}, but capacity is "
                f"{len(expanded_pool)} (base_pool={len(base_pool)}, "
                f"prompt_variants={len(PROMPT_VARIANTS[aff['id']])}). "
                "Add more prompt variants or lower target_total_samples."
            )

        chosen = _deterministic_pick(expanded_pool, aff_target, rng)
        variant_counter = Counter(c["prompt_variant_index"] for c in chosen)
        per_aff_variant_counts[aff["id"]] = {str(k): int(v) for k, v in sorted(variant_counter.items())}

        for c in chosen:
            item = c["base_item"]
            pv_idx = c["prompt_variant_index"]
            variant_prompt = c["prompt"]
            all_samples.append(
                {
                    "benchmark_id": f"{aff['id']}_pv{pv_idx}_{item['image_id']}",
                    "image_id": item["image_id"],
                    "image_file": item["image_file"],
                    "image_path": os.path.join(args.coco_val_img, item["image_file"]),
                    "width": item["width"],
                    "height": item["height"],
                    "prompt": variant_prompt,
                    "affordance_id": aff["id"],
                    "canonical_affordance_prompt": aff["prompt"],
                    "prompt_variant_index": pv_idx,
                    "prompt_variant_count": c["prompt_variant_count"],
                    "prompt_variant_id": f"{aff['id']}_pv{pv_idx}",
                    "weak_prompt_variant": True,
                    "human_verified_prompt": False,
                    "target_categories": list(aff["target_categories"]),
                    "num_annotations": item["num_annotations"],
                    "num_distractor_annotations": item["num_distractor_annotations"],
                    "category_histogram": item["category_histogram"],
                    "positive_category_histogram": item["positive_category_histogram"],
                    "positive_targets": item["positives"],
                }
            )

        per_aff_stats.append(
            {
                "affordance_id": aff["id"],
                "canonical_prompt": aff["prompt"],
                "prompt_variant_count": len(PROMPT_VARIANTS[aff["id"]]),
                "target_categories": aff["target_categories"],
                "base_candidate_pool_size": len(base_pool),
                "expanded_prompt_pool_size": len(expanded_pool),
                "selected": len(chosen),
            }
        )

    all_samples = _assign_splits_allow_zero_dev(all_samples, args.dev_ratio, rng)
    all_samples.sort(
        key=lambda x: (
            x["affordance_id"],
            x["split"],
            x["prompt_variant_index"],
            x["image_id"],
        )
    )

    split_counts = Counter(s["split"] for s in all_samples)
    aff_counts = Counter(s["affordance_id"] for s in all_samples)
    unique_images = len(set(s["image_id"] for s in all_samples))
    unique_image_affordance = len(set((s["image_id"], s["affordance_id"]) for s in all_samples))
    unique_image_prompt = len(set((s["image_id"], s["prompt"]) for s in all_samples))

    payload = {
        "status": "ok",
        "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "benchmark_name": "affordance_coco_val_promptvar_weak5k_v2",
        "version": 2,
        "label_quality": "weak",
        "source_dataset": {
            "name": "COCO",
            "split": "val2017",
            "annotation_file": args.coco_val_ann,
            "image_dir": args.coco_val_img,
        },
        "construction_protocol": {
            "model_inference_used": False,
            "sampling": (
                "deterministic random (seeded), balanced by affordance target count, "
                "prompt-variant expansion within affordance"
            ),
            "seed": args.seed,
            "target_total_samples": args.target_total_samples,
            "target_per_affordance": {
                aff["id"]: int(n) for aff, n in zip(AFFORDANCES, per_aff_target)
            },
            "dev_ratio": args.dev_ratio,
            "min_total_anns": args.min_total_anns,
            "min_distractor_anns": args.min_distractor_anns,
            "allow_image_reuse_across_affordances": True,
            "allow_image_reuse_across_prompt_variants": True,
            "allow_duplicate_image_prompt_pairs": False,
            "excluded_prior_visual_ids_count": len(excluded_ids),
            "human_verified_prompts": False,
            "human_verified_labels": False,
            "notes": (
                "Prompt variants are auto-authored weak paraphrases mapped to the same "
                "COCO category target sets per affordance."
            ),
        },
        "affordances": AFFORDANCES,
        "prompt_variants": PROMPT_VARIANTS,
        "summary": {
            "total_samples": len(all_samples),
            "unique_images": unique_images,
            "unique_image_affordance_pairs": unique_image_affordance,
            "unique_image_prompt_pairs": unique_image_prompt,
            "split_counts": dict(split_counts),
            "affordance_counts": dict(aff_counts),
            "expanded_total_capacity": total_capacity,
            "per_affordance_pool_and_selection": per_aff_stats,
            "per_affordance_prompt_variant_selection": per_aff_variant_counts,
        },
        "samples": all_samples,
    }

    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)

    md_path = os.path.splitext(args.output)[0] + "_stats.md"
    with open(md_path, "w") as f:
        f.write("# Affordance Weak 5K Benchmark Stats\n\n")
        f.write("- label_quality: weak (auto prompt variants, no human verification)\n")
        f.write(f"- total_samples: {len(all_samples)}\n")
        f.write(f"- unique_images: {unique_images}\n")
        f.write(f"- unique_image_affordance_pairs: {unique_image_affordance}\n")
        f.write(f"- unique_image_prompt_pairs: {unique_image_prompt}\n")
        f.write(f"- split_counts: {dict(split_counts)}\n")
        f.write(f"- excluded_prior_visual_ids_count: {len(excluded_ids)}\n")
        f.write(f"- expanded_total_capacity: {total_capacity}\n\n")
        f.write("| affordance | base_pool | prompt_vars | expanded_pool | selected |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for row in per_aff_stats:
            f.write(
                f"| {row['affordance_id']} | {row['base_candidate_pool_size']} | "
                f"{row['prompt_variant_count']} | {row['expanded_prompt_pool_size']} | "
                f"{row['selected']} |\n"
            )

    print("=" * 72)
    print("Built weak affordance benchmark (prompt-variant expanded)")
    print(f"JSON: {args.output}")
    print(f"Stats: {md_path}")
    print(
        f"total_samples={len(all_samples)} unique_images={unique_images} "
        f"unique_image_prompt_pairs={unique_image_prompt}"
    )
    print(f"split_counts={dict(split_counts)}")
    print("=" * 72)


if __name__ == "__main__":
    main()
