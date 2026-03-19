"""
InternVL 3.5-1B Baseline Test
=============================
Pure MLLM inference — no DINO, no adapters, no training.
Goal: See what the model can already do with image + text prompts.
Tests: captioning, object counting, spatial reasoning, detection-style queries.
"""

import os
import sys
import json
import time
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode

# ============================================================
# IMAGE LOADING (from InternVL official example)
# ============================================================
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1)
        for i in range(1, n + 1) for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        processed_images.append(resized_img.crop(box))
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


def load_image(image_file, input_size=448, max_num=6):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


# ============================================================
# TEST PROMPTS — designed to probe different capabilities
# ============================================================
TEST_PROMPTS = [
    # --- 1. Basic captioning ---
    {
        "name": "caption_basic",
        "category": "captioning",
        "prompt": "<image>\nDescribe this image in one sentence.",
    },
    # --- 2. Object listing ---
    {
        "name": "object_list",
        "category": "object_recognition",
        "prompt": "<image>\nList every distinct object you can see in this image.",
    },
    # --- 3. Object counting ---
    {
        "name": "object_count",
        "category": "counting",
        "prompt": "<image>\nHow many people are in this image? How many vehicles? Count carefully.",
    },
    # --- 4. Spatial reasoning ---
    {
        "name": "spatial_reasoning",
        "category": "spatial",
        "prompt": "<image>\nDescribe the spatial layout: what is in the foreground, middle ground, and background?",
    },
    # --- 5. Relational reasoning ---
    {
        "name": "relational",
        "category": "reasoning",
        "prompt": "<image>\nWhat objects are interacting with each other? Describe the relationships between objects.",
    },
    # --- 6. Detection-style query (specific object) ---
    {
        "name": "detect_specific",
        "category": "detection",
        "prompt": "<image>\nIs there a person in this image? If yes, describe where they are located (left/right/center, top/bottom).",
    },
    # --- 7. Grounding-style query ---
    {
        "name": "grounding",
        "category": "grounding",
        "prompt": "<image>\nFor each object you can identify, describe its approximate location in the image using coordinates or regions (e.g., top-left, center, bottom-right).",
    },
    # --- 8. Reasoning about scene ---
    {
        "name": "scene_reasoning",
        "category": "reasoning",
        "prompt": "<image>\nWhat is likely happening in this scene? What might happen next? Explain your reasoning.",
    },
    # --- 9. Attribute recognition ---
    {
        "name": "attributes",
        "category": "attributes",
        "prompt": "<image>\nDescribe the colors, sizes, and materials of the main objects in this image.",
    },
    # --- 10. Thinking mode (chain-of-thought) ---
    {
        "name": "think_detect",
        "category": "reasoning_cot",
        "prompt": "<image>\nThink step by step: First, identify all objects. Then, for each object, describe its location and any relationships with other objects. Finally, summarize what the scene is about.",
    },
]


# ============================================================
# MAIN
# ============================================================
def main():
    coco_val_dir = "/home/iibrohimm/project/next_step/dataSets/coco/val2017"
    output_dir = "/home/iibrohimm/project/next_step/thinkdet/tests/mllm_baseline/results"
    os.makedirs(output_dir, exist_ok=True)

    # Pick 5 diverse COCO val images
    test_images = [
        "000000000139.jpg",  # typically has people/objects
        "000000000285.jpg",
        "000000000632.jpg",
        "000000000724.jpg",
        "000000000776.jpg",
    ]

    # --- Load Model ---
    print("=" * 70)
    print("Loading InternVL 3.5-1B...")
    print("=" * 70)

    from transformers import AutoTokenizer, AutoModel

    model_path = "OpenGVLab/InternVL3_5-1B"
    t0 = time.time()

    # RTX 2080 Ti: no bf16 support, use float16
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).eval().cuda()

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=False,
    )

    load_time = time.time() - t0
    print(f"[OK] Model loaded in {load_time:.1f}s")

    # Check VRAM usage
    mem_allocated = torch.cuda.memory_allocated() / 1e9
    print(f"[INFO] GPU memory used: {mem_allocated:.2f} GB")

    generation_config = dict(max_new_tokens=512, do_sample=False)

    # --- Run Tests ---
    all_results = []

    for img_name in test_images:
        img_path = os.path.join(coco_val_dir, img_name)
        if not os.path.exists(img_path):
            print(f"[SKIP] {img_name} not found")
            continue

        print(f"\n{'=' * 70}")
        print(f"IMAGE: {img_name}")
        print(f"{'=' * 70}")

        pixel_values = load_image(img_path, max_num=6).to(torch.float16).cuda()
        print(f"[INFO] Image tiles: {pixel_values.shape[0]}")

        for test in TEST_PROMPTS:
            print(f"\n--- [{test['category']}] {test['name']} ---")
            print(f"Q: {test['prompt'][:80]}...")

            t0 = time.time()
            try:
                response = model.chat(tokenizer, pixel_values, test['prompt'], generation_config)
            except Exception as e:
                response = f"[ERROR] {e}"
            elapsed = time.time() - t0

            print(f"A: {response[:200]}")
            print(f"   ({elapsed:.1f}s)")

            all_results.append({
                "image": img_name,
                "test_name": test["name"],
                "category": test["category"],
                "prompt": test["prompt"],
                "response": response,
                "time_sec": round(elapsed, 2),
            })

    # --- Save Results ---
    results_path = os.path.join(output_dir, "internvl_baseline_results.json")
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n[SAVED] {results_path}")

    # --- Print Summary ---
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    categories = {}
    for r in all_results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = {"count": 0, "total_time": 0, "errors": 0}
        categories[cat]["count"] += 1
        categories[cat]["total_time"] += r["time_sec"]
        if r["response"].startswith("[ERROR]"):
            categories[cat]["errors"] += 1

    for cat, stats in sorted(categories.items()):
        avg_time = stats["total_time"] / max(stats["count"], 1)
        print(f"  {cat:20s}: {stats['count']} tests, avg {avg_time:.1f}s, {stats['errors']} errors")

    print(f"\nTotal: {len(all_results)} tests on {len(test_images)} images")
    print(f"Results: {results_path}")


if __name__ == "__main__":
    main()
