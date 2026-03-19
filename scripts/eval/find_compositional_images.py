"""
Find COCO val images that have compositional/ambiguous scenarios:
- Multiple instances of same category (e.g., multiple people, multiple dogs)
- Spatial relationships (person next to / on top of something)
- Attribute disambiguation needed
"""
import json
import os
from collections import Counter

ann_file = "/home/iibrohimm/project/next_step/dataSets/coco/annotations/instances_val2017.json"
with open(ann_file) as f:
    data = json.load(f)

cat_map = {c["id"]: c["name"] for c in data["categories"]}

# Group annotations by image
from collections import defaultdict
img_anns = defaultdict(list)
for ann in data["annotations"]:
    img_anns[ann["image_id"]].append(ann)

img_info = {i["id"]: i for i in data["images"]}

# Find images with:
# 1. Multiple people + objects (compositional)
# 2. Multiple animals of same type
# 3. Person interacting with objects

candidates = []
for img_id, anns in img_anns.items():
    cats = [cat_map[a["category_id"]] for a in anns]
    cat_counts = Counter(cats)
    unique_cats = set(cats)
    info = img_info[img_id]

    score = 0
    reasons = []

    # Multiple people + diverse objects
    if cat_counts.get("person", 0) >= 2 and len(unique_cats) >= 4:
        score += 3
        reasons.append(f"{cat_counts['person']} people + {len(unique_cats)} categories")

    # Multiple animals same type
    for animal in ["dog", "cat", "horse", "cow", "bird"]:
        if cat_counts.get(animal, 0) >= 2:
            score += 4
            reasons.append(f"{cat_counts[animal]} {animal}s")

    # Person + vehicle interaction
    vehicles = {"car", "truck", "bus", "motorcycle", "bicycle"}
    if "person" in unique_cats and unique_cats & vehicles:
        if cat_counts["person"] >= 2:
            score += 2
            reasons.append(f"people + {unique_cats & vehicles}")

    # Person + sports equipment
    sports = {"sports ball", "tennis racket", "baseball bat", "skateboard", "surfboard", "frisbee"}
    if "person" in unique_cats and unique_cats & sports:
        score += 2
        reasons.append(f"person + {unique_cats & sports}")

    # Multiple chairs/dining (spatial ambiguity)
    furniture = {"chair", "dining table", "couch"}
    if sum(cat_counts.get(f, 0) for f in furniture) >= 3:
        score += 2
        reasons.append(f"furniture scene")

    # Person + animal
    animals = {"dog", "cat", "horse", "cow", "bird", "elephant", "bear", "zebra", "giraffe"}
    if "person" in unique_cats and unique_cats & animals:
        score += 3
        reasons.append(f"person + {unique_cats & animals}")

    if score >= 4:
        candidates.append({
            "image_id": img_id,
            "file_name": info["file_name"],
            "width": info["width"],
            "height": info["height"],
            "num_anns": len(anns),
            "categories": dict(cat_counts),
            "score": score,
            "reasons": reasons,
        })

# Sort by score
candidates.sort(key=lambda x: -x["score"])

print(f"Found {len(candidates)} candidate images\n")
print("Top 30:")
for c in candidates[:30]:
    print(f"  {c['file_name']} (id={c['image_id']}) score={c['score']}")
    print(f"    {c['categories']}")
    print(f"    Reasons: {', '.join(c['reasons'])}")
    print()
