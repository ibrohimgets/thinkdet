#!/bin/bash
# Launch Think-then-Detect commonsense eval across 8 GPUs.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EVAL_SCRIPT="${SCRIPT_DIR}/eval_commonsense_think_then_detect.py"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR="/home/iibrohimm/project/next_step/thinkdet/results/eval"
NUM_SHARDS=8

echo "=========================================="
echo "Think-then-Detect eval on ${NUM_SHARDS} GPUs"
echo "Timestamp: ${TIMESTAMP}"
echo "=========================================="

PIDS=()
SHARD_OUTPUTS=()

for SHARD_ID in $(seq 0 $((NUM_SHARDS - 1))); do
    OUT_FILE="${OUT_DIR}/commonsense_ttd_shard${SHARD_ID}_${TIMESTAMP}.json"
    SHARD_OUTPUTS+=("${OUT_FILE}")
    echo "[shard ${SHARD_ID}] cuda:${SHARD_ID} -> ${OUT_FILE}"
    python "${EVAL_SCRIPT}" \
        --device "cuda:${SHARD_ID}" \
        --shard_id "${SHARD_ID}" \
        --num_shards "${NUM_SHARDS}" \
        --log_every 25 \
        --split test \
        --output "${OUT_FILE}" \
        2>&1 | sed "s/^/[GPU${SHARD_ID}] /" &
    PIDS+=($!)
done

echo ""
echo "Waiting for all ${NUM_SHARDS} shards..."

FAILED=0
for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}"; then
        echo "[shard ${i}] DONE"
    else
        echo "[shard ${i}] FAILED"
        FAILED=1
    fi
done

if [ "${FAILED}" -eq 1 ]; then
    echo "ERROR: One or more shards failed."
    exit 1
fi

echo ""
echo "Merging results..."

python - "${OUT_DIR}/commonsense_ttd_merged_${TIMESTAMP}.json" "${SHARD_OUTPUTS[@]}" <<'MERGE_SCRIPT'
import json, sys, os

out_path = sys.argv[1]
shard_paths = sys.argv[2:]

conditions = ["baseline_direct", "baseline_translated",
               "thinkdet_direct", "thinkdet_translated"]

all_per_object = {c: {} for c in conditions}
all_translations = []
meta = None

for sp in shard_paths:
    with open(sp) as f:
        data = json.load(f)
    if meta is None:
        meta = data
    all_translations.extend(data.get("translations_sample", []))
    for c in conditions:
        for obj_id, obj_m in data["results"][c]["per_object"].items():
            if obj_id not in all_per_object[c]:
                all_per_object[c][obj_id] = []
            all_per_object[c][obj_id].append(obj_m)

def merge_metric_lists(ml):
    total_n = sum(m["n_samples"] for m in ml)
    if total_n == 0:
        return {"n_samples": 0, "hit@0.5_top1": 0, "hit@0.5_topk": 0,
                "mean_best_iou_top1": 0, "mean_best_iou_topk": 0,
                "hit@0.75_top1": 0, "hit@0.75_topk": 0, "top_k": 5}
    merged = {}
    for key in ["hit@0.5_top1", "hit@0.5_topk", "mean_best_iou_top1",
                "mean_best_iou_topk", "hit@0.75_top1", "hit@0.75_topk"]:
        merged[key] = sum(m[key] * m["n_samples"] for m in ml) / total_n
    merged["n_samples"] = total_n
    merged["top_k"] = ml[0].get("top_k", 5)
    return merged

results = {}
for c in conditions:
    merged_po = {}
    for obj_id, ml in sorted(all_per_object[c].items()):
        merged_po[obj_id] = merge_metric_lists(ml)
    overall = merge_metric_lists(list(merged_po.values())) if merged_po else {}
    results[c] = {"overall": overall, "per_object": merged_po}

total_n = results[conditions[0]]["overall"].get("n_samples", 0)

payload = {
    "status": "ok",
    "benchmark_path": meta["benchmark_path"],
    "split": meta["split"],
    "n_samples": total_n,
    "top_k": meta["top_k"],
    "device": "multi-gpu (8 shards)",
    "thinkdet_checkpoint": meta["thinkdet_checkpoint"],
    "object_name_map": meta.get("object_name_map", {}),
    "translate_prompt_template": meta.get("translate_prompt_template", ""),
    "results": results,
    "translations_sample": all_translations[:50],
}

with open(out_path, "w") as f:
    json.dump(payload, f, indent=2)

name_map = payload["object_name_map"]
md_path = os.path.splitext(out_path)[0] + ".md"
with open(md_path, "w") as f:
    f.write("# Think-then-Detect Commonsense Results (Merged 8-GPU)\n\n")
    f.write(f"- n_samples: {total_n}\n\n")
    f.write("## Overall\n\n")
    f.write("| condition | hit@0.5_top1 | hit@0.5_topk | mean_iou_top1 |\n")
    f.write("|---|---:|---:|---:|\n")
    for c in conditions:
        r = results[c]["overall"]
        f.write(f"| {c} | {r['hit@0.5_top1']:.4f} | {r['hit@0.5_topk']:.4f} | "
                f"{r['mean_best_iou_top1']:.4f} |\n")
    f.write("\n## Per-Object (ThinkDet Translated, sorted by hit@0.5_top1)\n\n")
    f.write("| object | ttd_hit50 | direct_hit50 | baseline_hit50 | n |\n")
    f.write("|---|---:|---:|---:|---:|\n")
    sorted_objs = sorted(results["thinkdet_translated"]["per_object"].items(),
                         key=lambda x: x[1]["hit@0.5_top1"], reverse=True)
    for obj_id, m in sorted_objs:
        name = name_map.get(obj_id, obj_id)
        bd = results["baseline_direct"]["per_object"].get(obj_id, {}).get("hit@0.5_top1", 0)
        f.write(f"| {name} | {m['hit@0.5_top1']:.4f} | "
                f"{results['thinkdet_direct']['per_object'].get(obj_id, {}).get('hit@0.5_top1', 0):.4f} | "
                f"{bd:.4f} | {m['n_samples']} |\n")
    f.write("\n## Sample Translations\n\n")
    f.write("| object | original | translated |\n")
    f.write("|---|---|---|\n")
    seen = set()
    for t in all_translations:
        key = (t["object_name"], t["original_prompt"])
        if key not in seen and len(seen) < 30:
            seen.add(key)
            f.write(f"| {t['object_name']} | {t['original_prompt']} | {t['translated_prompt']} |\n")

print(f"[merged] {out_path}")
print(f"[merged] {md_path}")
MERGE_SCRIPT

echo ""
echo "=========================================="
echo "DONE. Merged:"
echo "  ${OUT_DIR}/commonsense_ttd_merged_${TIMESTAMP}.json"
echo "  ${OUT_DIR}/commonsense_ttd_merged_${TIMESTAMP}.md"
echo "=========================================="
