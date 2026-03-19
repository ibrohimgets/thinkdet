#!/bin/bash
# Launch ThinkDet with FULL fallback (InternVL feedback + prompt refinement)
# on the commonsense benchmark across a configurable set of GPUs.

set -e
set -x

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EVAL_SCRIPT="${SCRIPT_DIR}/eval_commonsense_benchmark_unified_fallback.py"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR="/home/iibrohimm/project/next_step/thinkdet/results/eval"

GPU_IDS_CSV="${GPU_IDS:-}"
if [ -z "${GPU_IDS_CSV}" ]; then
    GPU_IDS_CSV="$(python -c 'import torch; print(",".join(str(i) for i in range(torch.cuda.device_count())))')"
fi

if [ -z "${GPU_IDS_CSV}" ]; then
    echo "ERROR: No CUDA devices detected. Set GPU_IDS explicitly if needed."
    exit 1
fi

IFS=',' read -r -a GPU_ID_ARRAY <<< "${GPU_IDS_CSV}"
NUM_AVAILABLE_GPUS=${#GPU_ID_ARRAY[@]}
NUM_SHARDS="${NUM_SHARDS:-${NUM_AVAILABLE_GPUS}}"

if [ "${NUM_SHARDS}" -le 0 ]; then
    echo "ERROR: NUM_SHARDS must be >= 1"
    exit 1
fi

if [ "${NUM_SHARDS}" -gt "${NUM_AVAILABLE_GPUS}" ]; then
    echo "ERROR: NUM_SHARDS=${NUM_SHARDS} exceeds available GPU slots (${NUM_AVAILABLE_GPUS}) from GPU_IDS=${GPU_IDS_CSV}"
    exit 1
fi

EXTRA_ARGS=("$@")

echo "=========================================================="
echo "Commonsense FULL FALLBACK eval on ${NUM_SHARDS} GPUs"
echo "GPU IDs: ${GPU_IDS_CSV}"
echo "Timestamp: ${TIMESTAMP}"
echo "=========================================================="

PIDS=()
SHARD_OUTPUTS=()

for SHARD_ID in $(seq 0 $((NUM_SHARDS - 1))); do
    GPU_DEVICE="${GPU_ID_ARRAY[$SHARD_ID]}"
    OUT_FILE="${OUT_DIR}/commonsense_fallback_full_shard${SHARD_ID}_${TIMESTAMP}.json"
    SHARD_OUTPUTS+=("${OUT_FILE}")

    echo "[shard ${SHARD_ID}] cuda:${GPU_DEVICE} -> ${OUT_FILE}"

    # Use --enable_fallback and --prompt_refine for full fallback
    python "${EVAL_SCRIPT}" \
        --device "cuda:${GPU_DEVICE}" \
        --shard_id "${SHARD_ID}" \
        --num_shards "${NUM_SHARDS}" \
        --log_every 25 \
        --split test \
        --enable_fallback \
        --prompt_refine \
        --output "${OUT_FILE}" \
        "${EXTRA_ARGS[@]}" \
        2>&1 | sed "s/^/[GPU${SHARD_ID}] /" &

    PIDS+=($!)
done

echo ""
echo "Waiting for all ${NUM_SHARDS} shards to finish..."

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
echo "All shards complete. Merging results..."

# Merge shard outputs
python - "${OUT_DIR}/commonsense_fallback_full_merged_${TIMESTAMP}.json" "${NUM_SHARDS}" "${SHARD_OUTPUTS[@]}" <<'MERGE_SCRIPT'
import json, sys, os

out_path = sys.argv[1]
num_shards = int(sys.argv[2])
shard_paths = sys.argv[3:]

all_thinkdet_rows = []
all_fallback_rows = []
all_per_object = {}
all_fallback_stage_counts = {}
meta = None

for sp in shard_paths:
    with open(sp) as f:
        data = json.load(f)
    if meta is None:
        meta = data
    # Collect per-object metrics from each shard
    for obj_id, obj_metrics in data["results"]["thinkdet"]["per_object"].items():
        if obj_id not in all_per_object:
            all_per_object[obj_id] = {"thinkdet": [], "thinkdet_fallback": []}
        all_per_object[obj_id]["thinkdet"].append(obj_metrics)
    for obj_id, obj_metrics in data["results"]["thinkdet_fallback"]["per_object"].items():
        if obj_id not in all_per_object:
            all_per_object[obj_id] = {"thinkdet": [], "thinkdet_fallback": []}
        all_per_object[obj_id]["thinkdet_fallback"].append(obj_metrics)
    # Collect stage counts
    for stage, count in data.get("fallback_stage_counts", {}).items():
        all_fallback_stage_counts[stage] = all_fallback_stage_counts.get(stage, 0) + count

# Merge per-object: weight by n_samples
def merge_metric_lists(metric_list):
    total_n = sum(m["n_samples"] for m in metric_list)
    if total_n == 0:
        return {"n_samples": 0, "hit@0.5_top1": 0, "hit@0.5_topk": 0,
                "mean_best_iou_top1": 0, "mean_best_iou_topk": 0,
                "hit@0.75_top1": 0, "hit@0.75_topk": 0, "top_k": 5}
    merged = {}
    for key in ["hit@0.5_top1", "hit@0.5_topk", "mean_best_iou_top1",
                "mean_best_iou_topk", "hit@0.75_top1", "hit@0.75_topk"]:
        merged[key] = sum(m[key] * m["n_samples"] for m in metric_list) / total_n
    merged["n_samples"] = total_n
    merged["top_k"] = metric_list[0].get("top_k", 5)
    return merged

merged_per_object = {"thinkdet": {}, "thinkdet_fallback": {}}
for obj_id, data_dict in sorted(all_per_object.items()):
    if data_dict["thinkdet"]:
        merged_per_object["thinkdet"][obj_id] = merge_metric_lists(data_dict["thinkdet"])
    if data_dict["thinkdet_fallback"]:
        merged_per_object["thinkdet_fallback"][obj_id] = merge_metric_lists(data_dict["thinkdet_fallback"])

# Overall = merge all per-object
def overall_from_per_object(po):
    all_metrics = list(po.values())
    return merge_metric_lists(all_metrics) if all_metrics else {}

total_n = sum(m["n_samples"] for m in merged_per_object["thinkdet"].values())

payload = {
    "status": "ok",
    "benchmark_path": meta["benchmark_path"],
    "split": meta["split"],
    "n_samples": total_n,
    "top_k": meta["top_k"],
    "device": f"multi-gpu ({num_shards} shards)",
    "thinkdet_checkpoint": meta["thinkdet_checkpoint"],
    "checkpoint_meta": meta["checkpoint_meta"],
    "fallback_config": meta["fallback_config"],
    "object_name_map": meta.get("object_name_map", {}),
    "results": {
        "thinkdet": {
            "overall": overall_from_per_object(merged_per_object["thinkdet"]),
            "per_object": merged_per_object["thinkdet"],
        },
        "thinkdet_fallback": {
            "overall": overall_from_per_object(merged_per_object["thinkdet_fallback"]),
            "per_object": merged_per_object["thinkdet_fallback"],
        },
    },
    "fallback_stage_counts": all_fallback_stage_counts,
}

with open(out_path, "w") as f:
    json.dump(payload, f, indent=2)

# Also write markdown summary
md_path = os.path.splitext(out_path)[0] + ".md"
with open(md_path, "w") as f:
    f.write(f"# Commonsense Full Fallback Results (Merged {num_shards}-GPU)\n\n")
    f.write(f"- n_samples: {total_n}\n\n")
    f.write("## Overall\n\n")
    f.write("| model | hit@0.5_top1 | hit@0.5_topk | mean_iou_top1 | mean_iou_topk |\n")
    f.write("|---|---:|---:|---:|---:|\n")
    for model_name in ["thinkdet", "thinkdet_fallback"]:
        row = payload["results"][model_name]["overall"]
        f.write(f"| {model_name} | {row['hit@0.5_top1']:.4f} | {row['hit@0.5_topk']:.4f} | "
                f"{row['mean_best_iou_top1']:.4f} | {row['mean_best_iou_topk']:.4f} |\n")
    f.write("\n## Per-Object (ThinkDet Fallback, sorted by hit@0.5_top1)\n\n")
    f.write("| object | hit@0.5_top1 | mean_iou_top1 | n |\n")
    f.write("|---|---:|---:|---:|\n")
    name_map = payload["object_name_map"]
    sorted_objs = sorted(merged_per_object["thinkdet_fallback"].items(),
                         key=lambda x: x[1]["hit@0.5_top1"], reverse=True)
    for obj_id, m in sorted_objs:
        name = name_map.get(obj_id, obj_id)
        f.write(f"| {name} | {m['hit@0.5_top1']:.4f} | {m['mean_best_iou_top1']:.4f} | {m['n_samples']} |\n")

print(f"[merged] {out_path}")
print(f"[merged] {md_path}")
MERGE_SCRIPT

echo ""
echo "=========================================================="
echo "DONE. Merged output:"
echo "  ${OUT_DIR}/commonsense_fallback_full_merged_${TIMESTAMP}.json"
echo "  ${OUT_DIR}/commonsense_fallback_full_merged_${TIMESTAMP}.md"
echo "=========================================================="
