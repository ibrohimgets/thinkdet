"""
OmniLabel Evaluation for ThinkDet Stage or Baseline GDINO.

Evaluates top-1 grounding quality for each (annotation, description) pair:
    - top1_acc @ IoU threshold
    - mean_top1_iou
    - no_det_rate (optional via conf_thresh)

Supports multi-GPU evaluation via torchrun.
"""

import argparse
import datetime
import json
import os
import sys
import time

import torch
import torch.distributed as dist
from PIL import Image

ROOT = "/home/iibrohimm/project/next_step"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "GroundingDINO", "GroundingDINO"))

from groundingdino.util.inference import load_model as load_gd_model
from groundingdino.util.misc import nested_tensor_from_tensor_list
from thinkdet.data.refcoco_grounding import build_dino_transform, build_internvl_transform
from thinkdet.models.arch import DEFAULT_INJECTION_LAYERS, ThinkDetModel


GD_CONFIG = (
    f"{ROOT}/GroundingDINO/GroundingDINO/groundingdino/config/"
    "GroundingDINO_SwinT_OGC.py"
)
GD_WEIGHTS = (
    f"{ROOT}/GroundingDINO/GroundingDINO/weights/groundingdino_swint_ogc.pth"
)
INTERNVL_PATH = f"{ROOT}/InternVL3_5-1B"
DEFAULT_STAGE2 = (
    f"{ROOT}/thinkdet/checkpoints/stage2_tma/"
    "layer9_tma_m8_20260218_234913/"
    "thinkdet_tma_stage2_epoch2.pth"
)
DEFAULT_OMNILABEL_JSON = f"{ROOT}/dataSets/omnilabel/dataset_all_val_v0.1.3.json"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        type=str,
        default="thinkdet",
        choices=["thinkdet", "baseline"],
        help="Model to evaluate.",
    )
    parser.add_argument("--omnilabel_json", type=str, default=DEFAULT_OMNILABEL_JSON)
    parser.add_argument(
        "--subset",
        type=str,
        default="all",
        choices=["all", "coco", "object365", "openimagesv5"],
        help="Evaluate only a subset by image filename prefix.",
    )
    parser.add_argument("--coco_val_root", type=str, default=f"{ROOT}/dataSets/coco/val2017")
    parser.add_argument("--coco_train_root", type=str, default=f"{ROOT}/dataSets/coco/train2017")
    parser.add_argument("--object365_root", type=str, default=f"{ROOT}/dataSets/objects365/images")
    parser.add_argument("--openimages_root", type=str, default=f"{ROOT}/dataSets/openimagesv5/images")

    parser.add_argument("--thinkdet_checkpoint", type=str, default=DEFAULT_STAGE2)
    parser.add_argument("--gd_config", type=str, default=GD_CONFIG)
    parser.add_argument("--gd_weights", type=str, default=GD_WEIGHTS)
    parser.add_argument("--internvl_path", type=str, default=INTERNVL_PATH)

    parser.add_argument("--max_samples", type=int, default=0, help="0 means full set.")
    parser.add_argument("--iou_thresh", type=float, default=0.5)
    parser.add_argument("--conf_thresh", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--log_every", type=int, default=500)
    parser.add_argument("--output", type=str, default="")
    return parser.parse_args()


def setup_distributed(requested_device, use_distributed):
    if not use_distributed:
        return False, 0, 1, torch.device(requested_device if torch.cuda.is_available() else "cpu")

    if not dist.is_available():
        raise RuntimeError("torch.distributed not available")
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    return True, rank, world_size, device


def resolve_image_path(file_name, args):
    fn = file_name.strip().lstrip("/")
    if fn.startswith("coco/"):
        rel = fn[len("coco/") :]
        cand = [
            os.path.join(args.coco_val_root, rel),
            os.path.join(args.coco_train_root, rel),
        ]
        for p in cand:
            if os.path.exists(p):
                return p
        return cand[0]
    if fn.startswith("object365/"):
        rel = fn[len("object365/") :]
        return os.path.join(args.object365_root, rel)
    if fn.startswith("openimagesv5/"):
        rel = fn[len("openimagesv5/") :]
        return os.path.join(args.openimages_root, rel)

    # Fallback path probes
    base = os.path.basename(fn)
    cand = [
        os.path.join(args.coco_val_root, base),
        os.path.join(args.coco_train_root, base),
        os.path.join(args.object365_root, fn),
        os.path.join(args.openimages_root, fn),
    ]
    for p in cand:
        if os.path.exists(p):
            return p
    return cand[0]


def get_special_tokens(model):
    tokens = getattr(model, "special_tokens", None)
    if tokens is not None:
        return tokens
    tokens = getattr(model, "specical_tokens", None)
    if tokens is not None:
        return tokens
    return model.tokenizer.all_special_ids


def build_positive_map_for_query(tokenizer, special_tokens, query_text, max_text_len=512):
    tokenized = tokenizer(query_text, return_tensors="pt")
    input_ids = tokenized["input_ids"][0]
    positive_map = torch.zeros(1, max_text_len, dtype=torch.float32)
    special_set = set(int(x) for x in special_tokens)
    for pos, tok_id in enumerate(input_ids.tolist()):
        if pos >= max_text_len:
            break
        if tok_id not in special_set:
            positive_map[0, pos] = 1.0
    row_sum = positive_map.sum(dim=-1, keepdim=True).clamp(min=1e-6)
    return positive_map / row_sum


def box_cxcywh_to_xyxy(box):
    cx, cy, w, h = box.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)


def pairwise_iou_xyxy(boxes1, boxes2):
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter
    return inter / (union + 1e-6)


def score_outputs(outputs, positive_map_norm):
    logits = outputs["pred_logits"][0].clamp(-50, 50)
    boxes = outputs["pred_boxes"][0]
    pmap = positive_map_norm[:, : logits.shape[-1]].to(logits.device)
    probs = logits.sigmoid()
    scores = (probs * pmap).sum(dim=-1)
    return scores, boxes


def init_counters():
    return {
        "samples": 0,
        "top1_correct": 0,
        "iou_sum": 0.0,
        "top1_score_sum": 0.0,
        "no_det": 0,
    }


def merge_counters(dst, src):
    dst["samples"] += int(src["samples"])
    dst["top1_correct"] += int(src["top1_correct"])
    dst["iou_sum"] += float(src["iou_sum"])
    dst["top1_score_sum"] += float(src["top1_score_sum"])
    dst["no_det"] += int(src["no_det"])


def counters_to_tensor(c):
    return torch.tensor(
        [
            float(c["samples"]),
            float(c["top1_correct"]),
            float(c["iou_sum"]),
            float(c["top1_score_sum"]),
            float(c["no_det"]),
        ],
        dtype=torch.float64,
    )


def tensor_to_counters(t):
    return {
        "samples": int(round(t[0].item())),
        "top1_correct": int(round(t[1].item())),
        "iou_sum": float(t[2].item()),
        "top1_score_sum": float(t[3].item()),
        "no_det": int(round(t[4].item())),
    }


def summarize_counters(c):
    n = max(c["samples"], 1)
    return {
        "samples": c["samples"],
        "top1_acc": c["top1_correct"] / n,
        "mean_top1_iou": c["iou_sum"] / n,
        "mean_top1_score": c["top1_score_sum"] / n,
        "no_det_rate": c["no_det"] / n,
    }


def xywh_abs_to_cxcywh_norm(bbox_xywh, img_w, img_h, device):
    x, y, w, h = [float(v) for v in bbox_xywh]
    cx = (x + w / 2.0) / max(float(img_w), 1.0)
    cy = (y + h / 2.0) / max(float(img_h), 1.0)
    nw = w / max(float(img_w), 1.0)
    nh = h / max(float(img_h), 1.0)
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    nw = max(1e-6, min(1.0, nw))
    nh = max(1e-6, min(1.0, nh))
    return torch.tensor([[cx, cy, nw, nh]], dtype=torch.float32, device=device)


def load_omnilabel_samples(args, is_main):
    data = json.load(open(args.omnilabel_json))

    image_id_to_info = {}
    missing_images = 0
    dropped_prefix = 0
    for img in data.get("images", []):
        image_id = int(img["id"])
        file_name = img["file_name"]
        prefix = file_name.split("/", 1)[0] if "/" in file_name else "unknown"
        if args.subset != "all" and prefix != args.subset:
            dropped_prefix += 1
            continue
        image_path = resolve_image_path(file_name, args)
        if not os.path.exists(image_path):
            missing_images += 1
            continue
        image_id_to_info[image_id] = (file_name, image_path, prefix)

    desc_by_id = {}
    for d in data.get("descriptions", []):
        desc_by_id[int(d["id"])] = str(d.get("text", "")).strip()

    samples = []
    dropped_anns = 0
    for ann in data.get("annotations", []):
        image_id = int(ann["image_id"])
        info = image_id_to_info.get(image_id)
        if info is None:
            dropped_anns += 1
            continue
        bbox = ann.get("bbox", None)
        if not isinstance(bbox, list) or len(bbox) != 4:
            dropped_anns += 1
            continue
        desc_ids = ann.get("description_ids", [])
        if not desc_ids:
            dropped_anns += 1
            continue
        ann_id = int(ann.get("id", -1))
        for did in desc_ids:
            text = desc_by_id.get(int(did), "").strip()
            if not text:
                continue
            file_name, image_path, prefix = info
            samples.append(
                {
                    "image_id": image_id,
                    "ann_id": ann_id,
                    "desc_id": int(did),
                    "file_name": file_name,
                    "image_path": image_path,
                    "prefix": prefix,
                    "bbox_xywh": [float(v) for v in bbox],
                    "query_text": text.strip().lower() + " .",
                }
            )

    samples.sort(key=lambda x: (x["image_id"], x["ann_id"], x["desc_id"]))
    total_before_cap = len(samples)
    if args.max_samples > 0:
        samples = samples[: args.max_samples]

    meta = {
        "num_images_total_in_json": len(data.get("images", [])),
        "num_annotations_total_in_json": len(data.get("annotations", [])),
        "num_descriptions_total_in_json": len(data.get("descriptions", [])),
        "num_images_resolved_local": len(image_id_to_info),
        "num_images_missing_local": missing_images,
        "num_images_dropped_by_subset": dropped_prefix,
        "num_annotations_dropped": dropped_anns,
        "num_samples_total_before_cap": total_before_cap,
        "num_samples_after_cap": len(samples),
    }
    if is_main:
        print("[OmniLabel] sample build summary:")
        for k, v in meta.items():
            print(f"  {k}: {v}")
    return samples, meta


def load_thinkdet(args, device):
    gd = load_gd_model(args.gd_config, args.gd_weights, device="cpu")
    tokenizer = gd.tokenizer
    special_tokens = get_special_tokens(gd)

    ckpt = torch.load(args.thinkdet_checkpoint, map_location="cpu")
    extract_layers = ckpt.get("extract_layers", [9])
    injection_layers = ckpt.get("injection_layers", DEFAULT_INJECTION_LAYERS)
    tma_m = ckpt.get("tma_m", 8)
    tma_n_heads = ckpt.get("tma_n_heads", 8)

    model = ThinkDetModel(
        grounding_dino=gd,
        internvl_path=args.internvl_path,
        extract_layer=max(extract_layers),
        extract_layers=extract_layers,
        injection_layers=injection_layers,
        tma_m=tma_m,
        tma_n_heads=tma_n_heads,
    )
    missing, unexpected = model.load_state_dict(ckpt["trainable_state_dict"], strict=False)
    model = model.to(device).eval()
    return model, tokenizer, special_tokens, missing, unexpected


@torch.no_grad()
def main():
    args = parse_args()
    is_dist, rank, world_size, device = setup_distributed(args.device, args.distributed)
    is_main = rank == 0

    if not args.output:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"{ROOT}/thinkdet/results/eval/omnilabel_{args.model_type}_eval_{ts}.json"

    if is_main:
        print("=" * 80)
        print("OmniLabel Eval")
        print("=" * 80)
        print(f"model_type:    {args.model_type}")
        print(f"omnilabel_json: {args.omnilabel_json}")
        print(f"subset:        {args.subset}")
        print(f"device:        {device}")
        print(f"distributed:   {is_dist}  world_size: {world_size}")
        print(f"iou_thresh:    {args.iou_thresh}")
        print(f"conf_thresh:   {args.conf_thresh}")
        print(f"max_samples:   {args.max_samples}")
        if args.model_type == "thinkdet":
            print(f"checkpoint:    {args.thinkdet_checkpoint}")
        else:
            print(f"gd_weights:    {args.gd_weights}")
        print()

    samples, data_meta = load_omnilabel_samples(args, is_main=is_main)
    if not samples:
        raise RuntimeError("No valid OmniLabel samples found after filtering/path resolution.")

    # Use contiguous range per rank to maximize image-cache reuse.
    n = len(samples)
    chunk = (n + world_size - 1) // world_size
    start = rank * chunk
    end = min(n, (rank + 1) * chunk)
    local_samples = samples[start:end]

    if is_main:
        print(f"[Eval] total_samples={n}, per_rank_chunk={chunk}")

    if args.model_type == "thinkdet":
        model, tokenizer, special_tokens, missing, unexpected = load_thinkdet(args, device)
        if is_main:
            print(f"[Checkpoint] missing={len(missing)} unexpected={len(unexpected)}")
    else:
        model = load_gd_model(args.gd_config, args.gd_weights, device="cpu").to(device).eval()
        tokenizer = model.tokenizer
        special_tokens = get_special_tokens(model)

    internvl_tf = build_internvl_transform(input_size=448) if args.model_type == "thinkdet" else None
    dino_tf = build_dino_transform()

    counters = init_counters()
    t0 = time.time()

    cached_image_id = None
    cached_pil = None
    cached_iv = None
    cached_dino_nested = None
    cached_w = None
    cached_h = None

    for i, sample in enumerate(local_samples):
        image_id = sample["image_id"]
        if image_id != cached_image_id:
            cached_pil = Image.open(sample["image_path"]).convert("RGB")
            cached_w, cached_h = cached_pil.size

            if args.model_type == "thinkdet":
                iv_img = internvl_tf(cached_pil).unsqueeze(0).to(device)
            else:
                iv_img = None
            dino_img, _ = dino_tf(cached_pil, None)
            dino_img = dino_img.to(device)
            dino_nested = nested_tensor_from_tensor_list([dino_img])

            cached_image_id = image_id
            cached_iv = iv_img
            cached_dino_nested = dino_nested

        gt_box = xywh_abs_to_cxcywh_norm(sample["bbox_xywh"], cached_w, cached_h, device)
        query_text = sample["query_text"]
        pmap_norm = build_positive_map_for_query(tokenizer, special_tokens, query_text, max_text_len=512)

        if args.model_type == "thinkdet":
            dino_inputs = {"samples": cached_dino_nested, "captions": [query_text]}
            outputs, _ = model(cached_iv, [query_text], dino_inputs)
        else:
            outputs = model(samples=cached_dino_nested, captions=[query_text])

        scores, boxes = score_outputs(outputs, pmap_norm)
        best_idx = int(scores.argmax().item())
        top1_score = float(scores[best_idx].item())

        pred_top = boxes[best_idx].unsqueeze(0)
        top1_iou = float(
            pairwise_iou_xyxy(
                box_cxcywh_to_xyxy(pred_top),
                box_cxcywh_to_xyxy(gt_box),
            )[0, 0].item()
        )
        top1_correct = 1 if top1_iou >= args.iou_thresh else 0
        no_det = 1 if (args.conf_thresh > 0 and top1_score < args.conf_thresh) else 0

        merge_counters(
            counters,
            {
                "samples": 1,
                "top1_correct": top1_correct,
                "iou_sum": top1_iou,
                "top1_score_sum": top1_score,
                "no_det": no_det,
            },
        )

        if (i + 1) % max(1, args.log_every) == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 1e-6)
            print(
                f"[rank{rank}] {i+1}/{len(local_samples)} "
                f"({rate:.2f} samples/s) top1_acc={counters['top1_correct']/max(counters['samples'],1):.4f}"
            )

    if is_dist:
        t = counters_to_tensor(counters).to(device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        counters = tensor_to_counters(t.cpu())

    if not is_main:
        return

    summary = summarize_counters(counters)
    payload = {
        "status": "ok",
        "timestamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
        "dataset": {
            "name": "omnilabel",
            "json": args.omnilabel_json,
            "subset": args.subset,
        },
        "config": {
            "model_type": args.model_type,
            "iou_thresh": args.iou_thresh,
            "conf_thresh": args.conf_thresh,
            "max_samples": args.max_samples,
            "distributed": bool(is_dist),
            "world_size": int(world_size),
            "checkpoint": args.thinkdet_checkpoint if args.model_type == "thinkdet" else args.gd_weights,
        },
        "data_meta": data_meta,
        "summary": summary,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)

    print("\nSaved:", args.output)
    print("top1_acc:", summary["top1_acc"])
    print("mean_top1_iou:", summary["mean_top1_iou"])
    print("mean_top1_score:", summary["mean_top1_score"])


if __name__ == "__main__":
    main()
