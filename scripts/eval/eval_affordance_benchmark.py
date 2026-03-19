"""
Evaluate Baseline / Stage1 / Stage2 on held-out affordance benchmark.

Benchmark format expected from build_affordance_benchmark.py.
"""

import argparse
import csv
import datetime
import json
import os
import sys
from collections import defaultdict

import torch
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF

ROOT = "/home/iibrohimm/project/next_step"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "GroundingDINO", "GroundingDINO"))

from thinkdet.models.arch import ThinkDetModel, DEFAULT_INJECTION_LAYERS
from thinkdet.models.projector import InternVLFeatureExtractor
from thinkdet.inference.fallback import InternVLYesNoReranker
from groundingdino.util.inference import load_model as load_gd_model
from groundingdino.util.misc import NestedTensor


DEFAULT_BENCH = f"{ROOT}/thinkdet/data/benchmarks/affordance_coco_val_heldout_v1.json"
DEFAULT_OUT = f"{ROOT}/thinkdet/results/eval/affordance_benchmark_eval_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

GD_CONFIG = f"{ROOT}/GroundingDINO/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GD_WEIGHTS = f"{ROOT}/GroundingDINO/GroundingDINO/weights/groundingdino_swint_ogc.pth"
INTERNVL_PATH = f"{ROOT}/InternVL3_5-1B"
STAGE1_CKPT = (
    f"{ROOT}/thinkdet/checkpoints/stage1_tma/"
    "layer9_tma_m8_full_2ep_7gpu_20260218_152649/"
    "thinkdet_tma_stage1_epoch2.pth"
)
STAGE2_CKPT = (
    f"{ROOT}/thinkdet/checkpoints/stage2_tma/"
    "layer9_tma_m8_20260218_234913/"
    "thinkdet_tma_stage2_epoch2.pth"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=str, default=DEFAULT_BENCH)
    parser.add_argument("--split", type=str, default="test", choices=["dev", "test", "all"])
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output", type=str, default=DEFAULT_OUT)

    parser.add_argument("--gd_config", type=str, default=GD_CONFIG)
    parser.add_argument("--gd_weights", type=str, default=GD_WEIGHTS)
    parser.add_argument("--internvl_path", type=str, default=INTERNVL_PATH)
    parser.add_argument("--stage1_ckpt", type=str, default=STAGE1_CKPT)
    parser.add_argument("--stage2_ckpt", type=str, default=STAGE2_CKPT)
    parser.add_argument(
        "--shard_rank",
        type=int,
        default=-1,
        help="Shard rank for data-parallel eval (0-based).",
    )
    parser.add_argument(
        "--shard_world_size",
        type=int,
        default=1,
        help="Total number of shards for data-parallel eval.",
    )

    parser.add_argument(
        "--llm_rerank",
        action="store_true",
        help="Enable InternVL candidate-box reranking via yes/no feedback.",
    )
    parser.add_argument(
        "--llm_rerank_weight",
        type=float,
        default=0.2,
        help="Combined score = detector_score + weight * llm_score.",
    )
    parser.add_argument(
        "--llm_rerank_max_new_tokens",
        type=int,
        default=6,
        help="Max generation tokens for InternVL yes/no response.",
    )
    parser.add_argument(
        "--llm_rerank_temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for InternVL reranking generation.",
    )
    return parser.parse_args()


def build_dino_transform():
    normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def transform(image_pil):
        w, h = image_pil.size
        scale = 800 / min(w, h)
        if scale * max(w, h) > 1333:
            scale = 1333 / max(w, h)
        nw, nh = int(w * scale), int(h * scale)
        img = image_pil.resize((nw, nh), Image.BILINEAR)
        t = TF.to_tensor(img)
        return normalize(t)

    return transform


def build_internvl_transform():
    return T.Compose([
        T.Resize((448, 448), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def iou_xyxy(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (aa + bb - inter + 1e-9)


def xywh_to_xyxy(box):
    x, y, w, h = box
    return [x, y, x + w, y + h]


InternVLBoxReranker = InternVLYesNoReranker


def get_special_tokens(model):
    tokens = getattr(model, "specical_tokens", None)
    if tokens is not None:
        return tokens
    tokens = getattr(model, "special_tokens", None)
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


def score_outputs_for_query(outputs, positive_map_norm):
    logits = outputs["pred_logits"][0].clamp(-50, 50)
    boxes = outputs["pred_boxes"][0]
    text_len = logits.shape[-1]
    pmap = positive_map_norm[:, :text_len].to(logits.device)
    probs = logits.sigmoid()
    scores = (probs * pmap).sum(dim=-1)
    return scores, boxes


def resolve_query_scoring_assets(model_fn):
    if hasattr(model_fn, "grounding_dino"):
        dino_model = model_fn.grounding_dino
    else:
        dino_model = model_fn
    if not hasattr(dino_model, "tokenizer"):
        raise AttributeError("Could not resolve GroundingDINO tokenizer for query scoring")
    return dino_model.tokenizer, get_special_tokens(dino_model)


@torch.no_grad()
def run_topk(
    model_fn,
    image_pil,
    prompt,
    device,
    dino_tf,
    ivl_tf,
    top_k,
    tokenizer,
    special_tokens,
    reranker=None,
):
    img_w, img_h = image_pil.size
    dino_t = dino_tf(image_pil).unsqueeze(0).to(device)
    mask = torch.zeros(1, dino_t.shape[2], dino_t.shape[3], dtype=torch.bool, device=device)
    nested = NestedTensor(dino_t, mask)
    ivl_t = ivl_tf(image_pil).unsqueeze(0).to(device)
    query_text = prompt if prompt.strip().endswith(".") else (prompt.strip() + " .")
    dino_inputs = {"samples": nested, "captions": [query_text]}

    out = model_fn(ivl_t, [prompt], dino_inputs)
    outputs = out[0] if isinstance(out, tuple) else out

    positive_map_norm = build_positive_map_for_query(
        tokenizer, special_tokens, query_text, max_text_len=512
    )
    scores, boxes = score_outputs_for_query(outputs, positive_map_norm)
    vals, idx = scores.topk(min(top_k, scores.shape[0]))

    results = []
    for j in range(len(vals)):
        cx, cy, bw, bh = boxes[idx[j]].tolist()
        x1 = (cx - bw / 2) * img_w
        y1 = (cy - bh / 2) * img_h
        x2 = (cx + bw / 2) * img_w
        y2 = (cy + bh / 2) * img_h
        results.append({
            "score": float(vals[j].item()),
            "box_abs_xyxy": [x1, y1, x2, y2],
        })

    if reranker is not None and results:
        try:
            results = reranker.rerank_predictions(image_pil, prompt, results)
        except Exception as exc:
            # Keep detector-only ranking if reranker fails on a sample.
            print(f"  [WARN] LLM rerank failed: {type(exc).__name__}: {exc}")

    return results


class MetricAccumulator:
    def __init__(self, top_k):
        self.top_k = top_k
        self.n = 0
        self.sum_best_iou_top1 = 0.0
        self.sum_best_iou_topk = 0.0
        self.hit50_top1 = 0
        self.hit50_topk = 0
        self.hit75_top1 = 0
        self.hit75_topk = 0
        self.prec50_top1_sum = 0.0
        self.prec50_topk_sum = 0.0
        self.rec50_top1_sum = 0.0
        self.rec50_topk_sum = 0.0

    def update(self, pred_boxes, gt_boxes):
        self.n += 1
        k = min(self.top_k, len(pred_boxes))
        if k == 0:
            return

        # For each predicted box, best IoU vs any GT target.
        best_iou_per_pred = []
        for p in pred_boxes[:k]:
            ious = [iou_xyxy(p["box_abs_xyxy"], g) for g in gt_boxes]
            best_iou_per_pred.append(max(ious) if ious else 0.0)

        best1 = best_iou_per_pred[0]
        bestk = max(best_iou_per_pred)
        self.sum_best_iou_top1 += best1
        self.sum_best_iou_topk += bestk

        m50 = [x >= 0.5 for x in best_iou_per_pred]
        m75 = [x >= 0.75 for x in best_iou_per_pred]

        self.hit50_top1 += 1 if m50[0] else 0
        self.hit50_topk += 1 if any(m50) else 0
        self.hit75_top1 += 1 if m75[0] else 0
        self.hit75_topk += 1 if any(m75) else 0

        # Precision/recall view for single-query grounding:
        # precision@1 equals recall@1 (top1 hit), while @K differs.
        self.prec50_top1_sum += 1.0 if m50[0] else 0.0
        self.prec50_topk_sum += float(sum(m50)) / float(k)
        self.rec50_top1_sum += 1.0 if m50[0] else 0.0
        self.rec50_topk_sum += 1.0 if any(m50) else 0.0

    def to_dict(self):
        n = max(self.n, 1)
        return {
            "n_samples": self.n,
            "mean_best_iou_top1": self.sum_best_iou_top1 / n,
            "mean_best_iou_topk": self.sum_best_iou_topk / n,
            "hit@0.5_top1": self.hit50_top1 / n,
            "hit@0.5_topk": self.hit50_topk / n,
            "hit@0.75_top1": self.hit75_top1 / n,
            "hit@0.75_topk": self.hit75_topk / n,
            "precision@0.5_top1": self.prec50_top1_sum / n,
            "precision@0.5_topk": self.prec50_topk_sum / n,
            "recall@0.5_top1": self.rec50_top1_sum / n,
            "recall@0.5_topk": self.rec50_topk_sum / n,
        }


def evaluate_model(model_label, model_fn, samples, args, dino_tf, ivl_tf, device, reranker=None):
    overall = MetricAccumulator(args.top_k)
    per_aff = defaultdict(lambda: MetricAccumulator(args.top_k))
    per_sample = []
    tokenizer, special_tokens = resolve_query_scoring_assets(model_fn)

    for i, s in enumerate(samples):
        image_pil = Image.open(s["image_path"]).convert("RGB")
        preds = run_topk(
            model_fn,
            image_pil,
            s["prompt"],
            device,
            dino_tf,
            ivl_tf,
            args.top_k,
            tokenizer,
            special_tokens,
            reranker=reranker,
        )
        gt_boxes = [xywh_to_xyxy(t["bbox_xywh"]) for t in s["positive_targets"]]

        overall.update(preds, gt_boxes)
        per_aff[s["affordance_id"]].update(preds, gt_boxes)

        # Keep compact per-sample top1 record for significance checks later.
        best_top1 = 0.0
        if preds:
            best_top1 = max(iou_xyxy(preds[0]["box_abs_xyxy"], g) for g in gt_boxes)
        per_sample.append({
            "benchmark_id": s["benchmark_id"],
            "image_id": s["image_id"],
            "affordance_id": s["affordance_id"],
            "prompt": s["prompt"],
            "top1_iou": best_top1,
            "top1_score": preds[0]["score"] if preds else 0.0,
            "top1_llm_score": preds[0].get("llm_score", 0.0) if preds else 0.0,
            "top1_combined_score": preds[0].get("combined_score", preds[0]["score"]) if preds else 0.0,
        })

        if (i + 1) % args.log_every == 0:
            cur = overall.to_dict()
            print(
                f"  [{model_label}] {i+1}/{len(samples)} "
                f"hit@0.5_top1={cur['hit@0.5_top1']:.3f} "
                f"miou_top1={cur['mean_best_iou_top1']:.3f}"
            )

    return {
        "overall": overall.to_dict(),
        "per_affordance": {k: v.to_dict() for k, v in sorted(per_aff.items())},
        "per_sample": per_sample,
    }


def save_markdown_and_csv(output_json, payload):
    base = os.path.splitext(output_json)[0]
    md_path = base + ".md"
    csv_path = base + ".csv"

    models = ["baseline", "stage1", "stage2"]
    metrics = [
        "mean_best_iou_top1",
        "hit@0.5_top1",
        "hit@0.5_topk",
        "hit@0.75_top1",
        "precision@0.5_top1",
        "precision@0.5_topk",
        "recall@0.5_top1",
        "recall@0.5_topk",
    ]

    with open(md_path, "w") as f:
        f.write("# Held-Out Affordance Benchmark Results\n\n")
        f.write(f"- benchmark: `{payload['benchmark_path']}`\n")
        f.write(f"- split: `{payload['split']}`\n")
        f.write(f"- n_samples: {payload['n_samples']}\n")
        f.write(f"- top_k: {payload['top_k']}\n\n")

        f.write("## Overall\n\n")
        f.write("| model | " + " | ".join(metrics) + " |\n")
        f.write("|---|" + "|".join(["---:"] * len(metrics)) + "|\n")
        for m in models:
            row = payload["results"][m]["overall"]
            vals = [f"{row[k]:.4f}" for k in metrics]
            f.write(f"| {m} | " + " | ".join(vals) + " |\n")

        f.write("\n## Per-Affordance Hit@0.5 (Top1)\n\n")
        affs = sorted(payload["results"]["baseline"]["per_affordance"].keys())
        f.write("| affordance | baseline | stage1 | stage2 |\n")
        f.write("|---|---:|---:|---:|\n")
        for aff in affs:
            b = payload["results"]["baseline"]["per_affordance"][aff]["hit@0.5_top1"]
            s1 = payload["results"]["stage1"]["per_affordance"][aff]["hit@0.5_top1"]
            s2 = payload["results"]["stage2"]["per_affordance"][aff]["hit@0.5_top1"]
            f.write(f"| {aff} | {b:.4f} | {s1:.4f} | {s2:.4f} |\n")

    with open(csv_path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["model"] + metrics)
        for m in models:
            row = payload["results"][m]["overall"]
            wr.writerow([m] + [f"{row[k]:.6f}" for k in metrics])

    return md_path, csv_path


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    with open(args.benchmark, "r") as f:
        bench = json.load(f)

    samples = bench["samples"]
    if args.split != "all":
        samples = [s for s in samples if s.get("split") == args.split]
    if args.shard_world_size < 1:
        raise RuntimeError("shard_world_size must be >= 1")
    if args.shard_world_size > 1:
        if not (0 <= args.shard_rank < args.shard_world_size):
            raise RuntimeError(
                f"Invalid shard_rank={args.shard_rank} for shard_world_size={args.shard_world_size}"
            )
        samples = samples[args.shard_rank::args.shard_world_size]
    if not samples:
        raise RuntimeError(
            f"No samples for split={args.split}, "
            f"shard_rank={args.shard_rank}, shard_world_size={args.shard_world_size}"
        )

    print("=" * 70)
    print("Affordance benchmark evaluation")
    print(f"benchmark: {args.benchmark}")
    print(f"split: {args.split}  n_samples: {len(samples)}  top_k: {args.top_k}")
    if args.shard_world_size > 1:
        print(f"shard: rank={args.shard_rank}/{args.shard_world_size}")
    print(f"device: {device}")
    if args.llm_rerank:
        print(
            "llm_rerank: enabled  "
            f"weight={args.llm_rerank_weight}  "
            f"max_new_tokens={args.llm_rerank_max_new_tokens}  "
            f"temp={args.llm_rerank_temperature}"
        )
    else:
        print("llm_rerank: disabled")
    print("=" * 70)

    dino_tf = build_dino_transform()
    ivl_tf = build_internvl_transform()

    results = {}

    # Baseline
    print("\n[1/3] Loading baseline...")
    gd_base = load_gd_model(args.gd_config, args.gd_weights, device="cpu").to(device).eval()
    baseline_reranker = None
    baseline_reranker_extractor = None

    if args.llm_rerank:
        print("[1/3] Loading InternVL reranker for baseline...")
        baseline_reranker_extractor = InternVLFeatureExtractor(
            model_path=args.internvl_path,
            extract_layer=9,
            extract_layers=[9],
            freeze=True,
        ).to(device).eval()
        baseline_reranker = InternVLBoxReranker(
            internvl_model=baseline_reranker_extractor.internvl,
            tokenizer=baseline_reranker_extractor.tokenizer,
            device=device,
            image_transform=ivl_tf,
            weight=args.llm_rerank_weight,
            max_new_tokens=args.llm_rerank_max_new_tokens,
            temperature=args.llm_rerank_temperature,
        )

    def baseline_fn(ivl_imgs, queries, dino_inputs):
        return gd_base(**dino_inputs), {}
    baseline_fn.tokenizer = gd_base.tokenizer
    baseline_fn.specical_tokens = get_special_tokens(gd_base)

    print("[1/3] Evaluating baseline...")
    results["baseline"] = evaluate_model(
        "baseline",
        baseline_fn,
        samples,
        args,
        dino_tf,
        ivl_tf,
        device,
        reranker=baseline_reranker,
    )
    del gd_base, baseline_fn, baseline_reranker, baseline_reranker_extractor
    torch.cuda.empty_cache()

    # Stage1
    print("\n[2/3] Loading stage1...")
    ckpt1 = torch.load(args.stage1_ckpt, map_location="cpu")
    inject_layers1 = ckpt1.get("injection_layers", DEFAULT_INJECTION_LAYERS)
    tma_m1 = ckpt1.get("tma_m", 8)
    tma_nheads1 = ckpt1.get("tma_n_heads", 8)
    ext_layers1 = ckpt1.get("extract_layers") or [ckpt1.get("extract_layer", 9)]
    fusion_mode1 = ckpt1.get("fusion_mode", "concat")
    gd_s1 = load_gd_model(args.gd_config, args.gd_weights, device="cpu")
    model_s1 = ThinkDetModel(
        grounding_dino=gd_s1,
        internvl_path=args.internvl_path,
        extract_layer=max(ext_layers1),
        extract_layers=ext_layers1,
        injection_layers=inject_layers1,
        tma_m=tma_m1,
        tma_n_heads=tma_nheads1,
        fusion_mode=fusion_mode1,
    )
    model_s1.load_state_dict(ckpt1["trainable_state_dict"], strict=False)
    model_s1 = model_s1.to(device).eval()
    del ckpt1

    stage1_reranker = None
    if args.llm_rerank:
        stage1_reranker = InternVLBoxReranker(
            internvl_model=model_s1.feature_extractor.internvl,
            tokenizer=model_s1.feature_extractor.tokenizer,
            device=device,
            image_transform=ivl_tf,
            weight=args.llm_rerank_weight,
            max_new_tokens=args.llm_rerank_max_new_tokens,
            temperature=args.llm_rerank_temperature,
        )

    print("[2/3] Evaluating stage1...")
    results["stage1"] = evaluate_model(
        "stage1",
        model_s1,
        samples,
        args,
        dino_tf,
        ivl_tf,
        device,
        reranker=stage1_reranker,
    )
    del model_s1, stage1_reranker
    torch.cuda.empty_cache()

    # Stage2
    print("\n[3/3] Loading stage2...")
    ckpt2 = torch.load(args.stage2_ckpt, map_location="cpu")
    inject_layers = ckpt2.get("injection_layers", DEFAULT_INJECTION_LAYERS)
    tma_m = ckpt2.get("tma_m", 8)
    tma_nheads = ckpt2.get("tma_n_heads", 8)
    ext_layers = ckpt2.get("extract_layers") or [ckpt2.get("extract_layer", 9)]
    fusion_mode2 = ckpt2.get("fusion_mode", "concat")
    gd_s2 = load_gd_model(args.gd_config, args.gd_weights, device="cpu")
    model_s2 = ThinkDetModel(
        grounding_dino=gd_s2,
        internvl_path=args.internvl_path,
        extract_layer=max(ext_layers),
        extract_layers=ext_layers,
        injection_layers=inject_layers,
        tma_m=tma_m,
        tma_n_heads=tma_nheads,
        fusion_mode=fusion_mode2,
    )
    model_s2.load_state_dict(ckpt2["trainable_state_dict"], strict=False)
    model_s2 = model_s2.to(device).eval()
    del ckpt2

    stage2_reranker = None
    if args.llm_rerank:
        stage2_reranker = InternVLBoxReranker(
            internvl_model=model_s2.feature_extractor.internvl,
            tokenizer=model_s2.feature_extractor.tokenizer,
            device=device,
            image_transform=ivl_tf,
            weight=args.llm_rerank_weight,
            max_new_tokens=args.llm_rerank_max_new_tokens,
            temperature=args.llm_rerank_temperature,
        )

    print("[3/3] Evaluating stage2...")
    results["stage2"] = evaluate_model(
        "stage2",
        model_s2,
        samples,
        args,
        dino_tf,
        ivl_tf,
        device,
        reranker=stage2_reranker,
    )
    del model_s2, stage2_reranker
    torch.cuda.empty_cache()

    payload = {
        "status": "ok",
        "timestamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
        "benchmark_path": args.benchmark,
        "split": args.split,
        "n_samples": len(samples),
        "top_k": args.top_k,
        "shard": {
            "rank": int(args.shard_rank),
            "world_size": int(args.shard_world_size),
        },
        "llm_rerank": {
            "enabled": bool(args.llm_rerank),
            "weight": float(args.llm_rerank_weight),
            "max_new_tokens": int(args.llm_rerank_max_new_tokens),
            "temperature": float(args.llm_rerank_temperature),
        },
        "checkpoints": {
            "stage1": args.stage1_ckpt,
            "stage2": args.stage2_ckpt,
        },
        "results": results,
    }

    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)

    md_path, csv_path = save_markdown_and_csv(args.output, payload)

    print("\n" + "=" * 70)
    print("Completed affordance benchmark evaluation")
    print(f"JSON: {args.output}")
    print(f"MD:   {md_path}")
    print(f"CSV:  {csv_path}")
    print("=" * 70)
    for m in ["baseline", "stage1", "stage2"]:
        o = results[m]["overall"]
        print(
            f"{m:>8} | hit@0.5_top1={o['hit@0.5_top1']:.4f} "
            f"miou_top1={o['mean_best_iou_top1']:.4f} "
            f"hit@0.5_top{args.top_k}={o['hit@0.5_topk']:.4f}"
        )


if __name__ == "__main__":
    main()
