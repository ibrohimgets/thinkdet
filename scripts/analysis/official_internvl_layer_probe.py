#!/usr/bin/env python
"""
Official InternVL-style layer probe for functional prompts.

This script does not train or run ThinkDet detector heads. It follows the
InternVL chat construction:

  1. Build a chat prompt containing <image>.
  2. Replace <image> with <img><IMG_CONTEXT>...</img>.
  3. Tokenize the full prompt normally.
  4. Extract visual embeddings with model.extract_feature(pixel_values).
  5. Replace <IMG_CONTEXT> token embeddings with those visual embeddings.
  6. Run the Qwen language model and collect hidden states.
  7. Score image-token positions against the prompt-token representation.

The default benchmark is the balanced COCO affordance subset for:
  - something to drink from
  - something to cut with
  - something to sit on
"""

import argparse
import datetime as _datetime
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image


ROOT = "/home/iibrohimm/project/next_step"
sys.path.insert(0, ROOT)

from thinkdet.models.projector import build_internvl_transform


DEFAULT_INTERNVL = f"{ROOT}/InternVL3_5-1B"
DEFAULT_BENCH = (
    f"{ROOT}/thinkdet/data/benchmarks/"
    "affordance_coco_val_three_prompts_test_balanced30.json"
)
DEFAULT_OUT = (
    f"{ROOT}/thinkdet/results/official_internvl_layer_probe/"
    f"official_internvl_layer_probe_{_datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
)

PROMPT_BY_AFFORDANCE = {
    "drink_from": "something to drink from",
    "cut_with": "something to cut with",
    "sit_on": "something to sit on",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=str, default=DEFAULT_BENCH)
    parser.add_argument("--internvl_path", type=str, default=DEFAULT_INTERNVL)
    parser.add_argument("--layers", type=int, nargs="+", default=[7, 9, 11, 14])
    parser.add_argument(
        "--affordances",
        type=str,
        nargs="+",
        default=["drink_from", "cut_with", "sit_on"],
    )
    parser.add_argument("--split", type=str, default="test", choices=["dev", "test", "all"])
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
    )
    parser.add_argument("--softmax_temperature", type=float, default=0.07)
    parser.add_argument("--topk_tokens", type=int, nargs="+", default=[1, 5, 16])
    parser.add_argument(
        "--rank_by",
        type=str,
        default="top5_token_hit",
        choices=["top1_token_hit", "top5_token_hit", "top16_token_hit", "mass_lift", "gt_bg_z"],
    )
    parser.add_argument("--output", type=str, default=DEFAULT_OUT)
    parser.add_argument("--log_every", type=int, default=5)
    return parser.parse_args()


def torch_dtype(name):
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def xywh_to_xyxy(box):
    x, y, w, h = box
    return [float(x), float(y), float(x + w), float(y + h)]


def token_overlap_mask(image_size, gt_boxes_xyxy, n_tokens):
    img_w, img_h = image_size
    side = int(round(math.sqrt(n_tokens)))
    if side * side != n_tokens:
        raise ValueError(f"Expected square visual-token grid, got {n_tokens} tokens")

    positives = []
    overlap_fracs = []
    cell_w = float(img_w) / side
    cell_h = float(img_h) / side
    cell_area = max(cell_w * cell_h, 1e-9)

    for idx in range(n_tokens):
        row = idx // side
        col = idx % side
        x1 = col * cell_w
        y1 = row * cell_h
        x2 = (col + 1) * cell_w
        y2 = (row + 1) * cell_h

        overlap_area = 0.0
        for gx1, gy1, gx2, gy2 in gt_boxes_xyxy:
            ix1 = max(x1, gx1)
            iy1 = max(y1, gy1)
            ix2 = min(x2, gx2)
            iy2 = min(y2, gy2)
            overlap_area += max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)

        frac = min(1.0, overlap_area / cell_area)
        positives.append(frac > 0.0)
        overlap_fracs.append(frac)

    return (
        torch.tensor(positives, dtype=torch.bool),
        torch.tensor(overlap_fracs, dtype=torch.float32),
    )


def find_subsequence(sequence, pattern):
    if not pattern or len(pattern) > len(sequence):
        return []
    limit = len(sequence) - len(pattern) + 1
    for start in range(limit):
        if sequence[start:start + len(pattern)] == pattern:
            return list(range(start, start + len(pattern)))
    return []


def prompt_token_positions(tokenizer, input_ids, prompt):
    sequence = input_ids.tolist()
    candidates = [
        prompt,
        "\n" + prompt,
        " " + prompt,
        prompt + ".",
        prompt + " .",
    ]
    for text in candidates:
        pattern = tokenizer(text, add_special_tokens=False).input_ids
        positions = find_subsequence(sequence, pattern)
        if positions:
            return positions, "matched_prompt_subsequence"
    return [], "fallback_last_non_image_token"


class OfficialInternVLLayerProbe:
    def __init__(self, model_path, layers, device, dtype):
        self.model_path = model_path
        self.layers = sorted(set(int(x) for x in layers))
        self.device = device
        self.dtype = dtype

        self._load_model()

    def _load_model(self):
        from packaging import version
        import transformers
        from transformers import AutoModel, AutoTokenizer

        if version.parse(transformers.__version__) < version.parse("4.37.0"):
            raise RuntimeError(
                "InternVL3.5 remote code requires transformers >= 4.37.0; "
                f"current version is {transformers.__version__}."
            )

        self.model = AutoModel.from_pretrained(
            self.model_path,
            torch_dtype=self.dtype,
            trust_remote_code=True,
            use_flash_attn=False,
            low_cpu_mem_usage=False,
        ).to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            use_fast=False,
            fix_mistral_regex=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.img_context_token_id = self.tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
        self.model.img_context_token_id = self.img_context_token_id
        self.num_image_token = int(self.model.num_image_token)
        self.num_layers = int(self.model.language_model.config.num_hidden_layers)
        self.hidden_size = int(self.model.language_model.config.hidden_size)

        for layer in self.layers:
            if layer < 0 or layer >= self.num_layers:
                raise ValueError(f"Layer {layer} outside valid range 0..{self.num_layers - 1}")

    def build_query(self, prompt, num_patches=1):
        question = prompt if "<image>" in prompt else "<image>\n" + prompt
        template = self.model.conv_template.copy()
        template.system_message = self.model.system_message
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()

        image_tokens = (
            "<img>"
            + "<IMG_CONTEXT>" * self.num_image_token * int(num_patches)
            + "</img>"
        )
        query = query.replace("<image>", image_tokens, 1)
        return query

    @torch.no_grad()
    def extract_layer_scores(self, pixel_values, prompt):
        pixel_values = pixel_values.to(device=self.device, dtype=self.dtype)
        vit_embeds = self.model.extract_feature(pixel_values)

        query = self.build_query(prompt, num_patches=pixel_values.shape[0])
        model_inputs = self.tokenizer(query, return_tensors="pt")
        input_ids = model_inputs["input_ids"].to(self.device)
        attention_mask = model_inputs["attention_mask"].to(self.device)

        image_token_mask = input_ids == self.img_context_token_id
        expected_tokens = vit_embeds.reshape(-1, vit_embeds.shape[-1]).shape[0]
        actual_tokens = int(image_token_mask.sum().item())
        if actual_tokens != expected_tokens:
            raise RuntimeError(
                f"<IMG_CONTEXT> token count mismatch: tokenizer={actual_tokens}, "
                f"visual_features={expected_tokens}"
            )

        input_embeds = self.model.language_model.get_input_embeddings()(input_ids).clone()
        bsz, seq_len, hidden = input_embeds.shape
        flat_embeds = input_embeds.reshape(bsz * seq_len, hidden)
        flat_mask = image_token_mask.reshape(bsz * seq_len)
        flat_embeds[flat_mask] = vit_embeds.reshape(-1, hidden).to(flat_embeds.dtype)
        input_embeds = flat_embeds.reshape(bsz, seq_len, hidden)

        outputs = self.model.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )

        prompt_positions, query_pool = prompt_token_positions(
            self.tokenizer, input_ids[0].detach().cpu(), prompt
        )
        if not prompt_positions:
            non_image = (~image_token_mask[0]) & attention_mask[0].bool()
            prompt_positions = [int(non_image.nonzero()[-1].item())]

        per_layer = {}
        for layer in self.layers:
            # HF hidden_states[0] is the embedding output; hidden_states[L + 1]
            # is the output after decoder layer L.
            hidden_states = outputs.hidden_states[layer + 1][0]
            image_h = hidden_states[image_token_mask[0]]
            query_h = hidden_states[prompt_positions].mean(dim=0)

            image_h = F.normalize(image_h.float(), dim=-1)
            query_h = F.normalize(query_h.float(), dim=0)
            scores = image_h @ query_h
            per_layer[layer] = scores.detach().cpu()

        return per_layer, {
            "query_pool": query_pool,
            "prompt_token_count": len(prompt_positions),
            "sequence_length": int(input_ids.shape[1]),
            "num_image_tokens": actual_tokens,
        }


def compute_metrics(token_scores, positive_mask, overlap_fracs, topk_tokens, temperature):
    scores = token_scores.float()
    pos = positive_mask.to(scores.device)
    overlap = overlap_fracs.to(scores.device)

    if not pos.any() or pos.all():
        return None

    order = scores.argsort(descending=True)
    out = {}
    for k in topk_tokens:
        kk = min(int(k), len(order))
        out[f"top{k}_token_hit"] = int(pos[order[:kk]].any().item())

    mass = torch.softmax(scores / max(float(temperature), 1e-6), dim=0)
    mass_in_positive_tokens = mass[pos].sum()
    positive_token_frac = pos.float().mean()
    mass_lift = mass_in_positive_tokens / positive_token_frac.clamp_min(1e-6)

    pos_scores = scores[pos]
    bg_scores = scores[~pos]
    pooled_std = scores.std(unbiased=False).clamp_min(1e-6)
    gt_bg_z = (pos_scores.mean() - bg_scores.mean()) / pooled_std

    out.update({
        "mass_in_positive_tokens": float(mass_in_positive_tokens.item()),
        "positive_token_frac": float(positive_token_frac.item()),
        "mass_lift": float(mass_lift.item()),
        "gt_bg_z": float(gt_bg_z.item()),
        "mean_score": float(scores.mean().item()),
        "max_score": float(scores.max().item()),
        "weighted_overlap_mass": float((mass * overlap).sum().item()),
    })
    return out


def summarize_rows(rows, topk_tokens):
    n = max(len(rows), 1)
    summary = {
        "n_samples": len(rows),
        "mass_in_positive_tokens": sum(r["mass_in_positive_tokens"] for r in rows) / n,
        "positive_token_frac": sum(r["positive_token_frac"] for r in rows) / n,
        "mass_lift": sum(r["mass_lift"] for r in rows) / n,
        "gt_bg_z": sum(r["gt_bg_z"] for r in rows) / n,
        "mean_score": sum(r["mean_score"] for r in rows) / n,
        "max_score": sum(r["max_score"] for r in rows) / n,
        "weighted_overlap_mass": sum(r["weighted_overlap_mass"] for r in rows) / n,
    }
    for k in topk_tokens:
        key = f"top{k}_token_hit"
        summary[key] = sum(r[key] for r in rows) / n
    return summary


def best_layer_from_payload(payload):
    rank_by = payload["rank_by"]
    rows = [
        (int(layer), item["summary"])
        for layer, item in payload["per_layer"].items()
    ]
    return sorted(
        rows,
        key=lambda item: (
            -item[1].get(rank_by, 0.0),
            -item[1].get("mass_lift", 0.0),
            -item[1].get("gt_bg_z", 0.0),
            item[0],
        ),
    )[0]


def write_markdown(payload, out_json):
    rows = [
        (int(layer), item["summary"])
        for layer, item in sorted(payload["per_layer"].items(), key=lambda kv: int(kv[0]))
    ]
    winner_layer, winner_summary = best_layer_from_payload(payload)
    layer14 = payload["per_layer"].get("14", {}).get("summary")
    layer14_strongest = winner_layer == 14

    lines = [
        "# Official InternVL-Style Layer Probe",
        "",
        f"Benchmark: `{payload['benchmark']}`",
        f"Samples: {payload['n_samples']}",
        f"Layers: {payload['layers']}",
        f"Rank key: `{payload['rank_by']}`",
        "",
        "| Layer | Top-1 | Top-5 | Top-16 | Mass Lift | GT-BG z | Mean Score | N |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for layer, summary in rows:
        lines.append(
            f"| {layer} | "
            f"{100.0 * summary.get('top1_token_hit', 0.0):.2f}% | "
            f"{100.0 * summary.get('top5_token_hit', 0.0):.2f}% | "
            f"{100.0 * summary.get('top16_token_hit', 0.0):.2f}% | "
            f"{summary['mass_lift']:.3f} | "
            f"{summary['gt_bg_z']:.3f} | "
            f"{summary['mean_score']:.6f} | "
            f"{summary['n_samples']} |"
        )

    lines.extend([
        "",
        f"Best layer: **{winner_layer}** "
        f"({payload['rank_by']}={winner_summary.get(payload['rank_by'], 0.0):.4f}, "
        f"mass_lift={winner_summary['mass_lift']:.3f}, "
        f"gt_bg_z={winner_summary['gt_bg_z']:.3f}).",
        f"Layer 14 strongest: **{'yes' if layer14_strongest else 'no'}**.",
    ])
    if layer14 is not None:
        lines.append(
            "Layer 14 summary: "
            f"Top-5={100.0 * layer14.get('top5_token_hit', 0.0):.2f}%, "
            f"mass_lift={layer14['mass_lift']:.3f}, "
            f"gt_bg_z={layer14['gt_bg_z']:.3f}."
        )

    lines.extend([
        "",
        "## Metric Definitions",
        "",
        "- Top-k token hit: whether any of the top-k image tokens by query-image cosine score overlaps a GT target box.",
        "- Mass lift: softmax score mass on GT-overlapping tokens divided by the GT token fraction; 1.0 is uniform.",
        "- GT-BG z: mean raw cosine score on GT tokens minus background tokens, divided by the std over all image tokens.",
        "",
        "## Limitations",
        "",
        "- This is representation probing only; it does not train or inject into the detector.",
        "- It uses one 448x448 resized image, not InternVL dynamic tiling, so box-to-token mapping stays 16x16.",
        "- In causal Qwen, image-token states do not attend to later prompt words; prompt signal is measured through same-layer prompt-token representations compared to image tokens.",
        "- GT boxes are projected to a coarse 16x16 visual-token grid, so small objects can be noisy.",
    ])

    md_path = Path(out_json).with_suffix(".md")
    md_path.write_text("\n".join(lines) + "\n")
    return str(md_path), winner_layer


def load_samples(args):
    with open(args.benchmark, "r") as f:
        bench = json.load(f)
    samples = bench["samples"] if isinstance(bench, dict) else bench

    affordances = set(args.affordances)
    samples = [s for s in samples if s.get("affordance_id") in affordances]
    if args.split != "all":
        samples = [s for s in samples if s.get("split") == args.split]
    if args.max_samples > 0:
        samples = samples[: args.max_samples]

    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard_index must be in [0, num_shards)")
    samples = [
        sample
        for idx, sample in enumerate(samples)
        if idx % int(args.num_shards) == int(args.shard_index)
    ]
    return samples


@torch.no_grad()
def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = torch_dtype(args.dtype if device.type == "cuda" else "float32")

    samples = load_samples(args)
    if not samples:
        raise RuntimeError("No samples selected. Check --benchmark, --split, and --affordances.")

    print(f"[official-probe] samples={len(samples)} device={device} dtype={dtype}")
    print(f"[official-probe] loading InternVL: {args.internvl_path}")
    probe = OfficialInternVLLayerProbe(args.internvl_path, args.layers, device, dtype)

    transform = build_internvl_transform()
    rows_by_layer = {layer: [] for layer in probe.layers}
    per_affordance = {layer: defaultdict(list) for layer in probe.layers}
    skipped = []
    query_pool_counts = defaultdict(int)

    for idx, sample in enumerate(samples, 1):
        image_path = sample["image_path"]
        if not os.path.exists(image_path):
            skipped.append({"benchmark_id": sample.get("benchmark_id"), "reason": "missing_image"})
            continue

        prompt = PROMPT_BY_AFFORDANCE.get(sample["affordance_id"], sample["prompt"].replace(" .", "").strip())
        image_pil = Image.open(image_path).convert("RGB")
        pixel_values = transform(image_pil).unsqueeze(0)
        gt_boxes = [xywh_to_xyxy(t["bbox_xywh"]) for t in sample["positive_targets"]]

        layer_scores, meta = probe.extract_layer_scores(pixel_values, prompt)
        query_pool_counts[meta["query_pool"]] += 1

        n_tokens = len(next(iter(layer_scores.values())))
        positive_mask, overlap_fracs = token_overlap_mask(image_pil.size, gt_boxes, n_tokens)

        for layer, scores in layer_scores.items():
            metrics = compute_metrics(
                scores,
                positive_mask,
                overlap_fracs,
                args.topk_tokens,
                args.softmax_temperature,
            )
            if metrics is None:
                continue
            row = {
                "benchmark_id": sample["benchmark_id"],
                "image_id": int(sample["image_id"]),
                "affordance_id": sample["affordance_id"],
                "prompt": prompt,
                "layer": int(layer),
                **metrics,
            }
            rows_by_layer[layer].append(row)
            per_affordance[layer][sample["affordance_id"]].append(row)

        if idx % max(1, args.log_every) == 0 or idx == len(samples):
            print(f"[official-probe] {idx}/{len(samples)}")

    per_layer = {}
    for layer in probe.layers:
        rows = rows_by_layer[layer]
        per_layer[str(layer)] = {
            "summary": summarize_rows(rows, args.topk_tokens),
            "per_affordance": {
                aff: summarize_rows(aff_rows, args.topk_tokens)
                for aff, aff_rows in sorted(per_affordance[layer].items())
            },
            "per_sample": rows,
        }

    payload = {
        "status": "ok",
        "timestamp": _datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
        "benchmark": args.benchmark,
        "internvl_path": args.internvl_path,
        "model_note": "InternVL3.5 image features inserted at <IMG_CONTEXT> positions, then Qwen hidden states probed.",
        "num_llm_layers": probe.num_layers,
        "hidden_size": probe.hidden_size,
        "layers": probe.layers,
        "prompts": PROMPT_BY_AFFORDANCE,
        "split": args.split,
        "affordances": args.affordances,
        "n_samples": sum(len(v) for v in rows_by_layer.values()) // max(len(probe.layers), 1),
        "skipped": skipped,
        "topk_tokens": [int(k) for k in args.topk_tokens],
        "softmax_temperature": float(args.softmax_temperature),
        "rank_by": args.rank_by,
        "query_pool_counts": dict(query_pool_counts),
        "metric_note": (
            "Scores are cosine(image token hidden state, mean prompt-token hidden state) "
            "from the same Qwen layer after official <IMG_CONTEXT> visual-token replacement."
        ),
        "per_layer": per_layer,
    }

    winner_layer, winner_summary = best_layer_from_payload(payload)
    payload["best_layer"] = int(winner_layer)
    payload["layer14_strongest"] = winner_layer == 14
    payload["best_layer_summary"] = winner_summary

    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)
    md_path, _ = write_markdown(payload, args.output)

    print("Saved:", args.output)
    print("Saved:", md_path)
    print("best_layer:", winner_layer)
    print("layer14_strongest:", "yes" if winner_layer == 14 else "no")


if __name__ == "__main__":
    main()
