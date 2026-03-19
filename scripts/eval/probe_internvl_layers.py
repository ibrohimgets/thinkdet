#!/usr/bin/env python3
"""
Layer Probing Experiment for ThinkDet.

For each of InternVL's 28 LLM layers, extract visual token features and measure
how well they spatially correspond to the ground-truth object bounding box.

Approach:
  1. Run InternVL forward, collecting visual hidden states at ALL layers.
  2. Compute text query embedding (from the same LLM's embed_tokens).
  3. For each layer, compute cosine similarity between each visual token
     and the mean text embedding -> spatial "attention" map over the 16x16 grid.
  4. Map the GT box to the 16x16 grid and measure:
     - Mean attention inside the box vs. outside (signal/noise ratio)
     - "Pointing accuracy": does the max-attention token fall inside the box?

No training needed — this is purely diagnostic.
"""

import argparse
import json
import math
import os
import sys

import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

ROOT = "/home/iibrohimm/project/next_step"
sys.path.insert(0, ROOT)

INTERNVL_PATH = f"{ROOT}/InternVL3_5-1B"
DEFAULT_BENCH = (
    f"{ROOT}/thinkdet/data/benchmarks/commonsense_refcoco_objects_coco_val_v1.json"
)
DEFAULT_OUT = f"{ROOT}/thinkdet/results/eval/layer_probe_results.json"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", type=str, default=DEFAULT_BENCH)
    p.add_argument("--internvl_path", type=str, default=INTERNVL_PATH)
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--max_samples", type=int, default=200,
                    help="Max samples to probe (keep small for speed)")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--output", type=str, default=DEFAULT_OUT)
    return p.parse_args()


def build_transform(input_size=448):
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


def xywh_to_xyxy(box):
    x, y, w, h = box
    return [x, y, x + w, y + h]


def box_to_grid_mask(box_xyxy, img_w, img_h, grid_size=16):
    """Convert a bounding box in pixel coords to a binary mask on the grid."""
    x1, y1, x2, y2 = box_xyxy
    # Normalize to [0, 1]
    nx1 = max(0.0, x1 / img_w)
    ny1 = max(0.0, y1 / img_h)
    nx2 = min(1.0, x2 / img_w)
    ny2 = min(1.0, y2 / img_h)
    # Map to grid
    gx1 = int(nx1 * grid_size)
    gy1 = int(ny1 * grid_size)
    gx2 = max(gx1 + 1, int(math.ceil(nx2 * grid_size)))
    gy2 = max(gy1 + 1, int(math.ceil(ny2 * grid_size)))
    gx2 = min(gx2, grid_size)
    gy2 = min(gy2, grid_size)

    mask = torch.zeros(grid_size, grid_size, dtype=torch.bool)
    mask[gy1:gy2, gx1:gx2] = True
    return mask


@torch.no_grad()
def probe_all_layers(model, tokenizer, pixel_values, text_query, device):
    """
    Run InternVL forward and extract visual features at every LLM layer.

    Returns: dict[layer_idx] -> [256, D] visual token features
    """
    # Visual: InternViT -> pixel_shuffle -> MLP -> [1, 256, D]
    vit_embeds = model.extract_feature(pixel_values)
    num_vis = vit_embeds.shape[1]  # 256

    # Text: tokenize -> embed
    text_inputs = tokenizer(
        [text_query],
        return_tensors='pt',
        padding='max_length',
        truncation=True,
        max_length=128,
    ).to(device)

    text_embeds = model.language_model.get_input_embeddings()(text_inputs.input_ids)

    # Concat [visual; text]
    input_embeds = torch.cat([vit_embeds, text_embeds], dim=1)
    B, seq_len, _ = input_embeds.shape

    # Attention mask
    vis_mask = torch.ones(B, num_vis, device=device, dtype=torch.long)
    attention_mask = torch.cat([vis_mask, text_inputs.attention_mask], dim=1)

    # Forward through ALL layers
    llm = model.language_model.model
    hidden_states = input_embeds

    pad_mask = attention_mask[:, None, None, :].to(hidden_states.dtype)
    attn_mask = (1.0 - pad_mask) * torch.finfo(hidden_states.dtype).min

    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

    position_embeddings = None
    if hasattr(llm, 'rotary_emb'):
        position_embeddings = llm.rotary_emb(hidden_states, position_ids)

    num_layers = len(llm.layers)
    layer_features = {}

    # Also get the text embedding for similarity computation
    # Use the mean of text token embeddings as the "query" vector
    text_mask = text_inputs.attention_mask[0].bool()
    text_query_embed = text_embeds[0][text_mask].mean(dim=0)  # [D]

    for i in range(num_layers):
        layer_out = llm.layers[i](
            hidden_states,
            attention_mask=attn_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            use_cache=False,
        )
        hidden_states = layer_out[0] if isinstance(layer_out, tuple) else layer_out

        # Extract visual tokens only
        vis_feats = hidden_states[0, :num_vis, :]  # [256, D]
        layer_features[i] = vis_feats.cpu()

    return layer_features, text_query_embed.cpu()


def score_layer(vis_feats, text_embed, gt_grid_mask, grid_size=16):
    """
    Score how well a layer's visual features spatially correspond to the GT box.

    Returns dict with metrics.
    """
    # Cosine similarity between each visual token and text query
    vis_norm = F.normalize(vis_feats, dim=-1)  # [256, D]
    txt_norm = F.normalize(text_embed.unsqueeze(0), dim=-1)  # [1, D]
    sim = (vis_norm @ txt_norm.T).squeeze(-1)  # [256]

    # The 256 tokens correspond to a 16x16 grid (after pixel_shuffle)
    sim_grid = sim.view(grid_size, grid_size)

    # Metrics
    inside_mask = gt_grid_mask
    outside_mask = ~gt_grid_mask

    if inside_mask.sum() == 0 or outside_mask.sum() == 0:
        return None

    mean_inside = sim_grid[inside_mask].mean().item()
    mean_outside = sim_grid[outside_mask].mean().item()
    contrast = mean_inside - mean_outside

    # Pointing accuracy: is the max-sim token inside the box?
    max_pos = sim.argmax().item()
    max_row, max_col = max_pos // grid_size, max_pos % grid_size
    pointing_hit = bool(gt_grid_mask[max_row, max_col].item())

    # Top-4 pointing: are any of the top-4 tokens inside?
    topk_pos = sim.topk(4).indices.tolist()
    topk_hits = sum(
        1 for p in topk_pos
        if gt_grid_mask[p // grid_size, p % grid_size].item()
    )

    return {
        "mean_inside": mean_inside,
        "mean_outside": mean_outside,
        "contrast": contrast,
        "pointing_hit": 1.0 if pointing_hit else 0.0,
        "topk4_hits": topk_hits / 4.0,
    }


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    with open(args.benchmark) as f:
        bench = json.load(f)

    samples = bench["samples"]
    if args.split != "all":
        samples = [s for s in samples if s.get("split", "test") == args.split]
    if args.max_samples > 0:
        samples = samples[: args.max_samples]

    print(f"Layer probing: {len(samples)} samples on {device}")
    print(f"Loading InternVL from {args.internvl_path}...")

    from transformers import AutoTokenizer, AutoModel
    model = AutoModel.from_pretrained(
        args.internvl_path,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        use_flash_attn=False,
        low_cpu_mem_usage=False,
    ).to(device).eval()

    tokenizer = AutoTokenizer.from_pretrained(
        args.internvl_path, trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    num_layers = model.language_model.config.num_hidden_layers
    print(f"InternVL: {num_layers} LLM layers, "
          f"hidden_dim={model.language_model.config.hidden_size}")

    transform = build_transform()

    # Accumulate per-layer metrics
    layer_metrics = {i: [] for i in range(num_layers)}

    for idx, sample in enumerate(samples):
        image_pil = Image.open(sample["image_path"]).convert("RGB")
        img_w, img_h = image_pil.size
        pixel_values = transform(image_pil).unsqueeze(0).to(device)

        gt_boxes = [xywh_to_xyxy(t["bbox_xywh"]) for t in sample["positive_targets"]]
        # Union of all GT boxes for the grid mask
        union_mask = torch.zeros(16, 16, dtype=torch.bool)
        for box in gt_boxes:
            union_mask |= box_to_grid_mask(box, img_w, img_h)

        prompt = sample["prompt"]

        layer_feats, text_embed = probe_all_layers(
            model, tokenizer, pixel_values, prompt, device
        )

        for layer_idx in range(num_layers):
            result = score_layer(layer_feats[layer_idx], text_embed, union_mask)
            if result is not None:
                layer_metrics[layer_idx].append(result)

        if (idx + 1) % 25 == 0:
            # Print progress with a few layer highlights
            highlights = []
            for li in [0, 9, 14, 18, 22, num_layers - 1]:
                if li < num_layers and layer_metrics[li]:
                    n = len(layer_metrics[li])
                    c = sum(r["contrast"] for r in layer_metrics[li]) / n
                    p = sum(r["pointing_hit"] for r in layer_metrics[li]) / n
                    highlights.append(f"L{li}:c={c:.3f}/p={p:.3f}")
            print(f"  [{idx+1}/{len(samples)}] {' | '.join(highlights)}")

    # Aggregate
    summary = {}
    for layer_idx in range(num_layers):
        rows = layer_metrics[layer_idx]
        n = max(len(rows), 1)
        summary[layer_idx] = {
            "layer": layer_idx,
            "n_samples": len(rows),
            "mean_contrast": sum(r["contrast"] for r in rows) / n,
            "mean_pointing_hit": sum(r["pointing_hit"] for r in rows) / n,
            "mean_topk4_hits": sum(r["topk4_hits"] for r in rows) / n,
            "mean_inside_sim": sum(r["mean_inside"] for r in rows) / n,
            "mean_outside_sim": sum(r["mean_outside"] for r in rows) / n,
        }

    # Find best layers
    best_contrast = max(summary.values(), key=lambda x: x["mean_contrast"])
    best_pointing = max(summary.values(), key=lambda x: x["mean_pointing_hit"])

    payload = {
        "status": "ok",
        "n_samples": len(samples),
        "num_layers": num_layers,
        "best_contrast_layer": best_contrast["layer"],
        "best_pointing_layer": best_pointing["layer"],
        "per_layer": summary,
    }

    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)

    # Markdown summary
    md_path = os.path.splitext(args.output)[0] + ".md"
    with open(md_path, "w") as f:
        f.write("# InternVL Layer Probing Results\n\n")
        f.write(f"- Samples: {len(samples)}\n")
        f.write(f"- Layers: {num_layers}\n")
        f.write(f"- Best contrast layer: **{best_contrast['layer']}** "
                f"({best_contrast['mean_contrast']:.4f})\n")
        f.write(f"- Best pointing layer: **{best_pointing['layer']}** "
                f"({best_pointing['mean_pointing_hit']:.4f})\n\n")
        f.write("## All Layers\n\n")
        f.write("| layer | contrast | pointing_acc | topk4_acc | inside_sim | outside_sim |\n")
        f.write("|---:|---:|---:|---:|---:|---:|\n")
        for i in range(num_layers):
            s = summary[i]
            f.write(f"| {i} | {s['mean_contrast']:.4f} | "
                    f"{s['mean_pointing_hit']:.4f} | {s['mean_topk4_hits']:.4f} | "
                    f"{s['mean_inside_sim']:.4f} | {s['mean_outside_sim']:.4f} |\n")

    print(f"\n{'='*60}")
    print(f"Best contrast layer: {best_contrast['layer']} "
          f"(contrast={best_contrast['mean_contrast']:.4f})")
    print(f"Best pointing layer: {best_pointing['layer']} "
          f"(pointing_acc={best_pointing['mean_pointing_hit']:.4f})")
    print(f"{'='*60}")
    print(f"[saved] {args.output}")
    print(f"[saved] {md_path}")


if __name__ == "__main__":
    main()
