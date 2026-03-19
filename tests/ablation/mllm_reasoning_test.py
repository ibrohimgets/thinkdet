#!/usr/bin/env python3
"""
ThinkDet v2 — MLLM Reasoning Test

Ask InternVL a complex reasoning question about a specific image and
compare layer-level feature responses between the complex query and
simple literal queries.

Usage:
    python mllm_reasoning_test.py
"""

import os
import sys
import json
import time
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

ROOT = '/home/iibrohimm/project/next_step'
sys.path.insert(0, ROOT)

import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

IMAGE_PATH = f'{ROOT}/dataSets/flickr30k_entities/images/134206.jpg'
OUTPUT_DIR = f'{ROOT}/thinkdet/results/mllm_reasoning_test'

# The complex reasoning query
COMPLEX_QUERY = (
    "do we have a player in the field holding a black thing in his hand "
    "but not the one with a number 7 in his back t shirt"
)

# Simple/literal queries for comparison
SIMPLE_QUERIES = [
    "baseball player .",
    "person holding a glove .",
    "player with number 7 .",
    "black glove .",
    "player in the field .",
]


def build_transform(input_size=448):
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size),
                 interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = 'cuda:0'

    print("=" * 70)
    print("  MLLM Reasoning Test — InternVL3.5-1B")
    print("=" * 70)
    print(f"  Image: {IMAGE_PATH}")
    print(f"  Query: \"{COMPLEX_QUERY}\"")
    print()

    # ── 1. Load InternVL ──
    from transformers import AutoModel, AutoTokenizer

    model_path = f'{ROOT}/InternVL3_5-1B'
    print("[1/4] Loading InternVL ...")
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        use_flash_attn=False,
        low_cpu_mem_usage=False,
    ).to(device).eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    num_layers = model.language_model.config.num_hidden_layers
    hidden_dim = model.language_model.config.hidden_size
    num_vis = model.num_image_token
    print(f"  Layers: {num_layers}, hidden: {hidden_dim}, vis tokens: {num_vis}")

    # ── 2. Chat: ask the question ──
    print(f"\n[2/4] Asking InternVL the question (chat mode) ...")
    pil_img = Image.open(IMAGE_PATH).convert('RGB')
    transform = build_transform(448)
    pixel_values = transform(pil_img).unsqueeze(0).to(device)

    # Build chat prompt
    chat_query = f"<image>\n{COMPLEX_QUERY}"
    generation_config = dict(max_new_tokens=512, do_sample=False)

    try:
        response = model.chat(tokenizer, pixel_values, chat_query, generation_config)
        print(f"\n  QUESTION: {COMPLEX_QUERY}")
        print(f"  ANSWER:   {response}")
    except Exception as e:
        print(f"  Chat failed: {e}")
        response = f"[Chat error: {e}]"

    # ── 3. Layer-by-layer feature extraction ──
    print(f"\n[3/4] Extracting features from all {num_layers} layers ...")
    print(f"  Queries: 1 complex + {len(SIMPLE_QUERIES)} simple")

    all_queries = [COMPLEX_QUERY] + SIMPLE_QUERIES
    query_labels = ['COMPLEX'] + [f'simple_{i}' for i in range(len(SIMPLE_QUERIES))]

    @torch.no_grad()
    def extract_all_layers(pixel_values, text_query):
        """Extract visual features from every LLM layer for a given query."""
        vit_embeds = model.extract_feature(pixel_values)
        nv = vit_embeds.shape[1]

        text_inputs = tokenizer(
            [text_query], return_tensors='pt',
            padding='max_length', truncation=True, max_length=256,
        ).to(device)

        text_embeds = model.language_model.get_input_embeddings()(
            text_inputs.input_ids
        )
        input_embeds = torch.cat([vit_embeds, text_embeds], dim=1)
        B, seq_len, _ = input_embeds.shape

        vis_mask = torch.ones(B, nv, device=device, dtype=torch.long)
        attention_mask = torch.cat([vis_mask, text_inputs.attention_mask], dim=1)
        dtype = input_embeds.dtype
        pad_mask = attention_mask[:, None, None, :].to(dtype)
        attn_mask = (1.0 - pad_mask) * torch.finfo(dtype).min

        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

        llm = model.language_model.model
        position_embeddings = None
        if hasattr(llm, 'rotary_emb'):
            position_embeddings = llm.rotary_emb(input_embeds, position_ids)

        features = {}
        hidden_states = input_embeds
        for i in range(num_layers):
            layer_out = llm.layers[i](
                hidden_states,
                attention_mask=attn_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                use_cache=False,
            )
            hidden_states = layer_out[0] if isinstance(layer_out, tuple) else layer_out
            features[i] = hidden_states[0, :nv, :].float().cpu()

        return features

    # Extract features for all queries
    query_features = {}
    for qi, (query, label) in enumerate(zip(all_queries, query_labels)):
        print(f"  [{qi+1}/{len(all_queries)}] {label}: \"{query[:60]}...\"")
        query_features[label] = extract_all_layers(pixel_values, query)

    # ── 4. Compute reasoning metrics per layer ──
    print(f"\n[4/4] Computing reasoning metrics ...")

    results = {
        'image': IMAGE_PATH,
        'complex_query': COMPLEX_QUERY,
        'simple_queries': SIMPLE_QUERIES,
        'chat_response': response,
        'num_layers': num_layers,
        'layers': [],
    }

    complex_feats = query_features['COMPLEX']

    print(f"\n{'Layer':>5}  {'Complex-Simple L2':>18}  {'Complex-Simple Cos':>18}  "
          f"{'TokenDiv(Complex)':>17}  {'TokenDiv(Simple)':>16}  {'Norm(Complex)':>13}")
    print("-" * 95)

    for li in range(num_layers):
        fc = complex_feats[li]  # [256, D]

        # Mean L2 distance: complex vs each simple query
        l2_dists = []
        cos_dists = []
        simple_divs = []
        for si, slabel in enumerate(query_labels[1:]):
            fs = query_features[slabel][li]
            # L2
            l2 = (fc - fs).norm(dim=-1).mean().item()
            l2_dists.append(l2)
            # Cosine
            fc_norm = F.normalize(fc, dim=-1)
            fs_norm = F.normalize(fs, dim=-1)
            cos = F.cosine_similarity(fc_norm, fs_norm, dim=-1).mean().item()
            cos_dists.append(1.0 - cos)
            # Simple token diversity
            sim_s = fs_norm @ fs_norm.T
            n = sim_s.shape[0]
            mask = ~torch.eye(n, dtype=torch.bool)
            simple_divs.append(1.0 - sim_s[mask].mean().item())

        # Complex token diversity
        fc_norm = F.normalize(fc, dim=-1)
        sim_c = fc_norm @ fc_norm.T
        n = sim_c.shape[0]
        mask = ~torch.eye(n, dtype=torch.bool)
        complex_div = 1.0 - sim_c[mask].mean().item()

        mean_l2 = float(np.mean(l2_dists))
        mean_cos = float(np.mean(cos_dists))
        mean_simple_div = float(np.mean(simple_divs))
        feat_norm = fc.norm(dim=-1).mean().item()

        # Per-simple-query breakdown
        per_simple = {}
        for si, slabel in enumerate(query_labels[1:]):
            per_simple[SIMPLE_QUERIES[si]] = {
                'l2_dist': round(l2_dists[si], 6),
                'cos_dist': round(cos_dists[si], 6),
            }

        layer_result = {
            'layer': li,
            'mean_l2_complex_vs_simple': round(mean_l2, 6),
            'mean_cos_complex_vs_simple': round(mean_cos, 6),
            'token_diversity_complex': round(complex_div, 6),
            'token_diversity_simple_avg': round(mean_simple_div, 6),
            'diversity_delta': round(complex_div - mean_simple_div, 6),
            'feature_norm': round(feat_norm, 4),
            'per_simple_query': per_simple,
        }
        results['layers'].append(layer_result)

        print(f"{li:>5}  {mean_l2:>18.4f}  {mean_cos:>18.6f}  "
              f"{complex_div:>17.4f}  {mean_simple_div:>16.4f}  {feat_norm:>13.2f}")

    # ── Spatial heatmap: which of the 16x16 tokens activate most differently ──
    print(f"\n  Computing spatial activation maps for layers 7-12 ...")

    spatial_maps = {}
    for li in [7, 8, 9, 10, 11, 12]:
        fc = complex_feats[li]  # [256, D]
        # Average simple features
        simple_avg = torch.stack(
            [query_features[sl][li] for sl in query_labels[1:]], dim=0
        ).mean(dim=0)  # [256, D]

        # Per-token L2 difference
        diff = (fc - simple_avg).norm(dim=-1)  # [256]
        # Reshape to 16x16 spatial grid
        heatmap = diff.reshape(16, 16).numpy()
        spatial_maps[li] = heatmap.tolist()

    results['spatial_activation_maps'] = spatial_maps

    # ── Save ──
    result_path = os.path.join(OUTPUT_DIR, 'reasoning_test_134206.json')
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved: {result_path}")

    # ── Plot ──
    try:
        generate_plots(results, OUTPUT_DIR)
    except Exception as e:
        print(f"  Plot failed: {e}")

    print(f"\n{'=' * 70}")
    print(f"  SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Chat answer: {response}")
    print(f"\n  Top 3 layers by cosine distance (complex vs simple):")
    sorted_layers = sorted(results['layers'],
                           key=lambda r: r['mean_cos_complex_vs_simple'],
                           reverse=True)
    for i, r in enumerate(sorted_layers[:3]):
        print(f"    #{i+1} Layer {r['layer']}: cos_dist={r['mean_cos_complex_vs_simple']:.6f}  "
              f"diversity_delta={r['diversity_delta']:.4f}")

    print(f"\n  Top 3 layers by diversity delta (complex richer than simple):")
    sorted_div = sorted(results['layers'],
                        key=lambda r: r['diversity_delta'],
                        reverse=True)
    for i, r in enumerate(sorted_div[:3]):
        print(f"    #{i+1} Layer {r['layer']}: delta={r['diversity_delta']:.4f}  "
              f"complex_div={r['token_diversity_complex']:.4f}")
    print(f"{'=' * 70}")


def generate_plots(results, output_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    layers_data = results['layers']
    num_layers = len(layers_data)
    xs = list(range(num_layers))

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    fig.patch.set_facecolor('#0d1117')
    fig.suptitle(
        'ThinkDet — MLLM Reasoning Test\n'
        f'Query: "{results["complex_query"][:70]}..."',
        fontsize=12, color='#f0f6fc', fontweight='bold', y=0.995,
    )

    # 1. Cosine distance (complex vs simple)
    cos_vals = [r['mean_cos_complex_vs_simple'] for r in layers_data]
    ax = axes[0]
    ax.set_facecolor('#0d1117')
    ax.bar(xs, cos_vals, color='#58a6ff', edgecolor='#30363d', width=0.75, alpha=0.85)
    ax.plot(xs, cos_vals, color='#58a6ff', linewidth=1.5, marker='o', markersize=3)
    best_i = int(np.argmax(cos_vals))
    ax.annotate(f'Best: L{best_i} ({cos_vals[best_i]:.5f})',
                xy=(best_i, cos_vals[best_i]),
                xytext=(min(best_i+4, num_layers-1), cos_vals[best_i]*1.05),
                fontsize=9, color='#f0f6fc', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#f0f6fc', lw=1.2))
    ax.set_title('Cosine Distance: Complex Query vs Simple Queries (higher = more reasoning)',
                 fontsize=10, color='#58a6ff', fontweight='bold', pad=8)
    ax.tick_params(colors='#8b949e', labelsize=7)
    for spine in ax.spines.values(): spine.set_color('#30363d')
    ax.grid(axis='y', alpha=0.1, color='#484f58')

    # 2. Diversity delta
    div_vals = [r['diversity_delta'] for r in layers_data]
    ax = axes[1]
    ax.set_facecolor('#0d1117')
    colors = ['#3fb950' if v > 0 else '#f85149' for v in div_vals]
    ax.bar(xs, div_vals, color=colors, edgecolor='#30363d', width=0.75, alpha=0.85)
    ax.axhline(y=0, color='#484f58', linewidth=0.5)
    ax.set_title('Diversity Delta: Complex - Simple (positive = complex query adds spatial info)',
                 fontsize=10, color='#3fb950', fontweight='bold', pad=8)
    ax.tick_params(colors='#8b949e', labelsize=7)
    for spine in ax.spines.values(): spine.set_color('#30363d')
    ax.grid(axis='y', alpha=0.1, color='#484f58')

    # 3. Spatial heatmaps for layers 7-12
    spatial = results.get('spatial_activation_maps', {})
    ax = axes[2]
    ax.set_facecolor('#0d1117')
    if spatial:
        # Combine heatmaps side by side
        hm_layers = sorted(int(k) for k in spatial.keys())
        def _get_spatial_map(layer_idx):
            if layer_idx in spatial:
                return spatial[layer_idx]
            return spatial[str(layer_idx)]
        combined = np.concatenate(
            [np.array(_get_spatial_map(li)) for li in hm_layers], axis=1
        )
        im = ax.imshow(combined, cmap='inferno', aspect='auto',
                       interpolation='nearest')
        # Labels
        for idx, li in enumerate(hm_layers):
            ax.text(idx * 16 + 8, -1, f'L{li}', ha='center', va='bottom',
                    fontsize=9, color='#f0f6fc', fontweight='bold')
        ax.set_title('Spatial Activation Map: Where complex query differs from simple (16x16 tokens)',
                     fontsize=10, color='#d29922', fontweight='bold', pad=8)
        ax.set_yticks([])
        ax.set_xticks([])
    else:
        ax.text(0.5, 0.5, 'No spatial maps', ha='center', va='center',
                transform=ax.transAxes, color='#8b949e')

    axes[-1].set_xlabel('LLM Layer', fontsize=10, color='#8b949e')
    axes[-1].set_xticks(xs)
    axes[-1].set_xticklabels([str(i) for i in xs], fontsize=7, color='#8b949e')

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out_path = os.path.join(output_dir, 'reasoning_test_134206.png')
    plt.savefig(out_path, dpi=150, facecolor='#0d1117', bbox_inches='tight')
    plt.close()
    print(f"  Plot: {out_path}")


if __name__ == '__main__':
    main()
