#!/usr/bin/env python3
"""
ThinkDet v2 — Layer Sensitivity Probe (Forward Pass Only)

Goal: find which InternVL LLM layers best capture reasoning & uncertainty
by testing how visual features respond to semantically different prompts.

For each image × each layer:
    Run multiple prompts through InternVL and extract 256 visual tokens.

Metrics:
    (A) Text Sensitivity     ||H(prompt_i) - H(prompt_j)||
                              Higher = layer actively uses language to reshape features.

    (B) Token Diversity       1 - mean(off-diagonal cosine similarity among 256 tokens)
                              Higher = tokens carry diverse spatial/semantic info, not collapsed.

    (C) Cross-Image Sim       mean cosine between same-token features across different images
                              Lower = layer encodes image-specific content, not generic.

Composite ranking:
    high sensitivity + high diversity + low cross-image similarity

Prompts test 4 reasoning axes:
    - Literal:     "chair"
    - Affordance:  "object to sit on"
    - Function:    "rideable part"
    - Attribute:   "red thing"
    (+ more per group for robustness)

No training. Pure inference. ~500 COCO val images.

Usage:
    # Single GPU:
    python probe_layer_sensitivity.py --device cuda:0 --max_images 500

    # Multi-GPU (8x):
    torchrun --nproc_per_node=8 probe_layer_sensitivity.py --max_images 500

Output:
    thinkdet/results/layer_sensitivity/sensitivity_results.json
    thinkdet/results/layer_sensitivity/sensitivity_summary.png
"""

import os
import sys
import json
import time
import argparse

import torch
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
from PIL import Image
from itertools import combinations

ROOT = '/home/iibrohimm/project/next_step'
sys.path.insert(0, ROOT)

import torchvision.transforms as TV_T
from torchvision.transforms.functional import InterpolationMode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# ─────────────────────────────────────────────────────────────────────
# Prompt sets — each group tests a different reasoning axis
# ─────────────────────────────────────────────────────────────────────
PROMPT_GROUPS = {
    'literal_vs_affordance': [
        'chair .',
        'object to sit on .',
        'something you rest on .',
    ],
    'literal_vs_function': [
        'bicycle .',
        'rideable part .',
        'vehicle with pedals .',
    ],
    'literal_vs_attribute': [
        'red thing .',
        'bright colored object .',
        'warm toned item .',
    ],
    'spatial_reasoning': [
        'person on the left .',
        'object in the center .',
        'thing in the background .',
    ],
    'abstract_vs_concrete': [
        'dog .',
        'pet animal .',
        'furry companion .',
    ],
    'scene_understanding': [
        'food on the table .',
        'something edible .',
        'meal being served .',
    ],
}

# Flatten all prompts for cross-image metric
ALL_PROMPTS = []
for group_prompts in PROMPT_GROUPS.values():
    ALL_PROMPTS.extend(group_prompts)
ALL_PROMPTS = sorted(set(ALL_PROMPTS))


# ─────────────────────────────────────────────────────────────────────
# Distributed helpers
# ─────────────────────────────────────────────────────────────────────
def get_dist_info():
    """Read distributed env vars set by torchrun, but do NOT init process group
    yet — that must happen AFTER model loading to avoid PyTorch's meta-tensor
    device routing which breaks InternViT's torch.linspace().item()."""
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    if world_size <= 1:
        return False, 0, 1, 0
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    rank = int(os.environ.get('RANK', '0'))
    return True, rank, world_size, local_rank


def init_process_group(local_rank):
    """Call AFTER model loading to avoid meta-tensor issues."""
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')


def cleanup_distributed(is_dist):
    if is_dist and dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def build_transform(input_size=448):
    return TV_T.Compose([
        TV_T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        TV_T.Resize((input_size, input_size),
                    interpolation=InterpolationMode.BICUBIC),
        TV_T.ToTensor(),
        TV_T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ─────────────────────────────────────────────────────────────────────
# Multi-Layer Feature Extractor
# ─────────────────────────────────────────────────────────────────────
class AllLayerExtractor:
    """Extract visual features from ALL LLM layers in one forward pass."""

    def __init__(self, model_path, device='cuda:0'):
        from transformers import AutoModel, AutoTokenizer

        self.device = device
        print(f"  Loading InternVL from {model_path} -> {device} ...")

        # Load on CPU first (no device_map to avoid meta-tensor init),
        # then move to target GPU.  torch.cuda.set_device() is called
        # here AFTER model construction to avoid PyTorch's __torch_function__
        # routing during InternViT init.
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            use_flash_attn=False,
            low_cpu_mem_usage=False,
        ).to(device).eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.num_image_token = self.model.num_image_token  # 256
        llm_cfg = self.model.language_model.config
        self.num_layers = llm_cfg.num_hidden_layers
        self.hidden_dim = llm_cfg.hidden_size
        self.transform = build_transform(448)

        print(f"  LLM layers: {self.num_layers}  |  hidden: {self.hidden_dim}  "
              f"|  visual tokens: {self.num_image_token}")

    @torch.no_grad()
    def extract_all_layers(self, pixel_values, text_query):
        """
        Single forward pass -> visual features from every LLM layer.

        Returns:
            dict[layer_idx] -> Tensor [256, hidden_dim] float32 CPU
        """
        device = self.device
        B = pixel_values.shape[0]

        vit_embeds = self.model.extract_feature(pixel_values)
        num_vis = vit_embeds.shape[1]

        text_inputs = self.tokenizer(
            [text_query],
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=256,
        ).to(device)

        text_embeds = self.model.language_model.get_input_embeddings()(
            text_inputs.input_ids
        )
        input_embeds = torch.cat([vit_embeds, text_embeds], dim=1)
        seq_len = input_embeds.shape[1]

        # bidirectional attention (not causal) — visual tokens attend to text
        vis_mask = torch.ones(B, num_vis, device=device, dtype=torch.long)
        attention_mask = torch.cat([vis_mask, text_inputs.attention_mask], dim=1)
        dtype = input_embeds.dtype
        pad_mask = attention_mask[:, None, None, :].to(dtype)
        attn_mask = (1.0 - pad_mask) * torch.finfo(dtype).min

        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

        llm = self.model.language_model.model
        position_embeddings = None
        if hasattr(llm, 'rotary_emb'):
            position_embeddings = llm.rotary_emb(input_embeds, position_ids)

        features = {}
        hidden_states = input_embeds

        for i in range(self.num_layers):
            layer_out = llm.layers[i](
                hidden_states,
                attention_mask=attn_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                use_cache=False,
            )
            hidden_states = (layer_out[0] if isinstance(layer_out, tuple)
                             else layer_out)
            features[i] = hidden_states[0, :num_vis, :].float().cpu()

        return features


# ─────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────
def text_sensitivity_l2(feats_a, feats_b):
    """Mean L2 distance between two feature sets [256, D]."""
    diff = feats_a - feats_b  # [256, D]
    return diff.norm(dim=-1).mean().item()


def text_sensitivity_cosine(feats_a, feats_b):
    """1 - cosine similarity (mean-pooled)."""
    a = F.normalize(feats_a.mean(dim=0, keepdim=True), dim=-1)
    b = F.normalize(feats_b.mean(dim=0, keepdim=True), dim=-1)
    return 1.0 - F.cosine_similarity(a, b).item()


def token_diversity(feat):
    """
    1 - mean(off-diagonal cosine sim among 256 tokens).
    Higher = tokens are more diverse (not collapsed).
    """
    normed = F.normalize(feat, dim=-1)          # [256, D]
    sim = normed @ normed.T                     # [256, 256]
    n = sim.shape[0]
    mask = ~torch.eye(n, dtype=torch.bool)
    mean_sim = sim[mask].mean().item()
    return 1.0 - mean_sim


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--max_images', type=int, default=500,
                        help='Number of COCO val images to use')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir',
                        default=os.path.join(ROOT, 'thinkdet', 'results',
                                             'layer_sensitivity'))
    args = parser.parse_args()

    # Read dist info from env (do NOT init NCCL yet — see get_dist_info docstring)
    is_dist, rank, world_size, local_rank = get_dist_info()
    is_main = (rank == 0)
    run_device = f'cuda:{local_rank}' if is_dist else args.device

    os.makedirs(args.output_dir, exist_ok=True)
    np.random.seed(args.seed)

    if is_main:
        print("=" * 70)
        print("  ThinkDet v2 — Layer Sensitivity Probe")
        print(f"  {len(PROMPT_GROUPS)} prompt groups, {len(ALL_PROMPTS)} unique prompts")
        print(f"  Images: {args.max_images} COCO val")
        if is_dist:
            print(f"  GPUs: {world_size}")
        print("=" * 70)

    # ── 1. Model (load BEFORE dist init to avoid meta-tensor issue) ──
    model_path = os.path.join(ROOT, 'InternVL3_5-1B')
    extractor = AllLayerExtractor(model_path, device=run_device)
    num_layers = extractor.num_layers

    # ── 2. Now init NCCL (safe after model loading) ──
    if is_dist:
        init_process_group(local_rank)

    # ── 3. Collect COCO val images ──
    coco_val_dir = os.path.join(ROOT, 'dataSets', 'coco', 'val2017')
    if not os.path.isdir(coco_val_dir):
        coco_val_dir = os.path.join(ROOT, 'dataSets', 'refer', 'data',
                                    'images', 'mscoco', 'val2017')

    all_images = sorted([
        os.path.join(coco_val_dir, f)
        for f in os.listdir(coco_val_dir)
        if f.endswith('.jpg')
    ])
    rng = np.random.default_rng(args.seed)
    selected = rng.choice(len(all_images), size=min(args.max_images, len(all_images)),
                          replace=False)
    selected.sort()
    all_selected = [all_images[i] for i in selected]

    # shard images across GPUs
    image_paths = all_selected[rank::world_size] if is_dist else all_selected
    if is_main:
        print(f"  Selected {len(all_selected)} images from {coco_val_dir}")
        if is_dist:
            print(f"  Per-GPU shard: ~{len(image_paths)} images")

    # ── 3. Accumulators ──
    # (A) text sensitivity: per layer, list of floats
    sensitivity_l2 = [[] for _ in range(num_layers)]
    sensitivity_cos = [[] for _ in range(num_layers)]

    # (B) token diversity: per layer, list of floats
    tok_div = [[] for _ in range(num_layers)]

    # (C) cross-image: per layer per prompt, collect mean-pooled vectors
    #     shape: cross_vecs[layer][prompt_idx] = list of [D] vectors
    cross_vecs = [[[] for _ in ALL_PROMPTS] for _ in range(num_layers)]

    total_forwards = 0
    skipped = 0
    t0 = time.time()

    print(f"\n  Running {len(image_paths)} images x {len(ALL_PROMPTS)} prompts "
          f"= {len(image_paths) * len(ALL_PROMPTS)} forwards ...\n")

    for img_idx, img_path in enumerate(image_paths):
        try:
            pil = Image.open(img_path).convert('RGB')
        except Exception:
            skipped += 1
            continue

        pixel_values = extractor.transform(pil).unsqueeze(0).to(extractor.device)

        # extract features for ALL prompts on this image
        # group by prompt index for later use
        prompt_feats = {}   # prompt_str -> dict[layer] -> [256, D]

        oom = False
        for prompt in ALL_PROMPTS:
            try:
                feats = extractor.extract_all_layers(pixel_values, prompt)
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    torch.cuda.empty_cache()
                    oom = True
                    break
                raise
            prompt_feats[prompt] = feats
            total_forwards += 1

        if oom:
            skipped += 1
            continue

        # ── (A) Text Sensitivity: pairwise within each prompt group ──
        for group_name, group_prompts in PROMPT_GROUPS.items():
            for p_a, p_b in combinations(group_prompts, 2):
                if p_a not in prompt_feats or p_b not in prompt_feats:
                    continue
                for li in range(num_layers):
                    f_a = prompt_feats[p_a][li]
                    f_b = prompt_feats[p_b][li]
                    sensitivity_l2[li].append(text_sensitivity_l2(f_a, f_b))
                    sensitivity_cos[li].append(text_sensitivity_cosine(f_a, f_b))

        # ── (B) Token Diversity: per prompt per layer ──
        for prompt, feats in prompt_feats.items():
            for li in range(num_layers):
                tok_div[li].append(token_diversity(feats[li]))

        # ── (C) Cross-Image: store mean-pooled vectors ──
        for p_idx, prompt in enumerate(ALL_PROMPTS):
            if prompt not in prompt_feats:
                continue
            for li in range(num_layers):
                pooled = prompt_feats[prompt][li].mean(dim=0)  # [D]
                pooled = F.normalize(pooled.unsqueeze(0), dim=-1).squeeze(0)
                cross_vecs[li][p_idx].append(pooled)

        # progress
        if (img_idx + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = total_forwards / max(elapsed, 1e-6)
            remaining = (len(image_paths) - img_idx - 1) * len(ALL_PROMPTS)
            eta = remaining / max(rate, 1e-6)
            print(f"  [{img_idx+1:>4}/{len(image_paths)}]  "
                  f"{total_forwards} forwards  "
                  f"({rate:.1f} fwd/s  ETA {eta/60:.1f} min)")

    elapsed = time.time() - t0
    print(f"\n  [Rank {rank}] Done: {total_forwards} forwards in {elapsed/60:.1f} min  "
          f"(skipped {skipped} images)")

    # ── Distributed: gather partial results from all ranks ──
    if is_dist:
        import pickle as _pkl

        os.makedirs(args.output_dir, exist_ok=True)
        partial_path = os.path.join(args.output_dir, f'_partial_rank{rank}.pkl')
        with open(partial_path, 'wb') as f:
            _pkl.dump({
                'sensitivity_l2': sensitivity_l2,
                'sensitivity_cos': sensitivity_cos,
                'tok_div': tok_div,
                'cross_vecs': cross_vecs,
                'total_forwards': total_forwards,
                'skipped': skipped,
            }, f)

        dist.barrier()

        if is_main:
            print(f"\n  Merging results from {world_size} GPUs ...")
            for r in range(1, world_size):
                rpath = os.path.join(args.output_dir, f'_partial_rank{r}.pkl')
                with open(rpath, 'rb') as f:
                    rd = _pkl.load(f)
                for li in range(num_layers):
                    sensitivity_l2[li].extend(rd['sensitivity_l2'][li])
                    sensitivity_cos[li].extend(rd['sensitivity_cos'][li])
                    tok_div[li].extend(rd['tok_div'][li])
                    for p_idx in range(len(ALL_PROMPTS)):
                        cross_vecs[li][p_idx].extend(rd['cross_vecs'][li][p_idx])
                total_forwards += rd['total_forwards']
                skipped += rd['skipped']
                del rd
            print(f"  Merged: {total_forwards} total forwards, "
                  f"{skipped} total skipped")

            # Clean up partial files
            for r in range(world_size):
                p = os.path.join(args.output_dir, f'_partial_rank{r}.pkl')
                if os.path.exists(p):
                    os.remove(p)
        else:
            # Non-main ranks exit after saving partials
            cleanup_distributed(is_dist)
            return

    total_images = len(all_selected) - skipped

    # ── 4. Compute cross-image similarity per layer ──
    cross_img_sim = [0.0] * num_layers
    for li in range(num_layers):
        prompt_sims = []
        for p_idx in range(len(ALL_PROMPTS)):
            vecs = cross_vecs[li][p_idx]
            if len(vecs) < 2:
                continue
            mat = torch.stack(vecs, dim=0)      # [N_images, D]
            sim_matrix = mat @ mat.T            # [N, N]
            n = sim_matrix.shape[0]
            mask = ~torch.eye(n, dtype=torch.bool)
            mean_sim = sim_matrix[mask].mean().item()
            prompt_sims.append(mean_sim)
        if prompt_sims:
            cross_img_sim[li] = float(np.mean(prompt_sims))

    # free memory
    del cross_vecs
    torch.cuda.empty_cache()

    # ── 5. Aggregate results per layer ──
    results = []
    for li in range(num_layers):
        ts_l2 = float(np.mean(sensitivity_l2[li])) if sensitivity_l2[li] else 0.0
        ts_cos = float(np.mean(sensitivity_cos[li])) if sensitivity_cos[li] else 0.0
        td = float(np.mean(tok_div[li])) if tok_div[li] else 0.0
        ci = cross_img_sim[li]

        results.append({
            'layer': li,
            'text_sensitivity_l2': round(ts_l2, 6),
            'text_sensitivity_cosine': round(ts_cos, 6),
            'token_diversity': round(td, 6),
            'cross_image_similarity': round(ci, 6),
            'n_sensitivity_samples': len(sensitivity_l2[li]),
            'n_diversity_samples': len(tok_div[li]),
        })

    # ── 6. Composite ranking ──
    # normalize each metric to [0, 1]
    def norm01(vals, invert=False):
        mn, mx = min(vals), max(vals)
        rng_v = mx - mn if mx > mn else 1e-8
        normed = [(v - mn) / rng_v for v in vals]
        if invert:
            normed = [1.0 - v for v in normed]
        return normed

    ts_vals = [r['text_sensitivity_l2'] for r in results]
    td_vals = [r['token_diversity'] for r in results]
    ci_vals = [r['cross_image_similarity'] for r in results]

    ts_n = norm01(ts_vals, invert=False)    # higher is better
    td_n = norm01(td_vals, invert=False)    # higher is better
    ci_n = norm01(ci_vals, invert=True)     # LOWER is better -> invert

    for i, r in enumerate(results):
        r['composite_score'] = round(
            0.40 * ts_n[i] + 0.35 * td_n[i] + 0.25 * ci_n[i], 6)
        r['composite_components'] = {
            'sensitivity_norm': round(ts_n[i], 4),
            'diversity_norm': round(td_n[i], 4),
            'cross_image_inv_norm': round(ci_n[i], 4),
        }

    # ── 7. Find top layers ──
    sorted_layers = sorted(results, key=lambda r: r['composite_score'], reverse=True)
    top1 = sorted_layers[0]
    top2 = sorted_layers[1] if len(sorted_layers) > 1 else top1

    # ── 8. Print report ──
    print(f"\n{'=' * 78}")
    print(f"  LAYER SENSITIVITY RESULTS  "
          f"({total_forwards} forwards, {total_images} images, "
          f"{len(ALL_PROMPTS)} prompts)")
    print(f"{'=' * 78}")
    print(f"  {'Layer':>5}  {'Sens(L2)':>10}  {'Sens(Cos)':>10}  "
          f"{'TokDiv':>8}  {'CrossSim':>9}  {'Composite':>10}")
    print(f"  {'─'*5}  {'─'*10}  {'─'*10}  {'─'*8}  {'─'*9}  {'─'*10}")

    for r in results:
        tag = ''
        if r['layer'] == top1['layer']:
            tag = '  <-- #1'
        elif r['layer'] == top2['layer']:
            tag = '  <-- #2'
        print(f"  {r['layer']:>5}  {r['text_sensitivity_l2']:>10.4f}  "
              f"{r['text_sensitivity_cosine']:>10.6f}  "
              f"{r['token_diversity']:>8.4f}  "
              f"{r['cross_image_similarity']:>9.4f}  "
              f"{r['composite_score']:>10.4f}{tag}")

    print(f"\n  TOP 2 LAYERS:")
    print(f"    #1  Layer {top1['layer']}  "
          f"(composite={top1['composite_score']:.4f}  "
          f"sens_l2={top1['text_sensitivity_l2']:.4f}  "
          f"tok_div={top1['token_diversity']:.4f}  "
          f"cross_sim={top1['cross_image_similarity']:.4f})")
    print(f"    #2  Layer {top2['layer']}  "
          f"(composite={top2['composite_score']:.4f}  "
          f"sens_l2={top2['text_sensitivity_l2']:.4f}  "
          f"tok_div={top2['token_diversity']:.4f}  "
          f"cross_sim={top2['cross_image_similarity']:.4f})")
    print(f"\n  Composite = 40% sensitivity + 35% token_diversity "
          f"+ 25% (1 - cross_image_sim)")

    # ── 9. Save ──
    result_path = os.path.join(args.output_dir, 'sensitivity_results.json')
    with open(result_path, 'w') as f:
        json.dump({
            'config': {
                'num_images': total_images,
                'num_prompts': len(ALL_PROMPTS),
                'num_forwards': total_forwards,
                'prompt_groups': {k: v for k, v in PROMPT_GROUPS.items()},
                'all_prompts': ALL_PROMPTS,
                'max_images': args.max_images,
                'seed': args.seed,
                'time_min': round(elapsed / 60, 1),
                'composite_weights': '40% sensitivity_l2 + 35% token_diversity '
                                     '+ 25% (1 - cross_image_sim)',
            },
            'top_layers': [
                {'rank': 1, 'layer': top1['layer'],
                 'composite': top1['composite_score']},
                {'rank': 2, 'layer': top2['layer'],
                 'composite': top2['composite_score']},
            ],
            'layers': results,
        }, f, indent=2)
    print(f"\n  Results saved: {result_path}")

    # ── 10. Plot ──
    try:
        generate_plot(results, args.output_dir, top1['layer'], top2['layer'])
    except Exception as e:
        print(f"  Plot failed: {e}")

    print(f"{'=' * 78}")
    cleanup_distributed(is_dist)


# ─────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────
def generate_plot(results, output_dir, top1_layer, top2_layer):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    layers = [r['layer'] for r in results]
    sens_l2 = [r['text_sensitivity_l2'] for r in results]
    tok_div = [r['token_diversity'] for r in results]
    cross_sim = [r['cross_image_similarity'] for r in results]
    composite = [r['composite_score'] for r in results]

    fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=True)
    fig.patch.set_facecolor('#0d1117')
    fig.suptitle(
        'ThinkDet — Layer Sensitivity Probe\n'
        '(COCO val: literal vs affordance vs attribute vs spatial prompts)',
        fontsize=14, color='#f0f6fc', fontweight='bold', y=0.995)

    plots = [
        (axes[0], sens_l2,
         'Text Sensitivity (L2)  —  higher = layer uses language more',
         '#58a6ff', True),
        (axes[1], tok_div,
         'Token Diversity  —  higher = less collapsed spatial info',
         '#3fb950', True),
        (axes[2], cross_sim,
         'Cross-Image Similarity  —  lower = more image-specific',
         '#f85149', False),
        (axes[3], composite,
         'Composite  —  40% sens + 35% div + 25% (1-cross)',
         '#d29922', True),
    ]

    for ax, vals, title, color, higher_better in plots:
        ax.set_facecolor('#0d1117')

        best_idx = int(np.argmax(vals)) if higher_better else int(np.argmin(vals))

        bar_colors = []
        for i in range(len(vals)):
            if i == best_idx:
                bar_colors.append('#f0f6fc')
            else:
                bar_colors.append(color)

        ax.bar(range(len(vals)), vals, color=bar_colors,
               edgecolor='#30363d', linewidth=0.3, width=0.75, alpha=0.85)
        ax.plot(range(len(vals)), vals, color=color, linewidth=1.5,
                alpha=0.7, marker='o', markersize=3)

        # mark top 1 and top 2
        for tl, marker, label in [(top1_layer, '^', '#1'),
                                   (top2_layer, 's', '#2')]:
            if tl < len(vals):
                ax.plot(tl, vals[tl], marker=marker, markersize=10,
                        color='#f0f6fc', markeredgecolor='#f0f6fc',
                        markeredgewidth=1.5, zorder=5)

        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels([str(l) for l in layers], fontsize=7,
                           color='#8b949e')
        ax.set_title(title, fontsize=10, color=color, fontweight='bold',
                     pad=8)
        ax.tick_params(colors='#8b949e', labelsize=7)
        for spine in ax.spines.values():
            spine.set_color('#30363d')
        ax.grid(axis='y', alpha=0.1, color='#484f58')

        y_span = max(vals) - min(vals) if max(vals) != min(vals) else 0.01
        best_val = vals[best_idx]
        arrow_label = f'Best: L{layers[best_idx]} ({best_val:.4f})'

        ax.annotate(
            arrow_label,
            xy=(best_idx, best_val),
            xytext=(min(best_idx + 4, len(vals) - 1),
                    best_val + y_span * 0.12),
            fontsize=9, color='#f0f6fc', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#f0f6fc', lw=1.2))

    axes[-1].set_xlabel('LLM Layer', fontsize=10, color='#8b949e')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = os.path.join(output_dir, 'sensitivity_summary.png')
    plt.savefig(out_path, dpi=150, facecolor='#0d1117', bbox_inches='tight')
    plt.close()
    print(f"  Plot: {out_path}")


if __name__ == '__main__':
    main()
