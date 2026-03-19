#!/usr/bin/env python3
"""
ThinkDet v2 — Rigorous Reasoning Layer Probe (RefCOCO / RefCOCO+ / RefCOCOg / Flickr30k)

Goal:
Find which InternVL LLM layer best supports reasoning-oriented REC features.

This upgraded probe removes key confounds in the earlier quick test:
1) Uses ALL sentences per ref (or deterministic subset), not only sentence[0]
2) Separates paraphrase consistency from inter-ref discrimination
3) Computes image-level cross-image spread (not ref-weighted variance)
4) Adds an uncertainty proxy from positive-vs-negative similarity margins
5) Reports 95% bootstrap confidence intervals

Core metrics per layer:
- Expression Separation:   distance between different refs in the same image
- Paraphrase Stability:    similarity for different sentences of the SAME ref
- Disambiguation Margin:   separation - paraphrase_distance (higher is better)
- Token Diversity:         diversity across the 256 visual tokens
- Cross-Image Spread:      distance of image centroids to global centroid
- Uncertainty Rate:        fraction of refs with non-positive pos-neg margin

Usage:
    python probe_reasoning_layers.py --dataset refcoco --ref_split val --device cuda:0
    python probe_reasoning_layers.py --dataset refcocog --ref_split val --device cuda:0
    python probe_reasoning_layers.py --dataset flickr --flickr_split test --device cuda:0

Outputs:
    thinkdet/results/layer_probe/layer_probe_results.json
    thinkdet/results/layer_probe/layer_probe_summary.png
"""

import os
import sys
import json
import time
import argparse
import pickle
import zlib
from collections import defaultdict

import torch
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
from PIL import Image

ROOT = '/home/iibrohimm/project/next_step'
sys.path.insert(0, ROOT)

import torchvision.transforms as TV_T
from torchvision.transforms.functional import InterpolationMode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size=448):
    return TV_T.Compose([
        TV_T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        TV_T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        TV_T.ToTensor(),
        TV_T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def normalize_query(text):
    return text.strip().lower() + ' .'


def maybe_trim_sentences(expressions, max_sentences, seed):
    if max_sentences <= 0 or len(expressions) <= max_sentences:
        return expressions
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(expressions), size=max_sentences, replace=False))
    return [expressions[i] for i in idx]


def offdiag_mean(mat):
    n = mat.shape[0]
    if n <= 1:
        return None
    mask = ~torch.eye(n, dtype=torch.bool)
    return mat[mask].mean().item()


def token_diversity(feat):
    """
    1 - mean(off-diagonal cosine similarity) over visual tokens.
    Higher = token set is less collapsed.
    """
    normed = F.normalize(feat, dim=-1)
    sim = normed @ normed.T
    m = offdiag_mean(sim)
    if m is None:
        return 0.0
    return 1.0 - m


def bootstrap_mean_ci(values, num_boot=1000, seed=0):
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0, 0.0, 0.0

    mean = float(arr.mean())
    if arr.size == 1 or num_boot <= 1:
        return mean, mean, mean

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(num_boot, arr.size))
    boot_means = arr[idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return mean, float(lo), float(hi)


def summarize(values, num_boot, seed):
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            'mean': 0.0,
            'ci95': [0.0, 0.0],
            'std': 0.0,
            'n': 0,
        }

    mean, lo, hi = bootstrap_mean_ci(arr, num_boot=num_boot, seed=seed)
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    return {
        'mean': mean,
        'ci95': [lo, hi],
        'std': std,
        'n': int(arr.size),
    }


def rank01(values, higher_better=True):
    arr = np.asarray(values, dtype=np.float64)
    out = np.zeros_like(arr, dtype=np.float64)

    if not higher_better:
        arr = -arr

    finite = np.isfinite(arr)
    if finite.sum() == 0:
        return out.tolist()
    if finite.sum() == 1:
        out[finite] = 1.0
        return out.tolist()

    finite_vals = arr[finite]
    order = np.argsort(finite_vals, kind='mergesort')
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(finite_vals.size, dtype=np.float64)
    denom = max(finite_vals.size - 1, 1)

    out_vals = ranks / denom
    out[finite] = out_vals
    return out.tolist()


def ci_overlap(ci_a, ci_b):
    lo = max(ci_a[0], ci_b[0])
    hi = min(ci_a[1], ci_b[1])
    return lo <= hi


def setup_distributed():
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    if world_size <= 1:
        return False, 0, 1, 0

    if not torch.cuda.is_available():
        raise RuntimeError('Distributed mode requires CUDA')

    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    return True, rank, world_size, local_rank


def cleanup_distributed(is_distributed):
    if is_distributed and dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def merge_layer_lists(dst, src):
    for i in range(len(dst)):
        dst[i].extend(src[i])


# ─────────────────────────────────────────────────────────────────────
# Multi-Layer Feature Extractor
# ─────────────────────────────────────────────────────────────────────
class AllLayerExtractor:
    """Extract visual features from all LLM layers in one forward pass."""

    def __init__(self, model_path, device='cuda:0', dtype='float32'):
        from transformers import AutoModel, AutoTokenizer

        if dtype == 'bfloat16':
            torch_dtype = torch.bfloat16
        elif dtype == 'float16':
            torch_dtype = torch.float16
        elif dtype == 'float32':
            torch_dtype = torch.float32
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")

        self.device = device
        self.dtype_name = dtype
        self.input_dtype = torch_dtype

        print(f"  Loading InternVL from {model_path} ...")
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            use_flash_attn=False,
            low_cpu_mem_usage=False,
        ).to(device).eval()

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.num_image_token = self.model.num_image_token
        llm_cfg = self.model.language_model.config
        self.num_layers = llm_cfg.num_hidden_layers
        self.hidden_dim = llm_cfg.hidden_size
        self.transform = build_transform(448)

        print(
            f"  LLM layers: {self.num_layers}  |  hidden: {self.hidden_dim}  "
            f"|  visual tokens: {self.num_image_token}  |  dtype: {self.dtype_name}"
        )

    @torch.no_grad()
    def extract_all_layers(self, pixel_values, text_query):
        """
        Single forward pass -> visual features from every LLM layer.

        Args:
            pixel_values: [1, 3, 448, 448]
            text_query:   str

        Returns:
            dict[layer_idx] -> Tensor [256, hidden_dim] on CPU float32
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

        text_embeds = self.model.language_model.get_input_embeddings()(text_inputs.input_ids)
        input_embeds = torch.cat([vit_embeds, text_embeds], dim=1)
        seq_len = input_embeds.shape[1]

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
            hidden_states = layer_out[0] if isinstance(layer_out, tuple) else layer_out
            features[i] = hidden_states[0, :num_vis, :].float().cpu()

        return features


# ─────────────────────────────────────────────────────────────────────
# Refer-Style Dataset Loader
# ─────────────────────────────────────────────────────────────────────
DEFAULT_SPLIT_BY = {
    'refcoco': 'unc',
    'refcoco+': 'unc',
    'refcocog': 'umd',
}


def _filter_refer_refs(all_refs, split):
    if split in ['testA', 'testB']:
        return [r for r in all_refs if split[-1] in r['split']]
    if split == 'test':
        return [r for r in all_refs if 'test' in r['split']]
    return [r for r in all_refs if r['split'] == split]


def load_refer_split(data_root, dataset_name='refcocog', split_by=None, split='val', verbose=True):
    """Load Refer-style split, grouped by image_id, keeping all valid sentences."""
    split_by = split_by or DEFAULT_SPLIT_BY[dataset_name]
    ref_file = os.path.join(data_root, dataset_name, f'refs({split_by}).p')
    inst_file = os.path.join(data_root, dataset_name, 'instances.json')

    with open(ref_file, 'rb') as f:
        all_refs = pickle.load(f, encoding='latin1')
    with open(inst_file, 'r') as f:
        instances = json.load(f)

    ann_map = {a['id']: a for a in instances['annotations']}

    split_refs = _filter_refer_refs(all_refs, split)

    image_groups = defaultdict(list)
    total_sentences = 0
    multi_sent_refs = 0

    for ref in split_refs:
        ann = ann_map.get(ref['ann_id'])
        if ann is None or ann.get('area', 1) <= 0:
            continue

        expressions = []
        seen = set()
        for sent in ref.get('sentences', []):
            raw = sent.get('sent', '').strip()
            if not raw:
                continue
            key = raw.lower()
            if key in seen:
                continue
            seen.add(key)
            expressions.append(raw)

        if not expressions:
            continue

        if len(expressions) >= 2:
            multi_sent_refs += 1
        total_sentences += len(expressions)

        image_groups[ref['image_id']].append({
            'ref_id': ref['ref_id'],
            'bbox': ann['bbox'],
            'expressions': expressions,
        })

    total_refs = sum(len(g) for g in image_groups.values())
    multi_ref_images = sum(1 for g in image_groups.values() if len(g) >= 2)

    if verbose:
        print(f"  {dataset_name} {split} ({split_by}): {total_refs} refs across {len(image_groups)} images")
        print(f"  Sentences kept: {total_sentences}  "
              f"(avg {total_sentences / max(total_refs, 1):.2f} per ref)")
        print(f"  Refs with >=2 sentences: {multi_sent_refs}")
        print(f"  Images with >=2 refs: {multi_ref_images} "
              f"(used for inter-ref separation)")

    return image_groups


def load_flickr_split(flickr_root, split='test', verbose=True):
    """
    Load Flickr30k split and build ref-style groups:
      image_id -> list[{ref_id, expressions}]

    We merge the 5 caption entries per image by phrase_id so each "ref"
    keeps multiple paraphrastic expressions across captions.
    """
    ann_file = os.path.join(flickr_root, 'annotations.json')
    split_file = os.path.join(flickr_root, 'raw_source', f'{split}.txt')

    with open(ann_file, 'r') as f:
        all_items = json.load(f)
    with open(split_file, 'r') as f:
        split_ids = {line.strip().replace('.jpg', '') for line in f if line.strip()}

    per_image_phrase = defaultdict(dict)
    image_paths = {}
    caption_items = 0

    for item in all_items:
        image_id = str(item.get('image_id', '')).strip()
        if not image_id or image_id not in split_ids:
            continue

        caption_items += 1
        img_path = item.get('image_path')
        if not img_path:
            img_path = os.path.join(flickr_root, 'images', f'{image_id}.jpg')
        image_paths[image_id] = img_path

        for phrase_item in item.get('phrases', []):
            if phrase_item.get('is_nobox', False):
                continue
            boxes = phrase_item.get('boxes') or []
            if not boxes:
                continue

            phrase = (phrase_item.get('phrase') or '').strip()
            if not phrase:
                continue

            phrase_id = phrase_item.get('phrase_id')
            if phrase_id is None:
                continue
            phrase_key = str(phrase_id)

            rec = per_image_phrase[image_id].get(phrase_key)
            if rec is None:
                rec = {
                    'ref_id': f'{image_id}_{phrase_key}',
                    'expressions': [],
                    '_seen': set(),
                }
                per_image_phrase[image_id][phrase_key] = rec

            norm = phrase.lower()
            if norm not in rec['_seen']:
                rec['_seen'].add(norm)
                rec['expressions'].append(phrase)

    image_groups = defaultdict(list)
    valid_image_paths = {}
    total_sentences = 0
    multi_sent_refs = 0

    for image_id, phrase_map in per_image_phrase.items():
        refs = []
        for rec in phrase_map.values():
            expressions = rec['expressions']
            if not expressions:
                continue

            if len(expressions) >= 2:
                multi_sent_refs += 1
            total_sentences += len(expressions)

            refs.append({
                'ref_id': rec['ref_id'],
                'expressions': expressions,
            })

        if refs:
            image_groups[image_id] = refs
            valid_image_paths[image_id] = image_paths.get(
                image_id, os.path.join(flickr_root, 'images', f'{image_id}.jpg')
            )

    total_refs = sum(len(g) for g in image_groups.values())
    multi_ref_images = sum(1 for g in image_groups.values() if len(g) >= 2)

    if verbose:
        print(
            f"  Flickr30k {split}: {total_refs} refs across {len(image_groups)} images "
            f"from {caption_items} caption annotations"
        )
        print(
            f"  Sentences kept: {total_sentences} "
            f"(avg {total_sentences / max(total_refs, 1):.2f} per ref)"
        )
        print(f"  Refs with >=2 sentences: {multi_sent_refs}")
        print(
            f"  Images with >=2 refs: {multi_ref_images} "
            f"(used for inter-ref separation)"
        )

    return image_groups, valid_image_paths


def find_image(image_id, search_dirs):
    fname = f'{image_id:012d}.jpg'
    for d in search_dirs:
        p = os.path.join(d, fname)
        if os.path.exists(p):
            return p
    return None


def count_total_queries(group_items, max_sentences_per_ref):
    total = 0
    for _, refs in group_items:
        for ref in refs:
            n = len(ref['expressions'])
            total += n if max_sentences_per_ref <= 0 else min(n, max_sentences_per_ref)
    return total


def stable_seed_from_ref_id(ref_id):
    try:
        return int(ref_id)
    except Exception:
        return zlib.crc32(str(ref_id).encode('utf-8')) & 0xFFFFFFFF


def is_oom_error(err):
    return 'out of memory' in str(err).lower()


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='refcocog',
                        choices=['refcoco', 'refcoco+', 'refcocog', 'flickr'])
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--dtype', default='float32',
                        choices=['float32', 'bfloat16', 'float16'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--bootstrap_iters', type=int, default=1000)
    parser.add_argument('--max_sentences_per_ref', type=int, default=0,
                        help='0 means use all sentences')
    parser.add_argument('--max_images', type=int, default=0,
                        help='0 means use all val images')
    parser.add_argument('--log_every', type=int, default=100)
    parser.add_argument('--data_root',
                        default=os.path.join(ROOT, 'dataSets', 'refer', 'data'))
    parser.add_argument('--split_by', default=None,
                        help='Defaults: refcoco/refcoco+=unc, refcocog=umd')
    parser.add_argument('--ref_split', default='val',
                        help='Refer split: val, testA, testB, or test')
    parser.add_argument('--flickr_root',
                        default=os.path.join(ROOT, 'dataSets', 'flickr30k_entities'))
    parser.add_argument('--flickr_split', default='test',
                        choices=['val', 'test'])
    parser.add_argument('--model_path',
                        default=os.path.join(ROOT, 'InternVL3_5-1B'))
    parser.add_argument('--output_dir',
                        default=os.path.join(ROOT, 'thinkdet', 'results',
                                             'layer_probe'))
    args = parser.parse_args()

    is_distributed, rank, world_size, local_rank = setup_distributed()
    is_main = (rank == 0)

    if is_distributed:
        run_device = f'cuda:{local_rank}'
    else:
        run_device = args.device

    os.makedirs(args.output_dir, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if is_main:
        if args.dataset == 'flickr':
            ds_title = f'Flickr30k {args.flickr_split}'
        else:
            ds_title = f'{args.dataset} {args.ref_split}'
        print('=' * 78)
        print(f'  ThinkDet v2 — Rigorous Reasoning Layer Probe ({ds_title})')
        print('=' * 78)
        if is_distributed:
            print(f'  Distributed mode: world_size={world_size} rank={rank}')

    # ── 1) Data ──
    image_path_map = {}
    if args.dataset == 'flickr':
        image_groups, image_path_map = load_flickr_split(
            args.flickr_root, split=args.flickr_split, verbose=is_main
        )
        dataset_label = f'Flickr30k {args.flickr_split}'
    else:
        split_by = args.split_by or DEFAULT_SPLIT_BY[args.dataset]
        image_groups = load_refer_split(
            args.data_root,
            dataset_name=args.dataset,
            split_by=split_by,
            split=args.ref_split,
            verbose=is_main,
        )
        dataset_label = f'{args.dataset} {args.ref_split} ({split_by} split)'

    def sort_image_key(k):
        try:
            return int(k)
        except Exception:
            return str(k)

    full_group_items = sorted(image_groups.items(), key=lambda x: sort_image_key(x[0]))

    if args.max_images > 0:
        full_group_items = full_group_items[:args.max_images]
        if is_main:
            print(f"  Limiting to first {len(full_group_items)} images (--max_images)")

    group_items = full_group_items[rank::world_size] if is_distributed else full_group_items

    image_dirs = []
    if args.dataset != 'flickr':
        image_dirs = [
            os.path.join(args.data_root, 'images', 'mscoco', 'train2017'),
            os.path.join(args.data_root, 'images', 'mscoco', 'val2017'),
            os.path.join(ROOT, 'dataSets', 'coco', 'train2017'),
            os.path.join(ROOT, 'dataSets', 'coco', 'val2017'),
        ]

    est_queries = count_total_queries(group_items, args.max_sentences_per_ref)
    full_est_queries = count_total_queries(full_group_items, args.max_sentences_per_ref)

    # ── 2) Model ──
    extractor = AllLayerExtractor(args.model_path, device=run_device, dtype=args.dtype)
    num_layers = extractor.num_layers

    # ── 3) Accumulators ──
    token_div_vals = [[] for _ in range(num_layers)]
    norm_vals = [[] for _ in range(num_layers)]

    intra_dist_vals = [[] for _ in range(num_layers)]
    inter_dist_vals = [[] for _ in range(num_layers)]
    margin_img_vals = [[] for _ in range(num_layers)]

    # pos-neg margin per ref (for uncertainty proxy)
    margin_ref_vals = [[] for _ in range(num_layers)]

    # image centroid vectors per layer for cross-image spread
    cross_centroids = [[] for _ in range(num_layers)]

    processed_queries = 0
    processed_refs = 0
    processed_images = 0
    skipped_images = 0

    t0 = time.time()

    if is_main:
        if is_distributed:
            print(
                f"\n  Processing {len(full_group_items)} images across {world_size} ranks "
                f"(~{full_est_queries} query forwards total, local ~{est_queries}) "
                f"across {num_layers} layers ...\n"
            )
        else:
            print(
                f"\n  Processing {len(group_items)} images "
                f"(~{est_queries} query forwards) across {num_layers} layers ...\n"
            )

    for img_idx, (image_id, refs) in enumerate(group_items):
        if args.dataset != 'flickr':
            img_path = find_image(image_id, image_dirs)
        else:
            img_path = image_path_map.get(str(image_id))
        if img_path is None:
            skipped_images += 1
            continue

        try:
            pil_image = Image.open(img_path).convert('RGB')
        except Exception:
            skipped_images += 1
            continue

        pixel_values = extractor.transform(pil_image).unsqueeze(0).to(
            extractor.device, dtype=extractor.input_dtype
        )

        # each entry: list[layer_idx] -> list[tensor[D]] for one ref
        ref_sentence_vectors = []

        for ref in refs:
            expressions = maybe_trim_sentences(
                ref['expressions'], args.max_sentences_per_ref,
                seed=args.seed + stable_seed_from_ref_id(ref['ref_id'])
            )

            layer_vecs = [[] for _ in range(num_layers)]
            success = 0

            for expr in expressions:
                query = normalize_query(expr)
                try:
                    feats = extractor.extract_all_layers(pixel_values, query)
                except RuntimeError as e:
                    if is_oom_error(e):
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue
                    raise

                success += 1
                processed_queries += 1

                for li in range(num_layers):
                    f = feats[li]

                    norm_vals[li].append(f.norm(dim=-1).mean().item())
                    token_div_vals[li].append(token_diversity(f))

                    pooled = f.mean(dim=0)
                    pooled = F.normalize(pooled.unsqueeze(0), dim=-1).squeeze(0)
                    layer_vecs[li].append(pooled)

            if success > 0:
                ref_sentence_vectors.append(layer_vecs)
                processed_refs += 1

        if not ref_sentence_vectors:
            continue

        processed_images += 1

        # image-level metrics from per-ref sentence vectors
        for li in range(num_layers):
            reps = []
            pos_sims = []
            img_intra_dists = []

            for layer_vecs in ref_sentence_vectors:
                sent_vecs = layer_vecs[li]
                if not sent_vecs:
                    continue

                sent_mat = torch.stack(sent_vecs, dim=0)
                rep = F.normalize(sent_mat.mean(dim=0, keepdim=True), dim=-1).squeeze(0)
                reps.append(rep)

                if sent_mat.shape[0] >= 2:
                    sim = sent_mat @ sent_mat.T
                    m = offdiag_mean(sim)
                    if m is not None:
                        pos_sims.append(m)
                        intra_d = 1.0 - m
                        img_intra_dists.append(intra_d)
                        intra_dist_vals[li].append(intra_d)
                    else:
                        pos_sims.append(np.nan)
                else:
                    pos_sims.append(np.nan)

            if not reps:
                continue

            rep_mat = torch.stack(reps, dim=0)
            img_centroid = F.normalize(rep_mat.mean(dim=0, keepdim=True), dim=-1).squeeze(0)
            cross_centroids[li].append(img_centroid.numpy())

            if rep_mat.shape[0] >= 2:
                sim_ref = rep_mat @ rep_mat.T
                m = offdiag_mean(sim_ref)
                if m is not None:
                    inter_d = 1.0 - m
                    inter_dist_vals[li].append(inter_d)

                    if img_intra_dists:
                        margin_img_vals[li].append(inter_d - float(np.mean(img_intra_dists)))

                # per-ref confidence margin for uncertainty proxy
                for r_idx, pos_sim in enumerate(pos_sims):
                    if not np.isfinite(pos_sim):
                        continue
                    neg_sim = (
                        sim_ref[r_idx].sum().item() - sim_ref[r_idx, r_idx].item()
                    ) / (rep_mat.shape[0] - 1)
                    margin_ref_vals[li].append(pos_sim - neg_sim)

        if is_main and (img_idx + 1) % max(args.log_every, 1) == 0:
            elapsed = time.time() - t0
            rate = processed_queries / max(elapsed, 1e-6)
            eta = (est_queries - processed_queries) / max(rate, 1e-6)
            print(
                f"  [{img_idx + 1:>4}/{len(group_items)}] "
                f"{processed_queries}/{est_queries} query forwards "
                f"({rate:.2f} q/s  ETA {eta / 60:.1f} min)"
            )

    local_elapsed = time.time() - t0

    if is_distributed:
        local_payload = {
            'token_div_vals': token_div_vals,
            'norm_vals': norm_vals,
            'intra_dist_vals': intra_dist_vals,
            'inter_dist_vals': inter_dist_vals,
            'margin_img_vals': margin_img_vals,
            'margin_ref_vals': margin_ref_vals,
            'cross_centroids': cross_centroids,
            'processed_queries': processed_queries,
            'processed_refs': processed_refs,
            'processed_images': processed_images,
            'skipped_images': skipped_images,
            'elapsed': local_elapsed,
        }

        gathered = [None for _ in range(world_size)]
        dist.all_gather_object(gathered, local_payload)

        if not is_main:
            cleanup_distributed(is_distributed)
            return

        token_div_vals = [[] for _ in range(num_layers)]
        norm_vals = [[] for _ in range(num_layers)]
        intra_dist_vals = [[] for _ in range(num_layers)]
        inter_dist_vals = [[] for _ in range(num_layers)]
        margin_img_vals = [[] for _ in range(num_layers)]
        margin_ref_vals = [[] for _ in range(num_layers)]
        cross_centroids = [[] for _ in range(num_layers)]

        processed_queries = 0
        processed_refs = 0
        processed_images = 0
        skipped_images = 0
        elapsed = 0.0

        for payload in gathered:
            merge_layer_lists(token_div_vals, payload['token_div_vals'])
            merge_layer_lists(norm_vals, payload['norm_vals'])
            merge_layer_lists(intra_dist_vals, payload['intra_dist_vals'])
            merge_layer_lists(inter_dist_vals, payload['inter_dist_vals'])
            merge_layer_lists(margin_img_vals, payload['margin_img_vals'])
            merge_layer_lists(margin_ref_vals, payload['margin_ref_vals'])
            merge_layer_lists(cross_centroids, payload['cross_centroids'])

            processed_queries += payload['processed_queries']
            processed_refs += payload['processed_refs']
            processed_images += payload['processed_images']
            skipped_images += payload['skipped_images']
            elapsed = max(elapsed, payload['elapsed'])
    else:
        elapsed = local_elapsed

    print(
        f"\n  Done: {processed_queries} query forwards, {processed_refs} refs, "
        f"{processed_images} images in {elapsed / 60:.1f} min "
        f"(skipped {skipped_images} missing/corrupt images)"
    )

    # ── 4) Cross-image spread values ──
    cross_spread_vals = [[] for _ in range(num_layers)]
    for li in range(num_layers):
        if len(cross_centroids[li]) < 2:
            continue

        mat = np.stack(cross_centroids[li], axis=0).astype(np.float32)  # [N, D]
        mat_norm = np.linalg.norm(mat, axis=1, keepdims=True)
        mat = mat / np.clip(mat_norm, 1e-8, None)

        global_cent = mat.mean(axis=0, keepdims=True)
        global_cent = global_cent / np.clip(
            np.linalg.norm(global_cent, axis=1, keepdims=True), 1e-8, None
        )

        dists = 1.0 - (mat @ global_cent.T).squeeze(1)
        dists = np.clip(dists, 0.0, 2.0)
        cross_spread_vals[li] = dists.tolist()

    # ── 5) Layer statistics with 95% CI ──
    layer_stats = []
    for li in range(num_layers):
        base_seed = args.seed + 10007 * (li + 1)

        separation = summarize(inter_dist_vals[li], args.bootstrap_iters, base_seed + 1)
        margin = summarize(margin_img_vals[li], args.bootstrap_iters, base_seed + 3)
        tok_div = summarize(token_div_vals[li], args.bootstrap_iters, base_seed + 4)
        cross_spread = summarize(cross_spread_vals[li], args.bootstrap_iters, base_seed + 5)
        feat_norm = summarize(norm_vals[li], args.bootstrap_iters, base_seed + 6)

        stability_samples = [1.0 - v for v in intra_dist_vals[li]]
        paraphrase_stability = summarize(
            stability_samples, args.bootstrap_iters, base_seed + 7
        )

        uncertainty_samples = [1.0 if m <= 0.0 else 0.0 for m in margin_ref_vals[li]]
        uncertainty_rate = summarize(
            uncertainty_samples, args.bootstrap_iters, base_seed + 8
        )

        certainty_samples = [1.0 - v for v in uncertainty_samples]
        certainty_rate = summarize(
            certainty_samples, args.bootstrap_iters, base_seed + 9
        )

        layer_stats.append({
            'layer': li,
            'separation': separation,
            'paraphrase_stability': paraphrase_stability,
            'margin': margin,
            'token_diversity': tok_div,
            'cross_spread': cross_spread,
            'uncertainty_rate': uncertainty_rate,
            'certainty_rate': certainty_rate,
            'feature_norm': feat_norm,
        })

    # ── 6) Composite ranking (rank-normalized, robust to scale) ──
    margin_vals = [s['margin']['mean'] for s in layer_stats]
    stab_vals = [s['paraphrase_stability']['mean'] for s in layer_stats]
    td_vals = [s['token_diversity']['mean'] for s in layer_stats]
    cross_vals = [s['cross_spread']['mean'] for s in layer_stats]
    cert_vals = [s['certainty_rate']['mean'] for s in layer_stats]

    margin_n = rank01(margin_vals, higher_better=True)
    stab_n = rank01(stab_vals, higher_better=True)
    td_n = rank01(td_vals, higher_better=True)
    cross_n = rank01(cross_vals, higher_better=True)
    cert_n = rank01(cert_vals, higher_better=True)

    results = []
    for i, stats in enumerate(layer_stats):
        comp = (
            0.40 * margin_n[i]
            + 0.20 * stab_n[i]
            + 0.20 * td_n[i]
            + 0.15 * cross_n[i]
            + 0.05 * cert_n[i]
        )

        results.append({
            'layer': stats['layer'],

            # Core reasoning metrics
            'expression_separation': round(stats['separation']['mean'], 6),
            'expression_separation_ci95': [
                round(stats['separation']['ci95'][0], 6),
                round(stats['separation']['ci95'][1], 6),
            ],
            'paraphrase_stability': round(stats['paraphrase_stability']['mean'], 6),
            'paraphrase_stability_ci95': [
                round(stats['paraphrase_stability']['ci95'][0], 6),
                round(stats['paraphrase_stability']['ci95'][1], 6),
            ],
            'disambiguation_margin': round(stats['margin']['mean'], 6),
            'disambiguation_margin_ci95': [
                round(stats['margin']['ci95'][0], 6),
                round(stats['margin']['ci95'][1], 6),
            ],

            # Secondary diagnostics
            'token_diversity': round(stats['token_diversity']['mean'], 6),
            'token_diversity_ci95': [
                round(stats['token_diversity']['ci95'][0], 6),
                round(stats['token_diversity']['ci95'][1], 6),
            ],
            'cross_image_spread': round(stats['cross_spread']['mean'], 6),
            'cross_image_spread_ci95': [
                round(stats['cross_spread']['ci95'][0], 6),
                round(stats['cross_spread']['ci95'][1], 6),
            ],
            'uncertainty_rate': round(stats['uncertainty_rate']['mean'], 6),
            'uncertainty_rate_ci95': [
                round(stats['uncertainty_rate']['ci95'][0], 6),
                round(stats['uncertainty_rate']['ci95'][1], 6),
            ],
            'certainty_rate': round(stats['certainty_rate']['mean'], 6),
            'feature_norm': round(stats['feature_norm']['mean'], 6),

            # Counts
            'num_query_samples': stats['feature_norm']['n'],
            'num_intra_ref_samples': stats['paraphrase_stability']['n'],
            'num_inter_ref_images': stats['separation']['n'],
            'num_margin_images': stats['margin']['n'],
            'num_margin_refs': stats['uncertainty_rate']['n'],

            # Composite
            'composite_score': round(comp, 6),
            'composite_components': {
                'margin_rank01': round(margin_n[i], 6),
                'stability_rank01': round(stab_n[i], 6),
                'token_div_rank01': round(td_n[i], 6),
                'cross_spread_rank01': round(cross_n[i], 6),
                'certainty_rank01': round(cert_n[i], 6),
            },
        })

    # ── 7) Reporting ──
    best_comp = max(results, key=lambda r: r['composite_score'])
    best_margin = max(results, key=lambda r: r['disambiguation_margin'])
    best_stab = max(results, key=lambda r: r['paraphrase_stability'])

    sorted_comp = sorted(results, key=lambda r: r['composite_score'], reverse=True)
    runner = sorted_comp[1] if len(sorted_comp) > 1 else best_comp

    print(f"\n{'=' * 78}")
    print(
        f"  RIGOROUS LAYER PROBE RESULTS "
        f"({processed_queries} queries, {processed_images} images)"
    )
    print(f"{'=' * 78}")
    print(
        "  Layer  Margin    Stability  TokDiv    CrossSpr  Uncert   Composite"
    )
    print(
        "  -----  --------  ---------  --------  --------  -------  ---------"
    )

    for r in results:
        tag = '  <-- BEST' if r['layer'] == best_comp['layer'] else ''
        print(
            f"  {r['layer']:>5}  "
            f"{r['disambiguation_margin']:>8.4f}  "
            f"{r['paraphrase_stability']:>9.4f}  "
            f"{r['token_diversity']:>8.4f}  "
            f"{r['cross_image_spread']:>8.4f}  "
            f"{r['uncertainty_rate']:>7.4f}  "
            f"{r['composite_score']:>9.4f}{tag}"
        )

    print(f"\n  Best disambiguation margin: layer {best_margin['layer']} "
          f"({best_margin['disambiguation_margin']:.6f})")
    print(f"  Best paraphrase stability:  layer {best_stab['layer']} "
          f"({best_stab['paraphrase_stability']:.6f})")
    print(f"  Best composite (overall):   layer {best_comp['layer']} "
          f"({best_comp['composite_score']:.6f})")

    best_ci = best_comp['disambiguation_margin_ci95']
    runner_ci = runner['disambiguation_margin_ci95']
    overlap = ci_overlap(best_ci, runner_ci)
    print(
        f"  Top-2 margin CI overlap: {'YES' if overlap else 'NO'} "
        f"(L{best_comp['layer']} {best_ci} vs L{runner['layer']} {runner_ci})"
    )

    print(
        "  Composite weights: 40% margin + 20% stability + 20% token_div "
        "+ 15% cross_spread + 5% certainty"
    )

    # ── 8) Save ──
    result_path = os.path.join(args.output_dir, 'layer_probe_results.json')
    with open(result_path, 'w') as f:
        json.dump({
            'config': {
                'dataset': dataset_label,
                'dataset_name': args.dataset,
                'split_by': None if args.dataset == 'flickr' else (args.split_by or DEFAULT_SPLIT_BY[args.dataset]),
                'ref_split': args.ref_split if args.dataset != 'flickr' else None,
                'flickr_split': args.flickr_split if args.dataset == 'flickr' else None,
                'model_path': args.model_path,
                'dtype': args.dtype,
                'seed': args.seed,
                'bootstrap_iters': args.bootstrap_iters,
                'max_sentences_per_ref': args.max_sentences_per_ref,
                'max_images': args.max_images,
                'num_queries': processed_queries,
                'num_refs': processed_refs,
                'num_images': processed_images,
                'time_min': round(elapsed / 60, 2),
                'composite_weights': {
                    'disambiguation_margin': 0.40,
                    'paraphrase_stability': 0.20,
                    'token_diversity': 0.20,
                    'cross_image_spread': 0.15,
                    'certainty_rate': 0.05,
                },
            },
            'layers': results,
        }, f, indent=2)
    print(f"\n  Results: {result_path}")

    # ── 9) Plot ──
    try:
        generate_plot(results, args.output_dir, dataset_label=dataset_label)
    except Exception as e:
        print(f"  Plot failed: {e}")

    print('=' * 78)
    cleanup_distributed(is_distributed)


# ─────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────
def generate_plot(results, output_dir, dataset_label='RefCOCOg val (umd split)'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    layers = [r['layer'] for r in results]
    margin = [r['disambiguation_margin'] for r in results]
    stability = [r['paraphrase_stability'] for r in results]
    tok_div = [r['token_diversity'] for r in results]
    composite = [r['composite_score'] for r in results]

    fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=True)
    fig.patch.set_facecolor('#0d1117')
    fig.suptitle(
        'ThinkDet — Rigorous Reasoning Layer Probe\n'
        f'({dataset_label}: margin, stability, token structure, overall score)',
        fontsize=14, color='#f0f6fc', fontweight='bold', y=0.995
    )

    plots = [
        (axes[0], margin, 'Disambiguation Margin (higher better)', '#58a6ff'),
        (axes[1], stability, 'Paraphrase Stability (higher better)', '#3fb950'),
        (axes[2], tok_div, 'Token Diversity (higher better)', '#d29922'),
        (axes[3], composite, 'Composite Score (rank-normalized weighted)', '#ff7b72'),
    ]

    for ax, vals, title, color in plots:
        ax.set_facecolor('#0d1117')
        best_idx = int(np.argmax(vals))

        bar_colors = [color if i != best_idx else '#f0f6fc' for i in range(len(vals))]
        ax.bar(
            range(len(vals)), vals, color=bar_colors,
            edgecolor='#30363d', linewidth=0.3, width=0.75, alpha=0.85
        )
        ax.plot(
            range(len(vals)), vals, color=color, linewidth=1.5,
            alpha=0.75, marker='o', markersize=3
        )

        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels([str(l) for l in layers], fontsize=7, color='#8b949e')
        ax.set_title(title, fontsize=10, color=color, fontweight='bold', pad=8)
        ax.tick_params(colors='#8b949e', labelsize=7)

        for spine in ax.spines.values():
            spine.set_color('#30363d')
        ax.grid(axis='y', alpha=0.1, color='#484f58')

        y_span = max(vals) - min(vals)
        y_offset = (y_span * 0.1) if y_span > 1e-8 else 0.01

        ax.annotate(
            f"Best: L{layers[best_idx]} ({vals[best_idx]:.4f})",
            xy=(best_idx, vals[best_idx]),
            xytext=(min(best_idx + 3, len(vals) - 1), vals[best_idx] + y_offset),
            fontsize=9, color='#f0f6fc', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#f0f6fc', lw=1.2)
        )

    axes[-1].set_xlabel('LLM Layer', fontsize=10, color='#8b949e')

    plt.tight_layout(rect=[0, 0, 1, 0.965])
    out_path = os.path.join(output_dir, 'layer_probe_summary.png')
    plt.savefig(out_path, dpi=150, facecolor='#0d1117', bbox_inches='tight')
    plt.close()
    print(f"  Plot: {out_path}")


if __name__ == '__main__':
    main()
