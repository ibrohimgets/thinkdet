"""
Diagnose whether ThinkDet TMA injection is actually affecting GroundingDINO.

This script measures:
  - learned gate values per injected decoder layer
  - residual delta norm vs original DINO text-memory norm
  - InternVL layer feature similarity after the trained TMA projection
  - score/logit changes relative to gate-off baseline
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from types import SimpleNamespace

import torch
from PIL import Image

ROOT = "/home/iibrohimm/project/next_step"
THINKDET_ROOT = os.path.join(ROOT, "thinkdet")
GROUNDING_DINO_ROOT = os.path.join(ROOT, "GroundingDINO")
sys.path.insert(0, ROOT)
sys.path.insert(0, GROUNDING_DINO_ROOT)
sys.path.insert(0, os.path.join(THINKDET_ROOT, "scripts", "eval"))

from groundingdino.util.misc import NestedTensor
from eval_affordance_unified import (  # noqa: E402
    DEFAULT_CKPT,
    GD_CONFIG,
    GD_WEIGHTS,
    INTERNVL_PATH,
    build_dino_transform,
    build_internvl_transform,
    build_positive_map_for_query,
    load_unified_model,
    patch_groundingdino_ms_deform_attn,
    resolve_query_scoring_assets,
    score_outputs_for_query,
)


DEFAULT_BENCH = os.path.join(
    THINKDET_ROOT,
    "data/benchmarks/affordance_coco_val_three_prompts_test_balanced30.json",
)
DEFAULT_OUT = os.path.join(
    THINKDET_ROOT,
    "results/analysis/injection_effect_diagnostic_layers7_11_14.json",
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", default=DEFAULT_BENCH)
    p.add_argument("--checkpoint", default=DEFAULT_CKPT)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max_samples", type=int, default=30)
    p.add_argument("--output", default=DEFAULT_OUT)
    return p.parse_args()


def set_extractor_layers(model, layers, fusion="mean"):
    layers = sorted(set(int(x) for x in layers))
    model.feature_extractor.extract_layers = layers
    model.feature_extractor.extract_layer = max(layers)
    model.feature_extractor.layer_fusion = fusion


def make_dino_inputs(image_pil, dino_tf, device):
    dino_t = dino_tf(image_pil).unsqueeze(0).to(device)
    mask = torch.zeros(1, dino_t.shape[2], dino_t.shape[3], dtype=torch.bool, device=device)
    return NestedTensor(dino_t, mask)


def mean(xs):
    return float(sum(xs) / max(len(xs), 1))


def summarize(values):
    return {k: mean(v) for k, v in sorted(values.items())}


def add_capture_hooks(model, bucket):
    handles = []
    for inj_idx, layer in zip(model.injection_layers, model.adapted_layers):
        if getattr(layer, "residual_fuser", None) is None:
            continue

        def hook(_module, inputs, output, inj_idx=inj_idx, layer=layer):
            memory_text = inputs[0].detach().float()
            aug_tokens = inputs[1].detach().float()
            out = output.detach().float()
            delta = out - memory_text
            memory_norm = memory_text.flatten(1).norm(dim=1).mean().item()
            delta_norm = delta.flatten(1).norm(dim=1).mean().item()
            aug_norm = aug_tokens.flatten(1).norm(dim=1).mean().item()
            bucket[int(inj_idx)]["gate"].append(float(torch.tanh(layer.augmenter.alpha).item()))
            bucket[int(inj_idx)]["memory_norm"].append(memory_norm)
            bucket[int(inj_idx)]["aug_norm"].append(aug_norm)
            bucket[int(inj_idx)]["delta_norm"].append(delta_norm)
            bucket[int(inj_idx)]["delta_over_memory"].append(delta_norm / max(memory_norm, 1e-12))
            bucket[int(inj_idx)]["aug_over_memory"].append(aug_norm / max(memory_norm, 1e-12))

        handles.append(layer.residual_fuser.register_forward_hook(hook))
    return handles


@torch.no_grad()
def compute_projection_stats(model, ivl_t, prompt, accum):
    set_extractor_layers(model, [7, 9, 11, 14], fusion="mean")
    selected = model.feature_extractor.extract_selected_layers(ivl_t, [prompt])
    variants = {
        "layer9": selected[9],
        "layer14": selected[14],
        "layers7_11_14": torch.stack([selected[7], selected[11], selected[14]], dim=0).mean(dim=0),
    }
    for layer in model.adapted_layers:
        projected = {}
        for name, h_vlm in variants.items():
            augmenter = layer.augmenter
            pre = augmenter.proj_out(augmenter.proj_act(augmenter.proj_in(h_vlm))).float()
            post = augmenter.kv_norm(pre).float()
            projected[name] = (pre, post)
            accum[name]["h_vlm_norm"].append(h_vlm.detach().float().flatten(1).norm(dim=1).mean().item())
            accum[name]["proj_pre_ln_norm"].append(pre.flatten(1).norm(dim=1).mean().item())
            accum[name]["proj_post_ln_norm"].append(post.flatten(1).norm(dim=1).mean().item())

        base_pre, base_post = projected["layer9"]
        for name in ["layer14", "layers7_11_14"]:
            pre, post = projected[name]
            pre_cos = torch.nn.functional.cosine_similarity(
                base_pre.flatten(1), pre.flatten(1), dim=1
            ).mean().item()
            post_cos = torch.nn.functional.cosine_similarity(
                base_post.flatten(1), post.flatten(1), dim=1
            ).mean().item()
            rel_pre_diff = (pre - base_pre).flatten(1).norm(dim=1).mean().item() / max(
                base_pre.flatten(1).norm(dim=1).mean().item(), 1e-12
            )
            accum[name]["proj_pre_ln_cos_vs_layer9"].append(pre_cos)
            accum[name]["proj_post_ln_cos_vs_layer9"].append(post_cos)
            accum[name]["proj_pre_ln_rel_diff_vs_layer9"].append(rel_pre_diff)

    accum["layer9"]["proj_pre_ln_cos_vs_layer9"].append(1.0)
    accum["layer9"]["proj_post_ln_cos_vs_layer9"].append(1.0)
    accum["layer9"]["proj_pre_ln_rel_diff_vs_layer9"].append(0.0)


@torch.no_grad()
def main():
    args = parse_args()
    patch_groundingdino_ms_deform_attn()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    loader_args = SimpleNamespace(
        gd_config=GD_CONFIG,
        gd_weights=GD_WEIGHTS,
        internvl_path=INTERNVL_PATH,
        thinkdet_checkpoint=args.checkpoint,
        extract_layer=None,
        extract_layers=[7, 9, 11, 14],
        layer_fusion="mean",
        force_gate_value=None,
    )
    model, ckpt_meta = load_unified_model(loader_args, device)
    model.eval()
    tokenizer, special_tokens = resolve_query_scoring_assets(model)
    dino_tf = build_dino_transform()
    ivl_tf = build_internvl_transform()

    with open(args.benchmark, "r") as f:
        bench = json.load(f)
    samples = [s for s in bench["samples"] if s.get("split") == "test"]
    samples = samples[: int(args.max_samples)]

    configs = {
        "layer9": [9],
        "layer14": [14],
        "layers7_11_14": [7, 11, 14],
    }
    delta_stats = {
        name: defaultdict(lambda: defaultdict(list))
        for name in configs
    }
    score_stats = {name: defaultdict(list) for name in configs}
    projection_stats = defaultdict(lambda: defaultdict(list))

    for idx, sample in enumerate(samples, 1):
        image_pil = Image.open(sample["image_path"]).convert("RGB")
        dino_inputs = make_dino_inputs(image_pil, dino_tf, device)
        ivl_t = ivl_tf(image_pil).unsqueeze(0).to(device)
        query_text = sample["prompt"] if sample["prompt"].strip().endswith(".") else sample["prompt"].strip() + " ."
        gd_inputs = {"samples": dino_inputs, "captions": [query_text]}
        pmap = build_positive_map_for_query(tokenizer, special_tokens, query_text, max_text_len=512)

        outputs_gate0, _ = model(ivl_t, [sample["prompt"]], gd_inputs, force_gate0=True)
        base_scores, _ = score_outputs_for_query(outputs_gate0, pmap)
        base_logits = outputs_gate0["pred_logits"].detach().float().clamp(-50, 50)

        compute_projection_stats(model, ivl_t, sample["prompt"], projection_stats)

        for name, layers in configs.items():
            set_extractor_layers(model, layers, fusion="mean")
            handles = add_capture_hooks(model, delta_stats[name])
            outputs, aux = model(ivl_t, [sample["prompt"]], gd_inputs, force_gate0=False)
            for handle in handles:
                handle.remove()

            scores, _ = score_outputs_for_query(outputs, pmap)
            logits = outputs["pred_logits"].detach().float().clamp(-50, 50)
            score_stats[name]["gate_mean"].append(float(aux.get("gate_mean", 0.0)))
            score_stats[name]["score_abs_mean_delta_vs_gate0"].append(
                (scores - base_scores).abs().mean().item()
            )
            score_stats[name]["score_abs_max_delta_vs_gate0"].append(
                (scores - base_scores).abs().max().item()
            )
            score_stats[name]["logit_abs_mean_delta_vs_gate0"].append(
                (logits - base_logits).abs().mean().item()
            )
            score_stats[name]["logit_abs_max_delta_vs_gate0"].append(
                (logits - base_logits).abs().max().item()
            )

        if idx % 5 == 0 or idx == len(samples):
            print(f"[diagnose] {idx}/{len(samples)}")

    delta_summary = {}
    for cfg_name, by_layer in delta_stats.items():
        delta_summary[cfg_name] = {
            str(inj_idx): summarize(metrics)
            for inj_idx, metrics in sorted(by_layer.items())
        }

    payload = {
        "status": "ok",
        "benchmark": args.benchmark,
        "n_samples": len(samples),
        "checkpoint": args.checkpoint,
        "checkpoint_meta": ckpt_meta,
        "adapter_trained_layers": ckpt_meta.get("checkpoint_extract_layers"),
        "scoring_check": {
            "uses_positive_map_query_projection": True,
            "plain_logits_max_used_for_ranking": False,
            "source": "eval_affordance_unified.score_outputs_for_query",
        },
        "delta_summary_by_eval_config_and_decoder_layer": delta_summary,
        "score_delta_vs_gate0": {
            name: summarize(metrics)
            for name, metrics in sorted(score_stats.items())
        },
        "projection_summary": {
            name: summarize(metrics)
            for name, metrics in sorted(projection_stats.items())
        },
    }
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
