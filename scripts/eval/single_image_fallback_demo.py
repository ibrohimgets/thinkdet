#!/usr/bin/env python3
"""
Single-image ThinkDet fallback demo.

Runs:
- GroundingDINO baseline
- ThinkDet primary
- Optional LLM feedback rerank
- Optional prompt-refinement retries

Saves a JSON trace and annotated images so qualitative fallback cases can be
shown directly in a thesis or advisor meeting.
"""

import argparse
import copy
import datetime
import json
import os
import sys

import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = "/home/iibrohimm/project/next_step"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "GroundingDINO", "GroundingDINO"))
sys.path.insert(0, SCRIPT_DIR)

from eval_grounding_benchmark_compare import (
    DEFAULT_CKPT,
    GD_CONFIG,
    GD_WEIGHTS,
    INTERNVL_PATH,
    build_dino_transform,
    build_feedback_predictions,
    build_internvl_transform,
    build_positive_map_for_query,
    build_scored_bundle,
    bundle_to_predictions,
    get_special_tokens,
    load_baseline_model,
    load_unified_model,
    outputs_to_predictions,
    run_baseline_outputs,
    run_thinkdet_outputs,
)
from thinkdet.inference.fallback import (
    FallbackPolicyConfig,
    InternVLPromptRefiner,
    InternVLYesNoReranker,
    ScoredCandidateSet,
    apply_fallback_policy,
    normalize_query,
    summarize_scores,
)


DEFAULT_OUTPUT_DIR = (
    f"{ROOT}/thinkdet/results/qualitative/"
    f"single_image_fallback_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
)


try:
    FONT_BOLD = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18
    )
    FONT_SMALL = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13
    )
except Exception:
    FONT_BOLD = FONT_SMALL = ImageFont.load_default()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--max_draw", type=int, default=3)
    parser.add_argument("--selected_max_draw", type=int, default=1)

    parser.add_argument("--gd_config", type=str, default=GD_CONFIG)
    parser.add_argument("--gd_weights", type=str, default=GD_WEIGHTS)
    parser.add_argument("--internvl_path", type=str, default=INTERNVL_PATH)
    parser.add_argument("--thinkdet_checkpoint", type=str, default=DEFAULT_CKPT)
    parser.add_argument("--extract_layer", type=int, default=None)
    parser.add_argument("--extract_layers", type=int, nargs="+", default=None)
    parser.add_argument("--layer_fusion", type=str, default=None, choices=["mean", "last"])

    parser.add_argument("--disable_feedback", action="store_true")
    parser.add_argument("--prompt_refine", action="store_true")
    parser.add_argument("--force_feedback", action="store_true")
    parser.add_argument("--force_refine", action="store_true")
    parser.add_argument("--llm_feedback_top_k", type=int, default=5)
    parser.add_argument("--llm_feedback_weight", type=float, default=0.20)
    parser.add_argument("--llm_feedback_max_new_tokens", type=int, default=6)
    parser.add_argument("--llm_feedback_temperature", type=float, default=0.0)
    parser.add_argument("--llm_feedback_improve_margin", type=float, default=0.01)
    parser.add_argument("--prompt_refine_max_variants", type=int, default=3)
    parser.add_argument("--prompt_refine_max_new_tokens", type=int, default=96)
    parser.add_argument("--prompt_refine_temperature", type=float, default=0.0)
    parser.add_argument("--prompt_refine_improve_margin", type=float, default=0.00)
    parser.add_argument(
        "--refine_mode",
        type=str,
        default="default",
        choices=[
            "default",
            "affordance_demo",
            "visible_tool_demo",
            "generic_substitute_demo",
            "structured_reasoning_demo",
        ],
    )
    parser.add_argument("--fallback_min_top1", type=float, default=0.20)
    parser.add_argument("--fallback_min_margin", type=float, default=0.02)
    parser.add_argument("--fallback_min_gate", type=float, default=None)
    parser.add_argument(
        "--prompt_refine_require_semantic_preservation",
        dest="prompt_refine_require_semantic_preservation",
        action="store_true",
    )
    parser.add_argument(
        "--no_prompt_refine_require_semantic_preservation",
        dest="prompt_refine_require_semantic_preservation",
        action="store_false",
    )
    parser.set_defaults(
        prompt_refine=True,
        prompt_refine_require_semantic_preservation=True,
    )
    parser.add_argument("--prompt_refine_min_shared_terms", type=int, default=1)
    return parser.parse_args()


def box_key(pred):
    return pred.get("box_abs_xyxy") or pred.get("box") or [0, 0, 0, 0]


def jsonable(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (float, int, str, bool)) or obj is None:
        return obj
    return str(obj)


def draw_predictions(image_pil, preds, title, out_path, max_draw=3):
    img = image_pil.copy().convert("RGB")
    draw = ImageDraw.Draw(img)
    colors = [(0, 220, 60), (255, 160, 20), (220, 40, 40), (70, 160, 255)]
    widths = [4, 2, 1, 1]

    for rank, pred in enumerate(preds[:max_draw]):
        x1, y1, x2, y2 = [float(v) for v in box_key(pred)]
        color = colors[min(rank, len(colors) - 1)]
        width = widths[min(rank, len(widths) - 1)]
        for off in range(width):
            draw.rectangle([x1 - off, y1 - off, x2 + off, y2 + off], outline=color)
        label = f"#{rank + 1} {float(pred.get('score', 0.0)):.3f}"
        if "combined_score" in pred:
            label += f" | c={float(pred['combined_score']):.3f}"
        if "llm_score" in pred:
            label += f" | llm={float(pred['llm_score']):.1f}"
        label_w = max(72, len(label) * 7)
        bar_y = max(0, int(y1) - 18)
        draw.rectangle([x1, bar_y, x1 + label_w, bar_y + 16], fill=color)
        draw.text((x1 + 2, bar_y + 1), label, fill=(255, 255, 255), font=FONT_SMALL)

    bar_h = 34
    panel = Image.new("RGB", (img.width, img.height + bar_h), (28, 28, 28))
    panel.paste(img, (0, bar_h))
    d = ImageDraw.Draw(panel)
    d.text((10, 7), title, fill=(255, 255, 255), font=FONT_BOLD)
    panel.save(out_path, quality=95)


def make_overview(image_pil, panels, prompt, out_path):
    gap = 8
    resized = []
    target_h = max(panel.height for panel in panels)
    for panel in panels:
        if panel.height != target_h:
            scale = target_h / float(panel.height)
            panel = panel.resize((int(panel.width * scale), target_h), Image.BILINEAR)
        resized.append(panel)

    total_w = sum(panel.width for panel in resized) + gap * (len(resized) - 1)
    header_h = 44
    canvas = Image.new("RGB", (total_w, target_h + header_h), (18, 18, 18))
    d = ImageDraw.Draw(canvas)
    d.text((10, 8), f'Prompt: "{prompt}"', fill=(120, 255, 140), font=FONT_BOLD)
    d.text(
        (10, 25),
        f"image={os.path.basename(image_pil.filename)}",
        fill=(180, 180, 180),
        font=FONT_SMALL,
    )
    x = 0
    for panel in resized:
        canvas.paste(panel, (x, header_h))
        x += panel.width + gap
    canvas.save(out_path, quality=95)


def save_panel(image_pil, preds, title, out_path, max_draw):
    draw_predictions(image_pil, preds, title, out_path, max_draw=max_draw)
    return Image.open(out_path).convert("RGB")


def scored_candidate_from_rerank(bundle_name, reranked_preds, aux, query_text):
    return ScoredCandidateSet(
        name=bundle_name,
        scores=[float(pred.get("combined_score", pred["score"])) for pred in reranked_preds],
        payload={"type": "reranked_preds", "preds": reranked_preds},
        aux=aux,
        query_text=query_text,
    )


def maybe_force_cfg(cfg, args):
    if not (args.force_feedback or args.force_refine):
        return cfg
    forced = copy.deepcopy(cfg)
    forced.min_top1 = 1.10
    forced.min_margin = 1.10
    forced.min_gate = None
    return forced


def is_bad_refinement(query_text):
    lowered = " ".join(str(query_text).strip().lower().split())
    bad_markers = [
        "here are",
        "alternatives",
        "alternative",
        "rewrite",
        "rewritten query",
        "option 1",
        "option 2",
        "option 3",
    ]
    return any(marker in lowered for marker in bad_markers)


class AffordanceDemoPromptRefiner(InternVLPromptRefiner):
    """Image-aware refinement prompt for qualitative affordance demos."""

    def _build_question(self, query_text, has_image):
        prefix = "<image>\n" if has_image else ""
        return (
            f"{prefix}"
            f"Rewrite this grounding query into up to {self.num_candidates} short alternatives.\n"
            "Rules:\n"
            "- preserve the task intent\n"
            "- if the ideal tool is absent, choose a visible substitute object that could still do the job\n"
            "- only name objects that are clearly visible in the image\n"
            "- do not mention knife unless a knife is actually visible\n"
            "- for soft-food cutting or separating, fork can be a valid substitute if visible\n"
            "- prefer concrete visible object names and short spatial cues\n"
            "- keep each alternative short and literal\n"
            "- do not explain or add any preface\n"
            "- output one query per line only\n"
            f"Original query: {normalize_query(query_text)}"
        )


class VisibleToolDemoPromptRefiner(InternVLPromptRefiner):
    """Direct visible-object selector for affordance demo cases."""

    def _build_question(self, query_text, has_image):
        prefix = "<image>\n" if has_image else ""
        return (
            f"{prefix}"
            f"Given this image and request, output up to {self.num_candidates} short noun phrases "
            "naming visible objects that could satisfy the request.\n"
            "Rules:\n"
            "- choose only objects that are clearly visible in the image\n"
            "- choose a hand-held tool or utensil, not the food itself\n"
            "- prefer a visible substitute if the ideal tool is absent\n"
            "- prefer explicit object names and short spatial cues\n"
            "- do not mention knife unless a knife is actually visible\n"
            "- do not answer with tofu, plate, bowl, table, or cutting board\n"
            "- for soft tofu, fork can be a valid cutting or separating tool if visible\n"
            "- if both fork and spoon are visible, prefer the fork for separating or cutting tofu pieces\n"
            "- output one phrase per line only\n"
            f"Request: {normalize_query(query_text)}"
        )


class GenericSubstitutePromptRefiner(InternVLPromptRefiner):
    """Generic image-aware substitute selector without object-specific hints."""

    def _build_question(self, query_text, has_image):
        prefix = "<image>\n" if has_image else ""
        return (
            f"{prefix}"
            f"Rewrite this request into up to {self.num_candidates} short grounding phrases.\n"
            "Rules:\n"
            "- preserve the request intent\n"
            "- choose only objects that are clearly visible in the image\n"
            "- if the ideal object is absent, choose a visible substitute that could still satisfy the request\n"
            "- do not answer with the acted-on object itself or generic scene words\n"
            "- prefer explicit object names and short spatial cues\n"
            "- keep each phrase short and literal\n"
            "- output one phrase per line only\n"
            f"Request: {normalize_query(query_text)}"
        )


def parse_final_grounding_phrase(text):
    import re

    raw_text = str(text or "")
    choice_match = re.search(
        r"(?:best visible object to use in this image|best visible substitute)\s*:\s*(.+)",
        raw_text,
        flags=re.IGNORECASE,
    )
    chosen_object = None
    if choice_match:
        chosen_object = choice_match.group(1).strip().strip("\"'`").splitlines()[0].strip()
        if chosen_object.lower() == "none":
            chosen_object = None

    match = re.search(
        r"(?:grounding phrase for chosen visible object|final grounding phrase)\s*:\s*(.+)",
        raw_text,
        flags=re.IGNORECASE,
    )
    phrase = None
    if match:
        phrase = match.group(1).strip().strip("\"'`")
        phrase = phrase.splitlines()[0].strip()
        if not phrase or phrase.lower() == "none":
            phrase = None

    if chosen_object:
        if not phrase:
            return chosen_object
        lowered_phrase = f" {phrase.lower()} "
        lowered_choice = f" {chosen_object.lower()} "
        if lowered_choice not in lowered_phrase:
            return chosen_object
    return phrase


class StructuredReasoningPromptRefiner(InternVLPromptRefiner):
    """Structured image-grounded reasoning prompt for qualitative demos."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_raw_response = None

    def _build_question(self, query_text, has_image):
        prefix = "<image>\n" if has_image else ""
        return (
            f"{prefix}"
            "Choose the best visible object in the image that could satisfy the request.\n"
            "Use only objects that are actually visible in the image.\n"
            "Do not invent absent objects.\n"
            "Do not answer with the acted-on object itself.\n"
            "If the ideal tool is absent, choose the best visible substitute.\n"
            "Respond in plain text with exactly these fields:\n"
            "Request: <request>\n"
            "Visible relevant objects: <comma-separated list>\n"
            "Ideal tool normally used: <object or none>\n"
            "Ideal tool visible: <yes or no>\n"
            "Best visible object to use in this image: <object or none>\n"
            "Why: <one short sentence>\n"
            "Grounding phrase for chosen visible object: <short phrase naming the same chosen visible object, or none>\n"
            "The grounding phrase must refer to the same object as 'Best visible object to use in this image'.\n"
            "Do not output the ideal tool as the grounding phrase unless it is actually visible in the image.\n"
            f"Request: {normalize_query(query_text)}"
        )

    @torch.no_grad()
    def refine_prompts(self, query_text, image_pil=None):
        pixel_values = None
        has_image = image_pil is not None and self.image_transform is not None
        if has_image:
            pixel_values = self.image_transform(image_pil).unsqueeze(0).to(self.device)

        response = self.model.chat(
            tokenizer=self.tokenizer,
            pixel_values=pixel_values,
            question=self._build_question(query_text, has_image=has_image),
            generation_config=dict(self.generation_config),
            history=None,
            return_history=False,
            verbose=False,
        )
        self.last_raw_response = response
        phrase = parse_final_grounding_phrase(response)
        if not phrase:
            return []
        return [normalize_query(phrase)]


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    image_pil = Image.open(args.image).convert("RGB")
    image_pil.filename = args.image

    dino_tf = build_dino_transform()
    ivl_tf = build_internvl_transform()

    print("=" * 68)
    print("Single-image ThinkDet fallback demo")
    print(f"image: {args.image}")
    print(f"prompt: {args.prompt}")
    print(f"device: {device}")
    print("=" * 68)

    print("[1/4] Loading GroundingDINO baseline...")
    baseline = load_baseline_model(args, device)
    print("[2/4] Loading ThinkDet...")
    thinkdet, ckpt_meta = load_unified_model(args, device)
    print("[3/4] Preparing tokenizers and fallback modules...")

    baseline_tokenizer = baseline.tokenizer
    baseline_special_tokens = get_special_tokens(baseline)
    thinkdet_tokenizer = thinkdet.grounding_dino.tokenizer
    thinkdet_special_tokens = get_special_tokens(thinkdet.grounding_dino)

    llm_reranker = None
    if not args.disable_feedback:
        llm_reranker = InternVLYesNoReranker(
            internvl_model=thinkdet.feature_extractor.internvl,
            tokenizer=thinkdet.feature_extractor.tokenizer,
            device=device,
            image_transform=ivl_tf,
            weight=args.llm_feedback_weight,
            max_new_tokens=args.llm_feedback_max_new_tokens,
            temperature=args.llm_feedback_temperature,
        )

    llm_prompt_refiner = None
    if args.prompt_refine:
        refiner_map = {
            "default": InternVLPromptRefiner,
            "affordance_demo": AffordanceDemoPromptRefiner,
            "visible_tool_demo": VisibleToolDemoPromptRefiner,
            "generic_substitute_demo": GenericSubstitutePromptRefiner,
            "structured_reasoning_demo": StructuredReasoningPromptRefiner,
        }
        refiner_cls = refiner_map[args.refine_mode]
        llm_prompt_refiner = refiner_cls(
            internvl_model=thinkdet.feature_extractor.internvl,
            tokenizer=thinkdet.feature_extractor.tokenizer,
            device=device,
            image_transform=ivl_tf,
            max_new_tokens=args.prompt_refine_max_new_tokens,
            temperature=args.prompt_refine_temperature,
            num_candidates=args.prompt_refine_max_variants,
        )

    fallback_cfg = FallbackPolicyConfig(
        min_top1=args.fallback_min_top1,
        min_margin=args.fallback_min_margin,
        min_gate=args.fallback_min_gate,
        feedback_improve_margin=args.llm_feedback_improve_margin,
        refine_improve_margin=args.prompt_refine_improve_margin,
        refine_require_semantic_preservation=args.prompt_refine_require_semantic_preservation,
        refine_min_shared_terms=args.prompt_refine_min_shared_terms,
    )
    selection_cfg = maybe_force_cfg(fallback_cfg, args)

    print("[4/4] Running image inference...")
    with torch.no_grad():
        base_query, base_out = run_baseline_outputs(
            baseline, image_pil, args.prompt, device, dino_tf
        )
        base_pmap = build_positive_map_for_query(
            baseline_tokenizer,
            baseline_special_tokens,
            base_query,
            max_text_len=512,
        )
        base_preds = outputs_to_predictions(base_out, base_pmap, image_pil.size, args.top_k)

        td_query, td_out, td_aux = run_thinkdet_outputs(
            thinkdet, image_pil, args.prompt, device, dino_tf, ivl_tf
        )
        td_pmap = build_positive_map_for_query(
            thinkdet_tokenizer,
            thinkdet_special_tokens,
            td_query,
            max_text_len=512,
        )
        primary_bundle = build_scored_bundle(
            "thinkdet",
            td_out,
            td_pmap,
            td_query,
            aux=td_aux,
        )
        primary_preds = bundle_to_predictions(primary_bundle, image_pil.size, args.top_k)
        primary_stats = summarize_scores(primary_bundle.scores, primary_bundle.aux)

        feedback_bundle = None
        feedback_preds = []
        feedback_stats = None
        run_feedback = False
        if llm_reranker is not None:
            run_feedback = args.force_feedback or (
                primary_stats["top1"] < selection_cfg.min_top1
                or primary_stats["margin"] < selection_cfg.min_margin
                or (
                    selection_cfg.min_gate is not None
                    and primary_stats["gate_mean"] < selection_cfg.min_gate
                )
            )
            if run_feedback:
                feedback_candidates = build_feedback_predictions(
                    td_out,
                    td_pmap,
                    image_pil.size,
                    top_k=args.llm_feedback_top_k,
                )
                feedback_preds = llm_reranker.rerank_predictions(
                    image_pil,
                    td_query,
                    feedback_candidates,
                )
                feedback_bundle = scored_candidate_from_rerank(
                    "thinkdet_llm_feedback",
                    feedback_preds,
                    aux=td_aux,
                    query_text=td_query,
                )
                feedback_stats = summarize_scores(feedback_bundle.scores, feedback_bundle.aux)

        refine_source_stats = feedback_stats or primary_stats
        run_refine = False
        refinements = []
        refined_candidates = []
        if llm_prompt_refiner is not None:
            run_refine = args.force_refine or (
                refine_source_stats["top1"] < selection_cfg.min_top1
                or refine_source_stats["margin"] < selection_cfg.min_margin
                or (
                    selection_cfg.min_gate is not None
                    and refine_source_stats["gate_mean"] < selection_cfg.min_gate
                )
            )
            if run_refine:
                raw_refinements = llm_prompt_refiner.refine_prompts(td_query, image_pil=image_pil)
                raw_reasoning_output = getattr(llm_prompt_refiner, "last_raw_response", None)
                seen_queries = set()
                for idx, refined_query in enumerate(raw_refinements, start=1):
                    norm_q = " ".join(refined_query.strip().split())
                    if not norm_q or norm_q in seen_queries or is_bad_refinement(norm_q):
                        continue
                    seen_queries.add(norm_q)
                    ref_q, ref_out, ref_aux = run_thinkdet_outputs(
                        thinkdet, image_pil, refined_query, device, dino_tf, ivl_tf
                    )
                    ref_pmap = build_positive_map_for_query(
                        thinkdet_tokenizer,
                        thinkdet_special_tokens,
                        ref_q,
                        max_text_len=512,
                    )
                    bundle = build_scored_bundle(
                        f"thinkdet_refine_{idx}",
                        ref_out,
                        ref_pmap,
                        ref_q,
                        aux=ref_aux,
                    )
                    preds = bundle_to_predictions(bundle, image_pil.size, args.top_k)
                    stats = summarize_scores(bundle.scores, bundle.aux)
                    refinements.append(
                        {
                            "name": bundle.name,
                            "query_text": ref_q,
                            "raw_generation": raw_reasoning_output,
                            "stats": stats,
                            "predictions": preds,
                        }
                    )
                    refined_candidates.append(bundle)
            else:
                raw_reasoning_output = None
        else:
            raw_reasoning_output = None

        rerank_fn = None
        if feedback_bundle is not None:
            rerank_fn = lambda _candidate_set, bundle=feedback_bundle: bundle

        selected_bundle = primary_bundle
        selected_info = {
            "selected_name": primary_bundle.name,
            "selected_query_text": primary_bundle.query_text,
            "selected_stage": "primary",
            "selected_stats": primary_stats,
            "trace": [],
        }
        if rerank_fn is not None or refined_candidates:
            selected_bundle, selected_info = apply_fallback_policy(
                primary=primary_bundle,
                rerank_fn=rerank_fn,
                refined_candidates=refined_candidates,
                config=selection_cfg,
            )

        selected_preds = bundle_to_predictions(selected_bundle, image_pil.size, args.top_k)

    baseline_path = os.path.join(args.output_dir, "baseline.jpg")
    thinkdet_path = os.path.join(args.output_dir, "thinkdet_primary.jpg")
    selected_path = os.path.join(args.output_dir, "fallback_selected.jpg")

    base_panel = save_panel(
        image_pil,
        base_preds,
        "GroundingDINO Baseline",
        baseline_path,
        args.max_draw,
    )
    primary_panel = save_panel(
        image_pil,
        primary_preds,
        "ThinkDet Primary",
        thinkdet_path,
        args.max_draw,
    )
    selected_title = (
        f"Fallback Selected ({selected_info['selected_stage']})"
        f" | {selected_info['selected_query_text']}"
    )
    selected_panel = save_panel(
        image_pil,
        selected_preds,
        selected_title,
        selected_path,
        args.selected_max_draw,
    )

    saved_refinement_images = []
    for idx, item in enumerate(refinements, start=1):
        path = os.path.join(args.output_dir, f"refined_{idx}.jpg")
        title = f"Refined {idx}: {item['query_text']}"
        save_panel(image_pil, item["predictions"], title, path, args.selected_max_draw)
        saved_refinement_images.append(path)

    if feedback_preds:
        feedback_path = os.path.join(args.output_dir, "feedback_reranked.jpg")
        save_panel(
            image_pil,
            feedback_preds,
            "LLM Feedback Rerank",
            feedback_path,
            args.max_draw,
        )
    else:
        feedback_path = None

    overview_path = os.path.join(args.output_dir, "overview.jpg")
    make_overview(image_pil, [base_panel, primary_panel, selected_panel], args.prompt, overview_path)

    result = {
        "image_path": args.image,
        "prompt": args.prompt,
        "device": str(device),
        "output_dir": args.output_dir,
        "settings": {
            "top_k": args.top_k,
            "max_draw": args.max_draw,
            "disable_feedback": bool(args.disable_feedback),
            "prompt_refine": bool(args.prompt_refine),
            "force_feedback": bool(args.force_feedback),
            "force_refine": bool(args.force_refine),
            "fallback_config": jsonable(selection_cfg.__dict__),
        },
        "checkpoint_meta": jsonable(ckpt_meta),
        "baseline": {
            "query_text": base_query,
            "predictions": jsonable(base_preds),
        },
        "thinkdet_primary": {
            "query_text": td_query,
            "stats": jsonable(primary_stats),
            "predictions": jsonable(primary_preds),
            "aux": jsonable(td_aux),
        },
        "feedback": {
            "evaluated": bool(feedback_bundle is not None),
            "predictions": jsonable(feedback_preds),
            "stats": jsonable(feedback_stats),
            "image_path": feedback_path,
        },
        "refinements": [
            {
                "name": item["name"],
                "query_text": item["query_text"],
                "raw_generation": item.get("raw_generation"),
                "stats": jsonable(item["stats"]),
                "predictions": jsonable(item["predictions"]),
            }
            for item in refinements
        ],
        "selection": jsonable(selected_info),
        "selected_predictions": jsonable(selected_preds),
        "artifacts": {
            "overview_image": overview_path,
            "baseline_image": baseline_path,
            "thinkdet_primary_image": thinkdet_path,
            "selected_image": selected_path,
            "refinement_images": saved_refinement_images,
        },
    }

    json_path = os.path.join(args.output_dir, "result.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

    print("\nSaved artifacts:")
    print(f"- JSON: {json_path}")
    print(f"- Overview: {overview_path}")
    print(f"- Selected stage: {selected_info['selected_stage']}")
    print(f"- Selected query: {selected_info['selected_query_text']}")
    if selected_preds:
        top1 = selected_preds[0]
        print(
            "- Selected top1: "
            f"score={float(top1.get('score', 0.0)):.4f} "
            f"box={json.dumps([round(float(v), 1) for v in box_key(top1)])}"
        )


if __name__ == "__main__":
    main()
