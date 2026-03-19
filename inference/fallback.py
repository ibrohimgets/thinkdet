"""Inference-time LLM feedback and refinement helpers.

This module is for inference-only recovery when ThinkDet looks uncertain.
The intended routing is:

1. Run ThinkDet normally.
2. If the result looks weak, ask the MLLM to rerank detector candidates.
3. If it still looks weak, ask the MLLM to rewrite the prompt and retry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch


GENERIC_QUERY_STOPWORDS = {
    "a",
    "an",
    "the",
    "this",
    "that",
    "these",
    "those",
    "and",
    "or",
    "but",
    "if",
    "then",
    "else",
    "with",
    "without",
    "for",
    "from",
    "into",
    "onto",
    "over",
    "under",
    "near",
    "beside",
    "around",
    "through",
    "across",
    "between",
    "of",
    "to",
    "in",
    "on",
    "at",
    "by",
    "is",
    "are",
    "was",
    "were",
    "be",
    "being",
    "been",
    "it",
    "its",
    "their",
    "there",
    "here",
    "something",
    "someone",
    "somebody",
    "thing",
    "things",
    "object",
    "objects",
    "item",
    "items",
}


@dataclass
class ScoredCandidateSet:
    """Prediction candidate set with enough metadata for fallback routing."""

    name: str
    scores: Sequence[float]
    payload: Any = None
    aux: Optional[Dict[str, Any]] = None
    query_text: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConfidenceCalibrator:
    """Lightweight post-hoc confidence mapper over score diagnostics."""

    feature_names: Sequence[str]
    feature_means: Sequence[float]
    feature_stds: Sequence[float]
    weights: Sequence[float]
    bias: float
    source_path: Optional[str] = None

    def score_stats(self, stats: Optional[Dict[str, float]]) -> Optional[float]:
        if not stats:
            return None

        logit = float(self.bias)
        for name, mean, std, weight in zip(
            self.feature_names,
            self.feature_means,
            self.feature_stds,
            self.weights,
        ):
            value = stats.get(name)
            if value is None:
                return None
            scale = float(std) if abs(float(std)) >= 1e-8 else 1.0
            z = (float(value) - float(mean)) / scale
            logit += float(weight) * z

        logit = max(min(logit, 50.0), -50.0)
        return 1.0 / (1.0 + math.exp(-logit))


@dataclass
class FallbackPolicyConfig:
    """Thresholds controlling when to keep primary vs try LLM fallback."""

    min_top1: float = 0.20
    min_confidence: Optional[float] = None
    min_margin: float = 0.02
    min_gate: Optional[float] = None
    confidence_calibrator: Optional[ConfidenceCalibrator] = field(default=None, repr=False)
    confidence_calibrator_path: Optional[str] = None
    feedback_improve_margin: float = 0.01
    refine_improve_margin: float = 0.01
    refine_require_semantic_preservation: bool = True
    refine_min_shared_terms: int = 1


def load_confidence_calibrator(path: str) -> ConfidenceCalibrator:
    with open(path, "r") as f:
        payload = json.load(f)

    feature_names = payload.get("feature_names") or []
    feature_means = payload.get("feature_means") or []
    feature_stds = payload.get("feature_stds") or []
    weights = payload.get("weights") or []
    bias = payload.get("bias")

    n = len(feature_names)
    if not n:
        raise ValueError(f"Invalid calibrator at {path}: missing feature_names")
    if len(feature_means) != n or len(feature_stds) != n or len(weights) != n:
        raise ValueError(
            f"Invalid calibrator at {path}: feature lengths do not match "
            f"(names={n}, means={len(feature_means)}, stds={len(feature_stds)}, weights={len(weights)})"
        )
    if bias is None:
        raise ValueError(f"Invalid calibrator at {path}: missing bias")

    return ConfidenceCalibrator(
        feature_names=[str(x) for x in feature_names],
        feature_means=[float(x) for x in feature_means],
        feature_stds=[float(x) for x in feature_stds],
        weights=[float(x) for x in weights],
        bias=float(bias),
        source_path=str(path),
    )


def normalize_query(text: str) -> str:
    text = " ".join(str(text).strip().lower().split())
    text = re.sub(r"\s*\.\s*$", "", text)
    return f"{text or 'object'} ."


def parse_llm_refinements(text: str, limit: int = 3) -> List[str]:
    """Parse one-query-per-line MLLM rewrite output into normalized prompts."""

    lines = re.split(r"[\n;]+", text or "")
    cleaned: List[str] = []
    seen = set()

    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        line = re.sub(r"^\s*(?:[-*•]|\d+[.)]|[A-Za-z][.)])\s*", "", line)
        line = line.strip().strip("\"'`")
        if not line:
            continue
        lowered = line.lower()
        if lowered.startswith(("here are", "rewrites", "alternatives", "output")):
            continue
        query = normalize_query(line)
        if query in seen:
            continue
        seen.add(query)
        cleaned.append(query)
        if len(cleaned) >= int(limit):
            break

    return cleaned


def normalize_term(text: str) -> str:
    token = re.sub(r"[^a-z0-9]+", "", str(text).strip().lower())
    if len(token) > 4 and token.endswith("ies"):
        token = token[:-3] + "y"
    elif len(token) > 5 and token.endswith("ing"):
        token = token[:-3]
    elif len(token) > 4 and token.endswith("ed"):
        token = token[:-2]
    elif len(token) > 4 and token.endswith("es"):
        token = token[:-2]
    elif len(token) > 3 and token.endswith("s"):
        token = token[:-1]
    return token


def extract_content_terms(text: Optional[str]) -> List[str]:
    if not text:
        return []

    terms: List[str] = []
    seen = set()
    for raw in re.findall(r"[A-Za-z0-9']+", text.lower()):
        term = normalize_term(raw)
        if not term or term in GENERIC_QUERY_STOPWORDS or len(term) <= 1:
            continue
        if term in seen:
            continue
        seen.add(term)
        terms.append(term)
    return terms


def semantic_preservation_check(
    source_query: Optional[str],
    refined_query: Optional[str],
    required_terms: Optional[Sequence[str]] = None,
    min_shared_terms: int = 1,
) -> Tuple[bool, Dict[str, Any]]:
    if not source_query or not refined_query:
        return True, {"reason": "missing_query_text"}

    source_terms = set(extract_content_terms(source_query))
    refined_terms = set(extract_content_terms(refined_query))
    required = {normalize_term(x) for x in (required_terms or []) if normalize_term(x)}
    shared = sorted(source_terms & refined_terms)
    matched_required = sorted(required & refined_terms)

    if required and not matched_required:
        return False, {
            "reason": "missing_required_terms",
            "source_terms": sorted(source_terms),
            "refined_terms": sorted(refined_terms),
            "shared_terms": shared,
            "required_terms": sorted(required),
            "matched_required_terms": matched_required,
        }

    if min_shared_terms > 0 and len(shared) < int(min_shared_terms):
        return False, {
            "reason": "low_term_overlap",
            "source_terms": sorted(source_terms),
            "refined_terms": sorted(refined_terms),
            "shared_terms": shared,
            "required_terms": sorted(required),
            "matched_required_terms": matched_required,
        }

    return True, {
        "reason": "ok",
        "source_terms": sorted(source_terms),
        "refined_terms": sorted(refined_terms),
        "shared_terms": shared,
        "required_terms": sorted(required),
        "matched_required_terms": matched_required,
    }


def summarize_scores(scores: Sequence[float], aux: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
    vals = sorted((float(x) for x in scores), reverse=True)
    top1 = vals[0] if vals else 0.0
    top2 = vals[1] if len(vals) > 1 else 0.0
    top3_mean = sum(vals[:3]) / float(min(3, len(vals))) if vals else 0.0
    margin = top1 - top2

    gate_mean = 0.0
    if aux:
        if aux.get("gate_mean") is not None:
            gate_mean = float(aux["gate_mean"])
        elif aux.get("alpha_mean") is not None:
            gate_mean = float(aux["alpha_mean"])

    reliability = top1 + 0.50 * margin + 0.10 * top3_mean
    return {
        "top1": top1,
        "top2": top2,
        "top3_mean": top3_mean,
        "margin": margin,
        "gate_mean": gate_mean,
        "reliability": reliability,
    }


def apply_confidence_calibrator_to_stats(
    stats: Dict[str, float],
    calibrator: Optional[ConfidenceCalibrator],
) -> Dict[str, float]:
    if not stats or calibrator is None:
        return stats
    if stats.get("calibrated_confidence") is not None:
        return stats

    calibrated = calibrator.score_stats(stats)
    if calibrated is None:
        return stats

    stats["calibrated_confidence"] = float(calibrated)
    return stats


def is_unreliable(stats: Dict[str, float], cfg: FallbackPolicyConfig) -> bool:
    apply_confidence_calibrator_to_stats(stats, cfg.confidence_calibrator)

    use_calibrated_confidence = (
        cfg.min_confidence is not None
        and stats.get("calibrated_confidence") is not None
    )
    if use_calibrated_confidence:
        if float(stats["calibrated_confidence"]) < float(cfg.min_confidence):
            return True
    elif stats["top1"] < cfg.min_top1:
        return True
    if stats["margin"] < cfg.min_margin:
        return True
    if cfg.min_gate is not None and stats["gate_mean"] < cfg.min_gate:
        return True
    return False


def apply_fallback_policy(
    primary: ScoredCandidateSet,
    rerank_fn: Optional[Callable[[ScoredCandidateSet], Optional[ScoredCandidateSet]]] = None,
    refined_candidates: Optional[Sequence[ScoredCandidateSet]] = None,
    config: Optional[FallbackPolicyConfig] = None,
) -> Tuple[ScoredCandidateSet, Dict[str, Any]]:
    """Choose the best available result using LLM-only fallback stages."""

    cfg = config or FallbackPolicyConfig()
    stats_by_name: Dict[str, Dict[str, float]] = {}

    def _record(candidate: ScoredCandidateSet) -> Dict[str, float]:
        stats = summarize_scores(candidate.scores, candidate.aux)
        payload = candidate.payload if isinstance(candidate.payload, dict) else None
        payload_type = payload.get("type") if payload else None
        if payload_type != "reranked_preds":
            apply_confidence_calibrator_to_stats(stats, cfg.confidence_calibrator)
        stats_by_name[candidate.name] = stats
        return stats

    primary_stats = _record(primary)
    selected = primary
    selected_stats = primary_stats
    selected_stage = "primary"
    trace = [
        {
            "stage": "primary",
            "candidate": primary.name,
            "selected": True,
            "stats": primary_stats,
        }
    ]

    if rerank_fn is not None and is_unreliable(selected_stats, cfg):
        reranked = rerank_fn(selected)
        if reranked is not None:
            reranked_stats = _record(reranked)
            if (
                reranked_stats["reliability"]
                >= selected_stats["reliability"] + cfg.feedback_improve_margin
            ):
                selected = reranked
                selected_stats = reranked_stats
                selected_stage = "llm_feedback"
            trace.append(
                {
                    "stage": "feedback_check",
                    "candidate": reranked.name,
                    "selected": selected is reranked,
                    "stats": reranked_stats,
                }
            )

    if refined_candidates and is_unreliable(selected_stats, cfg):
        refined_stats = []
        best_refined = None
        best_refined_stats = None

        for candidate in refined_candidates:
            semantic_info = None
            if cfg.refine_require_semantic_preservation:
                semantic_ok, semantic_info = semantic_preservation_check(
                    primary.query_text,
                    candidate.query_text,
                    required_terms=(
                        candidate.meta.get("semantic_anchor_terms")
                        or primary.meta.get("semantic_anchor_terms")
                    ),
                    min_shared_terms=cfg.refine_min_shared_terms,
                )
                if not semantic_ok:
                    refined_stats.append(
                        {
                            "candidate": candidate.name,
                            "selected": False,
                            "semantic_guard_ok": False,
                            "semantic_guard": semantic_info,
                        }
                    )
                    continue
            stats = _record(candidate)
            refined_stats.append(
                {
                    "candidate": candidate.name,
                    "stats": stats,
                    "semantic_guard_ok": True,
                    "semantic_guard": semantic_info,
                }
            )
            if best_refined is None or stats["reliability"] > best_refined_stats["reliability"]:
                best_refined = candidate
                best_refined_stats = stats

        if (
            best_refined is not None
            and best_refined_stats["reliability"]
            >= selected_stats["reliability"] + cfg.refine_improve_margin
        ):
            selected = best_refined
            selected_stats = best_refined_stats
            selected_stage = "prompt_refine"

        trace.append(
            {
                "stage": "refine_check",
                "selected": selected_stage == "prompt_refine",
                "candidates": refined_stats,
            }
        )

    info = {
        "selected_name": selected.name,
        "selected_stage": selected_stage,
        "selected_query_text": selected.query_text,
        "selected_stats": selected_stats,
        "stats_by_name": stats_by_name,
        "refine_semantic_guard_enabled": bool(cfg.refine_require_semantic_preservation),
        "trace": trace,
    }
    return selected, info


class InternVLYesNoReranker:
    """Candidate-box reranker using InternVL yes/no feedback."""

    def __init__(
        self,
        internvl_model,
        tokenizer,
        device,
        image_transform,
        weight: float = 0.2,
        max_new_tokens: int = 6,
        temperature: float = 0.0,
    ):
        self.model = internvl_model
        self.tokenizer = tokenizer
        self.device = device
        self.image_transform = image_transform
        self.weight = float(weight)
        self.base_generation_config = {
            "max_new_tokens": int(max_new_tokens),
            "do_sample": bool(temperature > 0),
            "top_p": 1.0,
            "pad_token_id": int(getattr(tokenizer, "eos_token_id", 0) or 0),
        }
        if temperature > 0:
            self.base_generation_config["temperature"] = float(temperature)

    @staticmethod
    def _clip_box(box, w, h):
        x1, y1, x2, y2 = box
        x1 = max(0.0, min(float(w - 1), float(x1)))
        y1 = max(0.0, min(float(h - 1), float(y1)))
        x2 = max(0.0, min(float(w), float(x2)))
        y2 = max(0.0, min(float(h), float(y2)))
        return [x1, y1, x2, y2]

    @staticmethod
    def _llm_text_to_score(text: str) -> float:
        t = text.strip().lower()
        if t.startswith("yes"):
            return 1.0
        if t.startswith("no"):
            return 0.0
        has_yes = "yes" in t
        has_no = "no" in t
        if has_yes and not has_no:
            return 1.0
        if has_no and not has_yes:
            return 0.0
        return 0.5

    @torch.no_grad()
    def rerank_predictions(
        self,
        image_pil,
        prompt: str,
        preds: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        preds = [dict(p) for p in preds]
        if not preds:
            return preds

        w, h = image_pil.size
        crops = []
        questions = []
        valid_idx = []

        for i, pred in enumerate(preds):
            x1, y1, x2, y2 = self._clip_box(pred["box_abs_xyxy"], w, h)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = image_pil.crop((int(x1), int(y1), int(x2), int(y2)))
            crops.append(self.image_transform(crop))
            questions.append(
                "Answer only yes or no. "
                f"Does this crop contain the target described by the query: {prompt}"
            )
            valid_idx.append(i)

        if not crops:
            for pred in preds:
                pred["llm_score"] = 0.5
                pred["combined_score"] = float(pred["score"]) + self.weight * 0.5
            return sorted(preds, key=lambda r: r["combined_score"], reverse=True)

        pixel_values = torch.stack(crops, dim=0).to(self.device)
        responses = self.model.batch_chat(
            tokenizer=self.tokenizer,
            pixel_values=pixel_values,
            questions=questions,
            generation_config=dict(self.base_generation_config),
            num_patches_list=[1] * len(crops),
            verbose=False,
        )

        llm_scores = [0.5] * len(preds)
        for idx_in_valid, pred_idx in enumerate(valid_idx):
            llm_scores[pred_idx] = self._llm_text_to_score(responses[idx_in_valid])

        for i, pred in enumerate(preds):
            llm_score = float(llm_scores[i])
            pred["llm_score"] = llm_score
            pred["combined_score"] = float(pred["score"]) + self.weight * llm_score

        return sorted(preds, key=lambda r: r["combined_score"], reverse=True)


class InternVLPromptRefiner:
    """LLM-based prompt refiner for a second-pass detection attempt."""

    def __init__(
        self,
        internvl_model,
        tokenizer,
        device,
        image_transform=None,
        max_new_tokens: int = 96,
        temperature: float = 0.0,
        num_candidates: int = 3,
    ):
        self.model = internvl_model
        self.tokenizer = tokenizer
        self.device = device
        self.image_transform = image_transform
        self.num_candidates = int(num_candidates)
        self.generation_config = {
            "max_new_tokens": int(max_new_tokens),
            "do_sample": bool(temperature > 0),
            "top_p": 1.0,
            "pad_token_id": int(getattr(tokenizer, "eos_token_id", 0) or 0),
        }
        if temperature > 0:
            self.generation_config["temperature"] = float(temperature)

    def _build_question(self, query_text: str, has_image: bool) -> str:
        prefix = "<image>\n" if has_image else ""
        return (
            f"{prefix}"
            f"Rewrite this grounding query into up to {self.num_candidates} clearer alternatives.\n"
            "Rules:\n"
            "- preserve the original meaning\n"
            "- use visible object words, attributes, or spatial relations\n"
            "- keep each alternative short and literal\n"
            "- output one query per line only\n"
            f"Original query: {normalize_query(query_text)}"
        )

    @torch.no_grad()
    def refine_prompts(self, query_text: str, image_pil=None) -> List[str]:
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
        return parse_llm_refinements(response, limit=self.num_candidates)
