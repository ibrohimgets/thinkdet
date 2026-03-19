"""
Sanity checks for inference-time LLM fallback routing.

These tests stay lightweight: they validate routing and parsing logic without
requiring GroundingDINO or InternVL weights.
"""

import sys

sys.path.insert(0, "/home/iibrohimm/project/next_step")

from thinkdet.inference.fallback import (  # noqa: E402
    FallbackPolicyConfig,
    ScoredCandidateSet,
    apply_fallback_policy,
    semantic_preservation_check,
    parse_llm_refinements,
)


def test_primary_kept_when_confident():
    cfg = FallbackPolicyConfig(min_top1=0.20, min_margin=0.02)
    primary = ScoredCandidateSet("thinkdet", scores=[0.62, 0.40, 0.31])

    selected, info = apply_fallback_policy(primary=primary, config=cfg)

    assert selected.name == "thinkdet"
    assert info["selected_stage"] == "primary"


def test_llm_feedback_can_replace_weak_primary():
    cfg = FallbackPolicyConfig(
        min_top1=0.20,
        min_margin=0.05,
        feedback_improve_margin=0.01,
    )
    primary = ScoredCandidateSet("thinkdet", scores=[0.19, 0.18, 0.17])
    reranked = ScoredCandidateSet("thinkdet_llm_feedback", scores=[0.37, 0.20, 0.10])

    selected, info = apply_fallback_policy(
        primary=primary,
        rerank_fn=lambda _: reranked,
        config=cfg,
    )

    assert selected.name == "thinkdet_llm_feedback"
    assert info["selected_stage"] == "llm_feedback"


def test_prompt_refinement_can_win_after_feedback():
    cfg = FallbackPolicyConfig(
        min_top1=0.25,
        min_margin=0.05,
        feedback_improve_margin=0.01,
        refine_improve_margin=0.01,
    )
    primary = ScoredCandidateSet("thinkdet", scores=[0.20, 0.19, 0.18], query_text="person in red shirt .")
    reranked = ScoredCandidateSet("thinkdet_llm_feedback", scores=[0.22, 0.20, 0.18])
    refined = ScoredCandidateSet(
        "thinkdet_refined",
        scores=[0.34, 0.20, 0.11],
        query_text="red shirt person .",
    )

    selected, info = apply_fallback_policy(
        primary=primary,
        rerank_fn=lambda _: reranked,
        refined_candidates=[refined],
        config=cfg,
    )

    assert selected.name == "thinkdet_refined"
    assert info["selected_stage"] == "prompt_refine"


def test_prompt_refinement_is_blocked_when_meaning_drifts():
    cfg = FallbackPolicyConfig(
        min_top1=0.25,
        min_margin=0.05,
        refine_improve_margin=0.01,
        refine_require_semantic_preservation=True,
        refine_min_shared_terms=1,
    )
    primary = ScoredCandidateSet(
        "thinkdet",
        scores=[0.18, 0.17, 0.16],
        query_text="something to talk on .",
    )
    refined = ScoredCandidateSet(
        "thinkdet_refined",
        scores=[0.42, 0.21, 0.10],
        query_text="a person holding a remote control .",
    )

    selected, info = apply_fallback_policy(
        primary=primary,
        refined_candidates=[refined],
        config=cfg,
    )

    assert selected.name == "thinkdet"
    assert info["selected_stage"] == "primary"
    refine_trace = next(x for x in info["trace"] if x["stage"] == "refine_check")
    assert refine_trace["candidates"][0]["semantic_guard_ok"] is False


def test_semantic_preservation_check_keeps_shared_action_term():
    ok, info = semantic_preservation_check(
        "something to carry things in .",
        "a person carrying a suitcase is crossing the street .",
        min_shared_terms=1,
    )

    assert ok is True
    assert "carry" in info["shared_terms"]


def test_parse_llm_refinements_strips_numbering_and_dedupes():
    text = """
    1. red shirt person
    2) person in red shirt
    - red shirt person
    """
    variants = parse_llm_refinements(text, limit=3)

    assert variants == [
        "red shirt person .",
        "person in red shirt .",
    ]


if __name__ == "__main__":
    test_primary_kept_when_confident()
    test_llm_feedback_can_replace_weak_primary()
    test_prompt_refinement_can_win_after_feedback()
    test_prompt_refinement_is_blocked_when_meaning_drifts()
    test_semantic_preservation_check_keeps_shared_action_term()
    test_parse_llm_refinements_strips_numbering_and_dedupes()
    print("fallback policy sanity checks passed")
