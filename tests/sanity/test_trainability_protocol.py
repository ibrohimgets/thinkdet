"""
Sanity check: current trainability protocol for ThinkDetModel.

This replaces older gate-centric checks that no longer match the live API.
"""

import os
import sys
from unittest import mock

import torch.nn as nn

ROOT = "/home/iibrohimm/project/next_step"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "GroundingDINO", "GroundingDINO"))

from thinkdet.models.arch import ThinkDetModel


class MockGroundingDINO(nn.Module):
    def __init__(self):
        super().__init__()
        self.class_embed = nn.Linear(256, 10)
        self.bbox_embed = nn.Linear(256, 4)
        self.enc_output = nn.Linear(256, 256)
        self.enc_score_head = nn.Linear(256, 10)
        self.enc_bbox_head = nn.Linear(256, 4)
        self.transformer = nn.Module()
        self.transformer.decoder = nn.Module()
        self.transformer.decoder.layers = nn.ModuleList(
            [nn.Identity() for _ in range(6)]
        )

    def forward(self, **kwargs):
        return {}


def _head_params(model):
    raw = model.grounding_dino
    modules = [
        raw.class_embed,
        raw.bbox_embed,
        raw.enc_output,
        raw.enc_score_head,
        raw.enc_bbox_head,
    ]
    return [p for module in modules for p in module.parameters()]


def _all_require_grad(params):
    return all(p.requires_grad for p in params)


def _all_frozen(params):
    return all(not p.requires_grad for p in params)


def test_trainability_protocol():
    print("=" * 60)
    print("SANITY CHECK: Trainability Protocol")
    print("=" * 60)

    gd = MockGroundingDINO()
    with mock.patch("thinkdet.models.arch.InternVLFeatureExtractor") as mock_feat:
        mock_feat.return_value.llm_hidden_dim = 1024

        model = ThinkDetModel(
            grounding_dino=gd,
            internvl_path="dummy",
            extract_layer=8,
            d_model=256,
            tma_n_heads=2,
            injection_layers=[1, 3, 5],
            fusion_mode="residual",
        )

    head_params = _head_params(model)
    tma_params = model.get_augmenter_params()

    model.set_tma_only()
    assert _all_require_grad(tma_params), "TMA params should train in Stage 1"
    assert _all_frozen(head_params), "Heads should stay frozen in Stage 1"
    print("  PASS: set_tma_only keeps only TMA/fusion params trainable")

    model.set_tma_and_heads()
    assert _all_require_grad(tma_params), "TMA params should train in Stage 2"
    assert _all_require_grad(head_params), "Heads should train in Stage 2"
    print("  PASS: set_tma_and_heads enables TMA + heads")

    print("=" * 60)
    print("ALL TRAINABILITY CHECKS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    test_trainability_protocol()
