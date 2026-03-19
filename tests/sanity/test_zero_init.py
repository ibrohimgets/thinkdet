"""
Sanity Check: Residual Fusion Zero-Init Identity

This validates the new baseline-preserving residual fusion path:
1) ResidualTextMemoryFusion starts as exact identity on memory_text.
2) ThinkDetDecoderLayer(residual mode) preserves baseline output at init.

Why this matters:
If these checks fail, the residual-fusion pivot cannot guarantee COCO/RefCOCO
non-regression at step 0.
"""

import sys
import torch
import torch.nn as nn

sys.path.insert(0, "/home/iibrohimm/project/next_step")
sys.path.insert(0, "/home/iibrohimm/project/next_step/GroundingDINO/GroundingDINO")

from thinkdet.models.cross_attention import ThinkDetTextAugmenter
from thinkdet.models.decoder_layer import ResidualTextMemoryFusion, ThinkDetDecoderLayer


class MockDecoderLayer(nn.Module):
    """
    Minimal stand-in for GroundingDINO decoder layer.

    Uses `memory_text` so wrapper-induced changes would show up in the output.
    """

    def __init__(self, d_model=256):
        super().__init__()
        self.proj_tgt = nn.Linear(d_model, d_model)
        self.proj_txt = nn.Linear(d_model, d_model)
        nn.init.eye_(self.proj_tgt.weight)
        nn.init.zeros_(self.proj_tgt.bias)
        nn.init.eye_(self.proj_txt.weight)
        nn.init.zeros_(self.proj_txt.bias)

    def forward(self, tgt, memory_text=None, text_attention_mask=None, **kwargs):
        out = self.proj_tgt(tgt)
        if memory_text is not None:
            # Simple deterministic dependence on text memory.
            txt = self.proj_txt(memory_text.mean(dim=1, keepdim=True))
            out = out + txt
        return out


def _max_abs_diff(a, b):
    return float((a - b).abs().max().item())


def test_residual_fuser_identity():
    print("=" * 60)
    print("SANITY CHECK: ResidualTextMemoryFusion Zero-Init Identity")
    print("=" * 60)

    torch.manual_seed(0)
    device = "cpu"
    B, T, M, D = 2, 12, 8, 256

    fuser = ResidualTextMemoryFusion(d_model=D, n_heads=8, hidden_mult=2).to(device).eval()
    memory_text = torch.randn(B, T, D, device=device)
    aug_tokens = torch.randn(B, M, D, device=device)

    with torch.no_grad():
        out = fuser(memory_text, aug_tokens)

    max_diff = _max_abs_diff(out, memory_text)
    print(f"  memory_text shape: {tuple(memory_text.shape)}")
    print(f"  aug_tokens  shape: {tuple(aug_tokens.shape)}")
    print(f"  max abs diff:      {max_diff:.10f}")
    assert torch.equal(out, memory_text), (
        f"ResidualTextMemoryFusion is not exact identity at init (max_diff={max_diff:.10f})"
    )
    print("  PASS: fuser output is exactly equal to input memory_text at init")


def test_decoder_wrapper_residual_identity():
    print("\n" + "=" * 60)
    print("SANITY CHECK: ThinkDetDecoderLayer Residual Mode Identity")
    print("=" * 60)

    torch.manual_seed(1)
    device = "cpu"
    B, NQ, T, D = 2, 20, 11, 256
    NVIS, MLLM_DIM = 32, 1024

    original = MockDecoderLayer(d_model=D).to(device).eval()
    augmenter = ThinkDetTextAugmenter(
        mllm_hidden_dim=MLLM_DIM,
        d_model=D,
        M=8,
        n_heads=8,
        alpha_init=0.3,
    ).to(device).eval()
    wrapped = ThinkDetDecoderLayer(
        original_layer=original,
        augmenter=augmenter,
        fusion_mode="residual",
        residual_hidden_mult=2,
    ).to(device).eval()

    tgt = torch.randn(B, NQ, D, device=device)
    memory_text = torch.randn(B, T, D, device=device)
    text_attention_mask = torch.zeros(B, T, dtype=torch.bool, device=device)
    h_vlm = torch.randn(B, NVIS, MLLM_DIM, device=device)

    with torch.no_grad():
        baseline_out = original(
            tgt=tgt,
            memory_text=memory_text,
            text_attention_mask=text_attention_mask,
        )
        wrapped.set_h_vlm(h_vlm)
        thinkdet_out = wrapped(
            tgt=tgt,
            memory_text=memory_text,
            text_attention_mask=text_attention_mask,
        )
        wrapped.clear_h_vlm()

    max_diff = _max_abs_diff(baseline_out, thinkdet_out)
    print(f"  tgt shape:         {tuple(tgt.shape)}")
    print(f"  memory_text shape: {tuple(memory_text.shape)}")
    print(f"  h_vlm shape:       {tuple(h_vlm.shape)}")
    print(f"  max abs diff:      {max_diff:.10f}")
    assert torch.equal(baseline_out, thinkdet_out), (
        f"Residual ThinkDetDecoderLayer changed baseline output at init (max_diff={max_diff:.10f})"
    )
    print("  PASS: wrapped layer output exactly matches baseline at init")

    # Bonus activation check: perturb zero-init residual output and confirm effect.
    last = wrapped.residual_fuser.delta_mlp[-1]
    with torch.no_grad():
        last.bias.fill_(1e-2)
        wrapped.set_h_vlm(h_vlm)
        modified_out = wrapped(
            tgt=tgt,
            memory_text=memory_text,
            text_attention_mask=text_attention_mask,
        )
        wrapped.clear_h_vlm()
    mod_diff = _max_abs_diff(baseline_out, modified_out)
    print(f"  bonus max diff after enabling residual bias: {mod_diff:.6f}")
    assert mod_diff > 0.0, "Residual branch perturbation did not affect output"
    print("  PASS: residual branch can modify output when zero-init is broken (as expected)")


def main():
    test_residual_fuser_identity()
    test_decoder_wrapper_residual_identity()
    print("\n" + "=" * 60)
    print("ALL RESIDUAL ZERO-INIT IDENTITY CHECKS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
