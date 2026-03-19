"""
Sanity Check 1: Text Conditioning

Same image + different prompts → H_vlm must change.
If not → the VLM is ignoring text, and the whole v2 design is broken.
"""

import sys
import torch
import numpy as np
from PIL import Image

sys.path.insert(0, '/home/iibrohimm/project/next_step')

from thinkdet.models.projector import InternVLFeatureExtractor, build_internvl_transform


def test_text_conditioning():
    print("=" * 60)
    print("SANITY CHECK 1: Text Conditioning")
    print("=" * 60)

    # ── Config ──
    INTERNVL_PATH = "/home/iibrohimm/project/next_step/InternVL3_5-1B"
    EXTRACT_LAYER = 8
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Load feature extractor ──
    print(f"\n[1/3] Loading InternVL from {INTERNVL_PATH} ...")
    extractor = InternVLFeatureExtractor(
        model_path=INTERNVL_PATH,
        extract_layer=EXTRACT_LAYER,
        freeze=True,
        use_flash_attn=False,
    ).to(DEVICE).eval()

    print(f"       LLM hidden dim = {extractor.llm_hidden_dim}")
    print(f"       num_image_token = {extractor.num_image_token}")
    print(f"       num_llm_layers  = {extractor.num_llm_layers}")

    # ── Create a dummy image (random noise as PIL) ──
    print("\n[2/3] Creating dummy image and two different prompts ...")
    dummy_pil = Image.fromarray(np.random.randint(0, 255, (448, 448, 3), dtype=np.uint8))
    transform = build_internvl_transform(448)
    pixel_values = transform(dummy_pil).unsqueeze(0).to(DEVICE)  # [1, 3, 448, 448]

    prompt_a = "person . dog . car ."
    prompt_b = "airplane . mountain . river ."

    # ── Forward with prompt A ──
    print("\n[3/3] Running forward passes ...")
    with torch.no_grad():
        h_vlm_a = extractor(pixel_values, [prompt_a])  # [1, 256, D]
        h_vlm_b = extractor(pixel_values, [prompt_b])  # [1, 256, D]

    # ── Check shapes ──
    D = extractor.llm_hidden_dim
    assert h_vlm_a.shape == (1, 256, D), f"Shape mismatch: {h_vlm_a.shape} vs expected (1, 256, {D})"
    assert h_vlm_b.shape == (1, 256, D), f"Shape mismatch: {h_vlm_b.shape} vs expected (1, 256, {D})"
    print(f"  ✓ Shape: {h_vlm_a.shape}")

    # ── Check that outputs differ ──
    cos_sim = torch.nn.functional.cosine_similarity(
        h_vlm_a.flatten(), h_vlm_b.flatten(), dim=0
    ).item()

    l2_diff = (h_vlm_a - h_vlm_b).norm().item()
    max_diff = (h_vlm_a - h_vlm_b).abs().max().item()

    print(f"\n  Cosine similarity:  {cos_sim:.6f}")
    print(f"  L2 distance:        {l2_diff:.4f}")
    print(f"  Max element diff:   {max_diff:.6f}")

    # They should be different — NOT identical
    assert cos_sim < 0.9999, (
        f"FAIL: H_vlm is nearly identical for different prompts "
        f"(cosine={cos_sim:.6f}). Text conditioning is NOT working."
    )
    assert l2_diff > 1e-4, (
        f"FAIL: L2 diff is {l2_diff:.8f}, too small. "
        f"Text conditioning is NOT working."
    )

    print("\n" + "=" * 60)
    print("✅ PASS: Different prompts → different H_vlm")
    print(f"   Text conditioning is working (cosine={cos_sim:.4f}, L2={l2_diff:.2f})")
    print("=" * 60)


if __name__ == "__main__":
    test_text_conditioning()
