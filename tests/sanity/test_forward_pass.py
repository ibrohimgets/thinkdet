"""
Sanity check: full ThinkDet forward pass with the current API.

This validates the active path only:
image + text -> InternVL feature extraction -> TMA injection -> GroundingDINO.
"""

import sys

import numpy as np
from PIL import Image
import torch

sys.path.insert(0, "/home/iibrohimm/project/next_step")
sys.path.insert(0, "/home/iibrohimm/project/next_step/GroundingDINO/GroundingDINO")

from thinkdet.models.arch import ThinkDetModel
from thinkdet.models.projector import build_internvl_transform


def load_grounding_dino(device="cpu"):
    from groundingdino.util.inference import load_model

    config = "/home/iibrohimm/project/next_step/GroundingDINO/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    weights = "/home/iibrohimm/project/next_step/GroundingDINO/GroundingDINO/weights/groundingdino_swint_ogc.pth"
    return load_model(config, weights, device=device)


def prepare_dino_input(device="cpu"):
    from groundingdino.util.misc import NestedTensor
    import groundingdino.datasets.transforms as T

    pil_img = Image.fromarray(
        np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    )
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    img_tensor, _ = transform(pil_img, None)
    return NestedTensor(
        img_tensor.unsqueeze(0).to(device),
        torch.zeros(
            1,
            img_tensor.shape[1],
            img_tensor.shape[2],
            dtype=torch.bool,
            device=device,
        ),
    )


def test_forward_pass():
    print("=" * 60)
    print("SANITY CHECK: Full Forward Pass")
    print("=" * 60)

    internvl_path = "/home/iibrohimm/project/next_step/InternVL3_5-1B"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n[1/5] Loading GroundingDINO (device={device}) ...")
    gd_model = load_grounding_dino(device=device)
    print(f"       Decoder layers: {len(gd_model.transformer.decoder.layers)}")

    print(f"\n[2/5] Building ThinkDetModel (InternVL={internvl_path}) ...")
    model = ThinkDetModel(
        grounding_dino=gd_model,
        internvl_path=internvl_path,
        extract_layer=8,
        d_model=256,
        tma_n_heads=8,
    ).to(device).eval()

    print(f"       mllm_hidden_dim: {model.mllm_hidden_dim}")
    print(f"       adapted layers:  {len(model.adapted_layers)}")
    print(f"       injection at:    {model.injection_layers}")

    print("\n[3/5] Preparing inputs ...")
    dummy_pil = Image.fromarray(
        np.random.randint(0, 255, (448, 448, 3), dtype=np.uint8)
    )
    transform = build_internvl_transform(448)
    internvl_image = transform(dummy_pil).unsqueeze(0).to(device)
    text_queries = ["person . dog . car ."]
    dino_inputs = {
        "samples": prepare_dino_input(device=device),
        "captions": ["person . dog . car ."],
    }

    print(f"       InternVL image: {internvl_image.shape}")
    print(f"       DINO samples:   {dino_inputs['samples'].tensors.shape}")
    print(f"       Text query:     {text_queries[0]}")

    print("\n[4/5] Running full forward pass ...")
    with torch.no_grad():
        outputs, aux = model(internvl_image, text_queries, dino_inputs)
        outputs_gate0, aux_gate0 = model(
            internvl_image, text_queries, dino_inputs, force_gate0=True
        )

    print("\n[5/5] Validating outputs ...")
    assert "pred_logits" in outputs, "FAIL: 'pred_logits' missing from outputs"
    assert "pred_boxes" in outputs, "FAIL: 'pred_boxes' missing from outputs"
    assert "h_vlm" in aux, "FAIL: 'h_vlm' missing from aux"
    assert "gate_values" in aux, "FAIL: 'gate_values' missing from aux"

    pred_logits = outputs["pred_logits"]
    pred_boxes = outputs["pred_boxes"]
    h_vlm = aux["h_vlm"]
    gate_values = aux["gate_values"]

    print(f"  ✓ pred_logits shape: {pred_logits.shape}")
    print(f"  ✓ pred_boxes  shape: {pred_boxes.shape}")
    print(f"  ✓ h_vlm shape:       {h_vlm.shape}")
    print(f"  ✓ gate values:       {gate_values}")

    d = model.mllm_hidden_dim
    assert pred_logits.dim() == 3
    assert pred_boxes.dim() == 3
    assert pred_boxes.shape[-1] == 4
    assert h_vlm.shape == (1, 256, d)
    assert len(gate_values) == len(model.adapted_layers)

    assert not torch.isnan(pred_logits).any(), "FAIL: NaN in pred_logits"
    assert not torch.isnan(pred_boxes).any(), "FAIL: NaN in pred_boxes"
    assert not torch.isnan(h_vlm).any(), "FAIL: NaN in h_vlm"
    assert not (pred_logits == float("inf")).any().item(), "FAIL: +Inf in logits"

    assert aux_gate0["h_vlm"] is None
    assert aux_gate0["force_gate0"] is True
    assert outputs_gate0["pred_logits"].shape == pred_logits.shape
    assert outputs_gate0["pred_boxes"].shape == pred_boxes.shape

    print("\n" + "=" * 60)
    print("PASS: full forward path matches current ThinkDet API")
    print("=" * 60)


if __name__ == "__main__":
    test_forward_pass()
