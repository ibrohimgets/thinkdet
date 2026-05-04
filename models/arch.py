"""
ThinkDet TMA - Main Architecture

Wires together:
    1. GroundingDINO (frozen backbone)
    2. InternVLFeatureExtractor (extracts H_vlm from layer 9)
    3. ThinkDetDecoderLayer wrappers at injection_layers
       Each wrapper fuses InternVL summary tokens into memory_text before
       DINO's native `ca_text` cross-attention runs.

Fusion modes:
    - residual (default): zero-init residual delta, same sequence length
    - concat   (legacy): append TMA tokens to memory_text

Training stages:
    Stage 1 (TMA warmup): Only TMA/fusion parameters trainable
    Stage 2 (joint):      TMA/fusion + detection heads
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple

from .projector import InternVLFeatureExtractor
from .cross_attention import ThinkDetTextAugmenter
from .decoder_layer import ThinkDetDecoderLayer


# DINO decoder layers where TMA injection is applied (legacy default).
DEFAULT_INJECTION_LAYERS = [1, 3, 5]

# Default TMA config
DEFAULT_TMA_M = 8       # summary tokens per injection layer
DEFAULT_TMA_HEADS = 8   # attention heads in augmenter


class ThinkDetModel(nn.Module):
    """
    ThinkDet TMA: VLM-Guided Object Detection via Text Memory Augmentation

    InternVL3.5-1B extracts text-conditioned visual features H_vlm,
    which are summarized into M=8 tokens and fused into DINO's
    memory_text at each injection layer. DINO's own ca_text then
    attends to reasoning-augmented text memory.

    Args:
        grounding_dino:   Pre-loaded GroundingDINO model
        internvl_path:    Path to InternVL model
        extract_layer:    LLM layer to extract H_vlm from
        extract_layers:   Optional list of LLM layers to fuse
        layer_fusion:     Fusion strategy ('mean', 'last', or 'learned')
        injection_layers: DINO decoder layer indices to augment
        d_model:          GroundingDINO hidden dim (256)
        mllm_hidden_dim:  InternVL LLM hidden dim (None = auto-detect)
        tma_m:            Number of summary tokens per injection layer
        tma_n_heads:      Attention heads in TextAugmenter
        tma_alpha_init:   Initial scaling for injected summary tokens
        fusion_mode:      'residual' for the diagram path, 'concat' for legacy runs
    """

    def __init__(
        self,
        grounding_dino,
        internvl_path: str,
        extract_layer: int = 9,
        extract_layers: Optional[List[int]] = None,
        layer_fusion: str = 'mean',
        injection_layers: Optional[List[int]] = None,
        d_model: int = 256,
        mllm_hidden_dim: Optional[int] = None,
        tma_m: int = DEFAULT_TMA_M,
        tma_n_heads: int = DEFAULT_TMA_HEADS,
        tma_alpha_init: float = 0.0,
        fusion_mode: str = "residual",
        residual_fusion_hidden_mult: int = 2,
        preserve_kd_enabled: bool = False,
        gate_after_delta: bool = False,
        use_official_internvl_extraction: bool = False,
    ):
        super().__init__()

        self.injection_layers = (
            list(injection_layers)
            if injection_layers is not None
            else DEFAULT_INJECTION_LAYERS
        )
        self.d_model = d_model
        self.tma_m = tma_m
        self.fusion_mode = fusion_mode
        self.residual_fusion_hidden_mult = residual_fusion_hidden_mult

        # 1. GroundingDINO backbone (always frozen)
        self.grounding_dino = grounding_dino
        self._freeze_grounding_dino()

        # 2. InternVL Feature Extractor (always frozen)
        self.feature_extractor = InternVLFeatureExtractor(
            model_path=internvl_path,
            extract_layer=extract_layer,
            extract_layers=extract_layers,
            layer_fusion=layer_fusion,
            freeze=True,
            use_official_prompt_extraction=use_official_internvl_extraction,
        )

        self.mllm_hidden_dim = (
            mllm_hidden_dim or self.feature_extractor.llm_hidden_dim
        )

        # 3. Inject TMA wrappers into decoder
        self._inject_augmenters(
            d_model,
            self.mllm_hidden_dim,
            tma_m,
            tma_n_heads,
            tma_alpha_init,
            fusion_mode,
            residual_fusion_hidden_mult,
            preserve_kd_enabled,
            gate_after_delta,
        )

        print(
            f"[ThinkDet TMA] injection_layers={self.injection_layers} "
            f"M={tma_m} n_heads={tma_n_heads} fusion={fusion_mode}"
        )
        self._print_param_summary()

    # ── Setup ─────────────────────────────────────────────────────────

    def _freeze_grounding_dino(self):
        for param in self.grounding_dino.parameters():
            param.requires_grad = False

    def _get_decoder(self):
        model = self.grounding_dino
        if hasattr(model, 'model'):
            model = model.model
        if hasattr(model, 'transformer'):
            return model.transformer.decoder
        if hasattr(model, 'decoder'):
            return model.decoder
        raise AttributeError("Cannot find decoder in GroundingDINO model")

    def _inject_augmenters(
        self,
        d_model,
        mllm_hidden_dim,
        tma_m,
        tma_n_heads,
        tma_alpha_init,
        fusion_mode,
        residual_fusion_hidden_mult,
        preserve_kd_enabled,
        gate_after_delta,
    ):
        """Replace specified decoder layers with ThinkDetDecoderLayer wrappers."""
        decoder = self._get_decoder()
        num_layers = len(decoder.layers)
        self.adapted_layers: List[ThinkDetDecoderLayer] = []

        for layer_idx in self.injection_layers:
            if layer_idx >= num_layers:
                print(
                    f"[ThinkDet TMA] WARNING: layer {layer_idx} >= {num_layers}, skipping"
                )
                continue

            augmenter = ThinkDetTextAugmenter(
                mllm_hidden_dim=mllm_hidden_dim,
                d_model=d_model,
                M=tma_m,
                n_heads=tma_n_heads,
                alpha_init=tma_alpha_init,
            )
            wrapped = ThinkDetDecoderLayer(
                original_layer=decoder.layers[layer_idx],
                augmenter=augmenter,
                fusion_mode=fusion_mode,
                residual_hidden_mult=residual_fusion_hidden_mult,
                preserve_kd_enabled=preserve_kd_enabled,
                gate_after_delta=gate_after_delta,
            )
            decoder.layers[layer_idx] = wrapped
            self.adapted_layers.append(wrapped)

            print(f"[ThinkDet TMA] Injected TextAugmenter at decoder layer {layer_idx}")

    # ── H_vlm dispatch ────────────────────────────────────────────────

    def _set_h_vlm(self, h_vlm: torch.Tensor):
        for layer in self.adapted_layers:
            layer.set_h_vlm(h_vlm)

    def _clear_h_vlm(self):
        for layer in self.adapted_layers:
            layer.clear_h_vlm()

    # ── Forward ───────────────────────────────────────────────────────

    def forward(self, images, text_queries, dino_inputs, force_gate0: bool = False):
        """
        Args:
            images:       [B, 3, 448, 448] for InternVL
            text_queries: List[str] passed to InternVL
            dino_inputs:  Dict with GroundingDINO inputs
                          (samples, captions, ...)
            force_gate0:  If True, zero out all adapter gates so the forward
                          pass is equivalent to the frozen baseline. Used for
                          the KD reference forward inside the training loop.

        Returns:
            outputs: GroundingDINO output dict
            aux:     Dict with H_vlm info for logging
        """
        # Pure baseline mode: no adapted layers, skip InternVL entirely
        if not self.adapted_layers:
            outputs = self.grounding_dino(**dino_inputs)
            return outputs, {'h_vlm': None}

        # force_gate0: skip InternVL (gate is 0, so h_vlm doesn't matter)
        if force_gate0:
            outputs = self.grounding_dino(**dino_inputs)
            return outputs, {'h_vlm': None, 'force_gate0': True}

        # Extract H_vlm from InternVL (single forward, no delta)
        selected = self.feature_extractor.extract_selected_layers(images, text_queries)
        if len(self.feature_extractor.extract_layers) == 1:
            h_vlm = selected[self.feature_extractor.extract_layer]
        else:
            h_vlm = self.feature_extractor.fuse_selected_layers(selected, weights=None)
        # h_vlm: [B, 256, mllm_hidden_dim]

        # Push H_vlm to each adapted decoder layer
        self._set_h_vlm(h_vlm)

        # Run GroundingDINO — adapted layers augment memory_text internally
        outputs = self.grounding_dino(**dino_inputs)

        self._clear_h_vlm()

        # Collect gate values (tanh(alpha)) from each adapted layer.
        # tanh keeps the adapter exactly closed at alpha=0; the residual
        # fuser also starts as an exact identity on memory_text.
        gates = [
            float(torch.tanh(layer.augmenter.alpha).item())
            for layer in self.adapted_layers
        ]
        preserve_kd_losses = [
            layer.last_preserve_kd_loss
            for layer in self.adapted_layers
            if getattr(layer, "last_preserve_kd_loss", None) is not None
        ]

        fusion_weights = None
        if getattr(self.feature_extractor, "layer_fusion_logits", None) is not None:
            fusion_weights = (
                torch.softmax(self.feature_extractor.layer_fusion_logits.detach(), dim=0)
                .cpu()
                .tolist()
            )

        aux = {
            'h_vlm': h_vlm,
            'h_vlm_norm_mean': float(h_vlm.flatten(1).norm(dim=1).mean().item()),
            'gate_values': gates,
            'gate_mean': float(sum(gates) / len(gates)) if gates else 0.0,
            'pre_adapter_kd_losses': preserve_kd_losses,
            'layer_fusion_weights': fusion_weights,
            # backward-compat alias
            'alpha_values': gates,
            'alpha_mean': float(sum(gates) / len(gates)) if gates else 0.0,
        }
        return outputs, aux

    # ── Training Stages ───────────────────────────────────────────────

    def set_tma_only(self):
        """Stage 1: Only TextAugmenters trainable. DINO + InternVL frozen."""
        for param in self.parameters():
            param.requires_grad = False
        for layer in self.adapted_layers:
            for param in layer.tma_parameters():
                param.requires_grad = True
        for param in self.feature_extractor.get_fusion_params():
            param.requires_grad = True
        self._print_param_summary("Stage 1 — TMA only")

    def set_tma_and_heads(self):
        """Stage 2: TextAugmenters + detection heads trainable."""
        self.set_tma_only()
        self._unfreeze_detection_heads()
        self._print_param_summary("Stage 2 — TMA + heads")

    def _unfreeze_detection_heads(self):
        model = self.grounding_dino
        if hasattr(model, 'model'):
            model = model.model
        for name in ['class_embed', 'bbox_embed', 'enc_output',
                     'enc_score_head', 'enc_bbox_head']:
            module = getattr(model, name, None)
            if module is not None:
                for param in module.parameters():
                    param.requires_grad = True

    def _print_param_summary(self, label: str = ""):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        pct = 100 * trainable / total if total > 0 else 0
        tag = f" [{label}]" if label else ""
        print(f"[ThinkDet TMA]{tag} {trainable:,}/{total:,} params trainable ({pct:.2f}%)")

    def get_augmenter_params(self):
        params = []
        for layer in self.adapted_layers:
            params.extend(layer.tma_parameters())
        params.extend(self.feature_extractor.get_fusion_params())
        return params
