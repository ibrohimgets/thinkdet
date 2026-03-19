"""
ThinkDet TMA - Modified GroundingDINO Decoder Layer

Wraps a GroundingDINO decoder layer and fuses InternVL-derived summary tokens
into `memory_text` before the original layer runs.

Supports two fusion modes:
  - `concat`   : append TMA tokens to memory_text (legacy behavior)
  - `residual` : baseline-safe delta fusion with zero-init residual branch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ResidualTextMemoryFusion(nn.Module):
    """
    Baseline-safe fusion of TMA summary tokens into memory_text.

    Queries come from the existing detector text memory and keys/values come from
    TMA tokens. The output passes through a zero-initialized residual MLP, so the
    module starts as an exact identity on `memory_text`.
    """

    def __init__(self, d_model: int, n_heads: int = 8, hidden_mult: int = 2):
        super().__init__()
        hidden_dim = max(d_model, int(d_model * hidden_mult))
        self.q_norm = nn.LayerNorm(d_model)
        self.kv_norm = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            batch_first=True,
        )
        self.delta_mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
        )
        self._init_weights()

    def _init_weights(self):
        # Start as exact baseline behavior (delta = 0).
        last = self.delta_mlp[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(self, memory_text: torch.Tensor, aug_tokens: torch.Tensor) -> torch.Tensor:
        q = self.q_norm(memory_text)
        kv = self.kv_norm(aug_tokens)
        delta, _ = self.cross_attn(q, kv, kv)
        delta = self.delta_mlp(delta)
        return memory_text + delta


class ThinkDetDecoderLayer(nn.Module):
    """
    Drop-in replacement for a GroundingDINO decoder layer.

    Augments memory_text with InternVL summary tokens before calling
    the original layer. Supports either concatenation (legacy) or a
    baseline-preserving residual fusion path.

    Args:
        original_layer: GroundingDINO DeformableTransformerDecoderLayer
        augmenter:      ThinkDetTextAugmenter instance
        fusion_mode:    'concat' or 'residual'
    """

    def __init__(
        self,
        original_layer,
        augmenter,
        fusion_mode: str = "concat",
        residual_hidden_mult: int = 2,
        preserve_kd_enabled: bool = False,
    ):
        super().__init__()
        self.original_layer = original_layer
        self.augmenter = augmenter
        self.h_vlm: Optional[torch.Tensor] = None
        self.fusion_mode = fusion_mode
        self.preserve_kd_enabled = preserve_kd_enabled
        self.last_preserve_kd_loss: Optional[torch.Tensor] = None

        if fusion_mode not in {"concat", "residual"}:
            raise ValueError(f"Unsupported fusion_mode={fusion_mode!r}")

        if self.fusion_mode == "residual":
            d_model = getattr(augmenter, "d_model", None)
            if d_model is None:
                d_model = augmenter.cross_attn.embed_dim
            n_heads = getattr(augmenter.cross_attn, "num_heads", 8)
            self.residual_fuser = ResidualTextMemoryFusion(
                d_model=d_model,
                n_heads=n_heads,
                hidden_mult=residual_hidden_mult,
            )
        else:
            self.residual_fuser = None

    def forward(
        self,
        tgt,
        tgt_query_pos=None,
        tgt_query_sine_embed=None,
        tgt_key_padding_mask=None,
        tgt_reference_points=None,
        memory_text=None,
        text_attention_mask=None,
        memory=None,
        memory_key_padding_mask=None,
        memory_level_start_index=None,
        memory_spatial_shapes=None,
        memory_pos=None,
        self_attn_mask=None,
        cross_attn_mask=None,
    ):
        self.last_preserve_kd_loss = None

        # Augment memory_text BEFORE the original layer runs
        if self.h_vlm is not None and memory_text is not None:
            aug = self.augmenter(self.h_vlm)                   # [B, M, d_model]
            if self.fusion_mode == "concat":
                B, M = aug.shape[0], aug.shape[1]
                memory_text = torch.cat([memory_text, aug], dim=1)  # [B, T+M, d_model]

                if text_attention_mask is not None:
                    # False = attend to token; extend mask to cover aug tokens
                    aug_mask = torch.zeros(
                        B, M, dtype=torch.bool, device=aug.device
                    )
                    text_attention_mask = torch.cat(
                        [text_attention_mask, aug_mask], dim=1
                    )
            else:
                pre_text = memory_text
                memory_text = self.residual_fuser(memory_text, aug)

                if self.preserve_kd_enabled:
                    # Optional baseline-preservation KD target:
                    # keep the adapted text memory close to the pre-adapter state.
                    target = pre_text.detach()
                    diff2 = (memory_text - target).pow(2)
                    if text_attention_mask is not None:
                        valid = (~text_attention_mask.bool()).to(diff2.dtype).unsqueeze(-1)
                        diff2 = diff2 * valid
                        denom = valid.sum().clamp_min(1.0) * diff2.shape[-1]
                        self.last_preserve_kd_loss = diff2.sum() / denom
                    else:
                        self.last_preserve_kd_loss = diff2.mean()

        # Run original DINO decoder layer with enriched memory_text
        tgt = self.original_layer(
            tgt=tgt,
            tgt_query_pos=tgt_query_pos,
            tgt_query_sine_embed=tgt_query_sine_embed,
            tgt_key_padding_mask=tgt_key_padding_mask,
            tgt_reference_points=tgt_reference_points,
            memory_text=memory_text,
            text_attention_mask=text_attention_mask,
            memory=memory,
            memory_key_padding_mask=memory_key_padding_mask,
            memory_level_start_index=memory_level_start_index,
            memory_spatial_shapes=memory_spatial_shapes,
            memory_pos=memory_pos,
            self_attn_mask=self_attn_mask,
            cross_attn_mask=cross_attn_mask,
        )

        return tgt

    def set_h_vlm(self, h_vlm: torch.Tensor):
        self.h_vlm = h_vlm

    def clear_h_vlm(self):
        self.h_vlm = None

    def tma_parameters(self):
        params = list(self.augmenter.parameters())
        if self.residual_fuser is not None:
            params.extend(self.residual_fuser.parameters())
        return params
