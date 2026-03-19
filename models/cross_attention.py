"""
ThinkDet TMA - Text Memory Augmenter

Summarizes InternVL H_vlm into M learnable text-space tokens.

This module only produces aligned summary tokens. How those tokens are fused
into GroundingDINO's text memory is handled by the decoder-layer wrapper
(`concat` vs baseline-safe `residual` fusion).
"""

import torch
import torch.nn as nn


class ThinkDetTextAugmenter(nn.Module):
    """
    Projects InternVL visual features into M summary tokens
    compatible with DINO's memory_text (d_model=256).

    Flow:
        H_vlm [B, N_vis, mllm_dim]
          -> MLP(mllm_dim -> hidden -> d_model)
          -> LayerNorm
          -> CrossAttn(M learnable queries)  pool N_vis -> M tokens
          -> LayerNorm
          -> aug_tokens [B, M, d_model]

    The M=8 learnable queries learn to extract the most detection-
    relevant summary from InternVL's 256 visual tokens.

    Args:
        mllm_hidden_dim: InternVL LLM hidden dim (1024 for InternVL3.5-1B)
        d_model:         DINO d_model (256)
        M:               Number of summary tokens to produce (default 8)
        n_heads:         Attention heads for pooling cross-attention
    """

    def __init__(
        self,
        mllm_hidden_dim: int = 1024,
        d_model: int = 256,
        M: int = 8,
        n_heads: int = 8,
        alpha_init: float = 0.0,
        proj_hidden_mult: int = 2,
    ):
        super().__init__()
        self.M = M
        self.d_model = d_model
        hidden_dim = max(d_model, int(d_model * proj_hidden_mult))

        # Project InternVL dim -> DINO dim with a slightly richer MLP.
        self.proj_in = nn.Linear(mllm_hidden_dim, hidden_dim, bias=False)
        self.proj_act = nn.GELU()
        self.proj_out = nn.Linear(hidden_dim, d_model, bias=False)
        self.kv_norm = nn.LayerNorm(d_model)

        # Learnable query vectors that pool over projected H_vlm
        self.queries = nn.Parameter(torch.zeros(M, d_model))

        # Cross-attention: queries attend to projected H_vlm
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            batch_first=True,
        )
        self.out_norm = nn.LayerNorm(d_model)

        # Learnable gate. Initialise to alpha_init (default 0.0 → tanh(0)=0 = closed).
        # Effective contribution = tanh(alpha) ∈ (-1, 1).
        # Starting at 0 means the adapter is a perfect identity at init.
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.queries, std=0.02)
        nn.init.xavier_uniform_(self.proj_in.weight, gain=0.1)
        nn.init.xavier_uniform_(self.proj_out.weight, gain=0.1)

    @property
    def gate_value(self) -> float:
        """Effective gate strength in (-1, 1). Useful for logging."""
        return float(torch.tanh(self.alpha).item())

    def forward(self, h_vlm: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_vlm: [B, N_vis, mllm_hidden_dim]
        Returns:
            aug_tokens: [B, M, d_model]  scaled by tanh(alpha) ∈ (-1, 1)
        """
        B = h_vlm.shape[0]

        # Project and normalize
        kv = self.proj_out(self.proj_act(self.proj_in(h_vlm)))
        kv = self.kv_norm(kv)                                 # [B, N_vis, d_model]

        # Expand learnable queries over batch
        q = self.queries.unsqueeze(0).expand(B, -1, -1)       # [B, M, d_model]

        # Attend: M query tokens attend over N_vis InternVL tokens
        aug_tokens, _ = self.cross_attn(q, kv, kv)            # [B, M, d_model]
        aug_tokens = self.out_norm(aug_tokens)

        # Scale by tanh(alpha): starts at 0 (closed gate), bounded to (-1, 1)
        gate = torch.tanh(self.alpha)
        aug_tokens = gate * aug_tokens

        return aug_tokens
