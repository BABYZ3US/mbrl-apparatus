"""LatentTransformer — a temporal latent encode-decode window model that REPLACES
the external imagination horizon with an internal half-window (PM 2026-06-21).

Given the past+current half-window  [z_{k-h/2} .. z_k]  (length L_in = h/2 + 1)
it predicts the future half-window  [z_k .. z_{k+h/2}]   (length L_out = h/2 + 1).
The internal `half_window` h/2 IS the model's horizon: nothing outside this module
decides how far to imagine. The k+1 slot of the predicted window is returned
separately as `out` — the single "output vector" the dynamics operator (tanh) and
policy operator (sigmoid) act on DOWNSTREAM; this module applies no operator.

Encode-decode approach (encoder-only + learned output queries):
  • The input window is embedded (in_proj) + positional-encoded and run through a
    stack of self-attention EncoderLayers (the ENCODE pass) → memory.
  • L_out learned "output query" tokens (one per future slot), also positional-
    encoded, are CONCATENATED after the encoded memory and run through the SAME
    layer stack a second time (the DECODE pass) so the queries cross-attend to the
    encoded window through ordinary self-attention; out_proj on the query slots
    yields the predicted future window. This is the cleanest faithful encoder-only
    form of the SequencePlanner query-slot pattern (planner.py) — a learned
    decoding set rather than a separate nn.TransformerDecoder — and keeps a single
    EncoderLayer definition whose attention is fully exposable.

Attention exposure (REQUIRED for error attribution): the EncoderLayer is hand-
written around nn.MultiheadAttention called with need_weights=True,
average_attn_weights=False, so each layer hands back per-head weights
[B, nhead, S, S]. We capture them ONLY on the encode pass (the L_in×L_in
state↔state map the trainer attributes prediction error to) and stack across
layers → [B, layers, nhead, L_in, L_in]. When need_attn=False we skip weight
computation entirely (cheap path) and return None.

House rules: deterministic given input (no RNG beyond optional dropout, default
0.0 ⇒ bitwise-resume safe); no global state; norm_first/GELU to match the planner.
"""
from __future__ import annotations

import math

import torch
from torch import nn, Tensor


class _SinusoidalPositionalEncoding(nn.Module):
    """Fixed (non-learned, deterministic) sinusoidal positional encoding, added to
    a [B, S, d_model] embedding. Buffer-backed so it moves with .to()/.cuda() and
    carries no parameters. Built once up to `max_len`; sliced to the live S."""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)      # (max_len,1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32)
                        * (-math.log(10000.0) / d_model))                  # (d_model/2,)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))                        # (1,max_len,d)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.pe[:, : x.size(1)]                                 # (B,S,d)


class _EncoderLayer(nn.Module):
    """Pre-norm (norm_first) self-attention + FFN block, written explicitly around
    nn.MultiheadAttention so per-head attention weights are returnable — the opaque
    nn.TransformerEncoder averages/hides them. Matches the planner's
    norm_first=True, GELU, dim_feedforward=4*d_model style (planner.py)."""

    def __init__(self, d_model: int, nhead: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout,
                                          batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor, *, need_attn: bool = False):
        """x: [B, S, d_model] → (out [B, S, d_model], attn [B, nhead, S, S] | None).
        Pre-norm residuals: x = x + Attn(LN(x)); x = x + FF(LN(x))."""
        h = self.norm1(x)
        # average_attn_weights=False ⇒ per-head weights [B, nhead, S, S]; only ask
        # for weights on the attribution path to keep need_attn=False cheap.
        a, w = self.attn(h, h, h, need_weights=need_attn,
                         average_attn_weights=False)
        x = x + self.drop(a)
        x = x + self.ff(self.norm2(x))
        return x, w


class LatentTransformer(nn.Module):
    """Encode a window [z_{k-h/2}..z_k], decode the future window [z_k..z_{k+h/2}];
    expose per-layer per-head attention for state↔state error attribution.

    forward(z_window, *, need_attn=False) -> dict with keys:
      'pred': [B, L_out, latent_dim]   future window [z_k..z_{k+h/2}], L_out = h/2+1
      'out':  [B, latent_dim]          the next-step (k+1) vector the operators use
      'attn': [B, layers, nhead, L_in, L_in] if need_attn else None  (encode pass)
    """

    def __init__(self, latent_dim: int, d_model: int = 128, nhead: int = 4,
                 layers: int = 2, half_window: int = 8, dropout: float = 0.0):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.d_model = int(d_model)
        self.nhead = int(nhead)
        self.layers = int(layers)
        self.half_window = int(half_window)
        # window lengths: L_in = L_out = half_window + 1 (endpoints inclusive; both
        # windows share the anchor z_k, so they are the same length).
        self.L_in = self.half_window + 1
        self.L_out = self.half_window + 1

        self.in_proj = nn.Linear(self.latent_dim, self.d_model)
        # max_len covers the encode pass (L_in) and the concatenated decode pass
        # (L_in + L_out); +1 slack for safety.
        self.pos = _SinusoidalPositionalEncoding(
            self.d_model, max_len=self.L_in + self.L_out + 1)
        # L_out learned output-query tokens, one per future slot (planner.py style).
        self.out_query = nn.Parameter(torch.randn(self.L_out, self.d_model) * 0.02)
        self.enc_layers = nn.ModuleList(
            [_EncoderLayer(self.d_model, self.nhead, dropout) for _ in range(self.layers)])
        self.out_proj = nn.Linear(self.d_model, self.latent_dim)

    def forward(self, z_window: Tensor, *, need_attn: bool = False) -> dict:
        # z_window: [B, L_in, latent_dim]
        B, L, _ = z_window.shape
        assert L == self.L_in, (
            f"expected window length L_in={self.L_in} (half_window+1), got {L}")

        # ---- ENCODE: embed + positional-encode the input window, run the stack ----
        h = self.pos(self.in_proj(z_window))                       # (B, L_in, d)
        attn_stack = [] if need_attn else None
        for layer in self.enc_layers:
            h, w = layer(h, need_attn=need_attn)                   # h: (B, L_in, d)
            if need_attn:
                attn_stack.append(w)                               # w: (B, nhead, L_in, L_in)
        memory = h                                                 # encoded window

        # ---- DECODE: L_out learned queries cross-attend to memory via the SAME
        # layer stack (self-attention over [memory ; queries]); read the query slots.
        q = self.out_query.unsqueeze(0).expand(B, -1, -1)          # (B, L_out, d)
        q = self.pos(q)                                            # add positions (shares table)
        dec = torch.cat([memory, q], dim=1)                        # (B, L_in+L_out, d)
        for layer in self.enc_layers:
            dec, _ = layer(dec, need_attn=False)                   # decode attn not attributed
        pred_tokens = dec[:, self.L_in:]                           # (B, L_out, d) query slots
        pred = self.out_proj(pred_tokens)                          # (B, L_out, latent_dim)

        # 'out' = predicted next-step vector = the k+1 slot of the future window.
        # pred[:, 0] is the anchor z_k (start of [z_k..z_{k+h/2}]); index 1 is k+1.
        out = pred[:, 1]                                           # (B, latent_dim)

        attn = None
        if need_attn:
            # stack layers → [B, layers, nhead, L_in, L_in]
            attn = torch.stack(attn_stack, dim=1)
        return {"pred": pred, "out": out, "attn": attn}
