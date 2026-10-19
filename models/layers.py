# models/layers.py
"""Fusion layers.

Fixed relative to the thesis code (audit M14 / M15 / M17):
  * no ``os.chdir`` to a hard-coded path and no global ``torch.manual_seed``
    at import time (seeding is done once in run.py);
  * the legacy Devign ``Conv`` / ``Net`` classes (dead code) are removed;
  * ``GraphCodeFusion`` can run without the gate (plain concatenation, for the
    no-gate ablation), has a gate temperature, and returns the gate weights
    instead of stashing them on the module.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionwiseFFN(nn.Module):
    def __init__(self, dim, hidden_dim=None, dropout=0.1):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim), nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class AttentionBlock(nn.Module):
    """Multi-head attention usable as self- or cross-attention (batch_first)."""

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads,
                                          dropout=dropout, batch_first=True)

    def forward(self, query, key, value, key_padding_mask=None, need_weights=False):
        out, w = self.attn(query=query, key=key, value=value,
                           key_padding_mask=key_padding_mask, need_weights=need_weights)
        return (out, w) if need_weights else out


class FusionBlock(nn.Module):
    """One BDCA layer: code<-graph and graph<-code cross-attention, each with
    residual + LayerNorm, followed by a position-wise FFN per stream."""

    def __init__(self, dim, num_heads, ffn_dim, dropout):
        super().__init__()
        self.cross_attn_code = AttentionBlock(dim, num_heads, dropout)
        self.norm2_code = nn.LayerNorm(dim)
        self.cross_attn_graph = AttentionBlock(dim, num_heads, dropout)
        self.norm2_graph = nn.LayerNorm(dim)
        self.ffn_code = PositionwiseFFN(dim, ffn_dim, dropout)
        self.norm3_code = nn.LayerNorm(dim)
        self.ffn_graph = PositionwiseFFN(dim, ffn_dim, dropout)
        self.norm3_graph = nn.LayerNorm(dim)

    def forward(self, code, graph, code_padding_mask, graph_padding_mask):
        c_cross = self.cross_attn_code(query=code, key=graph, value=graph, key_padding_mask=graph_padding_mask)
        code_next = self.norm2_code(code + c_cross)
        g_cross = self.cross_attn_graph(query=graph, key=code, value=code, key_padding_mask=code_padding_mask)
        graph_next = self.norm2_graph(graph + g_cross)
        code, graph = code_next, graph_next
        code = self.norm3_code(code + self.ffn_code(code))
        graph = self.norm3_graph(graph + self.ffn_graph(graph))
        return code, graph


def masked_mean(h, mask):
    """h: [B, L, D]; mask: [B, L] with 1 = valid. Returns [B, D]."""
    if mask is None:
        return h.mean(dim=1)
    m = mask.to(h.dtype).unsqueeze(-1)
    return (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-6)


class GraphCodeFusion(nn.Module):
    """Project both modalities to ``proj_dim``, run ``attn_layers`` FusionBlocks,
    pool, gate (optional) and map to ``fusion_out_dim``.

    forward() returns (fused [B, fusion_out_dim], gate_weights [B, 2] or None).
    """

    def __init__(self, code_dim: int, graph_dim: int, proj_dim: int = 200, num_heads: int = 4,
                 ffn_hidden_dim: int = None, fusion_out_dim: int = 200, dropout: float = 0.1,
                 attn_layers: int = 3, use_gate: bool = True, gate_temperature: float = 1.0):
        super().__init__()
        ffn_hidden_dim = ffn_hidden_dim or proj_dim * 4
        self.use_gate = use_gate
        self.gate_temperature = float(gate_temperature)
        self.proj_code = nn.Linear(code_dim, proj_dim)
        self.proj_graph = nn.Linear(graph_dim, proj_dim)
        self.layers = nn.ModuleList([FusionBlock(proj_dim, num_heads, ffn_hidden_dim, dropout)
                                     for _ in range(attn_layers)])
        self.gate_layer = nn.Linear(proj_dim * 2, 2) if use_gate else None   # 2 = modalities (code, graph), not classes
        fused_dim = proj_dim * 2
        self.ffn_final = PositionwiseFFN(fused_dim, hidden_dim=ffn_hidden_dim, dropout=dropout)
        self.norm_final = nn.LayerNorm(fused_dim)
        self.linear_out = nn.Linear(fused_dim, fusion_out_dim)
        self.last_weights = None  # {"code": float, "graph": float}; kept for logging

    def forward(self, code_emb, graph_emb, code_mask=None, graph_mask=None):
        code_h = self.proj_code(code_emb)
        graph_h = self.proj_graph(graph_emb)
        code_padding_mask = (code_mask == 0) if code_mask is not None else None
        graph_padding_mask = (graph_mask == 0) if graph_mask is not None else None
        for layer in self.layers:
            code_h, graph_h = layer(code_h, graph_h, code_padding_mask, graph_padding_mask)
        code_vec = masked_mean(code_h, code_mask)
        graph_vec = masked_mean(graph_h, graph_mask)

        gate_weights = None
        if self.use_gate:
            gate_logits = self.gate_layer(torch.cat([code_vec, graph_vec], dim=-1)) / self.gate_temperature
            gate_weights = F.softmax(gate_logits, dim=-1)                       # [B, 2]
            w_code, w_graph = gate_weights[:, 0:1], gate_weights[:, 1:2]
            fused = torch.cat([w_code * code_vec, w_graph * graph_vec], dim=-1)
            self.last_weights = {"code": float(w_code.mean().detach()), "graph": float(w_graph.mean().detach())}
        else:
            fused = torch.cat([code_vec, graph_vec], dim=-1)
            self.last_weights = None

        fused = self.ffn_final(fused)
        fused = self.norm_final(fused)
        fused = self.linear_out(fused)
        return fused, gate_weights
